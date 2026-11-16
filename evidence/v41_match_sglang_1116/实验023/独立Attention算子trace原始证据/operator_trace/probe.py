from pathlib import Path
import json,os,collections
from types import SimpleNamespace
import torch,torch_musa
import v41_verify_graph_inputs_test_module as native
from validate_short_context_attention import _random,_clone
r=Path('/expt/diagnostics/e23_operator_trace_v1')
args=SimpleNamespace(ratio=1,sparse_so='/work/artifacts/native_stage_i/lib/mate/libdeepseek_v41_sparse_h8_bf16.so',indexer_dir='/work/artifacts/native_stage_i/lib/indexer')
torch.manual_seed(2304);torch.set_num_threads(1);device,dtype='musa:0',torch.bfloat16
with torch.inference_mode():
        weights = native.NativeV41AttentionWeights()
        weights.projection = [_random((1280, 5120)), torch.ones(1280, device=device, dtype=dtype), _random((4096, 1280)), _random((512, 5120)), torch.ones(512, device=device, dtype=dtype), _random((1024, 4096)), _random((5120, 1024))]
        weights.sink = torch.zeros(8, device=device)
        weights.compressor_norm = torch.ones(512, device=device, dtype=dtype)
        weights.compressor_kv = _random((512, 5120)).float()
        weights.compressor_gate = _random((512, 5120)).float()
        weights.indexer = [_random((4096, 1280)), _random((32, 5120)), _random((128, 512)), torch.ones(128, device=device, dtype=dtype)]
        config = native.NativeV41AttentionConfig()
        config.mode = (native.NativeV41AttentionMode.SWA if args.ratio == 0
                       else native.NativeV41AttentionMode.FULL)
        config.layer = config.source_layer = 0
        if args.ratio == 0:
            config.source_layer = -1
            weights.indexer = []
        config.ratio, config.heads, config.groups = args.ratio, 8, 1
        if args.ratio == 1:
            weights.compressor_kv = weights.compressor_kv.to(dtype)
            weights.compressor_gate = None
        config.output_rank, config.window = 1024, 128
        config.index_heads, config.index_dim, config.index_topk = 32, 128, 512
        config.short_context_indexer = True
        config.candidate_source, config.candidate_blocks, config.candidate_block_size = 0, 64, 8
        config.epsilon, config.scale = 1e-6, 512 ** -0.5
        config.original_fp8_activation = True
        config.sparse_library_path = args.sparse_so
        config.indexer_score_directory = args.indexer_dir
        config.decode_cache_read_reuse = config.decode_fused_swa_read = True
        layer = native.NativeV41Attention(config, weights)
        angles = torch.arange(1024, device=device).float()[:, None] * torch.pow(10000.0, -torch.arange(0, 64, 2, device=device).float() / 64)[None, :]
        rope = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
        # One-page SWA forces physical ring overwrite; compressed pages are permuted.
        swa = torch.tensor([[3]], device=device, dtype=torch.int32)
        table = torch.tensor([[4, 1, 5, 2, 6, 0, 3, 7]], device=device, dtype=torch.int32)

        config.short_context_indexer=False
        baseline=native.NativeV41Attention(config,weights)
        seed=native.NativeV41AttentionCache()
        seed.swa=torch.zeros(8,128,512,device=device,dtype=dtype)
        seed.key=torch.zeros_like(seed.swa)
        seed.index=torch.zeros(8,128,128,device=device,dtype=dtype)
        baseline.forward(_random((256,5120)),torch.arange(256,device=device),rope,seed,swa,table,table,256,native.NativeV41AttentionRuntime())
        hidden=_random((4,5120));positions=torch.arange(256,260,device=device)
        results=[];outputs=[]
        for name,enabled in [('off',False),('on',True)]:
            config.short_context_indexer=enabled
            layer=native.NativeV41Attention(config,weights)
            caches=[_clone(seed) for _ in range(5)]
            def forward(cache):
                runtime=native.NativeV41AttentionRuntime();runtime.decode_capacity=512
                return layer.forward(hidden,positions,rope,cache,swa,table,table,260,runtime)
            for i in range(2):forward(caches[i])
            torch.musa.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.PrivateUse1],record_shapes=True) as prof:
                for i in range(3):
                    with torch.profiler.record_function(f'E23/{name}/actual_Attention/q4/ratio1/{i}'):
                        output=forward(caches[i+2])
                torch.musa.synchronize()
            outputs.append(output.clone())
            path=r/f'{name}.pt.trace.json';prof.export_chrome_trace(str(path))
            ev=json.loads(path.read_text())['traceEvents'];kernels=[x for x in ev if x.get('ph')=='X' and 'kernel' in x.get('cat','').lower()]
            results.append({'variant':name,'gpu_kernels':len(kernels),'aten_counts':dict(collections.Counter(x['name'] for x in ev if x.get('ph')=='X' and x.get('name','').startswith('aten::'))),'kernel_counts':dict(collections.Counter(x['name'] for x in kernels))})
        assert torch.equal(outputs[0],outputs[1])
        assert results[1]['aten_counts'].get('aten::linear',0)<results[0]['aten_counts'].get('aten::linear',0)
        assert results[1]['gpu_kernels']<results[0]['gpu_kernels']
        (r/'result.json').write_text(json.dumps({'status':'PASS','scope':'Isolated actual production Attention q4 static metadata, outside full Graph for node visibility; no whole-model Amdahl claim','output_exact':True,'results':results},indent=2))
