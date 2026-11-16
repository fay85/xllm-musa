from pathlib import Path
import json,os,collections,sys,statistics
from types import SimpleNamespace
import torch,torch_musa
import v41_verify_graph_inputs_test_module as native
from validate_short_context_attention import _random,_clone
r=Path('/expt/diagnostics/e26_attention_graph_v2'); ratio=int(sys.argv[1])
args=SimpleNamespace(ratio=ratio,sparse_so='/work/artifacts/native_stage_i/lib/mate/libdeepseek_v41_sparse_h8_bf16.so',indexer_dir='/work/artifacts/native_stage_i/lib/indexer')
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
        if args.ratio==2:
            seed.compressor_kv_state=torch.zeros(8,32,512,device=device)
            seed.compressor_score_state=torch.zeros_like(seed.compressor_kv_state)
        baseline.forward(_random((256,5120)),torch.arange(256,device=device),rope,seed,swa,table,table,256,native.NativeV41AttentionRuntime())
        hidden=_random((4,5120));positions=torch.arange(256,260,device=device)
        results=[]
        for name,enabled in ([('off',False)] if ratio==0 else [('off',False),('on',True)]):
            config.short_context_indexer=enabled
            layer=native.NativeV41Attention(config,weights)
            cache=_clone(seed)
            def forward():
                runtime=native.NativeV41AttentionRuntime();runtime.decode_capacity=512
                return layer.forward(hidden,positions,rope,cache,swa,table,table,260,runtime)
            stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(stream):
                for _ in range(2):forward()
                stream.synchronize();graph=torch.musa.MUSAGraph()
                with torch.musa.graph(graph,stream=stream):captured=forward()
                for _ in range(3):graph.replay()
                stream.synchronize();assert torch.isfinite(captured).all().item()
                times=[]
                for _ in range(3):
                    a=torch.musa.Event(enable_timing=True);b=torch.musa.Event(enable_timing=True);a.record(stream)
                    for _ in range(100):graph.replay()
                    b.record(stream);b.synchronize();times.append(a.elapsed_time(b)/100)
                graph.reset()
                stream.synchronize()
            results.append({'ratio':ratio,'variant':name,'graph_ms_samples':times,'graph_median_ms':statistics.median(times),'finite_output':True,'scope':'Synthetic weights; single local Attention graph excludes TP communication and model rank skew. No profiler in this process.'})
            (r/f'ratio{ratio}_result.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results),flush=True)
