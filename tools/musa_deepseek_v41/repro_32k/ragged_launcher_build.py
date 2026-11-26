"""Build the MP31 MATE S4FP8 segmented launchers on the pinned native 0.2.8 distribution."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
out=Path('/guide')
site=Path('/expt/long_opt_20261119/goal20k200k_20261119/mate_eval_028/native-r2/site')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
assert not list(Path('/dev').glob('mtgpu*')) and not list(Path('/dev/dri').glob('renderD*'))
os.environ.update(MATE_MUSA_ARCH_LIST='3.1',TVM_FFI_MUSA_ARCH_LIST='31',MATE_WORKSPACE_BASE='/guide/isolated-cache',MAX_JOBS='4',PYTHONDONTWRITEBYTECODE='1',MUSA_VISIBLE_DEVICES='-1',CUDA_VISIBLE_DEVICES='-1',TVM_FFI_GPU_BACKEND='musa')
os.environ.pop('MATE_MUBIN_VERIFY_DISABLED',None)
os.environ['MATE_MUBIN_DIR']=str(site/'mate_mubin/mubin')
os.environ['PATH']='/usr/local/musa/bin:'+os.environ['PATH']
sys.path.insert(0,str(site))
started=time.time();rc=1
try:
    import torch,mate
    from importlib import metadata
    from mate.jit.mubin.gemm.gemm_api import _get_gemm_mubin_runtime
    from mate.jit.mubin.gemm.types import MxScaleMode,TensorQuantMode
    from mate.jit.mubin.launcher import gen_mubin_launcher_spec,render_gemm_mubin_launcher
    assert Path(mate.__file__).resolve().parent==site/'mate'
    assert metadata.version('mate')==metadata.version('mate-mubin')=='0.2.8'
    assert sha(site/'mate/gemm.py')=='ce6986c6dbf823c7c3b35228af7d9880e54aa4721a7938074930b719e2ca2dac'
    pkg={i:metadata.version(i) for i in ('torch','torch-musa','apache-tvm-ffi','mate','mate-mubin')}
    save('torch-provenance.json',dict(packages=pkg,source=str(site),python=sys.executable))
    dispatcher=_get_gemm_mubin_runtime()
    maps=Path(dispatcher.module_dir)/'kernel_map.json'
    assert sha(maps)=='ea051c7e23e3ea57eb77140c632097392ceae938ecd6ff3c41272b921e6c8844'
    tools=out/'provider';tools.mkdir()
    rows={}
    for align in (128,256):
        for cache in (False,True):
            aid=dispatcher.get_w4a8_ragged_asm_id(type_a=torch.float8_e4m3fn,type_d=torch.bfloat16,type_scale_b=torch.bfloat16,tme_cache_hint_b=cache,mx_scale=MxScaleMode.NONE,quant_tile=128,alignment_m=align,quant_mode_a=TensorQuantMode.CHANNEL,quant_mode_b=TensorQuantMode.GROUP)
            entry,obj=dispatcher.resolve_kernel_artifact(aid)
            body=render_gemm_mubin_launcher(aid,entry.kernel_name)
            spec=gen_mubin_launcher_spec('gemm',entry.kernel_name,body)
            lib=spec.build(verbose=True)
            name='m'+str(align)+'-hint'+str(int(cache))
            dest=tools/name;dest.mkdir()
            shutil.copy2(lib,dest/'launcher.so');shutil.copy2(obj,dest/'kernel.mubin');(dest/'launcher.mu').write_text(body)
            rows[name]=dict(launch='run_'+entry.kernel_name,library=sha(dest/'launcher.so'),mubin=sha(dest/'kernel.mubin'),source=sha(dest/'launcher.mu'),vendor_object=str(obj))
            save('provider-progress.json',rows)
    save('artifact.json',dict(role='MATE ragged S4FP8, A per-token, B per row/K128',packs=rows,map_sha256=sha(maps)))
    rc=0
except Exception:
    traceback.print_exc()
finally:
    save('finished.json',dict(rc=rc,started=started,ended=time.time()))
sys.exit(rc)
