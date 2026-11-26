"""Build the pinned role in a new CPU-only container build directory."""
import hashlib,json,os,re,shutil,subprocess as sp,sys,time,traceback
from pathlib import Path
work=Path('/guide');source=Path('/workspace/xllm-mt-main')
parent=source/'build/cmake.linux-x86_64-cpython-310'
build=Path('/tmp/guide32k-pytorch-cmake')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 with (work/name).open('x') as f:json.dump(v,f,indent=2)
def go():
 assert not list(Path('/dev').glob('mtgpu*')) and not list(Path('/dev/dri').glob('renderD*'))
 meta=json.loads((work/'spec.json').read_text())
 expected=meta['sources_sha256']
 for name,digest in expected.items():
  overlay=work/'src'/name
  assert sha(overlay)==digest
  out=source/name;out.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists() or sha(out)!=digest:shutil.copyfile(overlay,out)
 sys.path.insert(0,str(work))
 from pytorch_musa_paths import package_paths
 py=package_paths();import torch
 assert torch.compiled_with_cxx11_abi()
 save('torch-provenance.json',dict(paths=py,torch=torch.__version__,python=sys.executable,abi=torch.compiled_with_cxx11_abi(),provider_sha={k:sha(p) for k,p in [('config',Path(py['torch'])/'share/cmake/Torch/TorchConfig.cmake'),('libtorch_cpu',Path(py['torch'])/'lib/libtorch_cpu.so'),('musa',Path(py['torch_musa'])/'lib/libmusa_python.so')]}))
 # The reference graph already links the same PyTorch wheel. Keep its
 # command spellings to reuse unaffected objects from the pinned role image.
 # All translation units depending on the changed generated class must rebuild.
 build=parent
 graph=parent/'build.sg438.ninja'
 final=source/'build/lib.linux-x86_64-cpython-310/xllm/xllm'
 old_sha=sha(final)
 old_graph=graph.read_text()
 for name in ('torch','torch_musa'):
  pkg=Path('/usr/local/lib/python3.10/dist-packages')/name
  assert pkg.resolve()==Path(py[name]) and (str(pkg) in old_graph or py[name] in old_graph)
 assert Path('/expt/builds/native_sg438_control/placement_new/device_placement_new.h').is_file()
 save('parent-graph.json',dict(initial_elf=old_sha,graph_sha=sha(graph),build=str(parent),torch=py))
 env=dict(os.environ,MUSA_HOME='/usr/local/musa',CUDA_HOME='/usr/local/musa',MUSAMAPPING_PATH='/usr/local/musa/tools/musamapping',
          LD_LIBRARY_PATH=py['torch_musa']+'/lib:'+py['torch']+'/lib:/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64:/usr/local/openmpi/lib',
          LD_PRELOAD='',PATH=str(build/'musa_link_wrapper')+':/opt/venv/bin:/usr/local/musa/bin:/usr/bin:/bin',
          TVM_FFI_LIB_DIR='/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib',LIBRARY_PATH=str(build/'xllm/core/layers/musa')+':/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64',
          PYTORCH_INSTALL_PATH=py['torch'],LIBTORCH_ROOT=py['torch'],Torch_DIR=py['torch']+'/share/cmake/Torch',TorchMusa_DIR=py['cmake']+'/TorchMusa',
          MUSA_VISIBLE_DEVICES='-1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MAX_JOBS=str(meta['jobs']))
 args=['bash',str(source/'scripts/ninja_safe.sh'),str(build),'-f',graph.name,'-j'+str(meta['jobs'])]
 for f in (graph,build/'CMakeCache.txt',build/'CMakeFiles/rules.ninja'):shutil.copyfile(f,work/f.name)
 dry=sp.run(args+['-n','-v','xllm'],cwd=source,env=env,capture_output=True,text=True)
 (work/'dry.log').write_text(dry.stdout+dry.stderr)
 assert dry.returncode==0 and 'Re-running CMake' not in dry.stdout,(dry.returncode,dry.stderr[-900:])
 commands=[l for l in dry.stdout.splitlines() if re.match(r'^\[\d+/\d+\]',l)]
 assert any('completion.proto' in l and 'protoc' in l for l in commands)
 assert any('request_params.cpp' in l and ' -c ' in l for l in commands)
 assert any('completion_service_impl.cpp' in l and ' -c ' in l for l in commands)
 assert any('completion.pb.cc' in l and ' -c ' in l for l in commands)
 assert any('/torch/include' in l for l in commands)
 save('dry-run.json',dict(actions=len(commands),native_compiles=sum(' -c ' in l for l in commands),graph=sha(graph),jobs=meta['jobs']))
 with (work/'compile.log').open('xb') as log:
  proc=sp.run(args+['-v','xllm'],cwd=source,env=env,stdout=log,stderr=sp.STDOUT)
 assert proc.returncode==0,proc.returncode
 assert final.is_file() and sha(final)!=old_sha,str(final)
 dest=work/'bin/xllm';dest.parent.mkdir();shutil.copy2(final,dest)
 save('artifact.json',dict(sha256=sha(dest),bytes=dest.stat().st_size,ldd=sp.run(['ldd',str(dest)],env=env,text=True,capture_output=True).stdout,elf=sp.check_output(['readelf','-d',str(dest)],text=True)))
try:
 go();rc=0
except BaseException:
 traceback.print_exc();rc=1
save('finished.json',dict(rc=rc,at=time.time()));sys.exit(rc)
