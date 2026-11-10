#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Start one session child; every pre-registration failure cleans that child."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--env',action='append',default=[]);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 if not cmd:raise SystemExit('missing command')
 expected=str(Path(cmd[0]).resolve(strict=True));env=os.environ.copy()
 for value in a.env:
  k,v=value.split('=',1);env[k]=v
 a.run.mkdir(parents=True,exist_ok=True);child=None
 try:
  with (a.run/(a.name+'.log')).open('ab') as log:
   child=subprocess.Popen(cmd,cwd=a.run,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   deadline=time.monotonic()+15
   while True:
    try:
     proc=Path('/proc')/str(child.pid);exe=os.readlink(proc/'exe');stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
     if exe==expected and stat[0]!='Z':break
    except FileNotFoundError:pass
    if child.poll() is not None:raise RuntimeError(f'{a.name} exited before identity establishment, rc={child.returncode}')
    if time.monotonic()>deadline:raise RuntimeError(f'{a.name}: executable transition timed out')
    time.sleep(.02)
   allowed_env=['XLLM_USE_FA3','XLLM_USE_FA3_DECODE','XLLM_SPEC_VERIFY_LARGE_BATCH_EAGER','XLLM_GDN_DECODE_BACKEND','XLLM_PIECEWISE_CAPTURE_FA3','XLLM_MUSA_PCP_CUSTOM_AR','XLLM_MUSA_POOL_COMPUTE_STREAM','XLLM_MATE_GDN_MTP','XLLM_MATE_GDN_PREFILL','XLLM_PACKED_PREFILL_PIECEWISE','MATE_MUSA_ARCH_LIST','MATE_HOME','MATE_MUBIN_DIR','MATE_WORKSPACE_BASE','LD_LIBRARY_PATH','LD_PRELOAD','MUSA_VISIBLE_DEVICES','MUSA_OPS_DIR','FLASHINFER_OPS_PATH','OMP_NUM_THREADS']
   safe_argv=[arg.split('=',1)[0]+'=<local-lab-default>' if arg.startswith('--internal_api_token=') else arg for arg in cmd]
   rec={'name':a.name,'pid':child.pid,'exe':expected,'starttime':stat[19],'argv':safe_argv,'runtime_env':{k:env[k] for k in allowed_env if k in env}}
   identities=a.run/'identities.json';saved=json.loads(identities.read_text()) if identities.exists() else []
   saved.append(rec);stage=identities.with_suffix('.tmp');stage.write_text(json.dumps(saved,indent=2)+'\n');stage.replace(identities)
   with (a.run/'pids.txt').open('a') as f:f.write(a.name+' '+str(child.pid)+'\n')
   print(json.dumps(rec))
 except BaseException:
  # This exact Popen handle was created here. Never search for names/PIDs to
  # compensate for an absent or failed identity file.
  if child is not None:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     child.kill();child.wait(timeout=10)
   else:child.wait()
  raise
 return 0
if __name__=='__main__':raise SystemExit(main())
