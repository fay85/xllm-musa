"""Same-binary OFF/TP-custom A/B, full performance then matched trace."""
from pathlib import Path
import importlib.util
import json
import concurrent.futures
import time

E=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('driver',E.parent/'ToDoList_2/02_代码/g02_ram_sequence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.E=E;m.STATE=E/'04_性能/sequence'
SHA='20376a77ad348ffd889d2592aa982c7a6b35962c0bb3908bbe866d7eca1d8489'
BINARY='/work/artifacts/native_t01_decode_tp_ar/bin/xllm'


def prepare(mode: str, run: str) -> None:
    code=fr"""import pathlib,json,hashlib
r=pathlib.Path({m.R!r});p=r/'artifacts/prepared'/{run!r};p.mkdir(exist_ok=False)
assert hashlib.sha256((r/'artifacts/native_t01_decode_tp_ar/bin/xllm').read_bytes()).hexdigest()=={SHA!r}
for h in ['sj9','sj10']:
 t=(r/'runs/g02safe_device_v1'/h/'role_config.json').read_text().replace('g02safe_device_v1',{run!r}).replace('g02safe-device-v1',{run.replace('_','-')!r}).replace('native_g02_v2_timing/bin/xllm','native_t01_decode_tp_ar/bin/xllm')
 c=json.loads(t);c['binary_sha256']={SHA!r};c['source_manifest']='artifacts/native_t01_decode_tp_ar/manifest.json';c['status']='T01_'+{mode!r}
 for rank in c['ranks']:
  env=rank['environment']
  for k in list(env):
   if k.startswith('XLLM_V41_LIGHT_'):del env[k]
  env['XLLM_MUSA_ENABLE_TP_CUSTOM_AR']='1' if {mode!r}=='CUSTOM' else '0'
 (p/(h+'.json')).write_text(json.dumps(c,indent=2))
print(json.dumps({{'prepared':str(p),'binary_sha256':{SHA!r}}}))
"""
    m.remote('sj9',code)


def launch_client(run: str) -> str:
    source=(E.parent/'ToDoList_2/04_性能/g02_sequence/g02ram_off_v1_client.py').read_text().replace('g02ram_off_v1',run).replace('native_g02_v2_timing/bin/xllm','native_t01_decode_tp_ar/bin/xllm')
    # Optional API log-probability samples use exactly the same prompts on
    # both sides. They are separate correctness probes, not formal timing.
    marker=" gate={'status':'PASS'"
    at=source.index(marker)
    extra=""" prefix_records=[]
 for length in [256,2048]:
  body=json.loads((r/f'artifacts/benchmark_inputs_v1/input{length}_output256.jsonl').read_text().splitlines()[0]);body.update(max_tokens=8,stream=False,logprobs=True,top_logprobs=5);body.pop('stream_options',None)
  response=requests.post(url+'/v1/chat/completions',json=body,timeout=(10,900))
  record={'input_tokens':length,'request':body,'status':response.status_code,'response':response.json()};prefix_records.append(record)
  (o/'prefix_logprobs.json').write_text(json.dumps(prefix_records,ensure_ascii=False,indent=2))
  if response.status_code not in [200,400]:response.raise_for_status()
  if response.status_code==400:record['scope']='API capability unavailable; not a numerical pass'
"""
    source=source[:at]+extra+source[at:]
    old="p=subprocess.run(cmd);rc=p.returncode"
    trace="""p=subprocess.run(cmd);p.check_returncode();rc=1
 trace_cmd=['python3',str(r/'repos/xllm-musa/tools/musa_deepseek_v41/capture_native_pd_trace.py'),'--url',url,'--prefill-control','http://10.121.32.7:22000','--decode-control','http://10.121.32.9:32000','--performance',str(o/'performance'),'--baseline-commit','28824b2d_PLACEHOLDER','--dataset',str(r/'artifacts/benchmark_inputs_v1/input256_output256.jsonl'),'--profile-root',str(r/'profiling/RUN_PLACEHOLDER'),'--output',str(o/'trace')]
 (o/'trace_command.json').write_text(json.dumps(trace_cmd,indent=2));subprocess.run(trace_cmd,check=True);rc=0"""
    # The baseline is the committed observation parent; exact candidate patch
    # and source hashes are recorded separately because it is not promoted.
    commit=(E/'02_代码/observation_parent_commit.txt').read_text().strip()
    trace=trace.replace('28824b2d_PLACEHOLDER',commit).replace('RUN_PLACEHOLDER',run)
    assert old in source;source=source.replace(old,trace)
    compile(source,'t01_client','exec');(m.STATE/(run+'_client.py')).write_text(source)
    code=fr"""import pathlib,subprocess,json
r=pathlib.Path({m.R!r});f=r/'scripts'/{(run+'_client.py')!r};f.write_text({source!r})
cmd=['docker','exec','-d',{m.container('sj10')!r},'sh','-c','echo $$ > /work/scripts/{run}_client.pid; exec python3 /work/scripts/{run}_client.py > /work/scripts/{run}_client.log 2>&1'];subprocess.run(cmd,check=True);print(json.dumps(cmd))
"""
    m.remote('sj10',code);return 'new_start_'+run


def await_client(run: str, output: str) -> dict:
    # Observation duration is not a terminal condition. Continue watching the
    # same process until authoritative completion/failure, without restarting.
    while True:
        code=fr"""import pathlib,json,subprocess
r=pathlib.Path({m.R!r});o=r/{output!r};f=o/'finished.json'
if f.exists():print(json.dumps({{'terminal':True,'result':json.loads(f.read_text()),'status':json.loads((o/'performance/status.json').read_text()) if (o/'performance/status.json').exists() else None,'trace':json.loads((o/'trace/result.json').read_text()) if (o/'trace/result.json').exists() else None}}))
else:
 p=subprocess.run(['docker','exec',{m.container('sj10')!r},'sh','-c','p=$(cat /work/scripts/{run}_client.pid); ps -p "$p" -o pid,stat,etime,args'],capture_output=True,text=True)
 print(json.dumps({{'terminal':False,'process_exit':p.returncode,'process':p.stdout}}))
"""
        state=json.loads(m.remote('sj10',code)['stdout']);(m.STATE/(run+'_observation.json')).write_text(json.dumps(state,indent=2))
        if state['terminal']:
            assert state['result']['exit_code']==0,state
            assert len(state['status'])==4 and all(x['exit_code']==0 for x in state['status']),state
            assert state['trace']['status']=='PROFILE_REQUEST_SAVED_REQUIRES_ALL_RANK_TRACE_VALIDATION',state
            return state
        assert state['process_exit']==0 and run+'_client.py' in state['process'],state
        time.sleep(15)


def request_normal_stop(run: str) -> None:
    for host in ['sj9','sj10']:
        code=fr"""import pathlib,json
p=pathlib.Path({m.R!r})/'runs'/{run!r}/{host!r}
if (p/'role_config.json').exists() and not (p/'result.json').exists() and not (p/'stop.request').exists():(p/'stop.request').write_text('Normal TERM after T01 experiment failure')
print(json.dumps({{'normal_stop_checked':str(p)}}))
"""
        m.remote(host,code)


def main() -> None:
    m.STATE.mkdir(exist_ok=False)
    for mode in ['OFF','CUSTOM']:
        run='t01_'+mode.lower()+'_v1';prepare(mode,run)
        try:
            m.launch_role('sj10',run);m.launch_role('sj9',run)
            output=launch_client(run);m.checkpoint('client_running',mode=mode,run=run,output=output)
            result=await_client(run,output)
            (m.STATE/(run+'_result.json')).write_text(json.dumps(result,indent=2));m.checkpoint('normal_stop',mode=mode,run=run)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:stopped=list(pool.map(lambda h:m.stop_role(h,run),['sj9','sj10']))
            (m.STATE/(run+'_exit.json')).write_text(json.dumps(stopped,indent=2));m.checkpoint('mode_complete',mode=mode,run=run)
        except Exception:
            request_normal_stop(run);raise
    m.checkpoint('both_modes_complete_requires_trace_and_correctness_audit')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if m.STATE.exists():m.checkpoint('failed_no_automatic_restart',error=repr(error))
        raise
