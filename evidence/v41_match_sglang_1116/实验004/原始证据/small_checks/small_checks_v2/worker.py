from pathlib import Path
import json,subprocess,os,datetime
r=Path('/expt/builds/attention_replica_v1/small_checks_v2');b=Path('/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310');cmd=json.loads((r/'link_command.json').read_text());(r/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
with (r/'link.log').open('wb') as f:p=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
(r/'finished.json').write_text(json.dumps({'exit_code':p.returncode,'at':datetime.datetime.now().astimezone().isoformat()}))
