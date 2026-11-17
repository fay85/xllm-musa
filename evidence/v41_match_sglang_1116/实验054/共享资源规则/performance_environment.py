"""Benchmark-side checks. No GPU calls, process signals, or automatic retries."""
from pathlib import Path
import datetime
import json
import runpy
import time

def wait_before_phase(run, phase, output=None):
    control=Path('/expt/00_control')
    out=output or run/'formal_client'
    deadline=time.monotonic()+3600
    attempts=[]
    while True:
        for host in ['sj9','sj10']:
            result=run/host/'result.json'
            state=run/host/'status.json'
            if result.exists() or (state.exists() and any(x is not None for x in json.loads(state.read_text()).get('exit_codes',[]))):
                raise RuntimeError('Model role terminated while awaiting CPU compilation; inspect external-use policy')
        try:
            runpy.run_path(str(control/'assert_compilation_clear.py'))
            attempts.append({'at':datetime.datetime.now().astimezone().isoformat(),'clear':True})
            (out/(phase+'_compilation_gate.json')).write_text(json.dumps(attempts,indent=2))
            return
        except (AssertionError,FileNotFoundError,json.JSONDecodeError) as e:
            attempts.append({'at':datetime.datetime.now().astimezone().isoformat(),'clear':False,'reason':str(e)})
            (out/(phase+'_compilation_gate.json')).write_text(json.dumps(attempts,indent=2))
            if time.monotonic()>=deadline:
                raise RuntimeError('Compilation wait exceeded one hour; keep failed record, no automatic relaunch') from e
            time.sleep(10)

def audit_after_formal(run, output=None, phase_name='formal'):
    control=Path('/expt/00_control')
    output=output or run/'formal_client'
    phase=next(v for v in json.loads((output/'status.json').read_text()) if v['phase']==phase_name)
    end=datetime.datetime.fromisoformat(phase['finished_at'])
    deadline=time.monotonic()+35
    while time.monotonic()<deadline:
        latest=[json.loads((control/'compilation_monitor'/h/'latest.json').read_text()) for h in ['sj9','sj10']]
        if all(datetime.datetime.fromisoformat(v['at'])>=end for v in latest):
            break
        time.sleep(1)
    return runpy.run_path(str(control/'compile_window_audit.py'))['audit'](run, output=output, phase_name=phase_name)
