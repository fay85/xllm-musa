from pathlib import Path
import datetime
import json

root=Path('/expt')
for run in ['p54_rope_proposal_off_v1','p54_rope_proposal_on_v1','p54_rope_proposal_reverse_off_v1']:
    p=root/'runs'/run
    assert json.loads((p/'formal_client/finished.json').read_text())['exit_code']==0
    for h in ['sj9','sj10']:
        assert json.loads((p/h/'result.json').read_text())['exit_codes']==[0]*8
value={'decision_complete':True,'run':'p54_rope_proposal_off_v1',
       'at':datetime.datetime.now().astimezone().isoformat(),
       'reason':'Retain E52 proposal3 champion; proposal5 did not improve measured results.',
       'caveat':'E54 lacks continuous compiler observations. Provisional rejection, not final acceptance; new comparisons use monitored quiet windows.'}
p=root/'builds/e55_native_draft_gate_v1/selected_baseline.json'
assert not p.exists()
p.write_text(json.dumps(value,indent=2));print(json.dumps(value))
