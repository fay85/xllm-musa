from pathlib import Path
import json
p=Path('/expt/00_control/e56_selected_baseline.json')
assert not p.exists()
v={'decision_complete': True, 'run': 'p54_rope_proposal_off_v1', 'production_source_commit': '980d9da1287ba99a12749d89b89c55e0d8ed233f', 'binary_sha256': '7d2ffa0662fd53f0074d6f29c7a28e0f4db5e057cb1c0ad6febc706bfc74303f', 'reason': 'E55 all four rounds valid; second ON does not reproduce first ON end-to-end gain. Do not promote. E56 inherits E52 configuration via p54 OFF.', 'e55_status': 'INCONCLUSIVE_NO_STABLE_GAIN_NOT_PROMOTED', 'e55_evidence_sha256': 'b254303a606468561f2f1f6037b867afc81850edeae21f20ac3f56fa46a9fe31', 'e55_formal_requests': 20, 'all20_text_equal': True, 'accepted_proposed': [(148, 321)], 'at': '2026-11-17T20:08:37.196963+08:00'}
for mode in ['off_v1','on_v1','reverse_off_v1','on_confirm_v1']:
 r=Path('/expt/runs')/('p55_draft_gate_'+mode)
 assert json.loads((r/'formal_client/finished.json').read_text())['exit_code']==0
 assert json.loads((r/'formal_client/compilation_window.json').read_text())['all_clear']
 for host in ['sj9','sj10']:
  assert json.loads((r/host/'result.json').read_text())['exit_codes']==[0]*8
p.write_text(json.dumps(v,indent=2))
print(json.dumps(v))
