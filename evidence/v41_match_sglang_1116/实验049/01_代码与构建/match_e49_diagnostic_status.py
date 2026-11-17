from pathlib import Path
import json,datetime
r=Path('/expt/runs/p49_full_draft_graph_sg438_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['diagnostic_client/finished.json','diagnostic_client/summary.json','diagnostic_client/error.txt','trace/finished.json']:
 if (r/n).exists():v[n]=(r/n).read_text()
fix={'scope':'Metadata correction only, preserve original files','formal_client/identity.json':{'wrong_candidate_delta_manifest':'/expt/builds/e46_native_full_v4/source_manifest.json','correct_candidate_delta_manifest':'/expt/builds/e48_native_timing_v1/source_manifest.json'},'diagnostic_client/identity.json':{'source_note':'same E48 binary; E49 full draft Graph configured via roles/role_config.json'},'binary_sha256':'0c3def50fab14e750e61dede83c36c43eb6ecd590b6451153fbd2491b0db1bc1'}
(r/'identity_correction.json').write_text(json.dumps(fix,indent=2));print(json.dumps(v))
