from pathlib import Path
import datetime,hashlib,json,runpy,shutil,sqlite3,statistics
run=Path('/expt/runs/p55_draft_gate_on_v1');o=run/'formal_client'
assert json.loads((o/'finished.json').read_text())['exit_code']==1
assert (o/'error.txt').read_text()=="AssertionError('Compilation interference or monitoring gap: do not promote this round')"
assert not (o/'summary.json').exists()
for host in ['sj9','sj10']:
 assert not (run/host/'result.json').exists()
 assert all(x is None for x in json.loads((run/host/'status.json').read_text())['exit_codes'])
backup=o/'audit_first_failure';backup.mkdir(exist_ok=False)
for name in ['finished.json','error.txt','compilation_window.json','status.json','identity.json']:
 shutil.copy2(o/name,backup/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
immutable=list(o.rglob('benchmark_data.db'))+list(o.glob('basic_*_sse.jsonl'))+[o/'status.json',o/'identity.json',o/'basic_gate.json']
before={str(p.relative_to(o)):sha(p) for p in immutable}
status=json.loads((o/'status.json').read_text());identity=json.loads((o/'identity.json').read_text())
assert identity['profiler_on'] is False
assert identity['binary_sha256']=='c8a4f4cf1819baad39f30b8ad9962de73bb69653b5a6bd71d75c2d600c7bef2d'
assert len(status)==2
basic=json.loads((o/'basic_gate.json').read_text());assert len(basic)==2 and all(x['passed'] for x in basic)
for phase,count in [('formal_warmup',1),('formal',5)]:
 item=next(x for x in status if x['phase']==phase)
 assert item['exit_code']==0 and item['count']==count and item['input_tokens']==256
 dbs=list((o/phase).rglob('benchmark_data.db'));assert len(dbs)==1
 with sqlite3.connect('file:'+str(dbs[0])+'?mode=ro',uri=True) as db:
  db.row_factory=sqlite3.Row
  rows=[dict(x) for x in db.execute('SELECT success,prompt_tokens,completion_tokens,latency,first_chunk_latency FROM result')]
 assert rows==item['rows']
 assert len(rows)==count and all(x['success'] and x['prompt_tokens']==256 and x['completion_tokens']==256 and x['latency']>x['first_chunk_latency']>0 for x in rows)
audit=runpy.run_path('/expt/00_control/performance_environment.py')['audit_after_formal'](run)
assert audit['all_clear']
latency=sum(x['latency'] for x in rows);tpots=[(x['latency']-x['first_chunk_latency'])/255*1000 for x in rows]
summary={'status':'FORMAL_SHORT_PASSED_NOT_FINAL_THREE_ROUND','accuracy_status':identity['accuracy_status'],'mean_TTFT_ms':statistics.mean(x['first_chunk_latency'] for x in rows)*1000,'mean_TPOT_ms_per_token':statistics.mean(tpots),'TPOT_sample_stdev_ms':statistics.stdev(tpots),'mean_E2E_s':latency/count,'output_tokens_per_latency_sum_second':count*256/latency,'formal_requests':count,'input_output':[256,256],'profiler_on':False,'binary_sha256':identity['binary_sha256'],'postprocessing_recovery':'Original client exited 1 on initial NFS history coverage read; original failure preserved in audit_first_failure. Original requests and timestamps unchanged; complete original CPU history now validates.'}
assert before=={str(p.relative_to(o)):sha(p) for p in immutable}
receipt={'at':datetime.datetime.now().astimezone().isoformat(),'mode':'POSTPROCESSING_ONLY_NO_HTTP_NO_GPU_NO_SIGNALS','original_client_exit_code':1,'validated_exit_code':0,'original_failure':'audit_first_failure','immutable_evidence_sha256':before,'complete_original_window':audit['formal_window'],'no_observed_compilation':audit['all_clear'],'client_source_sha256':sha(Path('/expt/scripts/match_e55_client_v1.py')),'auditor_sha256':sha(Path('/expt/00_control/performance_environment.py'))}
(o/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(o/'audit_recovery.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
(o/'finished.json').write_text(json.dumps({'at':receipt['at'],'exit_code':0,'original_client_exit_code':1,'postprocessing_only_recovery':'audit_recovery.json'},indent=2))
print(json.dumps({'summary':summary,'recovery':receipt},ensure_ascii=False))

