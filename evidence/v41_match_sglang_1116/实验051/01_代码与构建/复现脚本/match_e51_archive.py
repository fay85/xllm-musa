from pathlib import Path
import tarfile,hashlib,json,statistics
root=Path('/expt');p=root/'diagnostics/e51_gather_gpu_v1'
finished=json.loads((p/'finished.json').read_text());assert finished['exit_codes']==[0]*8
r=[json.loads((p/f'rank{i}.json').read_text()) for i in range(8)]
summary={'finished':finished,'check_count_per_rank':len(r[0]['checks']),'special_rank0':r[0]['special_value_mismatches'],'micro':[]}
for j,a in enumerate(r[0]['microbenchmarks']):
 xs=[statistics.median(v['microbenchmarks'][j]['us_per_call_samples']) for v in r]
 summary['micro'].append({k:a[k] for k in ['label','rows','local_width','dtype','mode']}|{'rank_medians_us':xs,'max_rank_median_us':max(xs)})
(p/'summary.json').write_text(json.dumps(summary,indent=2))
archive=root/'exports/e51_native_gather_checks_v1.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 t.add(p,arcname='e51_gather_gpu_v1')
 for f in [root/'diagnostics/e51_gather_gpu_launch_v1.json',root/'diagnostics/e51_gather_supervisor_v1.log']:
  t.add(f,arcname=f.name)
x={'path':str(archive),'size':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
archive.with_suffix('.manifest.json').write_text(json.dumps(x,indent=2));print(json.dumps(x));print(json.dumps(summary))
