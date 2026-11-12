"""Collect immutable completed T01 runs; never launch or stop any service."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

E = Path(__file__).resolve().parents[1]
W = next(p for p in E.parents if (p / 'sj9_12_登陆工具').is_dir())
sys.path.insert(0, str(W / 'sj9_12_登陆工具'))
import sj_access as access

R = '/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_sglang_mt_cpp'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['off', 'custom'])
    parser.add_argument('--failed', action='store_true', help='Archive an explicitly terminal failed client; role logs are a timestamped snapshot, not proof of exit')
    args = parser.parse_args()
    run = 't01_' + args.mode + '_v1'
    suffix = '_failed_raw' if args.failed else '_raw'
    destination = E / '04_性能' / (run + suffix + '.tar.gz')
    code = f'''import pathlib,json,tarfile,hashlib
r=pathlib.Path({R!r});run={run!r};o=r/('new_start_'+run)
failed={args.failed!r}
assert (json.loads((o/'finished.json').read_text())['exit_code']!=0)==failed
if not failed:
 for host in ['sj9','sj10']:
  assert json.loads((r/'runs'/run/host/'result.json').read_text())['exit_codes']==[0]*8
archive=r/('new_start_'+run+('_failed_evidence.tar.gz' if failed else '_small_evidence.tar.gz'))
if not archive.exists():
 temporary=archive.with_suffix('.incomplete')
 with tarfile.open(temporary,'w:gz') as t:
  for f in sorted(o.rglob('*')):
   if not f.is_file() or 'raw' in f.relative_to(o).parts:continue
   t.add(f,arcname='output/'+str(f.relative_to(o)),recursive=False)
  for host in ['sj9','sj10']:
   for f in sorted((r/'runs'/run/host).iterdir()):
    if f.is_file():t.add(f,arcname='roles/'+host+'/'+f.name,recursive=False)
  t.add(r/'artifacts/native_t01_decode_tp_ar/manifest.json',arcname='binary_manifest.json')
  t.add(r/'scripts'/(run+'_client.log'),arcname='client.log')
 temporary.rename(archive);archive.chmod(0o644)
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
print(json.dumps({{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'run':run,'failed':failed,'scope':'all official performance, gates, configs, server logs; failed-run role logs may be a snapshot rather than final; large raw traces retained in original NFS paths'}}))
'''
    with access.ssh_client('sj10') as client:
        response = access.execute(client, 'sj10', 'python3 -', root=True,
                                  timeout=60, payload=code)
        if response['exit_status'] != 0:
            raise RuntimeError(response)
        receipt = json.loads(response['stdout'])
        with client.open_sftp() as sftp:
            if not destination.exists():
                temporary = destination.with_suffix('.incomplete')
                sftp.get(receipt['path'], str(temporary))
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != receipt['sha256']:
                    raise ValueError('Downloaded archive SHA mismatch')
                temporary.rename(destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == receipt['sha256']
    (destination.with_suffix('.receipt.json')).write_text(json.dumps(receipt, indent=2))
    import tarfile
    extracted = E / '04_性能' / (run + suffix)
    if not extracted.exists():
        extracted.mkdir()
        with tarfile.open(destination) as archive:
            for item in archive.getmembers():
                target = (extracted / item.name).resolve()
                if not target.is_relative_to(extracted.resolve()) or item.issym() or item.islnk():
                    raise ValueError('Unexpected archive member')
            archive.extractall(extracted)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
