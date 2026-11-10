#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Stop only recorded PID+executable+starttime identities; never broad-match names."""
import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--timeout',type=float,default=15);p.add_argument('--force',action='store_true');a=p.parse_args()
f=a.run/'identities.json'
if not f.exists():raise SystemExit('no identities.json; no process was signalled')
records=json.loads(f.read_text())
def current(r):
 try:
  q=Path('/proc')/str(r['pid']);stat=(q/'stat').read_text().rsplit(')',1)[1].split()
  if stat[0]=='Z':return None
  return {'exe':os.readlink(q/'exe'),'starttime':stat[19]}
 except FileNotFoundError:return None

def valid(r):
 now=current(r)
 if now is None:return False
 if any(now[k]!=r[k] for k in ('exe','starttime')):
  print('IDENTITY MISMATCH: left untouched',r['name'],r['pid']);return False
 return True
for r in records:
 if valid(r):
  try:os.kill(r['pid'],signal.SIGTERM)
  except ProcessLookupError:pass
end=time.monotonic()+a.timeout
while time.monotonic()<end and any(valid(r) for r in records):time.sleep(.2)
remaining=[r for r in records if valid(r)]
if a.force:
 for r in remaining:
  if valid(r):
   try:os.kill(r['pid'],signal.SIGKILL)
   except ProcessLookupError:pass
 time.sleep(.2);remaining=[r for r in records if valid(r)]
if remaining:raise SystemExit('Owned processes remain; inspect or explicitly retry --force: '+','.join(str(r['pid']) for r in remaining))
print('Recorded identities have exited; unrelated processes were left untouched.')
