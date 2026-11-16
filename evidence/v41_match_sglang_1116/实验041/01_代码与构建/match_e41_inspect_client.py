from pathlib import Path
import json
p=Path('/expt/scripts/match_e39_formal_client.py');s=p.read_text();lines=s.splitlines();print(json.dumps({'first':s[:6000],'relevant':[(i+1,x) for i,x in enumerate(lines) if any(k in x for k in ['proposal','q4','p37','p39','speculative','run=','--run','tokens','identity'])][-60:]}))
