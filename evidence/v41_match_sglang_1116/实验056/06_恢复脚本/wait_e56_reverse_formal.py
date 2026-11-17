from pathlib import Path
import time
p=Path('/expt/runs/p56_short_indexer_reverse_off_v1/formal_client/finished.json');end=time.monotonic()+45
while not p.exists() and time.monotonic()<end:time.sleep(3)
print(p.read_text() if p.exists() else 'still_running')
