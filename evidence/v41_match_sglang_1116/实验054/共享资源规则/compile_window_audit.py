"""Archive CPU compilation observations covering actual benchmark phases."""
from pathlib import Path
import datetime
import json

def audit(run, control=Path('/expt/00_control'), output=None, phase_name='formal'):
    output = output or run / 'formal_client'
    status = json.loads((output / 'status.json').read_text())
    phase = next(x for x in status if x['phase'] == phase_name)
    start = datetime.datetime.fromisoformat(phase['started_at'])
    end = datetime.datetime.fromisoformat(phase['finished_at'])
    hosts = {}
    for host in ['sj9', 'sj10']:
        source = control / 'compilation_monitor' / host / 'observations.jsonl'
        rows = [json.loads(s) for s in source.read_text().splitlines() if s]
        rows = [v for v in rows if start - datetime.timedelta(seconds=15) <= datetime.datetime.fromisoformat(v['at']) <= end + datetime.timedelta(seconds=15)]
        times = [datetime.datetime.fromisoformat(v['at']) for v in rows]
        coverage = bool(times) and times[0] <= start and times[-1] >= end
        gaps = [(b-a).total_seconds() for a,b in zip(times,times[1:])]
        clear = coverage and max(gaps, default=0) < 30 and all(v.get('compilation_clear', False) for v in rows)
        hosts[host] = {'clear': clear, 'coverage': coverage, 'observations': rows}
    result = {'formal_window': {'start': start.isoformat(), 'end': end.isoformat()},
              'all_clear': all(v['clear'] for v in hosts.values()), 'hosts': hosts,
              'scope': '10-second sampled known compiler/build processes; not proof of zero background load'}
    (output / 'compilation_window.json').write_text(json.dumps(result, indent=2))
    assert result['all_clear'], 'Compilation interference or monitoring gap: do not promote this round'
    return result
