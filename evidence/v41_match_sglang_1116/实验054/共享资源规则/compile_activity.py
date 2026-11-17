"""Read-only CPU compilation inventory; never signals or queries GPUs."""
from pathlib import Path
import datetime
import json
import os
import re
import socket
import subprocess

BUILD_NAMES = {'ninja', 'ninja-build', 'make', 'gmake', 'cmake', 'cc', 'c++',
               'gcc', 'g++', 'cc1', 'cc1plus', 'clang', 'clang++', 'clang-18',
               'clang-19', 'clang-20', 'mcc', 'musa-clang', 'nvcc', 'ptxas',
               'ld', 'ld.lld', 'ld.gold', 'collect2', 'ccache', 'lto1',
               'rustc', 'cargo', 'as', 'musa-as', 'musa-link'}

def inventory():
    ids = subprocess.check_output(['docker', 'ps', '-q', '--no-trunc'], text=True).split()
    containers = json.loads(subprocess.check_output(['docker', 'inspect'] + ids)) if ids else []
    records = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            comm = (p / 'comm').read_text().strip()
            exe = os.readlink(p / 'exe')
            basename = Path(exe).name
            args = (p / 'cmdline').read_bytes().split(b'\0')
            # Only classify Python build tools by known script/module names.
            python_build = basename.startswith('python') and any(
                Path(a.decode(errors='replace')).name in {'setup.py', 'build_ext.py'}
                for a in args[1:5])
            versioned_compiler = re.fullmatch(r'(?:gcc|g\+\+|clang|clang\+\+)(?:-\d+)?', basename)
            if comm not in BUILD_NAMES and basename not in BUILD_NAMES and not python_build and not versioned_compiler:
                continue
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            cgroup = (p / 'cgroup').read_text()
            owners = [{'name': c['Name'], 'cid': c['Id']} for c in containers if c['Id'] in cgroup]
            records.append({'pid': int(p.name), 'ppid': int(stat[1]), 'state': stat[0],
                            'start_ticks': stat[19], 'cpu_ticks': int(stat[11]) + int(stat[12]),
                            'comm': comm, 'exe': exe, 'containers': owners,
                            'uid': (p / 'status').read_text().split('Uid:', 1)[1].splitlines()[0].strip()})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return {'at': datetime.datetime.now().astimezone().isoformat(),
            'hostname': socket.gethostname(), 'loadavg': Path('/proc/loadavg').read_text().strip(),
            'logical_cpus': os.cpu_count(), 'compilation_clear': not records,
            'build_processes': sorted(records, key=lambda x: x['pid']),
            'scope': 'Known compiler/build executable inventory; absence is not proof of zero background load.'}

if __name__ == '__main__':
    value = inventory()
    root = Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
    host = {'worker32007': 'sj9', 'worker32009': 'sj10'}[value['hostname']]
    history = root / 'compilation_snapshots'
    history.mkdir(exist_ok=True)
    name = host + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'
    (history / name).write_text(json.dumps(value, indent=2))
    (root / (host + '_compilation_latest.json')).write_text(json.dumps(value, indent=2))
    print(json.dumps(value))
