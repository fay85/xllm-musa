#!/bin/bash
# Stop live xLLM processes and reap zombie xllm children.
port_in_use() {
  local port="$1"
  ss -tln 2>/dev/null | grep -q ":${port} "
}
port_listener_pids() {
  local port="$1"
  if command -v nsenter >/dev/null 2>&1; then
    nsenter -t 1 -m -n -p -- python3 - "$port" <<'PY' 2>/dev/null || true
import os, sys
port = int(sys.argv[1])
port_hex = format(port, "X")
inodes = []
for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(fn).readlines()[1:]
    except OSError:
        continue
    for line in lines:
        parts = line.split()
        if len(parts) < 10 or parts[3] != "0A":
            continue
        local = parts[1]
        if local.endswith(":" + port_hex):
            inodes.append(parts[9])
for inode in inodes:
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fdpath = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fdpath):
                try:
                    if os.readlink(f"{fdpath}/{fd}") == f"socket:[{inode}]":
                        print(pid)
                except OSError:
                    pass
        except (FileNotFoundError, PermissionError):
            pass
PY
    return
  fi
  ss -tlnp 2>/dev/null | grep ":${port} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u
}
kill_port_listeners() {
  local port="$1"
  local pid
  for pid in $(port_listener_pids "$port"); do
    if command -v nsenter >/dev/null 2>&1; then
      nsenter -t 1 -m -p -- kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    else
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}
wait_ports_free() {
  local ports=("$@")
  local w=0
  while [[ "$w" -lt 60 ]]; do
    local busy=0
    for p in "${ports[@]}"; do
      port_in_use "$p" && busy=1
    done
    [[ "$busy" -eq 0 ]] && return 0
    sleep 2
    w=$((w + 2))
  done
  echo "ports still in use after cleanup: ${ports[*]}" >&2
  return 1
}
kill_zombie_xllm() {
  local ports=("$@")
  if command -v nsenter >/dev/null 2>&1; then
    nsenter -t 1 -m -p -- pkill -9 -x xllm 2>/dev/null || true
    nsenter -t 1 -m -p -- pkill -9 -f run_xllm_musa 2>/dev/null || true
  fi
  pkill -9 -x xllm 2>/dev/null || true
  pkill -9 -f run_xllm_musa 2>/dev/null || true
  pkill -9 -f build/lib.linux-x86_64-cpython-310/xllm/xllm 2>/dev/null || true
  sleep 1
  while read -r pid stat _; do
    [[ "$stat" != "Z" ]] && kill -9 "$pid" 2>/dev/null || true
  done < <(ps -o pid=,stat= -C xllm 2>/dev/null || true)
  if [[ ${#ports[@]} -gt 0 ]]; then
    for p in "${ports[@]}"; do
      kill_port_listeners "$p"
    done
  fi
  sleep 2
  if [[ ${#ports[@]} -gt 0 ]]; then
    wait_ports_free "${ports[@]}"
  fi
}
stop_xllm() { kill_zombie_xllm "$@"; }
