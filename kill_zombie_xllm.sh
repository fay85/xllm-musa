#!/bin/bash
# Stop live xLLM processes and reap zombie xllm children.
port_in_use() {
  local port="$1"
  ss -tln 2>/dev/null | grep -q ":${port} "
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
  pkill -9 -x xllm 2>/dev/null || true
  pkill -9 -f run_xllm_musa 2>/dev/null || true
  pkill -9 -f build/lib.linux-x86_64-cpython-310/xllm/xllm 2>/dev/null || true
  sleep 1
  while read -r pid stat _; do
    [[ "$stat" != "Z" ]] && kill -9 "$pid" 2>/dev/null || true
  done < <(ps -o pid=,stat= -C xllm 2>/dev/null || true)
  sleep 2
  if [[ $# -gt 0 ]]; then
    wait_ports_free "$@"
  fi
}
stop_xllm() { kill_zombie_xllm "$@"; }
