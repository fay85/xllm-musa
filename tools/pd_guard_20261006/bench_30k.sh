#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
arm=${1:?usage: bench_30k.sh b1|b16 UNIQUE_LABEL}
label=${2:?set a unique result label}
[[ "$label" =~ ^[A-Za-z0-9_-]+$ ]]
case "$arm" in b1) concurrency=1 ;; b16) concurrency=16 ;; *) exit 2 ;; esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
dataset=/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/logs/dspark_30k_1k_cases_20260926/cases_pd_model_id.jsonl
printf '%s  %s\n' f60ba0784dfd287d4aca59e9e74972f5c0b65bb33d9fa4be6a5551587d153d09 "$dataset" | sha256sum -c -
# The auditor deliberately keeps the original fixture path within the checkout.
fixture="$repo/logs/dspark_30k_1k_cases_20260926/cases_pd_model_id.jsonl"
if [[ ! -e "$fixture" ]]; then
  mkdir -p -- "$(dirname -- "$fixture")"
  cp -n -- "$dataset" "$fixture"
fi
printf '%s  %s\n' f60ba0784dfd287d4aca59e9e74972f5c0b65bb33d9fa4be6a5551587d153d09 "$fixture" | sha256sum -c -
for phase in warmup formal; do
  [[ ! -e "$repo/logs/${label}_${arm}_${phase}" ]]
done
curl --noproxy '*' -fsS --max-time 5 http://10.121.38.92:29888/v1/models
for phase in warmup formal; do
  start=304; count=16
  if [[ "$phase" == formal ]]; then start=320; count=64; fi
  name=${label}_${arm}_${phase}
  python3 -u "$repo/tools/run_dspark_stress.py" --dataset "$fixture" \
    --output-dir "$repo/logs/$name" --url http://10.121.38.92:29888/v1/chat/completions \
    --concurrency "$concurrency" --start "$start" --limit "$count" --timeout 600 --stop-errors 1
  python3 "$here/audit_counting.py" "$repo" "${name}_audit.json" "$name:$count"
done
