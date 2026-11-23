#!/usr/bin/env bash
# Receive a stop for exactly this launch on the container that owns its PIDs.
set -euo pipefail
cd "$(dirname "$0")"
role=$1
generation=$2
dir=pd_logs/${role,,}
while [[ $(cat "$dir/generation" 2>/dev/null || :) == "$generation" ]]; do
  if [[ -f "$dir/stop.$generation" ]]; then
    GLM5_EXPECT_GENERATION=$generation bash pd_role.sh "$role" stop && exit
  fi
  sleep 1
done
