#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Run on the relevant host/container: control -> decode -> prefill0/prefill1.
set -euo pipefail
role=${1:?usage: start_stack.sh control|decode|prefill0|prefill1 --config FILE [--plan]}
shift
exec python3 "$(dirname "$0")/run.py" start --role "$role" "$@"
