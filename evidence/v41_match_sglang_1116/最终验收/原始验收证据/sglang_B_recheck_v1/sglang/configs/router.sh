#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/venv/bin:$PATH
exec python -m sglang_router.launch_router --host 0.0.0.0 --port 28192 --disable-retries --request-timeout-secs 1800 --pd-disaggregation --prefill http://10.121.32.7:28586 28998 --decode http://10.121.32.9:28588 --worker-startup-timeout-secs 1800 --health-check-timeout-secs 30 --prometheus-port 29031
