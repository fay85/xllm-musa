#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
unset LD_PRELOAD
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec python3 -S "$HERE/scripts/stop_owned.py" "$@"
