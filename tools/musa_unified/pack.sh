#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
set -euo pipefail
exec python3 "$(dirname "$0")/create_pack.py" "$@"
