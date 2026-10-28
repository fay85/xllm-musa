#!/bin/bash
# MTP K=1 sweep. All other knobs from /workspace/config.json.
set -u
export VL_ENABLE_MTP=true
export VL_MTP_K=1
exec "$(cd "$(dirname "$0")" && pwd)/start_stack.sh"
