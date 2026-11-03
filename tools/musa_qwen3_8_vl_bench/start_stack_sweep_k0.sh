#!/bin/bash
# MTP-off sweep: no speculative decoding. All other knobs from /workspace/config.json.
set -u
export VL_ENABLE_MTP=false
export VL_MTP_K=0
exec "$(cd "$(dirname "$0")" && pwd)/start_stack.sh"
