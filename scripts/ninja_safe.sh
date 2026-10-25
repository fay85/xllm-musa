#!/bin/bash
# Run ninja via ninja_guard (lock + log backup). Prefer PATH setup in _build_cuda_graph_musa.sh.
# Usage: scripts/ninja_safe.sh <build_dir> [ninja args...]
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <build_dir> [ninja args...]" >&2
  exit 1
fi
GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ninja_guard/ninja"
BD="$(cd "$1" && pwd)"
shift
exec "${GUARD}" -C "${BD}" "$@"
