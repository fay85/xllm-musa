#!/bin/bash
# Run xllm-on-MUSA graph-mode correctness check with the given env vars and
# report PASS/FAIL + a short answer snippet. Single run per invocation.
#
# Usage:
#   ENV_OVERRIDES="ENABLE_FUSED_GDN_DECODE=0" bash bisect_graph.sh tag_name
#   ENV_OVERRIDES="MAX_TOKENS_FOR_GRAPH_MODE=1" bash bisect_graph.sh bucket1
TAG="${1:-untagged}"
LOG=/tmp/bisect_${TAG}.log
SERVERLOG=/workspace/xllm-git-master/log/xllm_bisect_${TAG}.log

# Reuse correctness_check + server lifecycle helpers
cd /workspace/xllm-git-master

pkill -9 xllm 2>/dev/null
sleep 3
rm -f /workspace/xllm-git-master/log/xllm_Qwen3.5-27B.log

echo "==> bisect tag=${TAG}, ENV_OVERRIDES=${ENV_OVERRIDES}"
env $ENV_OVERRIDES \
  ENABLE_GRAPH=1 \
  MUSA_VISIBLE_DEVICES=3 \
  MODEL_NAME=Qwen3.5-27B \
  PORT=8092 \
  START_SERVER=1 \
  bash correctness_check.sh > "$LOG" 2>&1
RC=$?

# Snapshot server log for postmortem
SLOG=$(ls -t /workspace/xllm-git-master/log/xllm_Qwen3.5-27B.log 2>/dev/null | head -1)
[ -n "$SLOG" ] && cp "$SLOG" "$SERVERLOG"

# Extract a tight summary of the answer + PASS/FAIL
echo ""
echo "##############################"
echo "## RESULT for tag=${TAG}"
echo "##############################"
echo "ENV_OVERRIDES=${ENV_OVERRIDES}"
echo ""
echo "Final CHECKS block:"
grep -E "CORRECTNESS|\[PASS\]|\[FAIL\]" "$LOG" | tail -10
echo ""
echo "Answer first 30 lines (after run1):"
sed -n "/^\[run1\]/,/^####/p" "$LOG" | head -35
echo ""
echo "Server log (last 5 lines):"
[ -n "$SLOG" ] && tail -5 "$SLOG"
echo "##############################"

pkill -9 xllm 2>/dev/null
sleep 2

exit $RC
