#!/usr/bin/env bash
export GLM5_ENABLE_GRAPH="${GLM5_ENABLE_GRAPH:-${GLM5_PREFILL_GRAPH:-true}}"
exec bash "$(dirname "$0")/pd_role.sh" PREFILL "$@"
