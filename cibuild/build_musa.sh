#!/bin/bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

IMAGE="${XLLM_MUSA_IMAGE:-registry.mthreads.com/presale/devtech/xllm:musa-cicd-20260820}"
WORKDIR="$(pwd)"
VCPKG_CACHE="${XLLM_MUSA_VCPKG_CACHE:-/export/home/musa_vcpkg_cache}"

CMD="$*"

mkdir -p "${VCPKG_CACHE}"

[[ ! -x $(command -v docker) ]] && \
  echo "ERROR: 'docker' command is missing." && exit 1

COMMAND="set -euo pipefail; \
unset VCPKG_ROOT VCPKG_BINARY_SOURCES \
  DEPENDENCES_ROOT FETCHCONTENT_SOURCE_DIR_VCPKG CMAKE_TOOLCHAIN_FILE
export VCPKG_DEFAULT_BINARY_CACHE=/root/.cache/vcpkg/archives
export VCPKG_DOWNLOADS=/root/.cache/vcpkg/downloads
${CMD}"
BUILD_JOBS="${MAX_JOBS:-16}"
MUSA_DEVICE_MASK="${MUSA_VISIBLE_DEVICES:-0}"

RUN_OPTS=(
  --rm
  --privileged
  --runtime=mthreads
  --ipc=host
  --network=host
  --ulimit memlock=-1
  --env "GITHUB_WORKSPACE=${WORKDIR}"
  --env "MAX_JOBS=${BUILD_JOBS}"
  --env "MUSA_VISIBLE_DEVICES=${MUSA_DEVICE_MASK}"
  --env "XLLM_HOST_GID=$(id -g)"
  --env "XLLM_HOST_UID=$(id -u)"
  --volume "${WORKDIR}:${WORKDIR}"
  --volume "${VCPKG_CACHE}:/root/.cache/vcpkg"
  --workdir "${WORKDIR}"
  --entrypoint /usr/local/bin/run-xllm-musa-ci
)

docker run "${RUN_OPTS[@]}" "${IMAGE}" run "${COMMAND}"
