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

set -euo pipefail

readonly IMAGE_REPOSITORY="registry.mthreads.com/presale/devtech/xllm"
readonly IMAGE_DIGEST="sha256:603451a86f8fa26beee6cf6877299fb7e23ab000f75c13ed3d50ad5df2a9ae52"
readonly IMAGE="${IMAGE_REPOSITORY}@${IMAGE_DIGEST}"
readonly WORKDIR="${GITHUB_WORKSPACE:-$(pwd)}"
readonly VCPKG_CACHE="${XLLM_MUSA_VCPKG_CACHE:-${HOME}/.cache/xllm/musa-vcpkg}"
readonly CCACHE_CACHE="${XLLM_MUSA_CCACHE:-${HOME}/.cache/xllm/musa-ccache}"

error() {
  echo "Require one build command, e.g. python setup.py build --device musa" >&2
  exit 1
}

if [[ $# -ne 1 || -z "$1" ]]; then
  error
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' command is missing." >&2
  exit 1
fi

if [[ ! -d "${WORKDIR}" ]]; then
  echo "ERROR: workspace does not exist: ${WORKDIR}" >&2
  exit 1
fi

mkdir -p "${VCPKG_CACHE}" "${CCACHE_CACHE}"
readonly COMMAND="$1"
readonly BUILD_JOBS="${MAX_JOBS:-16}"
readonly HOST_GID="$(id -g)"
readonly HOST_UID="$(id -u)"
readonly MUSA_DEVICE_MASK="${MUSA_VISIBLE_DEVICES:-0}"

RUN_OPTS=(
  --rm
  --privileged
  --ipc=host
  --network=host
  --shm-size=128g
  --ulimit memlock=-1
  --env "CCACHE_DIR=/root/.cache/ccache"
  --env "MAX_JOBS=${BUILD_JOBS}"
  --env "MUSA_VISIBLE_DEVICES=${MUSA_DEVICE_MASK}"
  --env "SKIP_EXPORT=1"
  --env "SKIP_TEST=1"
  --env "VCPKG_DEFAULT_BINARY_CACHE=/root/.cache/vcpkg"
  --volume "${WORKDIR}:${WORKDIR}"
  --volume "${VCPKG_CACHE}:/root/.cache/vcpkg"
  --volume "${CCACHE_CACHE}:/root/.cache/ccache"
  --workdir "${WORKDIR}"
)

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "
cleanup() {
  chown -R ${HOST_UID}:${HOST_GID} \
    build dist .git/hooks ./*.egg-info 2>/dev/null || true
}
trap cleanup EXIT
set -euo pipefail
${COMMAND}
"
