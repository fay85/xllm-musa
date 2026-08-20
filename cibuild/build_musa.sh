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

set -Eeuo pipefail

readonly IMAGE="${XLLM_MUSA_IMAGE:-registry.mthreads.com/presale/devtech/xllm:musa-cicd-20260820}"
readonly WORKDIR="${GITHUB_WORKSPACE:-$(pwd)}"
readonly CCACHE_CACHE="${XLLM_MUSA_CCACHE:-}"
readonly VCPKG_CACHE="${XLLM_MUSA_VCPKG_CACHE:-/export/home/musa_vcpkg_cache}"

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

if ! mkdir -p "${VCPKG_CACHE}/archives" "${VCPKG_CACHE}/downloads"; then
  echo "ERROR: cannot create vcpkg cache directories: ${VCPKG_CACHE}" >&2
  exit 1
fi

DOCKER_RUNTIMES="$(docker info --format '{{json .Runtimes}}')"
readonly DOCKER_RUNTIMES
if [[ "${DOCKER_RUNTIMES}" != *'"mthreads"'* ]]; then
  echo "ERROR: Docker runtime 'mthreads' is unavailable." >&2
  exit 1
fi

# Keep vcpkg source selection tied to CMake while reusing host downloads and
# binary archives.
readonly COMMAND="unset VCPKG_ROOT VCPKG_BINARY_SOURCES \
  DEPENDENCES_ROOT FETCHCONTENT_SOURCE_DIR_VCPKG CMAKE_TOOLCHAIN_FILE
export VCPKG_DEFAULT_BINARY_CACHE=/root/.cache/vcpkg/archives
export VCPKG_DOWNLOADS=/root/.cache/vcpkg/downloads
$1"
readonly BUILD_JOBS="${MAX_JOBS:-16}"
readonly MUSA_DEVICE_MASK="${MUSA_VISIBLE_DEVICES:-0}"
container_name="${JOBNAME:-xllm-musa-cibuild}-${BASHPID}"
container_name="${container_name//[^a-zA-Z0-9_.-]/-}"
readonly container_name
docker_pid=""

RUN_OPTS=(
  --rm
  --name "${container_name}"
  --privileged
  --runtime=mthreads
  --ipc=host
  --network=host
  --shm-size=128g
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

if [[ -n "${CCACHE_CACHE}" ]]; then
  if [[ ! -d "${CCACHE_CACHE}" ]]; then
    echo "ERROR: configured ccache directory does not exist: ${CCACHE_CACHE}" >&2
    exit 1
  fi
  RUN_OPTS+=(
    --env "CCACHE_DIR=/root/.cache/ccache"
    --volume "${CCACHE_CACHE}:/root/.cache/ccache"
  )
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "${docker_pid}" ]]; then
    docker stop --time 40 "${container_name}" >/dev/null 2>&1
    kill -TERM "${docker_pid}" >/dev/null 2>&1
    wait "${docker_pid}" >/dev/null 2>&1
    docker_pid=""
  fi
  docker rm --force "${container_name}" >/dev/null 2>&1
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker run "${RUN_OPTS[@]}" "${IMAGE}" run "${COMMAND}" &
docker_pid=$!
set +e
wait "${docker_pid}"
docker_status=$?
set -e
docker_pid=""
exit "${docker_status}"
