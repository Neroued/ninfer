#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${NINFER_GPU_UUID:?set NINFER_GPU_UUID to the UUID of the preferred V100 SXM2}"
readonly NINFER_GPU_UUID
readonly NINFER_EXECUTABLE="${NINFER_EXECUTABLE:-${REPOSITORY_ROOT}/build-v100/apps/ninfer}"

if [[ ! -x "${NINFER_EXECUTABLE}" ]]; then
    echo "ninfer executable is missing: ${NINFER_EXECUTABLE}" >&2
    exit 1
fi
if ! nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep -Fq \
    "${NINFER_GPU_UUID}, Tesla V100-SXM2-32GB"; then
    echo "the selected UUID is not a Tesla V100-SXM2-32GB: ${NINFER_GPU_UUID}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${NINFER_GPU_UUID}"
exec "${NINFER_EXECUTABLE}" "$@"
