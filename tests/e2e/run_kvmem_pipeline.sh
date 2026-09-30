#!/usr/bin/env bash
set -euo pipefail

# Usage: NINFER_MODEL=/absolute/model.ninfer PYTHON=/path/to/python3.11 \
#          bash tests/e2e/run_kvmem_pipeline.sh [smoke|regression|long|concurrency|vision|dflash2-vision]
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${NINFER_MODEL:?Set NINFER_MODEL to an explicit official v3 .ninfer artifact}"
PYTHON=${PYTHON:-python3.11}
PROFILE=${1:-smoke}
case "$PROFILE" in smoke|regression|long|concurrency|vision|dflash2-vision) ;; *) echo 'invalid profile' >&2; exit 2 ;; esac
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
test -f "$NINFER_MODEL"
exec 9>"${TMPDIR:-/tmp}/ninfer-kvmem-e2e.lock"
flock -n 9 || { echo 'another KVMem pipeline owns this GPU test lock' >&2; exit 2; }
export CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc}
export PATH="$(dirname "$CUDACXX"):$PATH"
BUILD=${NINFER_BUILD_DIR:-build}
OUTPUT=${NINFER_TEST_OUTPUT:-"$ROOT/out/kvmem-tests/$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv > "$OUTPUT/hardware.csv"
git rev-parse HEAD > "$OUTPUT/source-revision.txt"
git diff --stat > "$OUTPUT/source-diff.txt"
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120a -DPython3_EXECUTABLE="$(command -v "$PYTHON")"
cmake --build "$BUILD" -j --target ninfer-serve ninfer_kvmem_options_test \
    ninfer_qwen3_5_retrieval_test ninfer_qwen3_5_context_store_test ninfer_qwen3_5_frontend_test \
    ninfer_span_accumulate_test ninfer_softmax_attention_test ninfer_serve_options_test \
    ninfer_resource_manager_test ninfer_qwen3_5_runtime_mechanisms_test \
    ninfer_qwen3_5_state_image_test ninfer_qwen3_5_state_image_layout_test
"$PYTHON" -m unittest discover -s tests/e2e -p 'test_kvmem*.py'
ctest --test-dir "$BUILD" --output-on-failure --no-tests=error \
    --output-junit "$OUTPUT/unit.xml" \
    -R '^ninfer_(kvmem_options|qwen3_5_retrieval|qwen3_5_context_store|qwen3_5_frontend|qwen3_5_runtime_mechanisms|qwen3_5_state_image|qwen3_5_state_image_layout|span_accumulate|softmax_attention|serve_options|resource_manager)_test$'
"$PYTHON" - "$OUTPUT/unit.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
assert len(root.findall('testcase')) >= 11, 'missing required tests'
assert not root.findall('.//skipped'), 'CUDA tests skipped; this is not a passing GPU pipeline'
PY
COMMON=(--binary "$BUILD/apps/ninfer-serve" --model "$NINFER_MODEL")
# Fresh ports per process avoid inheriting accepted sockets' TIME_WAIT state.
PORT=${NINFER_TEST_PORT:-8095}
if [[ "$PROFILE" == dflash2-vision ]]; then
    # The artifact must include the matching DFlash2 companion and Vision weights.
    "$PYTHON" tests/e2e/kvmem_vision_suite.py "${COMMON[@]}" --output "$OUTPUT/dense-dflash2" \
        --spec dflash2 --window 0 --port "$PORT"
    for SPEC in none dflash2; do
        PORT=$((PORT + 1))
        "$PYTHON" tests/e2e/kvmem_vision_suite.py "${COMMON[@]}" --output "$OUTPUT/vision-$SPEC" \
            --spec "$SPEC" --port "$PORT"
    done
    echo "KVMem DFlash2 seven-draft vision pipeline passed: $OUTPUT"
    exit 0
fi
if [[ "$PROFILE" == vision ]]; then
    for SPEC in none mtp; do
        "$PYTHON" tests/e2e/kvmem_vision_suite.py "${COMMON[@]}" --output "$OUTPUT/vision-$SPEC" \
            --spec "$SPEC" --port "$PORT"
        PORT=$((PORT + 1))
    done
    echo "KVMem vision pipeline passed: $OUTPUT"
    exit 0
fi
if [[ "$PROFILE" == concurrency ]]; then
    for SPEC in none mtp; do
        "$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/dual-$SPEC" \
            --profile concurrency --concurrency 2 --spec "$SPEC" --host-mib 2048 --port "$PORT"
        PORT=$((PORT + 1))
    done
    echo "KVMem concurrency pipeline passed: $OUTPUT"
    exit 0
fi
"$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/smoke" --profile smoke --port "$PORT"
if [[ "$PROFILE" != smoke ]]; then
    "$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/mtp-regression" \
        --profile regression --chunk 2048 --port "$((PORT + 1))"
    "$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/ordinary-regression" \
        --profile regression --spec none --port "$((PORT + 2))"
    "$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/dense-smoke" \
        --profile smoke --window 0 --context 4096 --port "$((PORT + 3))"
    "$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/host-capacity" \
        --profile capacity --host-mib 32 --port "$((PORT + 4))"
fi
if [[ "$PROFILE" == long ]]; then
    "$PYTHON" tests/e2e/kvmem_suite.py "${COMMON[@]}" --output "$OUTPUT/long" \
        --profile long --window 1536 --context 262144 --host-mib 12288 --port "$((PORT + 5))"
fi
echo "KVMem pipeline passed: $OUTPUT"
