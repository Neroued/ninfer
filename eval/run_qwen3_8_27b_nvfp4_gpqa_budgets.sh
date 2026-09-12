#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build/apps/ninfer-serve"
artifact="${repo_dir}/out/qwen3_8_27b_nvfp4.ninfer"
config="${repo_dir}/eval/configs/qwen3_8_27b_nvfp4_gpqa_budgets.yaml"
eval_python="${repo_dir}/eval/.venv/bin/python"
campaign_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${repo_dir}/profiles/eval/qwen3_8_27b_nvfp4_gpqa_budgets_${campaign_stamp}"

for required_file in "${server_bin}" "${artifact}" "${config}" "${eval_python}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "missing required file: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "${server_bin}" || ! -x "${eval_python}" ]]; then
    echo "ninfer-serve and the evaluation Python must be executable" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the server health check" >&2
    exit 1
fi
if curl --fail --silent --show-error --max-time 2 \
    http://127.0.0.1:18080/health >/dev/null 2>&1; then
    echo "port 18080 already has a healthy service; stop it before running this campaign" >&2
    exit 1
fi

mkdir -p -- "${output_root}"
echo "campaign output: ${output_root}"

server_pid=""
cleanup_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        wait "${server_pid}" || true
    fi
    server_pid=""
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_tier() {
    local tier="$1"
    local suite="$2"
    local budget="$3"
    local max_context="$4"
    local kv_capacity="$5"
    local concurrency="$6"
    local tier_dir="${output_root}/${tier}"
    local server_log="${tier_dir}/server.log"
    local request_log="${tier_dir}/server.requests.jsonl"

    mkdir -p -- "${tier_dir}"
    echo "starting tier=${tier} budget=${budget} context=${max_context} kv=${kv_capacity} concurrency=${concurrency}"

    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 \
        --port 18080 \
        --model-id qwen3.8-27b \
        --max-context "${max_context}" \
        --kv-capacity "${kv_capacity}" \
        --max-concurrency "${concurrency}" \
        --max-pending-requests "${concurrency}" \
        --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 \
        --kv-dtype int8 \
        --spec mtp \
        --draft-tokens 3 \
        --lm-head-draft \
        --default-thinking-budget "${budget}" \
        --request-log-jsonl "${request_log}" \
        >"${server_log}" 2>&1 &
    server_pid=$!

    local ready=0
    for ((attempt = 1; attempt <= 180; ++attempt)); do
        if curl --fail --silent --show-error --max-time 2 \
            http://127.0.0.1:18080/health >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            wait "${server_pid}" || true
            echo "ninfer-serve exited before becoming ready; see ${server_log}" >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "ninfer-serve did not become ready within 180 seconds; see ${server_log}" >&2
        exit 1
    fi

    echo "running suite=${suite}; server_log=${server_log}; request_log=${request_log}"
    PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
        --config "${config}" \
        --suite "${suite}"

    cleanup_server
    echo "completed tier=${tier}"
}

if [[ $# -eq 0 ]]; then
    tiers=(8k 16k 32k)
else
    tiers=("$@")
fi

for tier in "${tiers[@]}"; do
    case "${tier}" in
        8k)
            run_tier 8k gpqa_8k 8192 12352 auto 8
            ;;
        16k)
            run_tier 16k gpqa_16k 16384 20544 auto 8
            ;;
        32k)
            run_tier 32k gpqa_32k 32768 36928 211200 6
            ;;
        *)
            echo "unknown tier: ${tier}; expected 8k, 16k, or 32k" >&2
            exit 2
            ;;
    esac
done

echo "campaign completed: ${output_root}"
