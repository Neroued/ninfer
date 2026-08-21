#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build/apps/ninfer-serve"
artifact="${repo_dir}/out/qwen3_8_27b.ninfer"
config="${repo_dir}/eval/configs/qwen3_8_27b_groupwise_reasoning.yaml"
eval_python="${repo_dir}/eval/.venv/bin/python"
log_dir="${repo_dir}/eval/server-logs"

usage() {
    echo "usage: $0 [--plan] [text|multimodal]" >&2
    echo "       with no step argument, text runs followed by multimodal" >&2
}

# Emits tab-separated fields: suite, max-context, has-vision, description.
step_config() {
    case "$1" in
        text)
            # The groupwise-int artifact leaves 13.53 GiB free after weights, so
            # the full 262,144-token context fits (the nvfp4 profile needed
            # 252,928). Auto sizing resolves a 324,224-token KV pool at C=4.
            printf 'text_full\t262144\t0\t%s\n' \
                "IFBench, AIME 2025, AIME 2026, and GPQA-Diamond"
            ;;
        multimodal)
            # The largest prepared prompt in the cached datasets is ERQA_75 at
            # 12,394 tokens. With the 65,536 output bound, 81,920 leaves 3,990 tokens.
            printf 'multimodal_full\t81920\t1\t%s\n' "ERQA and RealWorldQA"
            ;;
        *)
            return 2
            ;;
    esac
}

plan_only=0
if [[ "${1:-}" == "--plan" ]]; then
    plan_only=1
    shift
fi
if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi
case "${1:-}" in
    "")
        steps=(text multimodal)
        ;;
    text | multimodal)
        steps=("$1")
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ "${plan_only}" -eq 1 ]]; then
    for step in "${steps[@]}"; do
        IFS=$'\t' read -r suite _ _ _ <<<"$(step_config "${step}")"
        PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval plan \
            --config "${config}" --suite "${suite}" --check-runtime
    done
    exit 0
fi

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

mkdir -p -- "${log_dir}"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        wait "${server_pid}" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_step() {
    local step="$1"
    local suite max_context has_vision description
    local run_stamp server_log request_log
    local -a vision_args=()

    IFS=$'\t' read -r suite max_context has_vision description \
        <<<"$(step_config "${step}")"
    if [[ "${has_vision}" -eq 1 ]]; then
        vision_args=(--vision)
    fi

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    server_log="${log_dir}/qwen3_8_27b_groupwise-${step}-c4-${run_stamp}.server.log"
    request_log="${log_dir}/qwen3_8_27b_groupwise-${step}-c4-${run_stamp}.requests.jsonl"

    if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
        echo "port 18080 already has a healthy service; stop it before running this script" >&2
        exit 1
    fi

    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 \
        --port 18080 \
        --model-id qwen3.8-27b \
        --max-context "${max_context}" \
        --kv-capacity auto \
        --max-concurrency 4 \
        --max-pending-requests 4 \
        --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 \
        --kv-dtype int8 \
        --spec mtp \
        --draft-tokens 3 \
        --lm-head-draft \
        "${vision_args[@]}" \
        --request-log-jsonl "${request_log}" \
        >"${server_log}" 2>&1 &
    server_pid=$!

    ready=0
    for ((attempt = 1; attempt <= 120; ++attempt)); do
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
        echo "ninfer-serve did not become ready within 120 seconds; see ${server_log}" >&2
        exit 1
    fi

    echo "server log: ${server_log}"
    echo "request log: ${request_log}"
    echo "running ${description}"

    PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
        --config "${config}" --suite "${suite}"

    cleanup
    server_pid=""
}

for step in "${steps[@]}"; do
    run_step "${step}"
done
