#!/usr/bin/env bash
# Launch NInfer as a dedicated local LLM server: Qwen3.8-27B NVFP4, MTP3, C=8,
# 128K context. Tuned and measured on <host> (RTX 5090, 32 GiB).
#
# Every value here is explained in PRODUCTION.md next to this script. Override
# any of them from the environment rather than editing the file.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${NINFER_MODEL:-/path/to/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer}"
HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-8080}"
MAX_CONTEXT="${NINFER_MAX_CONTEXT:-131072}"
MAX_CONCURRENCY="${NINFER_MAX_CONCURRENCY:-8}"
DRAFT_TOKENS="${NINFER_DRAFT_TOKENS:-3}"
MIN_P="${NINFER_MIN_P:-0.05}"
DEFAULT_MAX_TOKENS="${NINFER_DEFAULT_MAX_TOKENS:-131072}"
# Sharp v22.1 prompt overlay is the default here: it cuts completion tokens sharply
# without changing correctness. Set NINFER_CHAT_STYLE=default to fall back to the
# artifact's stock chat template.
CHAT_STYLE="${NINFER_CHAT_STYLE:-sharp-v22.1}"

# The key is read from the environment and never written into this file, a log,
# or a shell history entry. NINFER_API_KEY wins; LLAMA_API_KEY is the fallback
# so the box reuses the key already exported for the benchmark harness.
API_KEY="${NINFER_API_KEY:-${LLAMA_API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "error: set NINFER_API_KEY (or LLAMA_API_KEY) before starting the server" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "error: artifact not found: $MODEL" >&2
  exit 1
fi

# One resident model owns the GPU. Refuse to start on top of another engine
# rather than OOMing ~25 ms into the launch.
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
if (( FREE_MIB < 30000 )); then
  echo "error: only ${FREE_MIB} MiB VRAM free -- stop the other engine first" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
  exit 1
fi

exec "$REPO/build/apps/ninfer-serve" "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --max-context "$MAX_CONTEXT" \
  --kv-capacity auto \
  --kv-dtype int8 \
  --max-concurrency "$MAX_CONCURRENCY" \
  --spec mtp --draft-tokens "$DRAFT_TOKENS" \
  --lm-head-draft \
  --min-p "$MIN_P" \
  --chat-style "$CHAT_STYLE" \
  --default-max-tokens "$DEFAULT_MAX_TOKENS" \
  --log-stats-interval-ms 5000 \
  "$@"
