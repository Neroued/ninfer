#!/bin/sh
# Entrypoint for the NInfer image.
#
# An explicit command after the image name runs verbatim, so the invocations in
# README.md's Docker section keep working. With no arguments -- or with flags
# only -- the image starts this host's production serving profile instead.
#
# Every value below is explained in docs/deployment.md. Override any of them
# from the environment rather than editing this file; flags passed after the
# image name are appended last and win, because the option parser is last-wins.
#
# NINFER_API_KEY is read by ninfer-serve itself and is deliberately never placed
# on the command line, so it stays out of `ps` and /proc/<pid>/cmdline.
set -eu

if [ "$#" -gt 0 ]; then
    case "$1" in
        --help | -h) exec ninfer-serve --help ;;
        -*) ;; # flags only: append them to the production profile below
        *) exec "$@" ;;
    esac
fi

MODEL="${NINFER_MODEL:-/models/qwen3_8_27b_nvfp4.ninfer}"
if [ ! -f "$MODEL" ]; then
    echo "error: artifact not found at $MODEL inside the container" >&2
    echo "       mount the directory holding it, for example" >&2
    echo "         --volume /path/to/models:/models:ro" >&2
    echo "       and set NINFER_MODEL to its path inside the container" >&2
    exit 1
fi

# The artifact must be argv[1]: the parser reads it positionally and rejects any
# flag placed before it. Caller-supplied flags stay at the tail so they win.
set -- "$MODEL" \
    --host "${NINFER_HOST:-0.0.0.0}" \
    --port "${NINFER_PORT:-11434}" \
    --max-context "${NINFER_MAX_CONTEXT:-131072}" \
    --kv-capacity auto \
    --kv-dtype int8 \
    --max-concurrency "${NINFER_MAX_CONCURRENCY:-8}" \
    --spec mtp --draft-tokens "${NINFER_DRAFT_TOKENS:-3}" \
    --lm-head-draft \
    --min-p "${NINFER_MIN_P:-0.05}" \
    --chat-style "${NINFER_CHAT_STYLE:-sharp-v22.1}" \
    --default-max-tokens "${NINFER_DEFAULT_MAX_TOKENS:-16384}" \
    --log-stats-interval-ms "${NINFER_LOG_STATS_INTERVAL_MS:-5000}" \
    "$@"

if [ -n "${NINFER_REQUEST_LOG_JSONL:-}" ]; then
    set -- "$@" --request-log-jsonl "$NINFER_REQUEST_LOG_JSONL"
fi

# exec, so ninfer-serve is PID 1 and receives SIGTERM directly. Without it the
# shell would absorb the signal and the engine would never drain.
exec ninfer-serve "$@"
