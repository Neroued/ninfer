"""Executable Serve profiles for the external TTFT campaigns."""

from __future__ import annotations


# qwen3_8_27b_nvfp4: StateImageLayout sums 48 GDN conv/state regions and
# continuation hidden, each aligned to 256 B (state/state_image.cpp). DFlash2 also
# carries 5 local K/V layers of 2048 positions. Context budgets below are explicit
# whole-block capacities; state and growing histories share them at runtime.
STATE_IMAGE_BYTES = 153_954_304
DFLASH2_STATE_IMAGE_BYTES = 195_897_344


def _bytes(capacity_bytes: int) -> str:
    return f"{capacity_bytes}B"


def _args(*values: str | int) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


COMMON_ARGS = _args(
    "--kv-dtype",
    "fp8",
    "--no-thinking",
    "--temperature", "0",
    "--stats-interval",
    1000,
)


PROFILE_ARGS: dict[str, tuple[str, ...]] = {
    "text-cold-8k": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "text-cold-64k": _args(
        "--max-context", 65536,
        "--context-capacity", _bytes(16 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "text-cold-256k": _args(
        "--max-context", 262144,
        "--context-capacity", _bytes(58 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "cache-hot": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(5 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--host-cache-capacity", "0MiB",
    ),
    "cache-state-working-set": _args(
        "--max-context", 32768,
        "--context-capacity", _bytes(15 * DFLASH2_STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--prefill-chunk", 1024,
        "--spec", "dflash2",
        "--draft-tokens", 7,
        "--proposal-head", "optimized",
        "--host-cache-capacity", _bytes(8 * DFLASH2_STATE_IMAGE_BYTES),
    ),
    "cache-private-working-set": _args(
        "--max-context", 32768,
        "--context-capacity", _bytes(9 * DFLASH2_STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--prefill-chunk", 1024,
        "--spec", "dflash2",
        "--draft-tokens", 7,
        "--proposal-head", "optimized",
        "--host-cache-capacity", _bytes(2 * DFLASH2_STATE_IMAGE_BYTES),
    ),
    "cache-pressure-device": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(8 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", "0MiB",
    ),
    "cache-pressure-state-host": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", _bytes(4 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-kv-host": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", "8192MiB",
    ),
    "cache-swap-64k-host": _args(
        "--max-context", 65536,
        "--context-capacity", _bytes(21 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", "4608MiB",
    ),
    "cache-rotation-55k-host": _args(
        "--max-context", 240000,
        "--context-capacity", _bytes(63 * STATE_IMAGE_BYTES),
        "--max-concurrency", 4,
        "--max-pending-requests", 32,
        "--pending-timeout", 120000,
        "--spec", "mtp",
        "--draft-tokens", 3,
        "--proposal-head", "optimized",
        "--host-cache-capacity", _bytes((49152 << 20) + 24 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-both-host": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(5 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", _bytes((8192 << 20) + 4 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-evict": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(5 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", "0MiB",
    ),
    "cache-off": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "shared-prefix": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(8 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", "0MiB",
    ),
    "prefix-competition": _args(
        "--max-context", 16384,
        "--context-capacity", _bytes(8 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--host-cache-capacity", "0MiB",
    ),
    "session-order": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(14 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--host-cache-capacity", "0MiB",
    ),
    "scheduler-overlap": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--prefill-chunk", 1024,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "scheduler-prefill-128": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "scheduler-prefill-4096": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--prefill-chunk", 4096,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "scheduler-kv-pressure": _args(
        "--max-context", 7744,
        "--context-capacity", _bytes(4 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--pending-timeout", 120000,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "lane-limit-8": _args(
        "--max-context", 4224,
        "--context-capacity", _bytes(16 * STATE_IMAGE_BYTES),
        "--max-concurrency", 8,
        "--max-pending-requests", 1,
        "--pending-timeout", 120000,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "pending-timeout": _args(
        "--max-context", 4224,
        "--context-capacity", _bytes(2 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--max-pending-requests", 1,
        "--pending-timeout", 100,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "context-boundary": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "vision-cache": _args(
        "--max-context", 32768,
        "--context-capacity", _bytes(11 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--vision",
        "--media-cache-capacity", "512MiB",
        "--media-live-capacity", "512MiB",
        "--host-cache-capacity", "0MiB",
    ),
    "vision-thread-1": _args(
        "--max-context", 32768,
        "--context-capacity", _bytes(11 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--vision",
        "--media-cache-capacity", "512MiB",
        "--media-live-capacity", "512MiB",
        "--media-preprocess-threads", 1,
        "--host-cache-capacity", "0MiB",
    ),
    "vision-concurrent": _args(
        "--max-context", 32768,
        "--context-capacity", _bytes(17 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--vision",
        "--media-cache-capacity", "512MiB",
        "--media-live-capacity", "1024MiB",
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "media-cache-tight": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--vision",
        "--media-cache-capacity", "16MiB",
        "--media-live-capacity", "128MiB",
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "vision-boundary": _args(
        "--max-context", 65536,
        "--context-capacity", _bytes(16 * STATE_IMAGE_BYTES),
        "--max-concurrency", 1,
        "--vision",
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "mixed-four": _args(
        "--max-context", 8192,
        "--context-capacity", _bytes(20 * STATE_IMAGE_BYTES),
        "--max-concurrency", 4,
        "--vision",
        "--host-cache-capacity", _bytes((8192 << 20) + 8 * STATE_IMAGE_BYTES),
    ),
    "preemption-replay": _args(
        "--max-context", 512,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-cache-capacity", "0MiB",
    ),
    "preemption-snapshot": _args(
        "--max-context", 512,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-cache-capacity", "512MiB",
    ),
    "shared-growth-recovery": _args(
        "--max-context", 2688,
        "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
        "--max-concurrency", 3,
        "--prefill-chunk", 128,
        "--host-cache-capacity", "0MiB",
    ),
    "host-history-pressure": _args(
        "--max-context", 512,
        "--context-capacity", _bytes(5 * STATE_IMAGE_BYTES),
        "--max-concurrency", 3,
        "--prefill-chunk", 128,
        "--host-cache-capacity", "384MiB",
    ),
    "vision-growth-replay": _args(
        "--max-context", 1024,
        "--context-capacity", _bytes(3 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--host-cache-capacity", "0MiB",
        "--no-prefix-reuse",
        "--vision",
        "--media-cache-capacity", "0MiB",
        "--media-live-capacity", "64MiB",
    ),
}


for _capacity, _blocks in (("roomy", 6), ("pressure", 5)):
    PROFILE_ARGS[f"resident-{_capacity}"] = _args(
        "--max-context", 8192, "--context-capacity", _bytes(_blocks * STATE_IMAGE_BYTES),
        "--max-concurrency", 3, "--prefill-chunk", 1024,
        "--no-prefix-reuse", "--host-cache-capacity", "0MiB",
    )

PROFILE_ARGS["mixed-arrivals"] = _args(
    "--max-context", 8192, "--context-capacity", _bytes(6 * STATE_IMAGE_BYTES),
    "--max-concurrency", 4, "--prefill-chunk", 1024,
    "--max-pending-requests", 16, "--pending-timeout", 120000,
    "--no-prefix-reuse", "--host-cache-capacity", "512MiB",
)

for _route, _host_capacity_value in (("replay", "0B"), ("snapshot", "512MiB")):
    PROFILE_ARGS[f"agent-{_route}"] = _args(
        "--max-context", 768, "--context-capacity", _bytes(5 * STATE_IMAGE_BYTES),
        "--max-concurrency", 2, "--prefill-chunk", 128,
        "--host-cache-capacity", _host_capacity_value,
    )

for _backend, _draft_tokens, _context_bytes in (
    ("mtp", 3, 6 * STATE_IMAGE_BYTES),
    ("dflash2", 7, 5 * DFLASH2_STATE_IMAGE_BYTES),
):
    PROFILE_ARGS[f"preemption-snapshot-{_backend}"] = _args(
        "--max-context", 512,
        "--context-capacity", _bytes(_context_bytes),
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-cache-capacity", "512MiB",
        "--spec", _backend,
        "--draft-tokens", _draft_tokens,
        "--proposal-head", "optimized",
    )
