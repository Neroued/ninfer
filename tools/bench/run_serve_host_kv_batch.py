#!/usr/bin/env python3
"""Benchmark the host-KV cache (PR #64) in a concurrent (batch) workload.

The host-KV cache parks evicted sequences in pinned host RAM so a follow-up turn can
restore a cached entry instead of re-prefilling the full context. It only helps at C>1
(other lanes must be parkable), so this benchmark drives a multi-turn conversation
workload at C>1 where follow-up turns match parked host-KV entries.

It A/Bs the PR against the original version: the baseline runs the server with the
host-KV cache disabled (--baseline-host-kv-mib 0, behaviorally identical to the pre-PR
version for the host-KV feature, since every host-KV code path is gated on
host_kv_cache_enabled()) and the candidate runs it with the cache enabled
(--host-kv-mib N). Both use the same binary, so the A/B isolates the host-KV cache
feature without a separate pre-PR build. For each run it reports:

- per-turn time-to-first-token (streaming), the prefill-sensitive metric (mean/median);
- the count of deferred probes that did NOT restore (the re-prefill fallback);
- the count of full_reset re-prefills vs restore_turn_checkpoint reuses (from the JSONL
  request log); the drop in full_reset from baseline to candidate is the recovered work;
- the workload makespan.

The workload is deterministic: K conversations, each a unique prompt (so no
cross-conversation prefix reuse) plus T short follow-up turns. --prompt-tokens may be a
single size or a per-conversation list (e.g. a mixed big/small agentic mix). The pool is
sized to hold a chosen number of the contexts, so some lanes are parked and a follow-up
whose lane was parked matches a parked entry - the geometry the evicting restore targets.

Four representative scenarios (C=4, 3 turns each):
  limit    - 6 x 100k contexts, pool 245760 (fully-saturated: the pool cannot hold all
             the contexts, so some must always re-prefill or be parked). The evicting
             restore fires (the baseline's deferred probes are recovered), but the
             follow-up TTFT is unchanged - the pool is saturated, so the follow-up is
             queued either way. In a steady-state saturated pool the total re-prefill
             work is conserved; the evicting restore reorders which conversation pays,
             not how much. This is the LIMIT.
  transient - one large (80k) conversation + one small one-off side request (2k,
             max_tokens 4096), pool 74000 (holds the large but not large+side). The side
             evicts the large session to host; its lane is retained (idle) when the large
             follow-up arrives. The follow-up restores the large entry (parking the small
             side lane) instead of re-prefilling ~72k tokens - the transient benefit the
             architecture targets: a large session's re-prefill is avoided by parking a
             small one-off. This is the BENEFIT (the designed case).
  mixed    - 3 x 60k + 3 x 2k contexts (agentic coding: a big/small mix), pool 130000.
             The evicting restore parks a small retained lane to restore a big entry
             (cheap, and the small lane re-prefills on its next request anyway). The
             follow-up TTFT improves - the evicting restore's benefit under multi-lane
             steady state.
  agentic  - one big (100k, 3 turns) session + N small one-offs (2k, 1 turn,
             --side-max-tokens) that never come back, pool sized to hold the big but not
             big+all smalls. The big turn 1 is sent first (admitted to lane 0); the smalls
             are sent --side-delay seconds later, while the big is still active, so they
             queue (needs --max-pending >= N). When the big completes, its lane is parked
             to host (the big entry); the smalls are admitted one per lane. When the big
             follow-up arrives, the candidate restores the big entry from host (a ~0.4s
             copy) instead of re-prefilling the full 90k context (~20s) - the re-prefill
             the original version pays. This is the AGENTIC-CODING BENEFIT: a big
             session's re-prefill is avoided by the host-KV cache (a net win, unlike the
             limit scenario where the parked sibling comes back and the work is conserved).

Usage (A/B, one geometry):
  python3 tools/bench/run_serve_host_kv_batch.py \
    --serve-baseline /path/to/ninfer-serve.baseline \
    --serve-candidate /path/to/ninfer-serve.evicting \
    --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
    --concurrency 4 --conversations 6 --turns 3 \
    --prompt-tokens 100000 --kv-capacity 320000 --host-kv-mib 16384 \
    --max-context 245760 --spec mtp --draft-tokens 3 \
    --output profiles/bench/host-kv-batch
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_corpus as corpus  # noqa: E402

# A short in-distribution English paragraph, tiled to reach the target prompt length.
# Each conversation gets a unique header so no two conversations share a prefix (the
# benchmark measures within-conversation reuse, not cross-conversation reuse).
_BASE_PARAGRAPH = (
    "The history of inference serving is a history of trade-offs between latency, "
    "throughput, and memory. A single request that reuses a long cached prefix avoids "
    "recomputing that prefix, but only if the cache still holds it when the request "
    "returns. Under concurrency, the device pool is shared, and a parked continuation "
    "may have to wait for another lane to release its pages before it can be restored. "
    "The question this workload measures is whether that wait is paid once, by parking "
    "a sibling lane, or paid every time, by re-prefilling the whole context from scratch."
)


def _tile_to_tokens(target_tokens: int, seed_label: str) -> str:
    # ~0.75 words/token for English prose; pad with a unique header so the prompt is
    # distinct per conversation while staying in-distribution.
    words_per_cycle = len(_BASE_PARAGRAPH.split())
    cycles = max(1, int((target_tokens * 0.75) / words_per_cycle))
    body = (" ".join(_BASE_PARAGRAPH.split()) + " ") * cycles
    return f"[{seed_label}]\n" + body.strip()


@dataclasses.dataclass
class TurnResult:
    conversation: int
    turn: int
    ttft_seconds: float
    prompt_tokens: int
    completion_tokens: int
    # The number of messages the client sent for this turn (2*turn-1: turn 1 is a
    # single user message, each follow-up adds an assistant + user pair). Used to
    # match the client's request to the server log's request_done record.
    message_count: int
    # Filled post-hoc by matching to the server log: the reuse path the server took
    # for this request (full_reset = a cold re-prefill, restore_turn_checkpoint = a
    # host-KV restore). None if no matching record was found.
    reuse_path: str | None = None
    # The server-side ttft for the matched request (the prefill+queue time the
    # evicting restore affects), distinct from the client-measured ttft_seconds.
    server_ttft_seconds: float | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-baseline", type=Path, required=True,
                        help="server binary for the baseline run (the original version). "
                             "For a PR-vs-original A/B, point this at the same binary as "
                             "--serve-candidate and set --baseline-host-kv-mib 0 (cache off).")
    parser.add_argument("--serve-candidate", type=Path, required=True,
                        help="server binary for the candidate run (the PR). For a "
                             "PR-vs-original A/B, point this at the same binary as "
                             "--serve-baseline and set --host-kv-mib N (cache on).")
    parser.add_argument("--artifact", action="append", required=True,
                        metavar="TARGET=PATH", help="artifact for a registered target")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="startup max concurrency and client worker count (default 4)")
    parser.add_argument("--conversations", type=int, default=6,
                        help="number of distinct conversations (default 6)")
    parser.add_argument("--turns", type=int, default=3,
                        help="turns per conversation, including the first (default 3)")
    parser.add_argument("--prompt-tokens", default="32768",
                        help="approximate first-turn prompt length in tokens. A single value "
                             "applies to every conversation, or a comma-separated list applies "
                             "per conversation (cycling if shorter than --conversations), e.g. "
                             "'60000,60000,60000,2000,2000,2000' for a mixed big/small workload "
                             "(default 32768)")
    parser.add_argument("--kv-capacity", type=int, default=131072,
                        help="shared Main KV pool in tokens; sized to saturate at C (default 131072)")
    parser.add_argument("--host-kv-mib", type=int, default=8192,
                        help="pinned host-KV budget in MiB for the CANDIDATE (default 8192)")
    parser.add_argument("--baseline-host-kv-mib", type=int, default=0,
                        help="pinned host-KV budget in MiB for the BASELINE (default 0 = the "
                             "host-KV cache is disabled, i.e. the original pre-PR behavior). "
                             "Point --serve-baseline at the same binary as --serve-candidate "
                             "and set this to 0 to A/B the PR against the original version "
                             "(no host-KV cache) without a separate pre-PR build.")
    parser.add_argument("--max-context", type=int, default=65536)
    parser.add_argument("--spec", default="mtp",
                        help="speculative backend: mtp, dflash, or none (default mtp)")
    parser.add_argument("--draft-tokens", type=int, default=3,
                        help="MTP draft window (default 3)")
    parser.add_argument("--output", type=Path, required=True, help="benchmark output directory")
    parser.add_argument("--port", type=int, default=8091, help="loopback serving port (default 8091)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lockstep", action="store_true",
                        help="advance all conversations in lockstep (turn 1 for all, then "
                             "turn 2 for all, ...). After each turn every lane is retained "
                             "(idle), so a follow-up can park a retained sibling. In a "
                             "steady-state pool (holds fewer than all conversations) the "
                             "total re-prefill work is conserved - the evicting restore "
                             "reorders which conversation pays, not how much (the limit "
                             "case).")
    parser.add_argument("--transient", action="store_true",
                        help="one large conversation (--prompt-tokens, --turns turns) plus "
                             "one small one-off side request (--side-tokens, "
                             "--side-max-tokens). The side is sent --side-delay seconds after "
                             "the large turn 1 (with --max-pending >= 2) so it is admitted to a "
                             "different lane while the large is still resident; its reservation "
                             "evicts the large session to host. Its lane is retained (idle) when "
                             "the large follow-up arrives, so the follow-up restores the large "
                             "entry (parking the small side lane) instead of re-prefilling. This "
                             "is the transient benefit the architecture targets: a large "
                             "session's re-prefill is avoided by parking a small one-off.")
    parser.add_argument("--side-tokens", type=int, default=2000,
                        help="transient scenario: the one-off side request's prompt size (default 2000)")
    parser.add_argument("--side-max-tokens", type=int, default=4096,
                        help="transient scenario: the one-off side request's max_completion_tokens "
                             "(its reservation is prompt + this, sized to evict the large session; "
                             "default 4096)")
    parser.add_argument("--large-max-tokens", type=int, default=64,
                        help="transient scenario: the large conversation's first-turn "
                             "max_completion_tokens. The parked entry's page entitlement is the "
                             "parking session's (context + this). Keeping it at 64 (the same as "
                             "the follow-ups) makes the entry's entitlement equal to the follow-up's "
                             "plan entitlement, so after the admission parks the retained side lane "
                             "the entry fits and the evicting restore succeeds (default 64)")
    parser.add_argument("--side-delay", type=float, default=2.0,
                        help="transient scenario: seconds to wait after starting the large turn 1 "
                             "before sending the side request. The large is a cold prefill (~14s), "
                             "so a short delay ensures the large is admitted to lane 0 and still "
                             "decoding when the side arrives, so the side is admitted to lane 1 "
                             "(not lane 0). The evicting restore then parks lane 1 (the side) to "
                             "restore the large entry (default 2.0)")
    parser.add_argument("--agentic", action="store_true",
                        help="agentic coding: one big multi-turn session (--prompt-tokens, "
                             "--turns turns) + --agentic-small-count small one-off requests "
                             "(--side-tokens, 1 turn, --side-max-tokens) that never come back. "
                             "The big and the smalls are sent in parallel (turn 1), so the smalls "
                             "land on different lanes than the big. The pool is sized so the smalls "
                             "evict the big to host; when the big follow-up arrives, the evicting "
                             "restore parks a small (cheap, it never returns) to restore the big "
                             "entry instead of re-prefilling. This is the agentic-coding benefit: "
                             "a big session's re-prefill is avoided by parking small one-offs.")
    parser.add_argument("--agentic-small-count", type=int, default=3,
                        help="agentic scenario: number of small one-off requests (default 3)")
    parser.add_argument("--max-pending", type=int, default=1,
                        help="server --max-pending-requests. The transient scenario needs >= 2 so "
                             "the side request is admitted in parallel (to lane 1) while the large "
                             "is still resident, not queued until the large completes (which would "
                             "admit it to lane 0, the same lane as the entry, so the probe parks it "
                             "directly and the evicting restore never fires) (default 1)")
    return parser.parse_args(argv)


def server_command(serve: Path, artifact: Path, args: argparse.Namespace,
                   server_log: Path, host_kv_mib: int) -> list[str]:
    command = [
        str(serve), str(artifact),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", str(args.max_context),
        "--kv-capacity", str(args.kv_capacity),
        "--max-concurrency", str(args.concurrency),
        "--max-pending-requests", str(args.max_pending),
        "--host-kv-cache-mib", str(host_kv_mib),
        "--device", str(args.device),
        "--request-log-jsonl", str(server_log),
        "--kv-dtype", "int8",
    ]
    if args.spec != "none":
        command.extend(["--spec", args.spec, "--draft-tokens", str(args.draft_tokens)])
    return command


def stream_ttft(connection: http.client.HTTPConnection, payload: dict[str, Any]) -> tuple[float, int, int]:
    """Send a streaming Chat Completions request; return (ttft, prompt_tokens, completion_tokens).

    ttft is the wall time from the request to the first non-empty content chunk. The
    prefill (restore vs re-prefill) dominates this for a large cached context.
    """
    payload = dict(payload)
    payload["stream"] = True
    started = time.monotonic()
    connection.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise corpus.CampaignError(f"streaming request failed: {response.status} {response.read()!r}")
    ttft = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
        choices = chunk.get("choices") or []
        if choices and (choices[0].get("delta") or {}).get("content"):
            if ttft == 0.0:
                ttft = time.monotonic() - started
    if ttft == 0.0:
        ttft = time.monotonic() - started
    return ttft, prompt_tokens, completion_tokens


def run_workload(
    serve: Path, artifact: Path, args: argparse.Namespace, server_log: Path,
    stdout_log: Path, host_kv_mib: int,
) -> dict[str, Any]:
    command = server_command(serve, artifact, args, server_log, host_kv_mib)
    with corpus.RunningServer(command, "127.0.0.1", args.port, server_log,
                              stdout_path=stdout_log) as server:
        server_start = server.wait_until_ready()
        model_id = server_start.get("server", {}).get("public_model_id", "bench")
        # One persistent connection per worker; each worker owns a round-robin slice of
        # the conversations and drives them turn by turn (the follow-up is sent only
        # after the previous turn's response completes, so the reuse frontier is live).
        results: list[TurnResult] = []
        results_lock = threading.Lock()
        failed = threading.Event()
        failure_reason: list[str] = []

        def worker(worker_index: int) -> None:
            # A fresh connection per request: the streaming response is not fully
            # drained on [DONE], so reusing one connection across turns trips
            # http.client's "Request-sent" guard. Loopback TCP setup is negligible
            # against a multi-second prefill and does not change server-side
            # concurrency (requests are still sent turn by turn per worker).
            for conv in range(worker_index, args.conversations, args.concurrency):
                target_tokens = args.prompt_tokens_list[conv % len(args.prompt_tokens_list)]
                history: list[dict[str, str]] = [
                    {"role": "user", "content": _tile_to_tokens(target_tokens, f"conversation {conv}")}
                ]
                for turn in range(1, args.turns + 1):
                    if turn > 1:
                        history.append({"role": "user", "content": f"Follow-up {turn - 1}: state the main point in one sentence."})
                    payload = {
                        "model": model_id,
                        "messages": history,
                        "max_completion_tokens": 64,
                    }
                    connection = http.client.HTTPConnection("127.0.0.1", args.port,
                                                            timeout=corpus.REQUEST_TIMEOUT_SECONDS)
                    try:
                        connection.connect()
                        ttft, prompt_tokens, completion_tokens = stream_ttft(connection, payload)
                    except Exception as exc:  # noqa: BLE001
                        failure_reason.append(f"worker {worker_index} conv {conv} turn {turn}: {exc}")
                        failed.set()
                        return
                    finally:
                        connection.close()
                    with results_lock:
                        results.append(TurnResult(conv, turn, ttft, prompt_tokens, completion_tokens,
                                                  message_count=2 * turn - 1))
                    history.append({"role": "assistant", "content": f"(turn {turn} response)"})

        def send_request(conv: int, turn: int, history: list[dict[str, str]],
                 max_completion_tokens: int) -> None:
            """Send one request on a fresh connection and record the result. The
            label is (conv, turn); max_completion_tokens varies (64 for the
            conversations, a large value for the transient side request)."""
            payload = {
                "model": model_id,
                "messages": history,
                "max_completion_tokens": max_completion_tokens,
            }
            connection = http.client.HTTPConnection("127.0.0.1", args.port,
                                                    timeout=corpus.REQUEST_TIMEOUT_SECONDS)
            try:
                connection.connect()
                ttft, prompt_tokens, completion_tokens = stream_ttft(connection, payload)
            except Exception as exc:  # noqa: BLE001
                failure_reason.append(f"conv {conv} turn {turn}: {exc}")
                failed.set()
                return
            finally:
                connection.close()
            with results_lock:
                results.append(TurnResult(conv, turn, ttft, prompt_tokens, completion_tokens,
                                          message_count=len(history)))

        def send_turn(conv: int, turn: int, history: list[dict[str, str]]) -> None:
            send_request(conv, turn, history, 64)

        if args.lockstep:
            # One thread per conversation. A semaphore caps active requests at C; a
            # barrier syncs the turns so every conversation finishes turn t before any
            # sends turn t+1. After each turn the lanes are retained (idle), so a
            # follow-up can park a retained sibling - the transient state the evicting
            # restore targets.
            semaphore = threading.Semaphore(args.concurrency)
            barrier = threading.Barrier(args.conversations)
            histories: list[list[dict[str, str]]] = [
                [{"role": "user", "content": _tile_to_tokens(
                    args.prompt_tokens_list[conv % len(args.prompt_tokens_list)],
                    f"conversation {conv}")}]
                for conv in range(args.conversations)
            ]

            def lockstep_worker(conv: int) -> None:
                for turn in range(1, args.turns + 1):
                    if turn > 1:
                        histories[conv].append({"role": "user", "content": f"Follow-up {turn - 1}: state the main point in one sentence."})
                    semaphore.acquire()
                    try:
                        send_turn(conv, turn, histories[conv])
                    finally:
                        semaphore.release()
                    histories[conv].append({"role": "assistant", "content": f"(turn {turn} response)"})
                    barrier.wait()

            threads = [threading.Thread(target=lockstep_worker, args=(c,))
                       for c in range(args.conversations)]
        elif args.transient:
            # One large conversation (conv 0) + one small one-off side request (conv 1).
            # The large turn 1 is sent FIRST (admitted to lane 0, a cold prefill). The
            # side request is sent --side-delay seconds later, while the large is still
            # decoding (lane 0 is occupied), so the side is admitted to lane 1 (not
            # lane 0). When the large completes, lane 0 is parked (the large entry).
            # When the side completes, lane 1 is retained (idle). The large follow-up
            # arrives: the probe defers (the entry does not fit alongside the side on
            # lane 1), and the admission's evicting restore parks lane 1 (the side,
            # cheap, and the side request never returns) to restore the large entry
            # instead of re-prefilling.
            #
            # The side MUST be on a different lane (lane 1) than the entry (lane 0):
            # if the side were on lane 0 (the same lane as the entry), the probe would
            # park it directly (the probe parks the selected lane's resident), and the
            # admission's evicting restore (which parks OTHER lanes) would never fire.
            large_tokens = args.prompt_tokens_list[0]
            large_history: list[dict[str, str]] = [
                {"role": "user", "content": _tile_to_tokens(large_tokens, "large")}
            ]
            side_history: list[dict[str, str]] = [
                {"role": "user", "content": _tile_to_tokens(args.side_tokens, "side")}
            ]

            def transient_driver() -> None:
                # Large turn 1 first (admitted to lane 0, a cold prefill). It runs in a
                # thread so the side can be sent while the large is still decoding
                # (lane 0 is occupied), so the side is admitted to lane 1.
                def send_large_turn1() -> None:
                    send_request(0, 1, large_history, args.large_max_tokens)
                    large_history.append({"role": "assistant", "content": "(turn 1 response)"})
                def send_side_after_delay() -> None:
                    # Wait for the large to be admitted to lane 0 and still decoding,
                    # so the side is admitted to lane 1 (not lane 0).
                    time.sleep(args.side_delay)
                    send_request(1, 1, side_history, args.side_max_tokens)
                large_thread = threading.Thread(target=send_large_turn1)
                side_thread = threading.Thread(target=send_side_after_delay)
                large_thread.start()
                side_thread.start()
                large_thread.join()
                side_thread.join()
                # Large follow-ups: the evicting restore parks the small retained lane
                # (lane 1) to restore the large entry instead of re-prefilling.
                for turn in range(2, args.turns + 1):
                    large_history.append({"role": "user", "content": f"Follow-up {turn - 1}: state the main point in one sentence."})
                    send_request(0, turn, large_history, 64)
                    large_history.append({"role": "assistant", "content": f"(turn {turn} response)"})

            threads = [threading.Thread(target=transient_driver)]
        elif args.agentic:
            # Agentic coding: one big multi-turn session (conv 0) + N small one-off
            # requests (convs 1..N) that never come back. The big and the smalls are
            # sent in parallel (turn 1), so the big is on lane 0 and the smalls are on
            # the other lanes. The pool is sized so the smalls evict the big to host;
            # when the big follow-up arrives, the evicting restore parks a small (cheap,
            # it never returns) to restore the big entry instead of re-prefilling. This
            # is the agentic-coding benefit: a big session's re-prefill is avoided by
            # parking small one-offs (a net win, unlike the limit scenario where the
            # parked sibling comes back and the work is conserved).
            big_tokens = args.prompt_tokens_list[0]
            big_history: list[dict[str, str]] = [
                {"role": "user", "content": _tile_to_tokens(big_tokens, "big")}
            ]
            small_histories: list[list[dict[str, str]]] = [
                [{"role": "user", "content": _tile_to_tokens(args.side_tokens, f"small {i}")}]
                for i in range(args.agentic_small_count)
            ]

            def agentic_driver() -> None:
                # Big turn 1 first (admitted to lane 0, a cold prefill). The small
                # one-offs are sent --side-delay seconds later, while the big is still
                # active (lane 0 is occupied and the pool is full), so they queue in
                # the pending queue (needs --max-pending >= small_count). When the big
                # completes, lane 0 is retained (the big's state); the smalls are then
                # admitted one per lane, and the first small's admission parks the big
                # to host (the evict_retained path). The smalls' lanes are retained
                # (idle) when the big follow-up arrives.
                def send_big_turn1() -> None:
                    send_request(0, 1, big_history, args.large_max_tokens)
                    big_history.append({"role": "assistant", "content": "(turn 1 response)"})
                def send_smalls_after_delay() -> None:
                    # Wait for the big to be admitted to lane 0 and still active, so
                    # the smalls queue behind it (not admitted before it, which would
                    # make the big park the smalls instead of the other way around).
                    time.sleep(args.side_delay)
                    small_threads = [threading.Thread(target=send_request,
                                                      args=(i + 1, 1, small_histories[i],
                                                            args.side_max_tokens))
                                     for i in range(args.agentic_small_count)]
                    for t in small_threads:
                        t.start()
                    for t in small_threads:
                        t.join()
                big_thread = threading.Thread(target=send_big_turn1)
                smalls_thread = threading.Thread(target=send_smalls_after_delay)
                big_thread.start()
                smalls_thread.start()
                big_thread.join()
                smalls_thread.join()
                # Big follow-ups: the probe defers (the big entry does not fit
                # alongside the smalls' residents), and the admission's evicting
                # restore parks a small (cheap, it never returns) to restore the big
                # entry instead of re-prefilling.
                for turn in range(2, args.turns + 1):
                    big_history.append({"role": "user", "content": f"Follow-up {turn - 1}: state the main point in one sentence."})
                    send_request(0, turn, big_history, 64)
                    big_history.append({"role": "assistant", "content": f"(turn {turn} response)"})

            threads = [threading.Thread(target=agentic_driver)]
        else:
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
        campaign_start = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        makespan = time.monotonic() - campaign_start
        if failed.is_set():
            raise corpus.CampaignError(f"a benchmark worker failed: {failure_reason[0] if failure_reason else 'unknown'}")
        results.sort(key=lambda r: (r.conversation, r.turn))

    # The host-KV park/restore/defer events are on the server's human-readable log
    # (stdout), not in the JSONL request log.
    deferred = 0
    restored = 0
    try:
        with stdout_log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "restore lane" in line and "deferred" in line:
                    deferred += 1
                elif "restored lane" in line and "from host" in line:
                    restored += 1
    except OSError:
        pass
    # The JSONL request log records each request's reuse path (full_reset = a cold
    # re-prefill, restore_turn_checkpoint = a host-KV restore). Count them.
    full_resets = 0
    restore_reuses = 0
    try:
        with server_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "request_done":
                    continue
                reuse = event.get("result", {}).get("prefix_reuse_path")
                if reuse == "full_reset":
                    full_resets += 1
                elif reuse == "restore_turn_checkpoint":
                    restore_reuses += 1
    except OSError:
        pass
    return {
        "results": results,
        "makespan_seconds": makespan,
        "deferred_probes": deferred,
        "restores": restored,
        # A recovered deferral is a deferred probe that a later restore undid (the
        # entry was restored instead of re-prefilled). In the candidate binary the
        # evicting restore prevents the deferral outright (deferred stays 0), so this
        # is 0; in the baseline the probe's cumulative parking recovers it, so it is
        # min(deferred, restored). The meaningful A/B signal is deferred_probes
        # itself (baseline > 0, candidate 0), not this recovered count.
        "recovered_deferrals": min(deferred, restored),
        "full_resets": full_resets,
        "restore_reuses": restore_reuses,
    }


def _size_class(prompt_tokens: int) -> str:
    if prompt_tokens > 50000:
        return "big"
    if prompt_tokens < 10000:
        return "small"
    return "medium"


def reuse_by_class(server_log: Path) -> dict[str, Any]:
    """Group the server log's request_done records by (turn, size-class) and report,
    per group, how many took the full_reset (re-prefill) vs restore path and the
    server-ttft range for each. Computed directly from the server log (message_count
    -> turn, prompt_tokens -> size-class), so it does not depend on matching the
    client's requests to the server's request ids (which differ between the two
    binaries). This is the per-request comparison that isolates the evicting
    restore's benefit (a big-context follow-up the baseline re-prefills but the
    candidate restores) from the limit (a fully-saturated follow-up both re-prefill).
    """
    groups: dict[tuple[int, str], list[tuple[str, float | None]]] = {}
    try:
        with server_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "request_done":
                    continue
                req = event.get("request", {})
                result = event.get("result", {})
                msg_count = req.get("message_count")
                prompt_tokens = result.get("prompt_tokens")
                if msg_count is None or prompt_tokens is None:
                    continue
                turn = (msg_count + 1) // 2
                reuse = result.get("prefix_reuse_path")
                ttft = event.get("timings_seconds", {}).get("ttft")
                groups.setdefault((turn, _size_class(prompt_tokens)), []).append((reuse, ttft))
    except OSError:
        return {}
    out: dict[str, Any] = {}
    for (turn, size), records in sorted(groups.items()):
        def _ttft_range(subset: list[tuple[str, float | None]]) -> list[float] | None:
            vals = [ttft for _reuse, ttft in subset if ttft is not None]
            return [round(min(vals), 1), round(max(vals), 1)] if vals else None
        full = [rec for rec in records if rec[0] == "full_reset"]
        restore = [rec for rec in records if rec[0] == "restore_turn_checkpoint"]
        out[f"turn{turn}_{size}"] = {
            "full_reset": len(full),
            "restore": len(restore),
            "full_reset_ttft_s": _ttft_range(full),
            "restore_ttft_s": _ttft_range(restore),
        }
    return out


def summarize(label: str, report: dict[str, Any], server_log: Path) -> dict[str, Any]:
    results: list[TurnResult] = report["results"]
    first_turn = [r.ttft_seconds for r in results if r.turn == 1]
    follow_ups = sorted(r.ttft_seconds for r in results if r.turn > 1)
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        mid = len(values) // 2
        return round(values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2, 2)
    return {
        "label": label,
        "conversations": report["results"][0].conversation if results else 0,
        "makespan_seconds": round(report["makespan_seconds"], 2),
        "first_turn_ttft_mean_s": round(sum(first_turn) / len(first_turn), 2) if first_turn else None,
        "follow_up_ttft_mean_s": round(sum(follow_ups) / len(follow_ups), 2) if follow_ups else None,
        "follow_up_ttft_median_s": _median(follow_ups),
        "follow_up_ttft_min_s": round(min(follow_ups), 2) if follow_ups else None,
        "follow_up_ttft_max_s": round(max(follow_ups), 2) if follow_ups else None,
        "deferred_probes": report["deferred_probes"],
        "restores": report["restores"],
        "recovered_deferrals": report["recovered_deferrals"],
        "full_resets": report["full_resets"],
        "restore_reuses": report["restore_reuses"],
        "reuse_by_class": reuse_by_class(server_log),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.prompt_tokens_list = [int(x) for x in args.prompt_tokens.split(",") if x.strip()]
    if not args.prompt_tokens_list:
        raise corpus.CampaignError("--prompt-tokens must name at least one size")
    artifacts = corpus.parse_artifacts(args.artifact)
    if len(artifacts) != 1:
        raise corpus.CampaignError("exactly one --artifact is required")
    target, artifact = artifacts[0]
    if not args.serve_baseline.is_file() or not args.serve_candidate.is_file():
        raise corpus.CampaignError("both serve binaries must exist")
    args.output.mkdir(parents=True, exist_ok=True)

    runs = [
        ("baseline", args.serve_baseline, args.baseline_host_kv_mib),
        ("candidate", args.serve_candidate, args.host_kv_mib),
    ]
    summaries = []
    for label, serve, host_kv_mib in runs:
        server_log = args.output / f"server_{label}.jsonl"
        stdout_log = args.output / f"server_{label}.log"
        print(f"=== {label}: {serve.name} (C={args.concurrency}, "
              f"{args.conversations} conversations x {args.turns} turns, "
              f"prompt={args.prompt_tokens} tok, kv={args.kv_capacity}, host-kv={host_kv_mib} MiB) ===",
              flush=True)
        report = run_workload(serve, artifact, args, server_log, stdout_log, host_kv_mib)
        summary = summarize(label, report, server_log)
        summaries.append(summary)
        print(f"  makespan={summary['makespan_seconds']}s "
              f"follow-up TTFT mean={summary['follow_up_ttft_mean_s']}s "
              f"median={summary['follow_up_ttft_median_s']}s "
              f"[{summary['follow_up_ttft_min_s']}-{summary['follow_up_ttft_max_s']}] "
              f"deferred={summary['deferred_probes']} restores={summary['restores']} "
              f"full_resets={summary['full_resets']} restore_reuses={summary['restore_reuses']}",
              flush=True)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.output / "summary.json").write_text(
        json.dumps({"target": target, "artifact": str(artifact), "args": args_dict,
                    "runs": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline = next((s for s in summaries if s["label"] == "baseline"), None)
    candidate = next((s for s in summaries if s["label"] == "candidate"), None)
    if baseline and candidate and baseline["follow_up_ttft_mean_s"] and candidate["follow_up_ttft_mean_s"]:
        speedup = baseline["follow_up_ttft_mean_s"] / candidate["follow_up_ttft_mean_s"]
        print(f"\nfollow-up TTFT speedup (baseline/candidate): {speedup:.1f}x "
              f"({baseline['follow_up_ttft_mean_s']}s -> {candidate['follow_up_ttft_mean_s']}s)",
              flush=True)
    if baseline and candidate:
        recovered = baseline["full_resets"] - candidate["full_resets"]
        print(f"recovered re-prefills (baseline full_reset - candidate full_reset): {recovered} "
              f"(baseline {baseline['full_resets']} -> candidate {candidate['full_resets']}; "
              f"deferred probes {baseline['deferred_probes']} -> {candidate['deferred_probes']})",
              flush=True)
        # Per (turn, size-class) comparison: for the same logical request, whether the
        # baseline re-prefilled (full_reset) and the candidate restored. This isolates
        # the evicting restore's benefit from its limit.
        b_class = baseline["reuse_by_class"]
        c_class = candidate["reuse_by_class"]
        for key in sorted(set(b_class) | set(c_class)):
            b = b_class.get(key, {})
            c = c_class.get(key, {})
            print(f"  {key}: baseline full_reset={b.get('full_reset',0)} "
                  f"({b.get('full_reset_ttft_s')}) restore={b.get('restore',0)} "
                  f"({b.get('restore_ttft_s')}) | candidate full_reset={c.get('full_reset',0)} "
                  f"({c.get('full_reset_ttft_s')}) restore={c.get('restore',0)} "
                  f"({c.get('restore_ttft_s')})",
                  flush=True)
    print(f"summary: {args.output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except corpus.CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)