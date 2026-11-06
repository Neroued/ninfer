"""Execution measurements from the public Serve observation endpoint."""

from __future__ import annotations

import http.client
import json
import time
from typing import Any


def snapshot(port: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", "/monitor/snapshot")
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"monitor/snapshot returned HTTP {response.status}")
        return json.loads(body)
    finally:
        connection.close()


def wait_idle(port: int, completed: int) -> dict[str, Any]:
    """A response can precede its final asynchronous retirement."""
    deadline = time.monotonic() + 30
    while True:
        value = snapshot(port)
        engine = value["engine"]
        if (
            engine["runtime"]["requests"]["completed"] >= completed
            and not engine["in_flight"]
            and not engine["requests"]
        ):
            return value
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Serve did not retire the completed measurement requests"
            )
        time.sleep(0.05)


def subtract(before: Any, after: Any) -> Any:
    if isinstance(after, dict):
        return {key: subtract(before[key], value) for key, value in after.items()}
    if isinstance(after, list):
        if len(before) != len(after):
            raise ValueError("observation counter shape changed during measurement")
        return [subtract(a, b) for a, b in zip(before, after)]
    result = after - before
    if result < 0:
        raise ValueError("observation counter decreased during measurement")
    return result


def execution_delta(before: dict, after: dict) -> dict:
    fields = (
        "units",
        "rows",
        "input_tokens",
        "scheduled_positions",
        "latency_bins",
        "result",
        "resolution",
        "retirement",
        "total",
    )
    result = {key: subtract(before[key], after[key]) for key in fields}
    result["mean_ms"] = (
        result["total"]["total_ns"] / 1e6 / result["units"] if result["units"] else None
    )
    result["stage_mean_ms"] = {
        key: (
            result[key]["total_ns"] / 1e6 / result["units"] if result["units"] else None
        )
        for key in ("result", "resolution", "retirement")
    }
    return result


def measurement(before: dict, after: dict) -> dict:
    if before["server_instance_id"] != after["server_instance_id"]:
        raise ValueError("Serve instance changed during measurement")
    a, b = before["engine"], after["engine"]
    if a["execution_latency_bounds_ns"] != b["execution_latency_bounds_ns"]:
        raise ValueError("execution histogram bounds changed during measurement")
    previous = {row["batch"]: row for row in a["decode_by_batch"]}
    batches = [
        {"batch": row["batch"], **execution_delta(previous[row["batch"]], row)}
        for row in b["decode_by_batch"]
    ]
    previous_ranks = {rank["rank"]: rank for rank in a["ranks"]}
    ranks = []
    for rank in b["ranks"]:
        old = previous_ranks[rank["rank"]]
        value = {"rank": rank["rank"]}
        for key in (
            "gpu_intervals",
            "attention_work",
            "communication_eager",
            "communication_graph",
        ):
            value[key] = subtract(old[key], rank[key])
        ranks.append(value)
    return {
        "runtime": {
            key: subtract(a["runtime"][key], b["runtime"][key])
            for key in (
                "computed_prefill_tokens",
                "committed_decode_tokens",
                "decode_rounds",
                "decode_row_rounds",
                "reused_prompt_tokens",
                "preemptions",
                "snapshot_restores",
                "replay_restores",
            )
        },
        "execution_latency_bounds_ns": b["execution_latency_bounds_ns"],
        "executions": {
            key: execution_delta(a["executions"][key], value)
            for key, value in b["executions"].items()
        },
        "decode_by_batch": batches,
        "ranks": ranks,
    }
