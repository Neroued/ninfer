from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest

from tools.bench.run_serve_corpus import (
    Fixture,
    RunSpec,
    build_result_record,
    parse_artifacts,
)
from tools.bench.run_serve_concurrency import build_points


def test_result_record_parses_request_host_exposure() -> None:
    fixture = Fixture(
        name="fixture",
        messages=[],
        thinking=True,
        max_new=8,
        suite="test",
    )
    spec = RunSpec(
        target="qwen3_6_27b",
        model_id="qwen3.6-27b",
        artifact=Path("/tmp/model.ninfer"),
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="greedy",
        fixture=fixture,
        seed=7,
    )
    payload = {"model": spec.model_id}
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 21,
        "event": "request_done",
        "request": {
            "model": spec.model_id,
            "requested_output_tokens": 8,
            "enable_thinking": True,
            "sampling": {"seed": 7},
        },
        "result": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "finish_reason": "output_limit",
        },
        "timings_seconds": {
            "prepare": 0.1,
            "vision": 0.0,
            "prefill": 0.2,
            "decode": 0.4,
            "total": 0.7,
        },
        "speculative": {
            "backend": "mtp",
            "rounds": 2,
            "drafted_tokens": 6,
            "accepted_tokens": 3,
            "fallback_steps": 0,
        },
        "engine_timing": {
            "queue_wait_seconds": 0.001,
            "host_exposed_seconds": {
                "engine_boundary": 0.001,
                "program_submit": 0.002,
                "program_post": 0.003,
                "engine_commit_output": 0.004,
                "engine_maintenance": 0.005,
                "total": 0.015,
            },
            "device_wait_exposed_seconds": 0.3,
            "decode": {
                "host_exposed_seconds": 0.01,
                "device_wait_exposed_seconds": 0.2,
                "rounds": 2,
            },
        },
    }

    record = build_result_record(spec, "measured-prefill-bindings", payload, response, event)
    assert record["schema_version"] == 7
    assert record["metrics"]["engine_host_exposed_ms"] == pytest.approx(15.0)
    assert record["metrics"]["decode_host_us_per_round"] == pytest.approx(5000.0)
    assert record["metrics"]["decode_device_wait_us_per_round"] == pytest.approx(
        100000.0
    )


def test_arbitrary_artifact_labels_reach_the_requested_backend(tmp_path: Path) -> None:
    artifact = tmp_path / "custom.ninfer"
    artifact.touch()
    artifacts = parse_artifacts([f"org/custom={artifact}", f"org%2Fcustom={artifact}"])
    points = build_points(
        artifacts,
        Namespace(mode=["dflash7", "dflash2_7"], suite=["decode-saturation"],
                  concurrency=[1], sampling="greedy"),
    )
    assert [(point.target, point.speculative_backend) for point in points] == [
        ("org/custom", "dflash"), ("org/custom", "dflash2"),
        ("org%2Fcustom", "dflash"), ("org%2Fcustom", "dflash2"),
    ]
    assert all(point.artifact == artifact and point.model_id == point.target for point in points)
    assert len({point.key for point in points}) == len(points)
    for point in points:
        (tmp_path / f"{point.key}.json").write_text("{}")
