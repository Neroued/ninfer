from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import json

import pytest

from tools.bench.run_serve_corpus import (
    Fixture,
    RunSpec,
    build_result_record,
    parse_artifacts,
)
from tools.bench.run_serve_concurrency import build_points
from tools.bench import run_serve_corpus as corpus
from tools.bench import run_serve_concurrency as concurrency


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
        kv_dtype="fp8",
        fixture=fixture,
        seed=7,
    )
    payload = {"model": spec.model_id}
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 22,
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

    record = build_result_record(
        spec, "measured-prefill-bindings", payload, response, event
    )
    assert record["schema_version"] == 8
    assert record["kv_dtype"] == "fp8"
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
        Namespace(
            mode=["dflash7", "dflash2_7"],
            suite=["decode-saturation"],
            concurrency=[1],
            sampling="greedy",
            kv_dtype="fp8",
        ),
    )
    assert [(point.target, point.speculative_backend) for point in points] == [
        ("org/custom", "dflash"),
        ("org/custom", "dflash2"),
        ("org%2Fcustom", "dflash"),
        ("org%2Fcustom", "dflash2"),
    ]
    assert all(
        point.artifact == artifact and point.model_id == point.target
        for point in points
    )
    assert len({point.key for point in points}) == len(points)
    for point in points:
        (tmp_path / f"{point.key}.json").write_text("{}")


@pytest.mark.parametrize(
    "kv_dtype,cache_name", [("int8", "int8-group64"), ("fp8", "fp8-e4m3-row256")]
)
@pytest.mark.parametrize("concurrent", [False, True])
def test_selected_kv_reaches_server_and_is_verified(
    tmp_path, kv_dtype, cache_name, concurrent
):
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    common = [
        "--artifact",
        f"model={artifact}",
        "--output",
        str(tmp_path),
        "--mode",
        "mtp0",
        "--kv-dtype",
        kv_dtype,
    ]
    engine = {
        "device": 0,
        "max_context": 262144,
        "kv_capacity": 262144,
        "prefill_chunk": 1024,
        "kv_cache": cache_name,
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": "none",
        "speculative_draft_window": 0,
        "proposal_head": "full",
    }
    event = {
        "artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
        "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION,
        "event": "server_start",
        "engine": engine,
        "sampling_defaults": {"greedy": False},
        "artifact": {"path": str(artifact), "prefill_signature": "bindings"},
        "server": {"public_model_id": "model"},
        "server_instance_id": "instance",
    }
    if concurrent:
        args = concurrency.parse_args(
            common + ["--suite", "decode-saturation", "--concurrency", "1"]
        )
        point = build_points([("model", artifact)], args)[0]
        command = concurrency.server_command(
            Path("serve"), point, tmp_path / "log", args
        )
        engine.update(
            kv_capacity_mode="explicit",
            max_concurrency=1,
            max_pending_requests=1,
            pending_timeout_ms=concurrency.PENDING_TIMEOUT_MS,
            log_stats_interval_ms=concurrency.STATS_INTERVAL_MS,
        )
        validate = lambda: concurrency.validate_server_start(event, point, args)
    else:
        args = corpus.parse_args(common)
        spec = RunSpec(
            "model",
            "model",
            artifact,
            "mtp0",
            "none",
            0,
            "stochastic",
            args.kv_dtype,
            Fixture("fixture", [], False, 8, "test"),
            7,
        )
        command = corpus.server_command(
            Path("serve"), spec, tmp_path / "log", args.port, args.device
        )
        validate = lambda: corpus.validate_server_start(event, spec, args.device)
    assert command[command.index("--kv-dtype") + 1] == kv_dtype
    assert validate() == ("instance", "bindings")
    engine["kv_cache"] = "bf16"
    with pytest.raises(corpus.CampaignError, match="configuration mismatch"):
        validate()


def test_resume_rejects_different_kv_dtype(tmp_path):
    spec = RunSpec(
        "model",
        "model",
        tmp_path / "model.ninfer",
        "mtp0",
        "none",
        0,
        "stochastic",
        "fp8",
        Fixture("fixture", [], False, 8, "test"),
        7,
    )
    record = {
        "artifact_type": corpus.RUN_ARTIFACT_TYPE,
        "schema_version": corpus.RUN_SCHEMA_VERSION,
        "target": spec.target,
        "speculative_mode": spec.speculative_mode,
        "sampling_mode": spec.sampling_mode,
        "fixture": spec.fixture.name,
        "seed": spec.seed,
        "artifact_path": str(spec.artifact),
        "kv_dtype": "int8",
    }
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(corpus.CampaignError, match="KV dtype differs"):
        corpus.load_existing_records(path, {spec.key: spec})
