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


@pytest.mark.parametrize(
    "section,field",
    [
        ("scheduling", "preemptions"),
        ("scheduling", "snapshot_restores"),
        ("scheduling", "replay_restores"),
        ("scheduling", "replayed_tokens"),
        ("scheduler", "paused"),
        ("scheduler", "replaying"),
    ],
)
def test_steady_throughput_excludes_recovery(section: str, field: str) -> None:
    steady = {
        "interval_seconds": 1.0,
        "tokens": {"computed_prefill": 0, "committed_decode": 400},
        "decode_batch": {"rounds": 50, "row_rounds": 400},
        "scheduler": {
            "running": 8,
            "prefilling": 0,
            "decode_ready": 8,
            "paused": 0,
            "replaying": 0,
        },
        "scheduling": {
            "preemptions": 0,
            "snapshot_restores": 0,
            "replay_restores": 0,
            "replayed_tokens": 0,
        },
    }
    recovery = {
        **steady,
        "interval_seconds": 10.0,
        section: {**steady[section], field: 1},
    }
    resumed = {
        **steady,
        "tokens": {"computed_prefill": 0, "committed_decode": 800},
        "decode_batch": {"rounds": 100, "row_rounds": 800},
    }

    assert concurrency.steady_metrics([steady, recovery, resumed], 8) == {
        "intervals": 2,
        "seconds": 2.0,
        "committed_decode_tokens": 1200,
        "decode_rounds": 150,
        "decode_row_rounds": 1200,
        "average_decode_batch": 8.0,
        "decode_tokens_per_second": 600.0,
    }
    with pytest.raises(corpus.CampaignError, match="no recovery-free full-batch"):
        concurrency.steady_metrics([recovery], 8)


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
        "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION,
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
                "session_call": 0.002,
                "executor_poll": 0.003,
                "engine_commit_output": 0.004,
                "engine_maintenance": 0.005,
                "total": 0.015,
            },
            "scheduler_yield_exposed_seconds": 0.3,
            "decode": {
                "host_exposed_seconds": 0.01,
                "scheduler_yield_exposed_seconds": 0.2,
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
    assert record["metrics"]["decode_scheduler_yield_us_per_round"] == pytest.approx(
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
            proposal_head="optimized",
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
@pytest.mark.parametrize(
    "concurrent,mode,head",
    [
        (False, "mtp0", "full"),
        (True, "mtp0", "full"),
        (False, "mtp5", "optimized"),
        (True, "mtp5", "optimized"),
        (True, "mtp5", "full"),
    ],
)
def test_selected_kv_reaches_server_and_is_verified(
    tmp_path, kv_dtype, cache_name, concurrent, mode, head
):
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    common = [
        "--serve",
        "serve",
        "--artifact",
        f"model={artifact}",
        "--output",
        str(tmp_path),
        "--mode",
        mode,
        "--kv-dtype",
        kv_dtype,
    ]
    backend, draft_tokens = corpus.SPECULATIVE_MODES[mode]
    engine = {
        "device": 0,
        "max_context": 262144,
        "kv_capacity": 262144,
        "prefill_chunk": 1024,
        "kv_cache": cache_name,
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": backend,
        "speculative_draft_window": draft_tokens,
        "proposal_head": head,
        "context_cache": {"extra_state_slots": 0, "host_capacity_bytes": 0},
    }
    event = {
        "artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
        "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION,
        "event": "server_start",
        "engine": engine,
        "sampling_defaults": {"server_overrides": {"temperature": 0.6}},
        "artifact": {"path": str(artifact), "prefill_signature": "bindings"},
        "server": {"public_model_id": "model"},
        "server_instance_id": "instance",
    }
    if concurrent:
        args = concurrency.parse_args(
            common + ["--suite", "decode-saturation", "--concurrency", "1",
                      "--proposal-head", head]
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
            mode,
            backend,
            draft_tokens,
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
    assert command[command.index("--extra-state-slots") + 1] == "0"
    assert command[command.index("--host-cache-capacity") + 1] == "0"
    assert ("--proposal-head" in command) == (head == "optimized")
    assert validate() == ("instance", "bindings")
    for field in ("extra_state_slots", "host_capacity_bytes"):
        engine["context_cache"][field] = 1
        with pytest.raises(corpus.CampaignError, match="context cache capacity differs"):
            validate()
        engine["context_cache"][field] = 0
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
