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
from tools.bench import serve_measurement as observation
from tools.bench.ttft.diagnostics import attach_generation_diagnostics


def test_configured_deployment_and_glm_workloads(tmp_path):
    config = tmp_path / "serve.toml"
    config.write_text("""
[engine]
max_context = 1048576
max_concurrency = 8
prefill_chunk = 4096
kv_dtype = "fp8"
[distributed.parallel]
tp = 2
adp = 2
[speculative]
backend = "mtp"
draft_tokens = 3
""")
    args = concurrency.parse_args(
        [
            "--serve",
            "serve",
            "--artifact",
            "glm=/model.ninfer",
            "--config",
            str(config),
            "--mode",
            "mtp3",
            "--thinking",
            "on",
            "--output",
            str(tmp_path),
            "--suite",
            "context-profile",
            "--suite",
            "decode-saturation",
            "--suite",
            "corpus-makespan",
            "--concurrency",
            "8",
            "--concurrency",
            "4",
            "--concurrency",
            "2",
            "--concurrency",
            "1",
        ]
    )
    points = build_points([("glm", Path("/model.ninfer"))], args)
    fixtures = corpus.load_fixtures(args.thinking)
    jobs = [
        concurrency.build_jobs(point, fixtures, args.decode_tokens) for point in points
    ]
    assert len(points) == 9 and sum(map(len, jobs)) == 335
    assert all(job.fixture.thinking for batch in jobs for job in batch)
    assert len(jobs[0]) == 20 and points[0].concurrency == 1
    assert all(
        job.max_tokens == 8192 and job.fixture.max_new == 8192 for job in jobs[1]
    )
    command = concurrency.server_command(
        Path("serve"), points[-1], tmp_path / "log", args
    )
    for flag, value in (
        ("--config", str(config)),
        ("--max-context", "1048576"),
        ("--max-concurrency", "8"),
        ("--prefill-chunk", "4096"),
        ("--kv-dtype", "fp8"),
    ):
        assert command[command.index(flag) + 1] == value
    override = corpus.parse_args(
        [
            "--serve",
            "serve",
            "--artifact",
            "glm=/model.ninfer",
            "--config",
            str(config),
            "--prefill-chunk",
            "2048",
            "--mode",
            "mtp0",
            "--output",
            str(tmp_path),
        ]
    )
    spec = RunSpec(
        "glm",
        "glm",
        Path("/model.ninfer"),
        "mtp0",
        "none",
        0,
        "greedy",
        "fp8",
        fixtures[corpus.WARMUP_FIXTURE],
        7,
    )
    command = corpus.server_command(Path("serve"), spec, tmp_path / "log", override)
    assert command[command.index("--prefill-chunk") + 1] == "2048"
    assert command[command.index("--spec") + 1] == "none"


def test_observation_measures_actual_batch_counters():
    def execution(count, ns):
        return dict(
            units=count,
            rows=count * 4,
            input_tokens=count * 4,
            scheduled_positions=count * 16,
            latency_bins=[count],
            total=dict(count=count, total_ns=ns),
            result=dict(count=count, total_ns=ns - count * 1000),
            resolution=dict(count=count, total_ns=0),
            retirement=dict(count=count, total_ns=count * 1000),
        )

    def snapshot(count, ns):
        counters = {
            key: count
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
        }
        return {
            "server_instance_id": "same",
            "engine": {
                "runtime": counters,
                "execution_latency_bounds_ns": [10000000],
                "executions": {"decode": execution(count, ns)},
                "decode_by_batch": [
                    {"batch": 1, **execution(0, 0)},
                    {"batch": 4, **execution(count, ns)},
                ],
                "ranks": [
                    {
                        "rank": rank,
                        "gpu_intervals": {"target": {"count": count, "total_ns": ns}},
                        "attention_work": {"decode": {"rows": count * (rank + 1)}},
                        "communication_eager": {},
                        "communication_graph": {},
                    }
                    for rank in (1, 0)
                ],
            },
        }

    before, after = snapshot(3, 21000000), snapshot(5, 37000000)
    measured = observation.measurement(before, after)
    assert measured["decode_by_batch"][0]["mean_ms"] is None
    actual = measured["decode_by_batch"][1]
    assert actual["units"] == 2 and actual["mean_ms"] == 8.0
    assert sum(actual["stage_mean_ms"].values()) == pytest.approx(8.0)
    assert [
        (r["rank"], r["attention_work"]["decode"]["rows"]) for r in measured["ranks"]
    ] == [(1, 4), (0, 2)]
    after["server_instance_id"] = "restarted"
    with pytest.raises(ValueError, match="instance changed"):
        observation.measurement(before, after)


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
    missing = concurrency.steady_metrics([recovery], 8)
    assert missing["decode_rounds"] == 0
    assert missing["decode_tokens_per_second"] is None


@pytest.mark.parametrize(
    "vision_samples,vision_seconds,expected_vision",
    [(0, 0.0, None), (1, 0.0, 0.0), (2, 0.125, 0.125)],
)
def test_result_record_parses_production_timings(
    vision_samples: int,
    vision_seconds: float,
    expected_vision: float | None,
) -> None:
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
            "ttft": 0.875,
            "vision_stream": vision_seconds,
            "prefill": 0.2,
            "decode": 0.4,
            "total": 1.3,
        },
        "execution": {
            "max_rank_gpu_intervals": {"vision_stream": {"count": vision_samples}},
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
                "executor_flush": 0.006,
                "engine_commit_output": 0.004,
                "engine_maintenance": 0.005,
                "total": 0.021,
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
    assert record["schema_version"] == 9
    assert record["kv_dtype"] == "fp8"
    assert record["metrics"]["server_ttft_ms"] == pytest.approx(875.0)
    assert record["metrics"]["vision_stream_seconds"] == expected_vision
    assert record["metrics"]["vision_stream_samples"] == vision_samples
    assert record["metrics"]["executor_flush_exposed_seconds"] == pytest.approx(0.006)
    assert record["metrics"]["engine_host_exposed_ms"] == pytest.approx(21.0)
    assert record["metrics"]["decode_host_us_per_round"] == pytest.approx(5000.0)
    assert record["metrics"]["decode_scheduler_yield_us_per_round"] == pytest.approx(
        100000.0
    )


@pytest.mark.parametrize("resource_case", ["single", "pair", "missing", "partial"])
def test_startup_resources_reach_concurrency_and_ttft_reports(
    tmp_path: Path,
    resource_case: str,
) -> None:
    ranks = [
        {
            "rank": rank,
            "node": rank,
            "device": 0,
            "hardware": "test-gpu",
            "shared_physical_memory": True,
            "budget": {"context_capacity_bytes": (rank + 1) * 1024},
            "resources": {"allocations": {"context_device_bytes": (rank + 1) * 1024}},
        }
        for rank in range(1 if resource_case == "single" else 2)
    ]
    resources = {
        "ranks": ranks,
        "total_allocations": {
            "context_device_bytes": sum(
                rank["budget"]["context_capacity_bytes"] for rank in ranks
            ),
        },
    }
    start = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION,
        "event": "server_start",
        "server_instance_id": "serve-test",
    }
    if resource_case != "missing":
        start["resources"] = resources
    if resource_case == "partial":
        del ranks[-1]["resources"]

    fixture = Fixture("fixture", [], False, 2, "test")
    point = concurrency.Point(
        "model",
        "model",
        tmp_path / "model.ninfer",
        "mtp0",
        "none",
        0,
        "full",
        "greedy",
        "fp8",
        "corpus-makespan",
        1,
    )
    done = {
        "event": "request_done",
        "result": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "computed_prefill_tokens": 10,
        },
        "speculative": {
            "rounds": 0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "fallback_steps": 0,
        },
    }
    interval = {
        "event": "throughput",
        "tokens": {"computed_prefill": 10, "committed_decode": 1},
        "decode_batch": {"rounds": 1, "row_rounds": 1},
    }
    result = concurrency.ClientResult(
        concurrency.Job(0, 0, fixture, 1, 2),
        0.0,
        1.0,
        10,
        2,
        "output_limit",
    )
    log = tmp_path / "server.jsonl"
    point_report = concurrency.analyze_point(
        point,
        [],
        log,
        start,
        "bindings",
        [done, interval],
        [result],
        0.0,
        1.0,
    )
    # Startup resources remain usable even when runtime intervals have not been logged yet.
    log.write_text(json.dumps(start) + "\n")
    run = {"requests": []}
    assert attach_generation_diagnostics(run, log) is None
    observed = run["global_diagnostics"]
    assert observed["reason"] == "runtime_intervals_missing"
    if resource_case in ("single", "pair"):
        expected = {"status": "available", **resources}
    else:
        reason = (
            "startup_resources_missing"
            if resource_case == "missing"
            else "startup_resources_invalid"
        )
        expected = {"status": "unavailable", "reason": reason}
    assert point_report["resources"] == expected
    assert observed["startup_resources"] == expected


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


@pytest.mark.parametrize("kv_dtype", ["int8", "fp8"])
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
    tmp_path, kv_dtype, concurrent, mode, head
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
        "context_capacity_mode": "auto",
        "prefill_chunk": 1024,
        "kv_cache": kv_dtype,
        "cuda_graph": True,
        "prefix_reuse": False,
        "speculative_backend": backend,
        "speculative_draft_window": draft_tokens,
        "proposal_head": head,
        "context_cache": {"host_capacity_bytes": 0},
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
            common
            + [
                "--suite",
                "decode-saturation",
                "--concurrency",
                "1",
                "--proposal-head",
                head,
            ]
        )
        point = build_points([("model", artifact)], args)[0]
        command = concurrency.server_command(
            Path("serve"), point, tmp_path / "log", args
        )
        engine.update(
            context_capacity_mode="auto",
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
        command = corpus.server_command(Path("serve"), spec, tmp_path / "log", args)
        validate = lambda: corpus.validate_server_start(event, spec, args)
    assert command[command.index("--kv-dtype") + 1] == kv_dtype
    assert command[command.index("--context-capacity") + 1] == "auto"
    assert command[command.index("--host-cache-capacity") + 1] == "0"
    if "--proposal-head" in command:
        assert command[command.index("--proposal-head") + 1] == head
    assert validate() == ("instance", "bindings")
    for field in ("host_capacity_bytes",):
        engine["context_cache"][field] = 1
        with pytest.raises(
            corpus.CampaignError, match="context cache capacity differs"
        ):
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
