from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tools.bench import run_serve_concurrency as campaign
from tools.bench import run_serve_corpus as corpus
from tools.bench.run_serve_corpus import (
    CampaignError,
    Fixture,
    RunSpec,
    build_result_record,
    require_server_log_identity,
    summary_row,
)


def test_request_log_v20_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 20,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=19)
    with pytest.raises(CampaignError):
        require_server_log_identity(stale, "server_start")


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
        "schema_version": 20,
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

    record = build_result_record(spec, "groupwise-int", payload, response, event)
    assert record["schema_version"] == 6
    assert record["metrics"]["engine_host_exposed_ms"] == pytest.approx(15.0)
    assert record["metrics"]["decode_host_us_per_round"] == pytest.approx(5000.0)
    assert record["metrics"]["decode_device_wait_us_per_round"] == pytest.approx(100000.0)


def test_summary_retains_one_canonical_weights_id() -> None:
    records = [{"weights_id": "nvfp4", "metrics": {}}]
    row = summary_row(
        "context_profile",
        "qwen3_6_27b",
        "fixture",
        "fixture",
        "mtp0",
        "greedy",
        records,
    )
    assert row["weights_id"] == "nvfp4"

    with pytest.raises(CampaignError):
        summary_row(
            "context_profile",
            "qwen3_6_27b",
            "fixture",
            "fixture",
            "mtp0",
            "greedy",
            [*records, {"weights_id": "groupwise-int", "metrics": {}}],
        )


def test_dflash2_c1_keeps_the_published_corpus_and_order() -> None:
    point = campaign.Point(
        target="qwen3_8_27b",
        model_id="qwen3.8-27b",
        artifact=Path("model.ninfer"),
        speculative_mode="dflash2_7",
        speculative_backend="dflash2",
        draft_tokens=7,
        sampling_mode="stochastic",
        suite="corpus-makespan",
        concurrency=1,
    )
    fixtures = corpus.load_fixtures()
    jobs = campaign.build_jobs(point, fixtures, 8192)
    reference = campaign.build_jobs(
        replace(point, speculative_mode="mtp3", speculative_backend="mtp", draft_tokens=3),
        fixtures,
        8192,
    )
    assert len(jobs) == 75
    assert [(j.fixture.name, j.seed, j.max_tokens) for j in jobs] == [
        (j.fixture.name, j.seed, j.max_tokens) for j in reference
    ]
    assert all(j.max_tokens == (65536 if j.fixture.thinking else 4096) for j in jobs)
    args = campaign.parse_args(
        [
            "--artifact", "qwen3_8_27b=model.ninfer",
            "--mode", "dflash2_7",
            "--suite", "corpus-makespan",
            "--concurrency", "1",
            "--max-context", "131072",
            "--kv-capacity", "auto",
            "--output", "results",
        ]
    )
    command = campaign.server_command(Path("ninfer-serve"), point, Path("server.jsonl"), args)
    for flag, value in [
        ("--spec", "dflash2"),
        ("--draft-tokens", "7"),
        ("--max-concurrency", "1"),
        ("--max-context", "131072"),
        ("--kv-dtype", "int8"),
        ("--temperature", "0.6"),
        ("--presence-penalty", "1.0"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--lm-head-draft" in command and "--no-prefix-reuse" in command
