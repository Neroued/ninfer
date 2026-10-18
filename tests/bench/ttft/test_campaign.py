from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import pytest

from tools.bench import run_serve_ttft_campaign as campaign


def test_profile_config_does_not_inherit_product_arguments(tmp_path: Path) -> None:
    config = tmp_path / "baseline.json"
    config.write_text(json.dumps({"profiles": {"old": ["--old-option", "4"]}}))
    common, profiles = campaign._load_profiles(config)
    assert common == ()
    assert profiles == {"old": ("--old-option", "4")}

    config.write_text(json.dumps({"profiles": {"old": ["--old-option", 4]}}))
    with pytest.raises(campaign.CampaignError, match="argument strings"):
        campaign._load_profiles(config)


def test_campaign_records_and_runs_explicit_baseline_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    serve = tmp_path / "baseline-serve"
    artifact = tmp_path / "custom.ninfer"
    serve.touch()
    artifact.write_bytes(b"artifact")
    config = tmp_path / "baseline.json"
    profile = campaign.CASES["cold-short"].profile
    common_args = ["--kv-dtype", "bf16", "--log-stats-interval-ms", "0"]
    profile_args = [
        "--max-context", "8192", "--old-capacity", "7",
        "--host-context-mib", "513.125", "--log-stats-interval-ms=2000",
    ]
    config.write_text(json.dumps({
        "common_args": common_args,
        "profiles": {profile: profile_args},
    }))
    started: list[list[str]] = []

    def running(command: list[str], *_args: object) -> object:
        started.append(command)
        return nullcontext()

    def run_client(command: list[str], _progress: Path) -> int:
        raw = Path(command[command.index("--output") + 1])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps({
            "artifact_type": "ninfer_serve_ttft_run",
            "server": {"model": "custom"},
            "constructed": True,
            "requests": [],
        }))
        return 0

    monkeypatch.setattr(campaign, "RunningServe", running)
    monkeypatch.setattr(campaign, "_run_client", run_client)
    monkeypatch.setattr(
        campaign, "write_campaign_summary",
        lambda output: ({}, {"markdown": output / "summary.md"}),
    )

    assert campaign.main([
        "--case", "cold-short", "--serve", "baseline-serve",
        "--artifact", "custom.ninfer", "--profile-config", "baseline.json",
        "--output-dir", "results",
    ]) == 0

    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert started == [manifest["plans"][0]["server_command"]]
    assert started[0] == [
        str(serve), str(artifact), "--host", campaign.HOST,
        "--port", str(campaign.PORT), "--request-log-jsonl",
        manifest["plans"][0]["request_log_jsonl"],
        "--kv-dtype", "bf16", "--max-context", "8192", "--old-capacity", "7",
        "--host-context-mib", "513.125", "--log-stats-interval-ms", "1000",
    ]
    assert manifest["serve"] == str(serve)
    assert manifest["weights_source"] == str(artifact)
    assert manifest["profile_config"] == str(config)
    assert manifest["common_args"] == common_args
    assert manifest["profile_args"] == {profile: profile_args}
    assert manifest["stats_interval_ms"] == 1000
    assert manifest["stats_interval_source"] == "campaign_override"
    assert manifest["model_profile"] == "artifact:custom"
    assert manifest["kv_dtype"] == "bf16"
    assert manifest["profile_kv_dtypes"] == {profile: "bf16"}


def test_missing_baseline_profile_fails_before_starting_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve = tmp_path / "serve"
    artifact = tmp_path / "custom.ninfer"
    config = tmp_path / "baseline.json"
    serve.touch()
    artifact.write_bytes(b"artifact")
    config.write_text(json.dumps({"profiles": {"other-profile": []}}))

    def running(*_args: object) -> None:
        pytest.fail("incomplete baseline config must fail before starting a server")

    monkeypatch.setattr(campaign, "RunningServe", running)
    with pytest.raises(SystemExit, match="no executable Serve profile"):
        campaign.main([
            "--case", "cold-short", "--serve", str(serve),
            "--artifact", str(artifact), "--profile-config", str(config),
        ])
