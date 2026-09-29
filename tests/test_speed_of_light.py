"""Accounting checks for the offline inference roof estimate."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.bench.speed_of_light import (
    _model_work,
    _projection_geometry,
    compare,
    estimate,
)


def fake_artifact(
    config: dict, byte_multipliers: dict[str, int] | None = None
):
    byte_multipliers = byte_multipliers or {}
    projections, expert_groups = _projection_geometry(config)
    shapes = dict(projections)
    shapes["text/output_head"] = config["hidden_size"] * config["vocab_size"]
    for experts in expert_groups:
        for expert in experts:
            for name in expert:
                shapes[name] = shapes[
                    name.replace("/experts/1/", "/experts/0/")
                ]
    objects = {}
    for name, count in shapes.items():
        objects[name] = TensorObject(
            name,
            (count,),
            "bf16",
            "row_major",
            0,
            count * byte_multipliers.get(name, 2),
        )
    return SimpleNamespace(
        directory=SimpleNamespace(
            components={"text": {"config": config}},
            bindings={name: {"object": name} for name in shapes},
        ),
        by_id=objects,
        artifact_id=b"\x01" * 16,
    )


class SpeedOfLightTests(unittest.TestCase):
    def test_real_artifact_directory_with_split_binding(self):
        config = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 2,
            "vocab_size": 4,
            "layer_types": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.ninfer"
            with ArtifactWriter(
                path,
                [TensorSpec("head", (8,), "bf16", "contiguous_le_v1")],
                components={"text": {"config": config}},
                bindings={
                    "text/output_head": {
                        "parts": [
                            {"object": "head", "range": [0, 4]},
                            {"object": "head", "range": [4, 8]},
                        ]
                    }
                },
            ) as writer:
                writer.write_object("head", bytes(16))
            with Artifact(path) as artifact:
                body, head, weight_bytes = _model_work(artifact)
        assert (body, head, weight_bytes) == (0, 16, 16)

    def test_prefill_head_runs_once_and_decode_head_runs_per_token(self):
        config = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 8,
            "layer_types": ["full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 6,
        }
        artifact = fake_artifact(config)
        report = {
            "artifact_type": "ninfer_bench_report",
            "schema_version": 15,
            "load": {"architecture": "Qwen3_5ForCausalLM", "name": "test"},
            "config": {"speculative_backend": "none"},
            "environment": {"gpu_name": "test gpu"},
            "memory": {"kv_capacity": 8, "kv_payload_bytes": 64},
            "tests": [
                {
                    "label": "pp2",
                    "kind": "pp",
                    "n_prompt": 2,
                    "n_gen": 0,
                    "prefill_seconds_mean": 0.5,
                    "decode_seconds_mean": 0,
                },
                {
                    "label": "tg3",
                    "kind": "tg",
                    "n_prompt": 0,
                    "n_gen": 3,
                    "prefill_seconds_mean": 0.1,
                    "decode_seconds_mean": 1,
                },
            ],
        }
        result = estimate(report, artifact, 1, 1)
        assert result["body_projection_flops_per_token"] == 272
        assert result["output_head_flops_per_call"] == 64
        assert (
            result["tests"][0]["phases"]["prefill"]["projection_flops"] == 608
        )
        assert (
            result["tests"][1]["phases"]["decode"]["projection_flops"] == 1008
        )
        assert (
            result["tests"][1]["phases"]["decode"][
                "estimated_weight_bytes_per_token"
            ]
            == 336
        )
        assert (
            result["tests"][1]["phases"]["decode"]["estimated_kv_read_bytes"]
            == 72
        )
        assert "decode" not in result["tests"][0]["phases"]
        assert "prefill" not in result["tests"][1]["phases"]

    def test_moe_traffic_uses_smallest_selectable_experts(self):
        config = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 8,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 3,
            "shared_expert_intermediate_size": 2,
        }
        expensive = {
            f"text/layers/0/moe/experts/1/{role}": 8
            for role in ("gate", "up", "down")
        }
        artifact = fake_artifact(config, expensive)
        body_flops, head_flops, selected_bytes = _model_work(artifact)
        assert body_flops == 240
        assert head_flops == 64
        assert selected_bytes == 304
        assert selected_bytes == _model_work(fake_artifact(config))[2]

    def test_speculative_report_is_rejected(self):
        config = {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 4,
            "vocab_size": 8,
            "layer_types": [],
        }
        artifact = fake_artifact(config)
        report = {
            "artifact_type": "ninfer_bench_report",
            "schema_version": 15,
            "load": {"architecture": "Qwen3_5ForCausalLM"},
            "config": {"speculative_backend": "mtp"},
        }
        with self.assertRaisesRegex(ValueError, "--spec none"):  # noqa: PT027
            estimate(report, artifact, 1, None)


def _sol_phase(
    tokens: int,
    seconds: float,
    compute_floor: float,
    weight_floor: float | None = None,
    kv_floor: float | None = None,
) -> dict:
    entry = {
        "tokens": tokens,
        "measured_seconds": seconds,
        "compute_floor_seconds": compute_floor,
        "compute_roof_fraction": compute_floor / seconds,
    }
    if weight_floor is not None:
        entry["weight_traffic_seconds"] = weight_floor
        entry["weight_roof_fraction"] = weight_floor / seconds
    if kv_floor is not None:
        entry["kv_traffic_seconds"] = kv_floor
        entry["kv_roof_fraction"] = kv_floor / seconds
    return entry


def _sol_result(
    artifact_id: str,
    phases: dict,
    peak_tflops: float = 1676.0,
    hbm_gbps: float | None = 1792.0,
) -> dict:
    return {
        "model": "qwen3.8-27b",
        "gpu": "NVIDIA GeForce RTX 5090",
        "artifact_id": artifact_id,
        "peak_tflops": peak_tflops,
        "hbm_gbps": hbm_gbps,
        "tests": [{"label": "pp2048+tg128", "phases": phases}],
    }


class SpeedOfLightCompareTests(unittest.TestCase):
    def test_change_closes_fraction_of_roof_gap(self):
        before = _sol_result(
            "a" * 32,
            {
                "decode": _sol_phase(
                    128, 1.765, 0.094, weight_floor=1.364, kv_floor=0.042
                )
            },
        )
        after = _sol_result(
            "b" * 32,
            {
                "decode": _sol_phase(
                    128, 1.610, 0.094, weight_floor=1.364, kv_floor=0.042
                )
            },
        )
        row = compare(before, after)["rows"][0]
        assert abs(row["measured_change_fraction"] + 0.155 / 1.765) < 1e-9
        weight = row["weight"]
        assert abs(weight["fraction_before"] - 1.364 / 1.765) < 1e-9
        assert abs(weight["fraction_after"] - 1.364 / 1.610) < 1e-9
        assert abs(weight["gap_closed_fraction"] - (0.401 - 0.246) / 0.401) < 1e-9

    def test_ceiling_change_is_rejected(self):
        before = _sol_result("a" * 32, {"decode": _sol_phase(128, 1.0, 0.1)})
        after = _sol_result(
            "b" * 32,
            {"decode": _sol_phase(128, 0.9, 0.1)},
            peak_tflops=1800.0,
        )
        with self.assertRaisesRegex(ValueError, "ceilings"):  # noqa: PT027
            compare(before, after)

    def test_model_change_is_rejected(self):
        before = _sol_result("a" * 32, {"decode": _sol_phase(128, 1.0, 0.1)})
        after = _sol_result("b" * 32, {"decode": _sol_phase(128, 0.9, 0.1)})
        after["model"] = "other-model"
        with self.assertRaisesRegex(ValueError, "models"):  # noqa: PT027
            compare(before, after)

    def test_token_count_change_is_rejected(self):
        before = _sol_result("a" * 32, {"decode": _sol_phase(128, 1.0, 0.1)})
        after = _sol_result("b" * 32, {"decode": _sol_phase(256, 0.9, 0.1)})
        with self.assertRaisesRegex(ValueError, "token counts"):  # noqa: PT027
            compare(before, after)

    def test_baseline_at_or_above_roof_reports_no_gap_closure(self):
        before = _sol_result("a" * 32, {"decode": _sol_phase(128, 0.5, 0.8)})
        after = _sol_result("b" * 32, {"decode": _sol_phase(128, 0.45, 0.8)})
        row = compare(before, after)["rows"][0]
        assert row["compute"]["gap_closed_fraction"] is None

    def test_unmatched_phase_is_listed_not_compared(self):
        before = _sol_result(
            "a" * 32,
            {
                "prefill": _sol_phase(2048, 0.241, 0.0595),
                "decode": _sol_phase(128, 1.765, 0.094, weight_floor=1.364),
            },
        )
        after = _sol_result(
            "b" * 32,
            {"decode": _sol_phase(128, 1.610, 0.094, weight_floor=1.364)},
        )
        comparison = compare(before, after)
        assert comparison["unmatched"] == ["pp2048+tg128 prefill"]
        assert [row["phase"] for row in comparison["rows"]] == ["decode"]


if __name__ == "__main__":
    unittest.main()
