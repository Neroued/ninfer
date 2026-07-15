from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tools.reference.qwen3_6_35b_a3b_rtx5090.bindings import ArtifactBinding
from tools.reference.qwen3_6_35b_a3b_rtx5090.weights import (
    WeightStore,
    estimate_fixed_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def binding():
    path = Path(
        os.environ.get(
            "NINFER_QWEN3_6_35B_A3B_ARTIFACT",
            PROJECT_ROOT / "out/qwen3_6_35b_a3b_rtx5090.ninfer",
        )
    )
    if not path.is_file():
        pytest.skip(f"real Qwen3.6-35B-A3B artifact is not available at {path}")
    with ArtifactBinding.open(path) as value:
        yield value


def test_real_artifact_binding_and_selected_expert_rows(
    binding: ArtifactBinding,
) -> None:
    assert len(binding.text.layers) == 40
    assert sum(layer.attention is not None for layer in binding.text.layers) == 10
    assert sum(layer.gdn is not None for layer in binding.text.layers) == 30

    moe = binding.text.layers[0].moe
    assert moe.router.block is moe.router_shared_gate
    assert moe.shared_gate.block is moe.router_shared_gate
    assert moe.shared_expert_gate.block is moe.shared_gate_up
    assert moe.shared_up.block is moe.shared_gate_up
    assert binding.mtp.token_embedding is binding.text.token_embedding
    assert binding.mtp.full_output_head is binding.text.output_head
    assert binding.mtp.optimized_proposal_head is binding.text.draft_head
    assert binding.vision.merger.fc2.shape == (2048, 4608)

    weights = WeightStore(
        binding,
        "cpu",
        capacity=1,
        kv_dtype="int8",
        text=True,
        mtp=True,
        vision=True,
        memory_bytes=0,
    )
    try:
        assert weights.representation(moe.routed_gate_up.block) == "stream"
        gate_up = weights.rows(
            moe.routed_gate_up.block,
            (
                0,
                511,
                512,
                1023,
                255 * 1024,
                255 * 1024 + 511,
                255 * 1024 + 512,
                255 * 1024 + 1023,
            ),
            dequant_dtype=torch.float32,
        )
        down = weights.rows(
            moe.routed_down.block,
            (0, 2047, 255 * 2048, 256 * 2048 - 1),
        )
        assert gate_up.shape == (8, 2048)
        assert down.shape == (4, 512)
        assert gate_up.dtype == torch.float32
        assert down.dtype == torch.bfloat16
    finally:
        weights.close()


def test_256k_int8_text_mtp_fixed_bytes() -> None:
    assert (
        estimate_fixed_bytes(
            262144,
            "int8",
            text=True,
            mtp=True,
            prefill_chunk=1024,
        )
        == 3_377_889_280
    )
