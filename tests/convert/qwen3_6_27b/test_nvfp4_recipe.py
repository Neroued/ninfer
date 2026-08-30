from __future__ import annotations

import struct

import pytest
import torch

from tools.convert.qwen3_6_27b import convert_nvfp4
from tools.convert.qwen3_6_27b import inventory_nvfp4 as inventory
from tools.convert.qwen3_6_27b import recipe_nvfp4 as recipe


FULL_ATTENTION = tuple(range(3, 64, 4))
EARLY_INPUT = (3, 7, 11, 15, 19, 23)
NVFP4_INPUT = (27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
BF16_OUTPUT = (3, 7)
NVFP4_OUTPUT = (11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
GDN = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION)


def test_closed_allocation_inventory_and_site_coverage():
    assert inventory.MODEL_ID == "qwen3.6-27b"
    assert inventory.WEIGHTS_ID == "nvfp4"
    assert inventory.FULL_ATTENTION_LAYERS == FULL_ATTENTION
    assert inventory.EARLY_ATTENTION_INPUT_LAYERS == EARLY_INPUT
    assert inventory.NVFP4_ATTENTION_INPUT_LAYERS == NVFP4_INPUT
    assert inventory.BF16_ATTENTION_OUTPUT_LAYERS == BF16_OUTPUT
    assert inventory.NVFP4_ATTENTION_OUTPUT_LAYERS == NVFP4_OUTPUT
    assert inventory.GDN_LAYERS == GDN
    assert inventory.BF16_GDN_OUTPUT_LAYERS == (4,)

    assert len(inventory.OBJECT_SPECS) == 1307
    assert len(inventory.NVFP4_TENSOR_SPECS) == 247
    assert len(inventory.INPUT_SCALE_DIVISOR_SPECS) == 247
    assert inventory.FORMAT_COUNTS == {
        "BF16": 591,
        "FP32": 343,
        "I32": 1,
        "Q4G64_F16S": 55,
        "Q5G64_F16S": 54,
        "Q6G64_F16S": 1,
        "W8G32_F16S": 9,
        "NVFP4": 247,
    }
    tensors = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    assert tensors["text/token_embedding"] == inventory.TensorSpec(
        "text/token_embedding",
        (248320, 5120),
        "W8G32_F16S",
        "row-split-k128-v1",
    )
    assert tensors["text/output_head"] == inventory.TensorSpec(
        "text/output_head",
        (248320, 5120),
        "W8G32_F16S",
        "row-split-k128-v1",
    )
    assert tensors["text/layers/3/attention/query_key_gate_value"] == inventory.TensorSpec(
        "text/layers/3/attention/query_key_gate_value",
        (14336, 5120),
        "BF16",
        "contiguous-le-v1",
    )
    assert tensors["text/layers/27/attention/query_key_gate_value"] == inventory.TensorSpec(
        "text/layers/27/attention/query_key_gate_value",
        (14336, 5120),
        "NVFP4",
        "blockscale-k16-m128x4-v1",
    )
    assert tensors["text/layers/0/gdn/query_key_value_z"] == inventory.TensorSpec(
        "text/layers/0/gdn/query_key_value_z",
        (16384, 5120),
        "NVFP4",
        "blockscale-k16-m128x4-v1",
    )
    assert "text/layers/3/attention/query_key" not in tensors
    assert "text/layers/3/attention/gate_value" not in tensors
    assert "text/layers/0/gdn/query_key" not in tensors
    assert "text/layers/0/gdn/value_z" not in tensors

    views = {view.name_pattern: view for view in inventory.LOGICAL_ROW_VIEW_SPECS}
    assert (
        views["text/layers/{l}/attention/value"].parent_pattern,
        views["text/layers/{l}/attention/value"].row_begin,
        views["text/layers/{l}/attention/value"].row_end,
    ) == ("text/layers/{l}/attention/query_key_gate_value", 13312, 14336)
    assert (
        views["text/layers/{l}/gdn/z"].parent_pattern,
        views["text/layers/{l}/gdn/z"].row_begin,
        views["text/layers/{l}/gdn/z"].row_end,
    ) == ("text/layers/{l}/gdn/query_key_value_z", 10240, 16384)

    assert convert_nvfp4.RECIPE_ID == "qwen3_6_27b_nvfp4-v1"


def test_fused_text_attention_dispatch_does_not_capture_mtp_parent():
    assert convert_nvfp4._is_fused_text_attention_parent(
        "text/layers/3/attention/query_key_gate_value"
    )
    assert not convert_nvfp4._is_fused_text_attention_parent(
        "mtp/layer/attention/query_key_gate_value"
    )


def test_input_divisor_materialization_preserves_positive_fp32_word():
    class Reader:
        def __init__(self, word: int):
            self.tensor = torch.frombuffer(
                bytearray(struct.pack("<I", word)), dtype=torch.float32
            )

        def get(self, _name: str) -> torch.Tensor:
            return self.tensor

    selected = recipe.INPUT_DIVISOR_RECIPES[0]
    positive_word = struct.unpack("<I", struct.pack("<f", 1.25))[0]
    actual = recipe.materialize_input_divisor(selected, Reader(positive_word))
    assert actual.shape == torch.Size([])
    assert int(actual.view(torch.int32)) & 0xFFFFFFFF == positive_word

    with pytest.raises(ValueError, match="finite and positive"):
        recipe.materialize_input_divisor(selected, Reader(0))


def test_converter_rejects_wrong_basename_before_source_or_file_creation(tmp_path):
    output = tmp_path / "qwen3_6_27b.ninfer"
    with pytest.raises(ValueError, match="output basename"):
        convert_nvfp4.convert(
            tmp_path / "missing-base",
            tmp_path / "missing-nvfp4",
            output,
            device="cpu",
        )
    assert not output.exists()
