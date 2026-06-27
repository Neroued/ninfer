from __future__ import annotations

import torch

from .. import qtypes as qt
from .. import tensor_plan as tp
from ..convert import _prepare_source
from ..layouts import encode_tensor


class _Reader:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get(self, name: str) -> torch.Tensor:
        assert name == "model.language_model.layers.0.linear_attn.conv1d.weight"
        return self.tensor


def _conv1d_spec():
    specs = tp.build_text_specs(["linear_attention"])
    matches = [s for s in specs if s.name.endswith("linear_attn.conv1d.weight")]
    assert len(matches) == 1
    return matches[0]


def test_gdn_conv1d_spec_uses_runtime_native_transform():
    spec = _conv1d_spec()
    assert spec.qtype == qt.QT_BF16
    assert spec.layout == qt.LAYOUT_CONTIGUOUS
    assert spec.source_kind == qt.SK_GDN_CONV1D
    assert spec.source_layer == 0
    assert spec.transform == tp.TRANSFORM_GDN_CONV1D_RUNTIME_NATIVE


def test_runtime_native_gdn_conv1d_reorders_payload_bytes():
    src = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[10.0, 11.0, 12.0, 13.0]],
            [[20.0, 21.0, 22.0, 23.0]],
        ],
        dtype=torch.float32,
    ).to(torch.bfloat16)

    got = tp.runtime_native_gdn_conv1d(src)

    assert tuple(got.shape) == (3, 4, 1)
    assert got.reshape(-1).float().tolist() == [
        0.0,
        10.0,
        20.0,
        1.0,
        11.0,
        21.0,
        2.0,
        12.0,
        22.0,
        3.0,
        13.0,
        23.0,
    ]


def test_prepare_source_applies_gdn_conv1d_transform_before_encoding():
    src = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[10.0, 11.0, 12.0, 13.0]],
            [[20.0, 21.0, 22.0, 23.0]],
        ],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    spec = _conv1d_spec()

    prepared = _prepare_source(_Reader(src), spec)
    payload, logical, padded, group, scale_dtype = encode_tensor(
        prepared, spec.qtype, spec.layout, torch.device("cpu")
    )

    assert logical == [3, 4, 1]
    assert padded == [3, 4, 1]
    assert group == 0
    assert scale_dtype == qt.SCALE_NONE
    decoded_words = torch.frombuffer(bytearray(payload), dtype=torch.int16).to(torch.int32)
    expected_words = prepared.reshape(-1).view(torch.int16).to(torch.int32)
    assert decoded_words.tolist() == expected_words.tolist()


def test_runtime_native_gdn_conv1d_rejects_unexpected_shape():
    bad = torch.zeros((3, 4), dtype=torch.bfloat16)
    try:
        tp.runtime_native_gdn_conv1d(bad)
    except ValueError as exc:
        assert "expected [C,1,K]" in str(exc)
    else:
        raise AssertionError("runtime_native_gdn_conv1d accepted a rank-2 tensor")
