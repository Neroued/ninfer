"""Round-trip tests for q5090 v4 packing, layouts and binary records.

Runs on CPU so it needs no GPU. Either:
  pytest tools/q5090_convert/tests/test_packing.py
  python -m tools.q5090_convert.tests.test_packing
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .. import format as fmt
from .. import qtypes as qt
from .. import layouts
from .. import packing
from ..packing import (
    high_bytes_per_group,
    nibble_bytes_per_group,
    pack_plane_split_groups,
    pack_plane_split_torch,
    pack_w8_groups,
    unpack_plane_split_groups,
    unpack_plane_split_torch,
    unpack_w8_groups,
)
from ..quantize import dequantize_rows, quantize_rows

DEV = torch.device("cpu")


def _pad_k(w: torch.Tensor, group_size: int) -> torch.Tensor:
    n, k = w.shape
    kp = (k + 127) // 128 * 128
    if kp == k:
        return w
    return F.pad(w, (0, kp - k, 0, 0), value=0.0)


def _ref_quantized(w: torch.Tensor, qtype: int) -> tuple[np.ndarray, np.ndarray]:
    spec = qt.QUANT_SPECS[qtype]
    wp = _pad_k(w, spec.group_size)
    return quantize_rows(wp, spec.group_size, spec.qmax, spec.qmin, DEV)


def test_plane_split_roundtrip_and_code_preservation():
    rng = np.random.default_rng(1)
    for bits, qmin, qmax in [(4, -8, 7), (5, -16, 15), (6, -32, 31)]:
        codes = rng.integers(qmin, qmax + 1, size=(777, 64)).astype(np.int8)
        codes[0] = np.resize(np.arange(qmin, qmax + 1, dtype=np.int8), 64)
        nibble, high = pack_plane_split_groups(codes, bits)
        assert nibble.shape == (777, nibble_bytes_per_group(64, bits))
        assert high.shape == (777, high_bytes_per_group(64, bits))
        back = unpack_plane_split_groups(nibble, high, bits, 64)
        assert np.array_equal(codes, back), f"bits={bits}"


def test_torch_matches_numpy_plane_split_packing():
    rng = np.random.default_rng(7)
    for bits, qmin, qmax in [(4, -8, 7), (5, -16, 15), (6, -32, 31)]:
        codes = rng.integers(qmin, qmax + 1, size=(999, 64)).astype(np.int8)
        np_nibble, np_high = pack_plane_split_groups(codes, bits)
        t_nibble, t_high = pack_plane_split_torch(torch.from_numpy(codes).to(DEV), bits)
        assert np.array_equal(np_nibble, t_nibble.cpu().numpy()), f"nibble pack bits={bits}"
        assert np.array_equal(np_high, t_high.cpu().numpy()), f"high pack bits={bits}"
        t_unpacked = unpack_plane_split_torch(t_nibble, t_high, bits, 64).cpu().numpy()
        assert np.array_equal(codes, t_unpacked), f"unpack bits={bits}"


def test_w8_roundtrip():
    rng = np.random.default_rng(2)
    for group_size in (32, 128):
        codes = rng.integers(-127, 128, size=(123, group_size)).astype(np.int8)
        back = unpack_w8_groups(pack_w8_groups(codes)).reshape(codes.shape)
        assert np.array_equal(codes, back)
        nibble, high = pack_plane_split_groups(codes, 8)
        assert high.shape == (123, 0)
        assert np.array_equal(unpack_plane_split_groups(nibble, high, 8, group_size), codes)


def test_quant_range_and_dequant():
    torch.manual_seed(3)
    w = torch.randn(64, 128)
    s, c = quantize_rows(w, 64, 7, -8, DEV)
    assert c.min() >= -8 and c.max() <= 7
    deq = dequantize_rows(s, c)
    assert np.allclose(deq, (c.astype(np.float32) * s.astype(np.float32)[:, :, None]).reshape(64, 128))


def test_row_split_encode_decode_quantized_bit_exact():
    torch.manual_seed(4)
    cases = [
        (torch.randn(17, 128), qt.QT_Q4G64),
        (torch.randn(19, 256), qt.QT_Q5G64),
        (torch.randn(23, 384), qt.QT_Q6G64),
        (torch.randn(11, 160), qt.QT_W8G32),
        (torch.randn(13, 256), qt.QT_W8G128),
    ]

    for w, qtype in cases:
        wb = w.bfloat16().float()
        payload, logical, padded, group, sd, nibble_bytes, high_bytes, scale_bytes = layouts.encode_tensor(
            wb, qtype, qt.LAYOUT_ROW_SPLIT, DEV
        )
        scale16, codes = layouts.decode_row_split_quantized(payload, padded, qtype, DEV)
        ref_scale16, ref_codes = _ref_quantized(wb, qtype)

        n, kp = padded
        spec = qt.QUANT_SPECS[qtype]
        groups = kp // spec.group_size
        nib = spec.nibble_bytes_per_group
        high = spec.high_bytes_per_group
        exp_nib, exp_high_off, exp_high, exp_scale_off, exp_scale, exp_payload = packing.row_split_plane_sizes(
            n, groups, nib, high
        )

        assert logical == [w.shape[0], w.shape[1]]
        assert padded == [w.shape[0], kp]
        assert group == spec.group_size
        assert sd == qt.SCALE_FP16
        assert nibble_bytes == exp_nib
        assert high_bytes == exp_high
        assert scale_bytes == exp_scale
        assert len(payload) == exp_payload
        assert bytes(payload[nibble_bytes:exp_high_off]) == b"\x00" * (exp_high_off - nibble_bytes)
        assert bytes(payload[exp_high_off + high_bytes:exp_scale_off]) == b"\x00" * (
            exp_scale_off - exp_high_off - high_bytes
        )

        row_nibble_bytes = groups * nib
        row_high_bytes = groups * high
        assert row_nibble_bytes % 16 == 0
        assert row_high_bytes % 16 == 0
        for row in range(n):
            assert (row * row_nibble_bytes) % 16 == 0
            assert (row * row_high_bytes) % 16 == 0

        assert np.array_equal(codes.cpu().numpy(), ref_codes), qt.QTYPE_NAME[qtype]
        assert np.array_equal(scale16.cpu().numpy().view(np.uint16), ref_scale16.view(np.uint16))

        got = layouts.decode_tensor(payload, qtype, qt.LAYOUT_ROW_SPLIT, logical, padded, DEV).cpu().numpy()
        ref = dequantize_rows(ref_scale16, ref_codes)[:, : w.shape[1]]
        assert np.array_equal(got, ref), qt.QTYPE_NAME[qtype]


def test_row_split_dequant_matches_policy_output():
    torch.manual_seed(11)
    cases = [
        (torch.randn(9, 192), qt.QT_Q4G64),
        (torch.randn(11, 192), qt.QT_Q5G64),
        (torch.randn(13, 320), qt.QT_Q6G64),
        (torch.randn(5, 160), qt.QT_W8G32),
        (torch.randn(7, 192), qt.QT_W8G128),
    ]

    for w, qtype in cases:
        wb = w.bfloat16().float()
        payload, logical, padded, *_ = layouts.encode_tensor(wb, qtype, qt.LAYOUT_ROW_SPLIT, DEV)
        got = layouts.decode_tensor(payload, qtype, qt.LAYOUT_ROW_SPLIT, logical, padded, DEV).cpu().numpy()

        ref_scale16, ref_codes = _ref_quantized(wb, qtype)
        policy_deq = dequantize_rows(ref_scale16, ref_codes)[:, : w.shape[1]]
        assert np.array_equal(got, policy_deq), qt.QTYPE_NAME[qtype]


def test_contiguous_exact_and_plane_metadata():
    torch.manual_seed(5)
    w = torch.randn(10, 1, 4).bfloat16()
    payload, logical, padded, group, sd, nibble_bytes, high_bytes, scale_bytes = layouts.encode_tensor(
        w, qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, DEV
    )
    got = layouts.decode_tensor(payload, qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, logical, padded, DEV).cpu().numpy()
    assert np.array_equal(got, w.float().numpy())
    assert logical == list(w.shape)
    assert padded == list(w.shape)
    assert group == 0
    assert sd == qt.SCALE_NONE
    assert nibble_bytes == len(payload)
    assert high_bytes == 0
    assert scale_bytes == 0

    w = torch.randn(48).bfloat16()
    payload, logical, padded, group, sd, nibble_bytes, high_bytes, scale_bytes = layouts.encode_tensor(
        w, qt.QT_FP32, qt.LAYOUT_CONTIGUOUS, DEV
    )
    got = layouts.decode_tensor(payload, qt.QT_FP32, qt.LAYOUT_CONTIGUOUS, logical, padded, DEV).cpu().numpy()
    assert np.array_equal(got, w.float().numpy())
    assert nibble_bytes == len(payload)
    assert high_bytes == 0
    assert scale_bytes == 0


def test_i32_ctrl_contiguous_roundtrip():
    ids = torch.tensor([0, 1, 42, 248319, -1, 131071], dtype=torch.int32)
    payload, logical, padded, group, sd, nibble_bytes, high_bytes, scale_bytes = layouts.encode_tensor(
        ids, qt.QT_I32, qt.LAYOUT_CONTIGUOUS, DEV
    )
    assert logical == [6]
    assert padded == [6]
    assert group == 0
    assert sd == qt.SCALE_NONE
    assert len(payload) == 6 * 4
    assert nibble_bytes == len(payload)
    assert high_bytes == 0
    assert scale_bytes == 0
    assert payload == ids.cpu().numpy().astype("<i4").tobytes()
    got = layouts.decode_tensor(payload, qt.QT_I32, qt.LAYOUT_CONTIGUOUS, logical, padded, DEV)
    assert got.dtype == torch.int32
    assert got.cpu().tolist() == ids.tolist()


def test_v4_1_records_pack_unpack_and_string_table():
    entry = fmt.TensorEntry(
        name="block.q4",
        qtype=qt.QT_Q4G64,
        layout=qt.LAYOUT_ROW_SPLIT,
        module_kind=qt.MODULE_TEXT,
        shape=[192, 128],
        padded_shape=[192, 128],
        group_size=64,
        scale_dtype=qt.SCALE_FP16,
        segment_count=2,
        source_layer=7,
        source_kind=qt.SK_OTHER,
        payload_offset=8192,
        payload_bytes=9984,
        crc32=0x12345678,
        segment_begin=3,
        fusion_group_id=1,
        fusion_index=1,
        nibble_plane_bytes=6144,
        high_plane_bytes=1024,
        scale_plane_bytes=768,
    )
    segments = [
        fmt.SegmentRecord("seg.q", qt.SK_ATTN_Q, 7, 0, 128),
        fmt.SegmentRecord("seg.k", qt.SK_ATTN_K, 7, 128, 64),
    ]
    table = fmt.build_string_table([entry], segments)
    assert table == b"block.q4\x00seg.q\x00seg.k\x00"

    packed_entry = fmt.pack_tensor_entry(entry)
    assert len(packed_entry) == fmt.TENSOR_ENTRY_SIZE
    assert packed_entry[124:] == b"\x00" * 4
    parsed_entry = fmt.unpack_tensor_entry(packed_entry)
    assert parsed_entry["name_offset"] == 0
    assert parsed_entry["name_len"] == len("block.q4")
    assert parsed_entry["name_hash"] == fmt.fnv1a_64("block.q4")
    assert parsed_entry["layout"] == qt.LAYOUT_ROW_SPLIT
    assert parsed_entry["segment_count"] == 2
    assert parsed_entry["segment_begin"] == 3
    assert parsed_entry["fusion_group_id"] == 1
    assert parsed_entry["fusion_index"] == 1
    assert parsed_entry["nibble_plane_bytes"] == 6144
    assert parsed_entry["high_plane_bytes"] == 1024
    assert parsed_entry["scale_plane_bytes"] == 768

    packed_segment = fmt.pack_segment_record(segments[1])
    assert len(packed_segment) == fmt.SEGMENT_RECORD_SIZE
    parsed_segment = fmt.unpack_segment_record(packed_segment)
    assert parsed_segment["source_kind"] == qt.SK_ATTN_K
    assert parsed_segment["row_begin"] == 128
    assert parsed_segment["row_count"] == 64
    assert parsed_segment["name_offset"] == len("block.q4") + 1 + len("seg.q") + 1
    assert parsed_segment["name_hash"] == fmt.fnv1a_64("seg.k")

    fusion = fmt.FusionGroupRecord(
        group_id=1,
        source_layer=7,
        block_count=2,
        shared_input_kind=qt.SK_OTHER,
        first_block_tensor_index=42,
        payload_offset=8192,
        payload_bytes=32768,
        total_n=384,
        shared_k=128,
    )
    packed_fusion = fmt.pack_fusion_group_record(fusion)
    assert len(packed_fusion) == fmt.FUSION_GROUP_RECORD_SIZE
    assert packed_fusion[48:] == b"\x00" * 16
    parsed_fusion = fmt.unpack_fusion_group_record(packed_fusion)
    assert parsed_fusion["group_id"] == 1
    assert parsed_fusion["block_count"] == 2
    assert parsed_fusion["first_block_tensor_index"] == 42
    assert parsed_fusion["total_n"] == 384
    assert parsed_fusion["shared_k"] == 128

    module = fmt.ModuleRecord(qt.MODULE_TEXT, fmt.VERSION, 0, 1, 8192, 9984)
    packed_module = fmt.pack_module_record(module)
    assert len(packed_module) == fmt.MODULE_RECORD_SIZE
    parsed_module = fmt.unpack_module_record(packed_module)
    assert parsed_module["module_version"] == fmt.VERSION
    assert parsed_module["payload_bytes"] == 9984
    assert parsed_module["reserved0"] == 0
    assert parsed_module["reserved1"] == 0

    tokenizer_record = fmt.TokenizerRecord(
        kind=fmt.TOKENIZER_JSON,
        encoding=fmt.TOKENIZER_RAW_UTF8,
        data_offset=7040,
        data_bytes=123,
        crc32=0xABCDEF01,
        sha256=b"\x22" * 32,
    )
    packed_tokenizer = fmt.pack_tokenizer_record(tokenizer_record)
    assert len(packed_tokenizer) == fmt.TOKENIZER_RECORD_SIZE
    parsed_tokenizer = fmt.unpack_tokenizer_record(packed_tokenizer)
    assert parsed_tokenizer["kind"] == fmt.TOKENIZER_JSON
    assert parsed_tokenizer["encoding"] == fmt.TOKENIZER_RAW_UTF8
    assert parsed_tokenizer["data_offset"] == 7040
    assert parsed_tokenizer["data_bytes"] == 123
    assert parsed_tokenizer["crc32"] == 0xABCDEF01
    assert parsed_tokenizer["reserved"] == 0
    assert parsed_tokenizer["sha256"] == b"\x22" * 32

    header = fmt.FileHeaderFields(
        tensor_count=1,
        module_count=1,
        layer_count=64,
        flags=1,
        segment_count=2,
        module_index_offset=fmt.HEADER_SIZE,
        module_index_bytes=fmt.MODULE_RECORD_SIZE,
        tensor_index_offset=fmt.HEADER_SIZE + fmt.MODULE_RECORD_SIZE,
        tensor_index_bytes=fmt.TENSOR_ENTRY_SIZE,
        string_table_offset=fmt.HEADER_SIZE + fmt.MODULE_RECORD_SIZE + fmt.TENSOR_ENTRY_SIZE,
        string_table_bytes=len(table),
        payload_offset=8192,
        payload_bytes=9984,
        hidden_size=5120,
        intermediate_size=17408,
        vocab_size=248320,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        gdn_key_heads=16,
        gdn_value_heads=48,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_width=4,
        full_attention_interval=4,
        max_position_embeddings=262144,
        fusion_group_count=1,
        segment_index_offset=6144,
        segment_index_bytes=2 * fmt.SEGMENT_RECORD_SIZE,
        fusion_group_index_offset=6208,
        fusion_group_index_bytes=fmt.FUSION_GROUP_RECORD_SIZE,
        sha256_safetensors_index=b"\x11" * 32,
        tokenizer_index_offset=6272,
        tokenizer_index_bytes=3 * fmt.TOKENIZER_RECORD_SIZE,
        tokenizer_data_offset=6464,
        tokenizer_data_bytes=321,
    )
    packed_header = fmt.pack_header(header)
    assert len(packed_header) == fmt.HEADER_SIZE
    assert packed_header[280:] == b"\x00" * (fmt.HEADER_SIZE - 280)
    parsed_header = fmt.unpack_header(packed_header)
    assert parsed_header["magic"] == fmt.MAGIC
    assert parsed_header["version"] == fmt.VERSION
    assert parsed_header["segment_count"] == 2
    assert parsed_header["fusion_group_count"] == 1
    assert parsed_header["segment_index_offset"] == 6144
    assert parsed_header["fusion_group_index_bytes"] == fmt.FUSION_GROUP_RECORD_SIZE
    assert parsed_header["format_minor"] == 1
    assert parsed_header["tokenizer_record_count"] == 3
    assert parsed_header["tokenizer_record_size"] == 64
    assert parsed_header["tokenizer_index_offset"] == 6272
    assert parsed_header["tokenizer_data_offset"] == 6464
    assert parsed_header["tokenizer_data_bytes"] == 321
    assert parsed_header["sha256_safetensors_index"] == b"\x11" * 32


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
