"""Round-trip tests for packing, quantization, layouts and the file container.

Runs on CPU so it needs no GPU. Either:
  pytest tools/q5090_convert/tests/test_packing.py
  python -m tools.q5090_convert.tests.test_packing
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from .. import format as fmt
from .. import qtypes as qt
from ..layouts import decode_tensor, encode_tensor
from ..packing import (
    pack_lowbit_groups,
    pack_lowbit_torch,
    pack_w8_groups,
    unpack_lowbit_groups,
    unpack_lowbit_torch,
    unpack_w8_groups,
)
from ..quantize import dequantize_rows, quantize_rows

DEV = torch.device("cpu")


def _ref_dequant(w: torch.Tensor, qtype: int, n_mult: int) -> np.ndarray:
    spec = qt.QUANT_SPECS[qtype]
    n, k = w.shape
    np_ = (n + n_mult - 1) // n_mult * n_mult
    kp = (k + spec.group_size - 1) // spec.group_size * spec.group_size
    wp = F.pad(w, (0, kp - k, 0, np_ - n), value=0.0)
    s, c = quantize_rows(wp, spec.group_size, spec.qmax, spec.qmin, DEV)
    return dequantize_rows(s, c)[:n, :k]


def test_lowbit_roundtrip():
    rng = np.random.default_rng(1)
    for bits, qmin, qmax in [(4, -8, 7), (5, -16, 15), (6, -32, 31)]:
        codes = rng.integers(qmin, qmax + 1, size=(777, 64)).astype(np.int8)
        packed = pack_lowbit_groups(codes, bits)
        assert packed.shape[1] == (64 * bits) // 8
        back = unpack_lowbit_groups(packed, bits, 64)
        assert np.array_equal(codes, back), f"bits={bits}"


def test_torch_matches_numpy_packing():
    rng = np.random.default_rng(7)
    for bits, qmin, qmax in [(4, -8, 7), (5, -16, 15), (6, -32, 31)]:
        codes = rng.integers(qmin, qmax + 1, size=(999, 64)).astype(np.int8)
        np_packed = pack_lowbit_groups(codes, bits)
        t_packed = pack_lowbit_torch(torch.from_numpy(codes).to(DEV), bits).cpu().numpy()
        assert np.array_equal(np_packed, t_packed), f"pack bits={bits}"
        t_unpacked = unpack_lowbit_torch(torch.from_numpy(np_packed).to(DEV), bits, 64).cpu().numpy()
        assert np.array_equal(codes, t_unpacked), f"unpack bits={bits}"


def test_w8_roundtrip():
    rng = np.random.default_rng(2)
    codes = rng.integers(-127, 128, size=(123, 128)).astype(np.int8)
    back = unpack_w8_groups(pack_w8_groups(codes)).reshape(codes.shape)
    assert np.array_equal(codes, back)


def test_quant_range_and_dequant():
    torch.manual_seed(3)
    w = torch.randn(64, 128)
    s, c = quantize_rows(w, 64, 7, -8, DEV)
    assert c.min() >= -8 and c.max() <= 7
    deq = dequantize_rows(s, c)
    # dequant must equal scale * code exactly
    assert np.allclose(deq, (c.astype(np.float32) * s.astype(np.float32)[:, :, None]).reshape(64, 128))


def test_encode_decode_layouts():
    torch.manual_seed(4)
    cases = [
        (torch.randn(192, 256), qt.QT_Q4G64, qt.LAYOUT_TILE_N64_K64, 64),
        (torch.randn(192, 256), qt.QT_Q5G64, qt.LAYOUT_TILE_N64_K64, 64),
        (torch.randn(128, 320), qt.QT_Q6G64, qt.LAYOUT_TILE_N64_K64, 64),
        (torch.randn(4304, 1152), qt.QT_Q4G64, qt.LAYOUT_TILE_N64_K64, 64),  # pad N
        (torch.randn(1152, 4304), qt.QT_Q5G64, qt.LAYOUT_TILE_N64_K64, 64),  # pad K
        (torch.randn(128, 256), qt.QT_W8G128, qt.LAYOUT_TILE_N64_K128, 64),
        (torch.randn(100, 128), qt.QT_Q6G64, qt.LAYOUT_ROW_GROUPED_G64, 1),  # no N pad
    ]
    for w, qtype, layout, nmult in cases:
        wb = w.bfloat16().float()
        payload, logical, padded, group, sd = encode_tensor(wb, qtype, layout, DEV)
        got = decode_tensor(payload, qtype, layout, logical, padded, DEV).cpu().numpy()
        ref = _ref_dequant(wb, qtype, nmult)
        assert list(got.shape) == [w.shape[0], w.shape[1]]
        assert np.array_equal(got, ref), f"{qt.QTYPE_NAME[qtype]}/{qt.LAYOUT_NAME[layout]}"


def test_contiguous_exact():
    torch.manual_seed(5)
    w = torch.randn(10, 1, 4).bfloat16()
    payload, logical, padded, group, sd = encode_tensor(w, qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, DEV)
    got = decode_tensor(payload, qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, logical, padded, DEV).cpu().numpy()
    assert np.array_equal(got, w.float().numpy())
    w = torch.randn(48).bfloat16()
    payload, logical, padded, group, sd = encode_tensor(w, qt.QT_FP32, qt.LAYOUT_CONTIGUOUS, DEV)
    got = decode_tensor(payload, qt.QT_FP32, qt.LAYOUT_CONTIGUOUS, logical, padded, DEV).cpu().numpy()
    assert np.array_equal(got, w.float().numpy())


def test_container_build_and_read():
    """Assemble a tiny .qus by hand (mirroring convert), then parse + decode it."""
    torch.manual_seed(6)
    tensors = [
        ("a.q4", qt.QT_Q4G64, qt.LAYOUT_TILE_N64_K64, torch.randn(64, 128)),
        ("b.w8", qt.QT_W8G128, qt.LAYOUT_TILE_N64_K128, torch.randn(64, 128)),
        ("c.norm", qt.QT_BF16, qt.LAYOUT_CONTIGUOUS, torch.randn(64)),
    ]
    entries, payloads, refs = [], [], []
    for name, qtype, layout, w in tensors:
        wb = w.bfloat16().float() if qtype != qt.QT_BF16 else w.bfloat16()
        payload, logical, padded, group, sd = encode_tensor(wb, qtype, layout, DEV)
        entries.append(fmt.TensorEntry(name=name, qtype=qtype, layout=layout, module_kind=qt.MODULE_TEXT,
                                       shape=logical, padded_shape=padded, group_size=group, scale_dtype=sd,
                                       source_layer=qt.NO_LAYER, source_kind=qt.SK_OTHER))
        payloads.append(payload)
        refs.append(decode_tensor(payload, qtype, layout, logical, padded, DEV).cpu().numpy())

    string_table = fmt.build_string_table(entries)
    module_count, tensor_count = 1, len(entries)
    mio = fmt.HEADER_SIZE
    mib = module_count * fmt.MODULE_RECORD_SIZE
    tio = mio + mib
    tib = tensor_count * fmt.TENSOR_ENTRY_SIZE
    sto = tio + tib
    stb = len(string_table)
    pstart = fmt.align_up(sto + stb, fmt.REGION_ALIGN)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.qus")
        with open(path, "wb") as f:
            f.seek(pstart)
            lo, hi = pstart, pstart
            for e, payload in zip(entries, payloads):
                pos = f.tell()
                pad = fmt.align_up(pos, fmt.PAYLOAD_ALIGN) - pos
                if pad:
                    f.write(b"\x00" * pad)
                    pos += pad
                f.write(payload)
                e.payload_offset = pos
                e.payload_bytes = len(payload)
                e.crc32 = fmt.crc32(payload)
                hi = pos + len(payload)
            file_size = f.tell()
            meta = bytearray()
            meta += fmt.pack_header(fmt.FileHeaderFields(
                tensor_count=tensor_count, module_count=module_count, layer_count=64, flags=1,
                module_index_offset=mio, module_index_bytes=mib,
                tensor_index_offset=tio, tensor_index_bytes=tib,
                string_table_offset=sto, string_table_bytes=stb,
                payload_offset=pstart, payload_bytes=file_size - pstart,
                hidden_size=5120, intermediate_size=17408, vocab_size=248320,
                num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                gdn_key_heads=16, gdn_value_heads=48, gdn_key_head_dim=128,
                gdn_value_head_dim=128, gdn_conv_width=4, full_attention_interval=4,
                max_position_embeddings=262144,
            ))
            meta += fmt.pack_module_record(fmt.ModuleRecord(qt.MODULE_TEXT, 1, 0, tensor_count, lo, hi - lo, qt.LOAD_RESIDENT))
            for e in entries:
                meta += fmt.pack_tensor_entry(e)
            meta += string_table
            meta = meta.ljust(pstart, b"\x00")
            f.seek(0)
            f.write(meta)

        # parse back
        from ..verify import _read_entries, _structural_checks
        hdr, modules, parsed, problems = _read_entries(path)
        problems += _structural_checks(path, hdr, parsed)
        assert not problems, problems
        assert hdr["tensor_count"] == tensor_count
        with open(path, "rb") as f:
            for e, ref in zip(parsed, refs):
                f.seek(e["payload_offset"])
                got = decode_tensor(f.read(e["payload_bytes"]), e["qtype"], e["layout"],
                                    e["shape"], e["padded_shape"], DEV).cpu().numpy()
                assert np.array_equal(got, ref), e["name"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
