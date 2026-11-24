from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.artifact.formats import get_format
from tools.convert.quantization.groupwise import quantize_matrix


def _fp16_decimal(word: int) -> Decimal:
    sign = -1 if word & 0x8000 else 1
    exp = (word >> 10) & 0x1F
    frac = word & 0x3FF
    if exp == 0:
        mag = Decimal(frac) * (Decimal(2) ** -24)
    else:
        mag = Decimal(1024 + frac) * (Decimal(2) ** (exp - 25))
    return mag if sign > 0 else -mag


def _decimal_exhaustive_oracle(
    group: list[float], qmin: int, qmax: int
) -> tuple[list[int], int]:
    """Independent exact 100-digit Decimal search over all nonzero finite FP16 scales."""
    with localcontext() as ctx:
        ctx.prec = 100
        xs = [Decimal(float(v)) for v in group]
        if not any(xs):
            return [0] * len(xs), 0

        xmax = max(Decimal(0), max(xs))
        xmin = max(Decimal(0), max(-v for v in xs))
        maxd = max(xmax, xmin)

        # Build exact sorted FP16 magnitude table in Decimal for the provably relevant range.
        def fp16_mag(word: int) -> Decimal:
            exp = (word >> 10) & 0x1F
            frac = word & 0x3FF
            return (
                Decimal(frac) * (Decimal(2) ** -24)
                if exp == 0
                else Decimal(1024 + frac) * (Decimal(2) ** (exp - 25))
            )

        def ceil_word(target: Decimal) -> int:
            if target <= fp16_mag(1):
                return 1
            if target >= Decimal(65504):
                return 0x7BFF
            lo, hi = 1, 0x7BFF
            while lo < hi:
                mid = (lo + hi) // 2
                if fp16_mag(mid) < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        def eval_candidate(sign: int, word: int):
            a = fp16_mag(word)
            codes = []
            sq = []
            for v in xs:
                y = v if sign > 0 else -v
                # Exact midpoint ties-to-even integer nearest in [qmin, qmax]
                q_est = int((y / a).to_integral_value(rounding="ROUND_HALF_EVEN"))
                q = max(qmin, min(qmax, q_est))
                if q < qmax:
                    hi = (Decimal(q) + Decimal("0.5")) * a
                    if y > hi or (y == hi and (q & 1) != 0):
                        q += 1
                if q > qmin:
                    lo = (Decimal(q) - Decimal("0.5")) * a
                    if y < lo or (y == lo and (q & 1) != 0):
                        q -= 1
                codes.append(q)
                diff = y - Decimal(q) * a
                sq.append(diff * diff)
            # Sum in Decimal (exact, independent of FP64 pairwise order)
            return sum(sq, Decimal(0)), codes

        w_pos0 = ceil_word(max(xmax / Decimal(qmax), xmin / Decimal(-qmin)))
        w_neg0 = ceil_word(max(xmax / Decimal(-qmin), xmin / Decimal(qmax)))
        w_hi = ceil_word(Decimal(2) * maxd)

        best_key = None
        best_result = None

        def consider(sign: int, word: int) -> None:
            nonlocal best_key, best_result
            err, codes = eval_candidate(sign, word)
            key = (err, 0 if sign > 0 else 1, word)
            if best_key is None or key < best_key:
                best_key = key
                best_result = (codes, (0x8000 if sign < 0 else 0) | word)

        consider(1, w_pos0)
        consider(-1, w_neg0)
        consider(1, ceil_word(maxd))

        # Exact exhaustive check over all words where clipping/zero bounds allow improvement
        for sign, qp, qn in ((1, qmax, -qmin), (-1, -qmin, qmax)):
            w0 = w_pos0 if sign > 0 else w_neg0
            for word in range(w0 - 1, 0, -1):
                a = fp16_mag(word)
                cp = max(Decimal(0), xmax - Decimal(qp) * a)
                cn = max(Decimal(0), xmin - Decimal(qn) * a)
                if cp * cp + cn * cn > best_key[0]:
                    break
                consider(sign, word)
            for word in range(w0, w_hi + 1):
                a = fp16_mag(word)
                half = a / Decimal(2)
                z = sum((v * v for v in xs if abs(v) <= half), Decimal(0))
                if z > best_key[0]:
                    break
                consider(sign, word)

        assert best_result is not None
        return best_result


@pytest.mark.parametrize(
    "format_name", ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"]
)
def test_grouped_mse_matches_independent_decimal_oracle(format_name):
    spec = get_format(format_name)
    g = spec.group_size
    gen = torch.Generator().manual_seed(2026 + spec.bits)

    rows = torch.randn((8, g), generator=gen).to(torch.bfloat16).float()
    # Row 0: all zeros
    rows[0].zero_()
    # Row 1: exact representable symmetric + asymmetric negative extreme (should pick +1.0)
    rows[1].zero_()
    rows[1, :6] = torch.tensor(
        [float(spec.qmin), -1.0, 0.0, 1.0, float(spec.qmax), 3.0]
    )
    # Row 2: positive extreme requiring negative FP16 scale to unlock |qmin| on positive side
    rows[2].zero_()
    rows[2, :5] = torch.tensor(
        [float(-spec.qmin), float(-spec.qmin), -1.0, 1.0, float(-spec.qmax)]
    )
    # Row 3: exact midpoint ties-to-even around scale 1.0
    rows[3].zero_()
    rows[3, :8] = torch.tensor(
        [float(spec.qmax), -float(spec.qmax), 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
    )
    # Row 4: sparse single nonzero element (tests upward sparse anchor & positive tie-break)
    rows[4].zero_()
    rows[4, 3] = 1.0
    # Row 5: tiny subnormal-underflow value
    rows[5].zero_()
    rows[5, 0] = torch.finfo(torch.float32).tiny

    references = [
        _decimal_exhaustive_oracle(rows[row].tolist(), spec.qmin, spec.qmax)
        for row in range(8)
    ]

    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        encoded = quantize_matrix(rows, format_name, device=device)
        assert encoded.codes.shape == (8, 128 // g, g)
        assert encoded.scales.shape == (8, 128 // g)

        codes_cpu = encoded.codes.cpu()
        scales_cpu = encoded.scales.cpu().view(torch.int16)
        for row, (ref_codes, ref_scale) in enumerate(references):
            assert codes_cpu[row, 0].tolist() == ref_codes
            assert (int(scales_cpu[row, 0].item()) & 0xFFFF) == ref_scale
            # Padded groups beyond logical K remain strictly zero
            assert torch.count_nonzero(codes_cpu[row, 1:]) == 0
            assert torch.count_nonzero(scales_cpu[row, 1:]) == 0

        # Specifically verify negative scale and asymmetric endpoint are actively used on row 2
        assert (int(scales_cpu[2, 0].item()) & 0xFFFF) == 0xBC00  # -1.0 in FP16
        assert int(codes_cpu[2, 0, 0].item()) == spec.qmin

        # Verify round-trip through row_split codec
        payload = encode_row_split(
            codes_cpu, encoded.scales.cpu(), format_name, tuple(rows.shape)
        )
        dec_scales, dec_codes = decode_row_split_codes(
            payload, format_name, tuple(rows.shape)
        )
        assert torch.equal(dec_scales, encoded.scales.cpu())
        assert torch.equal(dec_codes, codes_cpu)
        dequant = dequantize_row_split(
            payload, format_name, tuple(rows.shape), dtype=torch.float32
        )
        assert torch.equal(dequant[2, :2], torch.full((2,), float(-spec.qmin)))


@pytest.mark.parametrize(
    "format_name", ["q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"]
)
def test_cuda_matches_cpu_bit_exact_and_beats_absmax(format_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spec = get_format(format_name)
    gen = torch.Generator().manual_seed(9001 + spec.bits)
    # Non-aligned K exercises partially populated tail groups and zero padding
    weight = (torch.randn((23, 197), generator=gen) * 0.03).to(torch.bfloat16)

    cpu = quantize_matrix(weight, format_name, device="cpu")
    cuda = quantize_matrix(weight, format_name, device="cuda")

    assert torch.equal(cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16))
    assert torch.equal(cuda.codes.cpu(), cpu.codes)

    # Verify negative scales actually occur on random weights
    assert bool((cpu.scales < 0).any())

    # Compare reconstruction MSE against legacy grouped absmax
    w = weight.float()
    reconstructed = (
        cpu.codes.float() * cpu.scales.float().unsqueeze(-1)
    ).reshape(weight.shape[0], -1)[:, : weight.shape[1]]
    mse_opt = float(((w.double() - reconstructed.double()) ** 2).mean().item())

    k_pad = ((weight.shape[1] + 127) // 128) * 128
    padded = torch.zeros((weight.shape[0], k_pad), dtype=torch.float32)
    padded[:, : weight.shape[1]] = w
    grouped = padded.reshape(weight.shape[0], k_pad // spec.group_size, spec.group_size)
    abs_s = (grouped.abs().amax(-1).double() / spec.qmax).float().half()
    abs_sf = abs_s.float()
    abs_inv = torch.where(abs_sf > 0, abs_sf.double().reciprocal().float(), 0.0)
    abs_q = torch.clamp(torch.round(grouped * abs_inv.unsqueeze(-1)), spec.qmin, spec.qmax)
    abs_rec = (abs_q * abs_sf.unsqueeze(-1)).reshape(weight.shape[0], k_pad)[
        :, : weight.shape[1]
    ]
    mse_absmax = float(((w.double() - abs_rec.double()) ** 2).mean().item())

    assert mse_opt < mse_absmax * 0.90


def test_cuda_encoding_preserves_groups_across_chunk_boundaries():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = torch.randn(
        (37, 160), generator=torch.Generator().manual_seed(381)
    ).bfloat16()
    whole = quantize_matrix(values, "q4_g64_fp16", device="cuda")
    pieces = [
        quantize_matrix(values[i : i + 11], "q4_g64_fp16", device="cuda")
        for i in range(0, 37, 11)
    ]
    assert torch.equal(whole.codes, torch.cat([p.codes for p in pieces]))
    assert torch.equal(whole.scales, torch.cat([p.scales for p in pieces]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonfinite_input_is_rejected(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    bad = torch.zeros((1, 64), dtype=torch.float32)
    bad[0, 5] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        quantize_matrix(bad, "q4_g64_fp16", device=device)
