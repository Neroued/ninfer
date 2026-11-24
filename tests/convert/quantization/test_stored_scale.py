from decimal import Decimal, localcontext
import struct

import pytest
import torch

from tools.artifact.formats import NF4_CODEBOOK_WORDS
from tools.convert.quantization.stored_scale import quantize_stored_scale


def _oracle(group, nf4, divisor):
    # Decimal arithmetic and direct distances are independent of GPU midpoint
    # search, pruning and parallel FP64 reductions.
    with localcontext() as ctx:
        ctx.prec = 100
        table = (
            [
                Decimal(struct.unpack("<f", struct.pack("<I", w))[0])
                for w in NF4_CODEBOOK_WORDS
            ]
            if nf4
            else [Decimal(x) / 2 for x in (0, 1, 2, 3, 4, 6, 8, 12)]
        )
        x = [Decimal(float(w)) * Decimal(divisor) for w in group]
        if not any(x):
            return [7 if nf4 else 0] * 16, 0
        best = None
        for word in range(1, 127):
            e, m = divmod(word, 8)
            scale = (
                Decimal(m) / 512 if e == 0 else Decimal(8 + m) * Decimal(2) ** (e - 10)
            )
            codes, error = [], Decimal(0)
            for value in x:
                target = value if nf4 else abs(value)
                code = min(
                    range(len(table)),
                    key=lambda j: (
                        abs(target - table[j] * scale),
                        j if nf4 else j % 2,
                        j,
                    ),
                )
                error += (target - table[code] * scale) ** 2
                codes.append(code | (8 if not nf4 and value < 0 else 0))
            if best is None or error < best[0]:
                best = error, codes, word
        return best[1], best[2]


@pytest.mark.parametrize("format", ["nf4_g16_e4m3fn", "nvfp4"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stored_scale_matches_independent_decimal_search(format, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gen = torch.Generator().manual_seed(196)
    groups = torch.randn((12, 16), generator=gen).to(torch.bfloat16).float()
    groups[0].zero_()
    groups[1] = torch.tensor(
        [
            -0.25,
            0.25,
            -0.75,
            0.75,
            -1.25,
            1.25,
            -1.75,
            1.75,
            -2.5,
            2.5,
            -3.5,
            3.5,
            -5.0,
            5.0,
            -6.0,
            6.0,
        ]
    )
    groups[2].fill_(2.0**-15)
    groups[3] = torch.tensor([0.0, 1.0, -1.0, 0.0] * 4)
    divisor = struct.pack("<f", 1.75)
    encoded = quantize_stored_scale(groups, format, divisor, device=device)
    packed = encoded.codes.cpu()
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 16).tolist()
    for row, (actual_codes, actual_scale) in enumerate(
        zip(codes, encoded.scales.cpu().flatten().tolist())
    ):
        reference = _oracle(groups[row], format != "nvfp4", 1.75)
        assert (actual_codes, actual_scale) == reference


def test_cuda_encoding_preserves_groups_across_chunk_boundaries():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = torch.randn(
        (47, 160), generator=torch.Generator().manual_seed(381)
    ).bfloat16()
    divisor = struct.pack("<f", 73.125)
    whole = quantize_stored_scale(values, "nf4_g16_e4m3fn", divisor)
    pieces = [
        quantize_stored_scale(values[i : i + 13], "nf4_g16_e4m3fn", divisor)
        for i in range(0, 47, 13)
    ]
    assert torch.equal(whole.codes, torch.cat([p.codes for p in pieces]))
    assert torch.equal(whole.scales, torch.cat([p.scales for p in pieces]))


@pytest.mark.parametrize(
    "format,shape", [("nf4_g16_e4m3fn", (46, 16)), ("nvfp4", (256, 64))]
)
@pytest.mark.parametrize("source_cache_bytes", [0, 1024 * 1024])
def test_conversion_method_uses_complete_parent_amax(
    tmp_path, format, shape, source_cache_bytes
):
    from tools.artifact.codecs.nf4 import decode_nf4_words
    from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
    from tools.artifact.reader import Artifact
    from tools.convert.model import Model, Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.logical import array_source

    values = torch.zeros(shape, dtype=torch.bfloat16)
    values[0].fill_(1.0)
    values[-1].fill_(16.0)
    model = Model({"text": {"config": {}}})
    model.add(Parameter("table", shape, array_source(values, "values")))
    recipe = Recipe(model)
    recipe.assign("table", format=format, method="stored_scale_mse")
    path = tmp_path / "table.ninfer"
    convert(
        model,
        recipe,
        path,
        device="cpu",
        rows_per_chunk=17,
        source_cache_bytes=source_cache_bytes,
    )
    with Artifact(path) as artifact:
        obj = artifact.object(artifact.directory.bindings["table"]["object"])
        decode = decode_nvfp4_words if format == "nvfp4" else decode_nf4_words
        codes, scales, divisor = decode(artifact.read_object(obj.id), shape)
        assert divisor.item() == (168.0 if format == "nvfp4" else 28.0)
        assert bool((scales[0] == 0x5E).all()) and bool((scales[-1] == 0x7E).all())
        assert bool((scales[1:-1] == 0).all())
        endpoint, zero = (0x77, 0x00) if format == "nvfp4" else (0xFF, 0x77)
        assert bool((codes[0] == endpoint).all()) and bool(
            (codes[-1] == endpoint).all()
        )
        assert bool((codes[1:-1] == zero).all())
