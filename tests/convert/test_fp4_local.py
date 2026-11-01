"""FP4-Local (nvidia/Qwen3.8-Flash-Next-FP4-Local): CSF scales, MXFP8 and row-NVFP4 sources, the recipe.

References are written here from the format definitions; the CSF reference decoder and encoder are
transcriptions of vLLM PR #60646 (``csf_scales.py`` and its test, Apache-2.0), not of the code under
test. Tests marked ``real`` read the downloaded checkpoint at ``NINFER_TEST_FP4_LOCAL_CHECKPOINT``
(skipped when it is unset or holds no checkpoint) and compare against bytes read straight from the
safetensors files.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct

import pytest
import torch
from safetensors.torch import save_file

from tests.convert import qwen4_exp_fixture as fixture
from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales
from tools.artifact.formats import get_format
from tools.artifact.layouts import block_scale_geometry
from tools.artifact.reader import Artifact
from tools.convert.official_recipes import import_verbatim, qwen3_8_flash_next_fp4_local
from tools.convert.methods import import_encoded
from tools.convert.pipeline import convert
from tools.convert.qwen4_exp import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.csf import decode_csf, decode_csf_slabs, exception_words
from tools.convert.sources.modelopt import nvfp4_row_table_source, nvfp4_source
from tools.convert.sources.safetensors import SafetensorsSource

REAL = Path(os.environ.get("NINFER_TEST_FP4_LOCAL_CHECKPOINT", ""))
real = pytest.mark.skipif(
    not os.environ.get("NINFER_TEST_FP4_LOCAL_CHECKPOINT") or not (REAL / "model.safetensors.index.json").is_file(),
    reason="NINFER_TEST_FP4_LOCAL_CHECKPOINT does not name an FP4-Local checkpoint",
)
LM = "model.language_model."
MXFP8 = "mxfp8_e4m3fn_g32_ue8m0"
NVFP4_ROW = "nvfp4_row_e4m3_g16"


# --- references -----------------------------------------------------------------------------------


def _reference_csf_decode(fixed: torch.Tensor, exceptions: torch.Tensor) -> torch.Tensor:
    """vLLM PR #60646 ``decode_nvfp4_csf_scale`` (Apache-2.0), returning the U8 scale bytes."""
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    bases = fixed[:, :16].reshape(rows, 1).to(torch.int16)
    assert not bool((bases > 240).any())
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, cols)
    out = (bases + codes.to(torch.int16)).reshape(-1)
    words = exceptions.reshape(-1).view(torch.int32).to(torch.int64)
    words &= 0xFFFFFFFF
    positions = words & 0xFFFFFF
    assert not positions.numel() or int(positions.max()) < rows * cols
    out[positions] = (words >> 24).to(torch.int16)
    return out.to(torch.uint8).reshape(rows, cols)


def _reference_csf_encode(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM PR #60646 test encoder: row base = row min (capped at 240), outliers as exceptions."""
    raw = raw.to(torch.int64)
    rows, cols = raw.shape
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    codes = raw - bases
    outlier = (codes < 0) | (codes > 15)
    codes = torch.where(outlier, 0, codes)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat((bases.to(torch.uint8).reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), dim=1)
    positions = outlier.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return fixed, words.to(torch.int32).view(torch.uint32)


def _e4m3(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 0:
        return sign * mantissa / 8 * 2.0**-6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def _e2m1(nibble: int) -> float:
    sign = -1.0 if nibble & 8 else 1.0
    exponent, mantissa = (nibble >> 1) & 3, nibble & 1
    if exponent == 0:
        return sign * mantissa * 0.5
    return sign * (1 + mantissa / 2) * 2.0 ** (exponent - 1)


def _raw(root: Path, name: str, begin: int = 0, end: int | None = None) -> tuple[str, tuple, bytes]:
    """Bytes ``[begin, end)`` of one tensor, read straight from the indexed safetensors file."""
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    with open(root / weight_map[name], "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        info = json.loads(stream.read(length))[name]
        first, last = info["data_offsets"]
        end = last - first if end is None else end
        stream.seek(8 + length + first + begin)
        return info["dtype"], tuple(info["shape"]), stream.read(end - begin)


def _u8(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


# --- CSF ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(640, 160), (2560, 40), (16, 2)])
def test_csf_decoder_matches_reference_and_source_scales(shape):
    g = torch.Generator().manual_seed(shape[0])
    raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8, generator=g)
    count = min(64, raw.numel() // 4)
    raw.view(-1)[torch.randperm(raw.numel(), generator=g)[:count]] = 0x7E
    raw.view(-1)[torch.randperm(raw.numel(), generator=g)[:count]] = 0x01
    fixed, exceptions = _reference_csf_encode(raw)
    assert exceptions.numel() >= count
    decoded = decode_csf(fixed, exceptions)
    assert decoded.dtype == torch.uint8 and torch.equal(decoded, raw)
    assert torch.equal(decoded, _reference_csf_decode(fixed, exceptions))
    # Any run of slabs decodes alone to the same rows.
    rows, cols = shape
    positions, values = exception_words(exceptions, rows, cols)
    slabs = rows // 16
    for first, last in ((0, 1), (slabs // 2, slabs), (slabs - 1, slabs)):
        part = decode_csf_slabs(fixed[first:last], positions, values, first_slab=first, cols=cols)
        assert torch.equal(part, raw[first * 16 : last * 16])


def test_csf_rejects_malformed_streams():
    raw = torch.full((16, 4), 0x38, dtype=torch.uint8)
    raw[3, 1] = 0x7E
    fixed, exceptions = _reference_csf_encode(raw)
    bad = fixed.clone()
    bad[0, 0] = 241
    with pytest.raises(ValueError, match="base"):
        decode_csf(bad, exceptions)
    with pytest.raises(ValueError, match="exceeds"):
        decode_csf(fixed, torch.tensor([(0x7E << 24) | 64], dtype=torch.int64).to(torch.uint32))
    words = torch.tensor([(0x7E << 24) | 13, (0x7E << 24) | 5], dtype=torch.int64).to(torch.uint32)
    with pytest.raises(ValueError, match="increasing"):
        decode_csf(fixed, words)
    with pytest.raises(ValueError, match="shape"):
        decode_csf(fixed[:, :24], exceptions)


def test_nvfp4_source_decodes_csf_scales_for_any_row_range(tmp_path):
    g = torch.Generator().manual_seed(7)
    n, k = 64, 128
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    scales = torch.randint(0x28, 0x40, (n, k // 16), dtype=torch.uint8, generator=g)
    scales[5, 3], scales[40, 0] = 0x7E, 0x00
    fixed, exceptions = _reference_csf_encode(scales)
    save_file(
        {
            "e.weight": codes,
            "e.weight_scale.nvfp4_csf_fixed": fixed,
            "e.weight_scale.nvfp4_csf_exceptions": exceptions,
            "e.weight_scale_2": torch.tensor(0.25, dtype=torch.float32),
            "e.input_scale": torch.tensor(0.5, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = nvfp4_source(store, "e", (n, k))
        for begin, end in ((0, n), (0, 1), (3, 37), (16, 32), (63, 64)):
            words = source.read_encoded(begin, end)
            assert torch.equal(words.codes, codes[begin:end]) and torch.equal(words.scales, scales[begin:end])
        assert words.weight_divisor == struct.pack("<f", 4.0) and source.input_divisor() == struct.pack("<f", 2.0)
        row = source.rows(5, 6)[0]
        want = [
            _e2m1(int(codes[5, c // 2]) >> (4 * (c % 2)) & 15) * _e4m3(int(scales[5, c // 16])) * 0.25 for c in range(k)
        ]
        assert row.tolist() == want


# --- MXFP8 and row-NVFP4 sources ------------------------------------------------------------------


def test_mxfp8_source_keeps_words_and_decodes_exactly(tmp_path):
    g = torch.Generator().manual_seed(3)
    n, k = 6, 64
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0x11
    exponents = torch.randint(110, 140, (n, k // 32), dtype=torch.uint8, generator=g)
    save_file(
        {"p.weight": codes.view(torch.float8_e4m3fn), "p.weight_scale": exponents},
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "p.weight", (n, k))
        words = source.read_encoded(1, 4)
        assert words.format == MXFP8
        assert torch.equal(words.codes, codes[1:4]) and torch.equal(words.scales, exponents[1:4])
        want = [
            math.ldexp(_e4m3(int(codes[r, c])), int(exponents[r, c // 32]) - 127) for r in range(n) for c in range(k)
        ]
        assert source.values().tolist() == want
        assert source.values(70, 75).tolist() == want[70:75]
    bad = exponents.clone()
    bad[2, 1] = 255
    save_file({"p.weight": codes.view(torch.float8_e4m3fn), "p.weight_scale": bad}, str(tmp_path / "model.safetensors"))
    with SafetensorsSource(tmp_path) as store, pytest.raises(ValueError, match="255"):
        matrix_source(store, "p.weight", (n, k)).read_encoded(0, n)


def test_row_nvfp4_table_source_concatenates_shards_exactly(tmp_path):
    g = torch.Generator().manual_seed(5)
    width, counts = 32, (5, 7)
    shards, tensors = [], {}
    for index, count in enumerate(counts):
        name = f"t.shard_{index}"
        shards.append(name)
        tensors[name + ".weight"] = torch.randint(0, 256, (count, width // 2), dtype=torch.uint8, generator=g)
        scales = torch.randint(0x20, 0x50, (count, width // 16), dtype=torch.uint8, generator=g)
        tensors[name + ".weight_scale"] = scales.view(torch.float8_e4m3fn)
    tensors["t.weight_scale_2"] = torch.tensor([3.3242362e-05], dtype=torch.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    codes = torch.cat([tensors[s + ".weight"] for s in shards])
    scales = torch.cat([tensors[s + ".weight_scale"].view(torch.uint8) for s in shards])
    G = float(tensors["t.weight_scale_2"][0])
    with SafetensorsSource(tmp_path) as store:
        source = nvfp4_row_table_source(store, shards, "t.weight_scale_2", (sum(counts), width))
        for begin, end in ((0, 12), (3, 9), (5, 6), (4, 5)):
            words = source.read_encoded(begin, end)
            assert words.format == NVFP4_ROW
            assert torch.equal(words.codes, codes[begin:end]) and torch.equal(words.scales, scales[begin:end])
        want = []
        for r in range(sum(counts)):
            for c in range(width):
                nibble = int(codes[r, c // 2]) >> (4 * (c % 2)) & 15
                # One rounding to FP32 after the exact product (the contract's reference).
                want.append(
                    float(torch.tensor(_e2m1(nibble) * _e4m3(int(scales[r, c // 16])) * G, dtype=torch.float32))
                )
        assert source.values().tolist() == want


# --- synthetic FP4-Local checkpoint ---------------------------------------------------------------

_DENSE = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.indexer.index_qk_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)
PLE_DIM = 64  # n-gram rows of 16 values: one row-NVFP4 block


def _mxfp8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = values.shape
    groups = values.float().reshape(n, k // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp(min=2.0**-100)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-126, 127)
    codes = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).reshape(n, k)
    return codes, (exponent + 127).to(torch.uint8)


def write_fp4_local_checkpoint(path: Path) -> dict:
    """The qwen4_exp fixture rewritten in FP4-Local's formats; returns the original-form tensors too.

    MXFP8 dense Text projections, routed-expert scales CSF-compressed (the returned ``scales`` keep the
    uncompressed bytes), NVFP4 MTP experts without activation scales, F32 Text norms, a 16-wide
    row-NVFP4 n-gram table in two shards with one ``weight_scale_2``.
    """
    tensors = fixture.write_checkpoint(path.parent / (path.name + "-nvfp4"))
    config = fixture.config()
    config["text_config"]["ple_embed_dim"] = PLE_DIM
    text = config["text_config"]
    g = torch.Generator().manual_seed(11)
    scales = {}
    for name in list(tensors):
        prefix = name.removesuffix(".weight")
        if name.startswith(LM) and name.endswith(".weight") and prefix.endswith(_DENSE):
            tensors[name], tensors[prefix + ".weight_scale"] = _mxfp8(tensors[name])
        elif name.startswith(LM) and ".mlp.experts." in name and name.endswith(".weight_scale"):
            raw = tensors.pop(name).view(torch.uint8)
            raw[0, 0] = 0x7E  # at least one exception per matrix
            scales[name] = raw
            tensors[name + ".nvfp4_csf_fixed"], tensors[name + ".nvfp4_csf_exceptions"] = _reference_csf_encode(raw)
        elif name.startswith(LM) and name.endswith(("hc_norm.weight", "norm_key.weight", "norm_query.weight")):
            tensors[name] = tensors[name].float() + 2.0**-12  # not representable in BF16
        elif name.startswith(LM) and name.endswith("norm_conv.weight"):
            tensors[name] = tensors[name].float()
    for name in [n for n in tensors if n.startswith("mtp.") and ".mlp.experts." in n]:
        del tensors[name]
    for e in range(fixture.E):
        for role, shape in (
            ("gate", (fixture.IR, fixture.H)),
            ("up", (fixture.IR, fixture.H)),
            ("down", (fixture.H, fixture.IR)),
        ):
            prefix = f"mtp.layers.0.mlp.experts.{e}.{role}_proj"
            fixture._nvfp4_expert(tensors, prefix, shape, g)
            del tensors[prefix + ".input_scale"]
    ple = f"{LM}layers.{fixture.PLE_LAYER_IDS[0] - 1}.ple."
    table = ple + "ple_embedding.ngram_embedding"
    for name in [n for n in tensors if n.startswith(table + ".")]:
        del tensors[name]
    rows = fixture.ple_constants(text)["rows"]
    width = PLE_DIM // ((fixture.NGRAM - 1) * fixture.HEADS_PER_NGRAM)
    half = rows // 2
    for index, (first, last) in enumerate(((0, half), (half, rows))):
        tensors[f"{table}.shard_{index}.weight"] = torch.randint(
            0, 256, (last - first, width // 2), dtype=torch.uint8, generator=g
        )
        row_scales = torch.randint(0x20, 0x50, (last - first, width // 16), dtype=torch.uint8, generator=g)
        tensors[f"{table}.shard_{index}.weight_scale"] = row_scales.view(torch.float8_e4m3fn)
    tensors[table + ".weight_scale_2"] = torch.tensor([3.3242362e-05], dtype=torch.float32)
    wide = fixture.HC * fixture.H
    tensors[ple + "key_proj.weight"] = (torch.randn(wide, PLE_DIM, generator=g) * 0.05).to(torch.bfloat16)
    tensors[ple + "value_proj.weight"] = (torch.randn(fixture.H, PLE_DIM, generator=g) * 0.05).to(torch.bfloat16)

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=1))
    second = {k for k in tensors if k.startswith("mtp.") or ".ple_embedding." in k}
    shards = {
        "tensors/model-00001-of-00002.safetensors": {k: v for k, v in tensors.items() if k not in second},
        "tensors/model-00002-of-00002.safetensors": {k: tensors[k] for k in second},
    }
    (path / "tensors").mkdir(exist_ok=True)
    weight_map = {}
    for file, shard in shards.items():
        save_file({k: v.contiguous() for k, v in shard.items()}, str(path / file))
        weight_map.update({k: file for k in shard})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    fixture.write_tokenizer(path)
    return {"tensors": tensors, "scales": scales, "config": config}


def _plan(source_dir):
    store = SafetensorsSource(source_dir)
    model = build_model(store, components=("text", "mtp"))
    recipe = Recipe(model)
    qwen3_8_flash_next_fp4_local(model, recipe, {"base": store})
    return store, model, recipe


def test_fp4_local_recipe_keeps_every_checkpoint_format(tmp_path):
    written = write_fp4_local_checkpoint(tmp_path / "source")
    store, model, recipe = _plan(tmp_path / "source")
    with store:
        chosen = {name: recipe.selections[name] for name in model.parameters}
        assert all(len(items) == 1 for items in chosen.values())
        chosen = {name: items[0] for name, items in chosen.items()}
        ple = f"text/layers/{fixture.PLE_LAYER_IDS[0] - 1}/ple/"
        assert chosen[ple + "table"].format == NVFP4_ROW and chosen[ple + "table"].method is import_verbatim
        assert chosen[ple + "table"].layout == "row_interleaved_v1"
        assert chosen[ple + "table_global_scale"].format == "fp32"
        assert model.parameters[ple + "table_global_scale"].shape == (1,)
        assert model.parameters[ple + "table_global_scale"].source.values().tolist() == [
            float(written["tensors"][f"{LM}layers.0.ple.ple_embedding.ngram_embedding.weight_scale_2"][0])
        ]
        verbatim = {name for name, item in chosen.items() if item.method is import_verbatim}
        mxfp8 = {name for name in verbatim if chosen[name].format == MXFP8}
        for name in mxfp8:
            assert name.startswith("text/layers/") and chosen[name].layout == "block_scale_k32_v1"
        # Every MXFP8 checkpoint matrix of the Text layers, nothing else.
        attention = ("query", "key", "gate", "value", "output", "indexer_query", "indexer_key")
        gdn = ("query", "key", "value", "z", "a_projection", "b_projection", "output")
        want = set()
        for layer, kind in enumerate(fixture.LAYER_TYPES):
            p = f"text/layers/{layer}/"
            roles = (
                [p + "attention/" + r for r in attention] if kind == "full_attention" else [p + "gdn/" + r for r in gdn]
            )
            want.update(roles + [p + f"moe/shared/{r}" for r in ("gate", "up", "down")])
        assert mxfp8 == want
        experts = {name for name in chosen if "/moe/experts/" in name}
        for name in experts:
            assert chosen[name].format == "nvfp4" and chosen[name].method is import_encoded
            policy = "AllowA4" if name.startswith("text/") else "A16Only"
            assert {recipe.policies[(name, i)] for i in model.parameters[name].inputs} == {policy}
        for name, item in chosen.items():
            if name not in verbatim | experts and name != ple + "table_global_scale":
                assert item.format in ("bf16", "fp32") and item.method.__name__ == "cast_direct", name
        groups = {names for names, _ in recipe.explicit_groups}
        for layer, kind in enumerate(fixture.LAYER_TYPES):
            p = f"text/layers/{layer}/"
            if kind == "full_attention":
                assert tuple(p + "attention/" + r for r in ("query", "key", "gate", "value")) in groups
                assert (p + "attention/indexer_query", p + "attention/indexer_key") in groups
            else:
                assert tuple(p + "gdn/" + r for r in ("query", "key", "value", "z")) in groups
                assert (p + "gdn/a_projection", p + "gdn/b_projection") in groups
            assert (p + "moe/shared/gate", p + "moe/shared/up") in groups
        expert_groups = [names for names in groups if "/moe/experts/" in names[0]]
        assert len(expert_groups) == 2 * (len(fixture.LAYER_TYPES) + 1)
        # Text routed experts read their CSF scales back as the uncompressed bytes.
        for key, raw in written["scales"].items():
            layer, expert, role = key.split(".")[3], key.split(".")[6], key.split(".")[7].removesuffix("_proj")
            name = f"text/layers/{layer}/moe/experts/{expert}/{role}"
            words = model.parameters[name].source.read_encoded(0, raw.shape[0])
            assert torch.equal(words.scales, raw)


def test_fp4_local_recipe_refuses_vision(tmp_path):
    write_fp4_local_checkpoint(tmp_path / "source")
    with SafetensorsSource(tmp_path / "source") as store:
        model = build_model(store, components=("text",))
        model.components["vision"] = {"config": {}}  # the fixture has no Vision tower; the guard only reads the key
        with pytest.raises(ValueError, match="text and mtp only"):
            qwen3_8_flash_next_fp4_local(model, Recipe(model), {"base": store})


def _parts(directory, name):
    binding = directory.bindings[name]
    return binding["parts"] if "parts" in binding else [{"object": binding["object"]}]


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _formats_registered() -> bool:
    try:
        get_format(MXFP8), get_format(NVFP4_ROW)
    except (KeyError, ValueError):
        return False
    return True


@pytest.mark.skipif(not _formats_registered(), reason="the artifact formats arrive with the fp4l base piece")
def test_fp4_local_conversion_stores_checkpoint_words(tmp_path):
    written = write_fp4_local_checkpoint(tmp_path / "source")
    tensors = written["tensors"]
    store, model, recipe = _plan(tmp_path / "source")
    out = tmp_path / "fp4_local.ninfer"
    with store:
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        directory = artifact.directory

        def parent(name):
            (part,) = _parts(directory, name)
            return artifact.object(part["object"]), part

        # MXFP8 groups: code plane = the checkpoint codes of the members in group order, padded to
        # 256 bytes, then the scale plane = their UE8M0 bytes in the same order.
        d = fixture.HEAD_DIM
        heads = [(2 * h * d, 2 * h * d + d) for h in range(fixture.HEADS)]
        for layer, kind in enumerate(fixture.LAYER_TYPES):
            sp, p = f"{LM}layers.{layer}.", f"text/layers/{layer}/"
            groups = [
                (
                    (p + "moe/shared/gate", p + "moe/shared/up"),
                    [(sp + "mlp.shared_expert.gate_proj", None), (sp + "mlp.shared_expert.up_proj", None)],
                ),
                ((p + "moe/shared/down",), [(sp + "mlp.shared_expert.down_proj", None)]),
            ]
            if kind == "full_attention":
                q = sp + "self_attn.q_proj"
                groups += [
                    (
                        tuple(p + "attention/" + r for r in ("query", "key", "gate", "value")),
                        [
                            (q, heads),
                            (sp + "self_attn.k_proj", None),
                            (q, [(a + d, b + d) for a, b in heads]),
                            (sp + "self_attn.v_proj", None),
                        ],
                    ),
                    ((p + "attention/output",), [(sp + "self_attn.o_proj", None)]),
                    (
                        (p + "attention/indexer_query", p + "attention/indexer_key"),
                        [(sp + "self_attn.indexer.index_qk_proj", None)],
                    ),
                ]
            else:
                g = sp + "linear_attn."
                groups += [
                    (
                        tuple(p + "gdn/" + r for r in ("query", "key", "value", "z")),
                        [(g + "in_proj_qkv", None), (g + "in_proj_z", None)],
                    ),
                    (
                        (p + "gdn/a_projection", p + "gdn/b_projection"),
                        [(g + "in_proj_a", None), (g + "in_proj_b", None)],
                    ),
                    ((p + "gdn/output",), [(g + "out_proj", None)]),
                ]
            for names, members in groups:
                obj, _ = parent(names[0])
                assert all(parent(name)[0].id == obj.id for name in names)
                codes, scales = [], []
                for prefix, ranges in members:
                    weight, exponent = tensors[prefix + ".weight"], tensors[prefix + ".weight_scale"]
                    ranges = ranges or [(0, weight.shape[0])]
                    codes += [_bytes(weight[a:b]) for a, b in ranges]
                    scales += [_bytes(exponent[a:b]) for a, b in ranges]
                codes, scales = b"".join(codes), b"".join(scales)
                n = len(codes) // obj.shape[1]
                assert obj.format == MXFP8 and obj.layout == "block_scale_k32_v1" and obj.shape[0] == n
                pad = -len(codes) % 256
                assert artifact.read_object(obj.id) == codes + bytes(pad) + scales, names

        # The n-gram table: 16-value rows as 8 code bytes then 1 scale byte, and G beside it.
        table = f"{LM}layers.0.ple.ple_embedding.ngram_embedding"
        obj, _ = parent("text/layers/0/ple/table")
        assert obj.format == NVFP4_ROW and obj.layout == "row_interleaved_v1"
        rows = torch.cat(
            [
                torch.cat(
                    (
                        tensors[f"{table}.shard_{s}.weight"],
                        tensors[f"{table}.shard_{s}.weight_scale"].view(torch.uint8),
                    ),
                    dim=1,
                )
                for s in (0, 1)
            ]
        )
        assert obj.shape == (rows.shape[0], 16) and artifact.read_object(obj.id) == _bytes(rows)
        obj, _ = parent("text/layers/0/ple/table_global_scale")
        assert obj.format == "fp32" and artifact.read_object(obj.id) == _bytes(tensors[table + ".weight_scale_2"])

        # Routed experts: CSF scales stored as the original uncompressed bytes (swizzled plane).
        assert len(written["scales"]) == 3 * fixture.E * len(fixture.LAYER_TYPES)
        uses = {(u["parameter"], u["input"]): u for u in directory.uses}
        for key, raw_scales in written["scales"].items():
            layer, expert, role = key.split(".")[3], key.split(".")[6], key.split(".")[7].removesuffix("_proj")
            obj, part = parent(f"text/layers/{layer}/moe/experts/{expert}/{role}")
            k = obj.shape[1]
            row, last = part["range"][0] // k, part["range"][1] // k
            geometry = block_scale_geometry("nvfp4", obj.shape, obj.divisors)
            data = artifact.read_object(obj.id)
            span = data[geometry.scale_plane_offset + row * k // 16 :][: (last - row) * k // 16]
            natural = unswizzle_nvfp4_scales(torch.frombuffer(bytearray(span), dtype=torch.uint8), (last - row, k))
            assert torch.equal(natural.reshape(-1), raw_scales.reshape(-1)), key
            assert data[row * k // 2 : last * k // 2] == _bytes(tensors[key.removesuffix("_scale")])
            source_input = f"text/layers/{layer}/" + (
                f"moe/experts/{expert}/product" if role == "down" else "ffn_input"
            )
            use = uses[(f"text/layers/{layer}/moe/experts/{expert}/{role}", source_input)]
            assert use["activation_policy"] == "AllowA4" and "activation_input_divisor" in use.get("auxiliaries", {})
        # MTP experts: their own NVFP4 words, 16-bit activations, no activation divisor.
        obj, part = parent("mtp/layers/0/moe/experts/1/up")
        k = obj.shape[1]
        row = part["range"][0] // k
        data = artifact.read_object(obj.id)
        assert data[row * k // 2 :][: fixture.IR * k // 2] == _bytes(
            tensors["mtp.layers.0.mlp.experts.1.up_proj.weight"]
        )
        use = uses[("mtp/layers/0/moe/experts/1/up", "mtp/layers/0/ffn_input")]
        assert use["activation_policy"] == "A16Only" and not use.get("auxiliaries")

        # BF16 stays BF16; F32 norms are rounded once to BF16.
        obj, _ = parent("text/token_embedding")
        assert obj.format == "bf16" and artifact.read_object(obj.id) == _bytes(tensors[LM + "embed_tokens.weight"])
        norm = tensors[LM + "layers.1.attn_hyper_connection.hc_norm.weight"]
        obj, part = parent("text/layers/1/attn_hc/norm")
        assert norm.dtype == torch.float32 and obj.format == "bf16"
        assert artifact.read_object(obj.id)[: 2 * norm.numel()] == _bytes(norm.to(torch.bfloat16))


# --- the real checkpoint --------------------------------------------------------------------------


def _expert_matrices(count: int) -> list[str]:
    """Routed-expert projections from ``count`` different safetensors files, spread over layers and roles."""
    weight_map = json.loads((REAL / "model.safetensors.index.json").read_text())["weight_map"]
    chosen, files = [], set()
    roles = ("gate_proj", "up_proj", "down_proj")
    for step in range(48 * 512):
        layer, expert = (step * 7) % 48, (step * 131 + 17) % 512
        prefix = f"{LM}layers.{layer}.mlp.experts.{expert}.{roles[len(chosen) % 3]}"
        file = weight_map[prefix + ".weight_scale.nvfp4_csf_fixed"]
        if file not in files:
            files.add(file)
            chosen.append(prefix)
            if len(chosen) == count:
                return chosen
    raise AssertionError("not enough expert files")


@real
def test_real_csf_scales_match_the_reference_decoder():
    with SafetensorsSource(REAL) as store:
        for prefix in _expert_matrices(4):
            dtype, shape, data = _raw(REAL, prefix + ".weight_scale.nvfp4_csf_fixed")
            fixed = _u8(data).reshape(shape)
            assert dtype == "U8"
            dtype, _, data = _raw(REAL, prefix + ".weight_scale.nvfp4_csf_exceptions")
            assert dtype == "U32"
            exceptions = torch.frombuffer(bytearray(data), dtype=torch.int32).view(torch.uint32)
            want = _reference_csf_decode(fixed, exceptions)
            n, cols = want.shape
            assert int(want.max()) <= 0x7E  # finite, nonnegative E4M3FN
            source = nvfp4_source(store, prefix, (n, cols * 16))
            words = source.read_encoded(0, n)
            assert torch.equal(words.scales, want), prefix
            assert torch.equal(source.read_encoded(37, 301).scales, want[37:301])
            _, _, codes = _raw(REAL, prefix + ".weight")
            assert words.codes.numpy().tobytes() == codes
            _, _, multiplier = _raw(REAL, prefix + ".weight_scale_2")
            assert words.weight_divisor == struct.pack("<f", 1 / struct.unpack("<f", multiplier)[0])


@real
def test_real_mxfp8_matrix_and_ngram_rows_are_checkpoint_bytes():
    with SafetensorsSource(REAL) as store:
        prefix = f"{LM}layers.3.self_attn.o_proj"
        source = matrix_source(store, prefix + ".weight", (2560, 6144))
        words = source.read_encoded(0, 2560)
        assert words.format == MXFP8
        assert words.codes.numpy().tobytes() == _raw(REAL, prefix + ".weight")[2]
        assert words.scales.numpy().tobytes() == _raw(REAL, prefix + ".weight_scale")[2]
        model_rows = 128 * 2_500_012
        table = f"{LM}layers.1.ple.ple_embedding.ngram_embedding"
        shards = [f"{table}.shard_{i}" for i in range(128)]
        rows = nvfp4_row_table_source(store, shards, table + ".weight_scale_2", (model_rows, 160))
        begin = 2_500_012 * 5 - 500  # 1000 rows across the shard_4/shard_5 boundary
        words = rows.read_encoded(begin, begin + 1000)
        local = 2_500_012 - 500
        codes = (
            _raw(REAL, f"{table}.shard_4.weight", local * 80)[2] + _raw(REAL, f"{table}.shard_5.weight", 0, 500 * 80)[2]
        )
        scales = (
            _raw(REAL, f"{table}.shard_4.weight_scale", local * 10)[2]
            + _raw(REAL, f"{table}.shard_5.weight_scale", 0, 500 * 10)[2]
        )
        assert words.codes.numpy().tobytes() == codes and words.scales.numpy().tobytes() == scales
        assert int(words.scales.max()) <= 0x7E
