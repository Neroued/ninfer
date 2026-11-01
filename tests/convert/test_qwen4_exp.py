"""Qwen4Exp: config mapping, the Flash-Next official recipes and qwen4_bf16."""

from __future__ import annotations

import struct

import pytest
import torch

from tests.convert import qwen4_exp_fixture as fixture
from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales
from tools.artifact.layouts import block_scale_geometry
from tools.artifact.reader import Artifact
from tools.convert.official_recipes import (
    qwen4_bf16,
    qwen3_8_flash_next_nvfp4,
    qwen3_8_flash_next_nvfp4_dense,
    qwen3_8_flash_next_nvfp4_mix,
    qwen3_8_flash_next_nvfp4_gdn,
)
from tools.convert.pipeline import convert
from tools.convert.qwen4_exp import build_model, ngram_table_rows, text_config
from tools.convert.recipe import Recipe
from tools.convert.sources.modelopt import decode_nvfp4
from tools.convert.sources.safetensors import SafetensorsSource

H, IR, E = fixture.H, fixture.IR, fixture.E


def test_config_keeps_qwen4_fields_and_checks_fixed_mathematics():
    config = text_config(fixture.config(), mtp=True)
    assert config["architectures"] == ["Qwen4ExpForCausalLM"]
    assert config["model_type"] == "qwen4_exp_text"
    assert config["hc_count"] == fixture.HC and config["ple_layer_ids"] == [1]
    assert config["output_gate_type"] == "sigmoid" and config["eos_token_id"] == fixture.EOS
    bad = fixture.config()
    bad["text_config"]["ple_layer_ids"] = [2]  # one-indexed: block 1 is attention
    with pytest.raises(ValueError, match="GDN block"):
        text_config(bad, mtp=False)
    bad = fixture.config()
    bad["text_config"]["norm_topk_prob"] = False
    with pytest.raises(ValueError, match="norm_topk_prob"):
        text_config(bad, mtp=False)


@pytest.mark.parametrize("ple", [True, False])
def test_dense_config_without_moe_fields_and_optional_ple(ple):
    config = text_config(fixture.config(dense=True, ple=ple), mtp=True)
    assert (config["architectures"][0], config["model_type"]) == ("Qwen4ExpForCausalLM", "qwen4_exp_text")
    assert config["intermediate_size"] == fixture.DENSE_IR
    assert not {"num_experts", "moe_intermediate_size", "shared_expert_intermediate_size"} & set(config)
    assert config["output_gate_type"] == "sigmoid" and config["hc_count"] == fixture.HC
    ple_fields = {"ple_layer_ids", "ple_embed_dim", "ngram_size", "seed", "eos_token_id"}
    assert (ple_fields <= set(config)) if ple else not (ple_fields & set(config))


def test_dense_config_needs_an_ffn_width():
    bad = fixture.config(dense=True)
    del bad["text_config"]["intermediate_size"]
    with pytest.raises(ValueError, match="intermediate_size"):
        text_config(bad, mtp=False)


@pytest.mark.parametrize("ple", [True, False])
def test_qwen4_bf16_recipe_converts_a_dense_checkpoint(tmp_path, ple):
    source_dir = tmp_path / "source"
    fixture.write_checkpoint(source_dir, dense=True, ple=ple)
    with SafetensorsSource(source_dir) as source:
        model = build_model(source, components=("text", "mtp"))
        recipe = Recipe(model)
        qwen4_bf16(model, recipe, {"base": source})
        out = tmp_path / "dense.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with SafetensorsSource(source_dir) as store, Artifact(out) as artifact:
        directory = artifact.directory
        text = directory.components["text"]["config"]
        assert text["intermediate_size"] == fixture.DENSE_IR and "num_experts" not in text
        mtp = directory.components["mtp"]["config"]["architectures"]
        assert mtp == ["Qwen4ExpMTP"]
        assert ("text/layers/0/ple/table" in directory.bindings) == ple
        assert not any("/moe/" in name for name in directory.bindings)
        for prefix, source_prefix in (
            ("text/layers/1/", "model.language_model.layers.1."),
            ("mtp/layers/0/", "mtp.layers.0."),
        ):
            (gate,) = _parts(directory, prefix + "mlp/gate")
            (up,) = _parts(directory, prefix + "mlp/up")
            (down,) = _parts(directory, prefix + "mlp/down")
            # gate|up one BF16 parent (gate rows first), down its own; the values are the source's.
            assert gate["object"] == up["object"] and gate["range"][1] == up["range"][0]
            parent = artifact.object(gate["object"])
            assert parent.format == "bf16" and parent.shape == (2 * fixture.DENSE_IR, fixture.H)
            raw = artifact.read_object(gate["object"])
            want = torch.cat([store.read_flat(source_prefix + f"mlp.{r}_proj.weight") for r in ("gate", "up")])
            assert raw[: parent.shape[0] * parent.shape[1] * 2] == want.view(torch.uint8).numpy().tobytes()
            assert artifact.object(down["object"]).format == "bf16"
        if ple:
            (part,) = _parts(directory, "text/layers/0/ple/table")
            assert artifact.object(part["object"]).format == "fp8_e4m3fn_row_bf16"


def test_ngram_table_rows_match_official_shards_and_the_independent_constants():
    official = dict(
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
    )
    assert ngram_table_rows(official) == 128 * 2_500_012
    text = fixture.config()["text_config"]
    assert ngram_table_rows(text) == fixture.ple_constants(text)["rows"]


def _convert(tmp_path, recipe_function, name):
    source_dir = tmp_path / "source"
    if not source_dir.exists():
        fixture.write_checkpoint(source_dir)
    with SafetensorsSource(source_dir) as source:
        model = build_model(source, components=("text", "mtp"))
        recipe = Recipe(model)
        recipe_function(model, recipe, {"base": source})
        out = tmp_path / f"{name}.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    return out, SafetensorsSource(source_dir)


def _parts(directory, name):
    binding = directory.bindings[name]
    return binding["parts"] if "parts" in binding else [{"object": binding["object"]}]


@pytest.mark.parametrize("dense", [False, True, "mix", "gdn"])
def test_flash_next_recipes_import_experts_and_table_exactly(tmp_path, dense):
    recipe = {
        False: qwen3_8_flash_next_nvfp4,
        True: qwen3_8_flash_next_nvfp4_dense,
        "mix": qwen3_8_flash_next_nvfp4_mix,
        "gdn": qwen3_8_flash_next_nvfp4_gdn,
    }[dense]
    out, store = _convert(tmp_path, recipe, {False: "q8", True: "dense", "mix": "mix", "gdn": "gdn"}[dense])
    with store, Artifact(out) as artifact:
        directory = artifact.directory
        text = directory.components["text"]["config"]
        assert text["model_type"] == "qwen4_exp_text" and text["hc_count"] == fixture.HC
        assert directory.components["mtp"]["config"]["architectures"] == ["Qwen4ExpMTP"]
        uses = {(u["parameter"], u["input"]): u for u in directory.uses}
        for layer in range(len(fixture.LAYER_TYPES)):
            lp = f"text/layers/{layer}/moe/experts/"
            (gate_part,) = _parts(directory, lp + "0/gate")
            (down_part,) = _parts(directory, lp + "0/down")
            gate_up = artifact.object(gate_part["object"])
            down = artifact.object(down_part["object"])
            # One stacked parent per bank and layer, one divisor per source matrix.
            assert gate_up.format == "nvfp4" and gate_up.shape == (2 * E * IR, H)
            assert down.format == "nvfp4" and down.shape == (E * H, IR)
            assert gate_up.divisors == 2 * E and down.divisors == E
            banks = {}
            for bank in (gate_up, down):
                raw = artifact.read_object(bank.id)
                banks[bank.id] = (raw, block_scale_geometry("nvfp4", bank.shape, bank.divisors))
            for e in range(E):
                for index, role in enumerate(("gate", "up", "down")):
                    (part,) = _parts(directory, f"{lp}{e}/{role}")
                    bank = down if role == "down" else gate_up
                    assert part["object"] == bank.id
                    rows = H if role == "down" else IR
                    k = bank.shape[1]
                    row = e * H if role == "down" else (2 * e + index) * IR
                    assert part["range"] == [row * k, (row + rows) * k]
                    raw, g = banks[bank.id]
                    prefix = f"model.language_model.layers.{layer}.mlp.experts.{e}.{role}_proj"
                    # Codes: a contiguous row range of the code plane, bit-exact.
                    codes = raw[row * k // 2 : (row + rows) * k // 2]
                    assert codes == store.read_flat(prefix + ".weight").numpy().tobytes()
                    # Scales: rows of whole 128-row tiles are a contiguous range of the
                    # swizzled plane, and unswizzle to the checkpoint's natural scales.
                    first = g.scale_plane_offset + row * k // 16
                    span = raw[first : first + rows * k // 16]
                    natural = unswizzle_nvfp4_scales(torch.frombuffer(bytearray(span), dtype=torch.uint8), (rows, k))
                    want = store.read_flat(prefix + ".weight_scale").view(torch.uint8)
                    assert torch.equal(natural.reshape(-1), want.reshape(-1))
                    # Divisor: the source's own word, at the source's index.
                    source_index = row // rows
                    word = raw[g.weight_divisor_offset + 4 * source_index :][:4]
                    assert word == fixture.divisor_word(store.read_flat(prefix + ".weight_scale_2"))
                    # Activation divisor of the source's own Use.
                    use = uses[
                        (
                            f"{lp}{e}/{role}",
                            f"text/layers/{layer}/" + (f"moe/experts/{e}/product" if role == "down" else "ffn_input"),
                        )
                    ]
                    assert use["activation_policy"] == "AllowA4"
                    assert "activation_input_divisor" in use.get("auxiliaries", {})

        # The FP8 n-gram table keeps its codes and repeats the shared multiplier per row.
        (part,) = _parts(directory, "text/layers/0/ple/table")
        obj = artifact.object(part["object"])
        rows = fixture.ple_constants(text)["rows"]
        assert obj.format == "fp8_e4m3fn_row_bf16" and obj.shape == (rows, 8)
        raw = artifact.read_object(obj.id)
        table = torch.cat(
            [
                store.read_flat(
                    f"model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
                ).view(torch.uint8)
                for s in (0, 1)
            ]
        )
        assert raw[: rows * 8] == table.numpy().tobytes()
        scale = store.read_flat("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale").view(
            torch.uint8
        )
        scale_offset = (rows * 8 + 255) // 256 * 256
        assert raw[scale_offset : scale_offset + 2 * rows] == scale.numpy().tobytes() * rows

        # MTP experts: re-encoded from block FP8, one divisor per expert matrix, A16 activations.
        (part,) = _parts(directory, "mtp/layers/0/moe/experts/1/gate")
        bank = artifact.object(part["object"])
        assert bank.format == "nvfp4" and bank.divisors == 2 * E
        assert uses[("mtp/layers/0/moe/experts/1/gate", "mtp/layers/0/ffn_input")]["activation_policy"] == "A16Only"
        raw = artifact.read_object(bank.id)
        g = block_scale_geometry("nvfp4", bank.shape, bank.divisors)
        row = 2 * IR  # expert 1's gate
        codes = torch.frombuffer(bytearray(raw[row * H // 2 : (row + IR) * H // 2]), dtype=torch.uint8)
        first = g.scale_plane_offset + row * H // 16
        scales = unswizzle_nvfp4_scales(
            torch.frombuffer(bytearray(raw[first : first + IR * H // 16]), dtype=torch.uint8), (IR, H)
        )
        divisor = raw[g.weight_divisor_offset + 8 : g.weight_divisor_offset + 12]
        decoded = decode_nvfp4(codes.reshape(IR, H // 2), scales.reshape(IR, H // 16), divisor)
        prefix = "mtp.layers.0.mlp.experts.1.gate_proj"
        weight = store.read_flat(prefix + ".weight").view(torch.float8_e4m3fn).float().reshape(IR, H)
        reference = weight * store.read_flat(prefix + ".weight_scale_inv").float()[0]
        # nvfp4_mse maps a block's maximum to a target t in {6, 5.5, 5, 4.5, 4} code units, so
        # the step is s = max/t and the worst rounding error is half the widest E2M1 gap reached
        # (gap 2 above 4: 1 code unit when t >= 5, 0.5 otherwise), largest at t = 5: max/5. The
        # E4M3 block scale adds at most 2^-4 relative; the bound doubles that margin.
        block_max = reference.reshape(IR, H // 16, 16).abs().amax(-1).repeat_interleave(16, dim=1)
        assert bool(((decoded - reference).abs() <= block_max / 5 * (1 + 2**-3) + 1e-12).all())

        # Dense projections: Q8 everywhere, or NVFP4 on the selected projections and the head.
        def fmt(name):
            (part,) = _parts(directory, name)
            return artifact.object(part["object"]).format

        def expected(name):  # the mix keeps the shared expert Q8, the gdn recipe all but GDN
            if dense == "mix" and "/moe/shared/" in name:
                return "q8_g32_fp16"
            if dense == "gdn" and "/gdn/" not in name:
                return "q8_g32_fp16"
            return "nvfp4" if dense else "q8_g32_fp16"

        assert fmt("text/output_head") == ("nvfp4" if dense in (True, "mix") else "q6_g64_fp16")
        assert fmt("text/token_embedding") == "q8_g32_fp16"
        for name in (
            "text/layers/0/gdn/query",
            "text/layers/0/gdn/z",
            "text/layers/0/gdn/output",
            "text/layers/1/attention/query",
            "text/layers/1/attention/gate",
            "text/layers/1/attention/output",
            "text/layers/1/moe/shared/gate",
            "text/layers/1/moe/shared/down",
            "mtp/layers/0/attention/value",
        ):
            assert fmt(name) == expected(name), name
        for name in (
            "text/layers/0/gdn/a_projection",
            "text/layers/1/attention/indexer_query",
            "text/layers/1/attn_hc/down",
            "text/layers/0/ple/key",
            "mtp/hidden_projection",
        ):
            assert fmt(name) == "q8_g32_fp16", name
        for name in ("text/layers/1/moe/router", "text/layers/1/moe/shared_score"):
            assert fmt(name) == "bf16", name


def test_stacked_quantization_keeps_one_divisor_per_source():
    from tools.convert.methods import MethodInput, PrepareRequest, nvfp4_mse, nvfp4_mse_stacked
    from tools.artifact.schema import TensorSpec
    from tools.convert.sources.logical import LogicalSource

    g = torch.Generator().manual_seed(3)
    values = [torch.randn(128, 64, generator=g) * scale for scale in (1.0, 100.0)]

    def source(v):
        return LogicalSource(tuple(v.shape), "test", lambda b, e, v=v: v.reshape(-1)[b:e])

    inputs = tuple(MethodInput(f"p{i}", source(v), ()) for i, v in enumerate(values))
    with pytest.raises(ValueError, match="one divisor per complete source"):
        nvfp4_mse_stacked(PrepareRequest(TensorSpec("w", (256, 64), "nvfp4", "x", 1), inputs, {}, {}))
    with pytest.raises(ValueError, match="one divisor per parent"):
        nvfp4_mse(PrepareRequest(TensorSpec("w", (256, 64), "nvfp4", "x", 2), inputs, {}, {}))
    request = PrepareRequest(TensorSpec("w", (256, 64), "nvfp4", "x", 2), inputs, {}, {}, device="cpu")
    written = []

    class Output:
        def write_codes(self, row, codes, scales, divisor):
            written.append((row, codes, scales, divisor))

    nvfp4_mse_stacked(request).produce(Output())
    divisors = {row: struct.unpack("<f", divisor)[0] for row, _, _, divisor in written}
    assert set(divisors) == {0, 128}
    # The large source would underflow the small one's E4M3 scales under one shared divisor.
    assert divisors[0] == pytest.approx(divisors[128] * 100, rel=0.05)
