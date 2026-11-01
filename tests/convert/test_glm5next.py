from __future__ import annotations

import json
import struct

import pytest
import torch

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.fp8_block import dequantize_fp8_block_scaled
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.glm5next import build_model
from tools.convert.official_recipes import glm5next
from tools.convert.__main__ import main as convert_main
from tools.convert.proposal import add_official_proposal
from tools.convert.recipe import Recipe

from .glm5next_fixture import build_checkpoint


def parts(artifact, name):
    return binding_parts(artifact.directory.bindings[name], artifact.by_id)


def direct(artifact, name, shape):
    chunks = []
    for object_id, first, last in parts(artifact, name):
        obj = artifact.object(object_id)
        data = decode_direct(artifact.read_object(object_id), obj.format, obj.shape)
        chunks.append(data.flatten()[first:last])
    return torch.cat(chunks).reshape(shape)


def test_glm_checkpoint_layouts_and_official_recipe(tmp_path):
    torch.set_num_threads(1)
    with build_checkpoint(tmp_path / "source") as source:
        model = build_model(source, components=("text", "mtp", "vision"))
        recipe = Recipe(model)
        glm5next(model, recipe, {"base": source})
        add_official_proposal(recipe, rows=1, ranking=tmp_path / "no-ranking")
        output = tmp_path / "model.ninfer"
        convert_main(
            [
                "--model",
                str(source.path),
                "--recipe",
                "glm5next",
                "--components",
                "text,mtp,vision",
                "--proposal",
                "--ranking",
                str(tmp_path / "no-ranking"),
                "--out",
                str(output),
                "--device",
                "cpu",
                "--rows-per-chunk",
                "128",
            ]
        )
        with Artifact(output) as artifact:
            assert model.config["model_type"] == "glm5next_text"
            assert model.config["linear_gate_lowrank"] == 32
            assert model.components["text"]["proposal"] == {"domain": "full"}
            assert set(model.components["vision"]["resources"]) == {
                "processor_config.json"
            }
            assert "proposal/token_ids" not in artifact.directory.bindings
            assert (
                artifact.directory.bindings["mtp/token_embedding"]
                == artifact.directory.bindings["text/token_embedding"]
            )
            assert (
                artifact.directory.bindings["mtp/output_head"]
                == artifact.directory.bindings["text/output_head"]
            )
            # Main full precision and optional proposal are distinct payloads.
            main = parts(artifact, "text/output_head")[0][0]
            draft = parts(artifact, "proposal/head")[0][0]
            assert main != draft
            assert artifact.object(main).format == "bf16"
            assert artifact.object(draft).format == "q4_g64_fp16"
            assert artifact.object(draft).shape == (16, 128)
            assert torch.equal(
                direct(artifact, "text/output_head", (16, 128)),
                source.read_flat("lm_head.weight").reshape(16, 128),
            )

            # Each W_A prefix and the W_B query prefix are contiguous parent views.
            for prefix, source_layer in (
                ("text/layers/1/attention/", 1),
                ("mtp/layers/0/attention/", 2),
            ):
                names = ("latent", "index_key", "pool_gate", "query_a", "index_weight")
                spans = [parts(artifact, prefix + role)[0] for role in names]
                assert len({span[0] for span in spans}) == 1
                assert [(first, last) for _, first, last in spans] == [
                    (0, 16384),
                    (16384, 24576),
                    (24576, 32768),
                    (32768, 49152),
                    (49152, 49408),
                ]
                q, iq = (
                    parts(artifact, prefix + "query_b")[0],
                    parts(artifact, prefix + "index_query")[0],
                )
                assert (
                    q[0] == iq[0] and q[1:] == (0, 16384) and iq[1:] == (16384, 32768)
                )
                original = source.read_flat(
                    f"model.language_model.layers.{source_layer}.self_attn.kv_b_proj.weight"
                ).reshape(2, 128, 128)
                assert torch.equal(
                    direct(artifact, prefix + "key_up", (2, 64, 128)), original[:, :64]
                )
                assert torch.equal(
                    direct(artifact, prefix + "value_up", (2, 64, 128)),
                    original[:, 64:],
                )

            p = "text/layers/0/kda/"
            control = [
                parts(artifact, p + role)[0] for role in ("beta", "forget_a", "gate_a")
            ]
            assert len({part[0] for part in control}) == 1
            assert [part[1:] for part in control] == [
                (0, 256),
                (256, 4352),
                (4352, 8448),
            ]
            conv = torch.cat(
                [
                    source.read_flat(
                        f"model.language_model.layers.0.self_attn.{role}_conv1d.weight"
                    )
                    .reshape(128, 4)
                    .T
                    for role in ("q", "k", "v")
                ],
                1,
            )
            assert torch.equal(direct(artifact, p + "convolution", (4, 384)), conv)
            downsample = (
                source.read_flat("model.visual.downsample.weight")
                .reshape(128, 64, 2, 2)
                .permute(0, 2, 3, 1)
                .contiguous()
                .reshape(128, 256)
            )
            assert torch.equal(
                direct(artifact, "vision/downsample", (128, 256)), downsample
            )
            bias = source.read_flat("model.visual.blocks.0.attn.qkv.bias")
            for j, role in enumerate(("query", "key", "value")):
                assert torch.equal(
                    direct(
                        artifact, "vision/layers/0/attention/" + role + "_bias", (64,)
                    ),
                    bias[j * 64 : (j + 1) * 64],
                )

            # NVFP4 GU preserves each expert's code/scale words and own divisor.
            bank = parts(artifact, "text/layers/1/moe/experts/gate")[0][0]
            assert artifact.object(bank).shape == (2, 256, 128)
            codes, scales, divisors = decode_nvfp4_words(
                artifact.read_object(bank), (2, 256, 128)
            )
            assert bool((codes == 0x21).all()) and bool((scales == 0x38).all())
            assert divisors.tolist() == [1.0, 0.5]
            uses = {use["parameter"]: use for use in artifact.directory.uses}
            for role in ("gate", "up", "down"):
                name = "text/layers/1/moe/experts/" + role
                assert uses[name]["activation_policy"] == "AllowA4"
                aux = binding_parts(
                    uses[name]["auxiliaries"]["activation_input_divisor"],
                    artifact.by_id,
                )[0]
                raw = artifact.read_object(aux[0])[aux[1] * 4 : aux[2] * 4]
                assert struct.unpack("<f", raw)[0] == 0.5
                name = "mtp/layers/0/moe/experts/" + role
                assert uses[name]["activation_policy"] == "AllowA8"
                assert "auxiliaries" not in uses[name]
            for name in (
                "text/layers/0/kda/query",
                "text/layers/1/attention/output",
                "mtp/layers/0/moe/shared/gate",
            ):
                assert (
                    artifact.object(parts(artifact, name)[0][0]).format == "q8_g32_fp16"
                )
                assert uses[name]["activation_policy"] == "A16Only"
            mtp = parts(artifact, "mtp/layers/0/moe/experts/gate")[0][0]
            values = dequantize_fp8_block_scaled(
                artifact.read_object(mtp), (2, 256, 128)
            )
            expected = torch.stack(
                [
                    torch.cat(
                        [
                            source.read_flat(
                                f"model.language_model.layers.2.mlp.experts.{expert}.{role}_proj.weight"
                            ).reshape(128, 128)
                            for role in ("gate", "up")
                        ]
                    )
                    for expert in range(2)
                ]
            ).float()
            assert (
                torch.linalg.vector_norm(values - expected)
                / torch.linalg.vector_norm(expected)
                < 0.04
            )


def test_glm_optional_components_coverage_and_lazy_reads(tmp_path):
    with build_checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        assert set(model.components) == {"text"}
        assert (
            source.bytes_read == 0
        )  # Mapping only reads tensor metadata and resource files.
        assert all(name.startswith("text/") for name in model.parameters)
        config = json.loads((source.root / "config.json").read_text())
        config["text_config"]["layer_types"] = ["linear_attention"]
        source.config = config
        with pytest.raises(ValueError, match="cover every main layer"):
            build_model(source)


def test_glm_gu_preserves_source_calibration_contract(tmp_path):
    with build_checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        original = source.read_flat

        def read(name, begin=0, end=None):
            if name == "model.language_model.layers.0.mlp.up_proj.input_scale":
                return torch.tensor([4.0], dtype=torch.float32)
            return original(name, begin, end)

        source.read_flat = read
        with pytest.raises(ValueError, match="dense gate/up multipliers differ"):
            glm5next(model, Recipe(model), {"base": source})
