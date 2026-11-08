from __future__ import annotations

import struct

import numpy as np
import torch

from tools.artifact.codecs.csf32 import decode_csf32_scales
from tools.artifact.layouts import csf32_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.__main__ import main as convert_main
from tools.convert.glm5next import build_model
from tools.convert.sources.modelopt import ModelOptSource

from .glm5next_fixture import build_native_checkpoint
from .test_glm5next import direct, parts


def test_native_checkpoint_recipe_and_words(tmp_path):
    with build_native_checkpoint(tmp_path / "source") as source:
        model = build_model(source, components=("text", "mtp", "vision"))
        assert model.config["linear_num_heads"] == 2
        assert model.config["bos_token_id"] is None
        output = tmp_path / "native.ninfer"
        convert_main(
            [
                "--model",
                str(source.path),
                "--recipe",
                "glm5next_fp4_local",
                "--components",
                "text,mtp,vision",
                "--proposal",
                "--out",
                str(output),
                "--device",
                "cpu",
                "--rows-per-chunk",
                "128",
            ]
        )
        with Artifact(output) as artifact:

            def obj(name):
                return artifact.object(parts(artifact, name)[0][0])

            assert (
                artifact.directory.bindings["mtp/output_head"]
                == artifact.directory.bindings["text/output_head"]
            )
            assert obj("text/output_head").format == "mxfp8"
            assert obj("proposal/head").format == "q4_g64_fp16"
            for name, field in (
                ("text/final_norm", "model.language_model.norm.weight"),
                (
                    "text/layers/0/mlp_hc/base",
                    "model.language_model.layers.0.hc_ffn_base",
                ),
            ):
                assert obj(name).format == "fp32"
                assert torch.equal(
                    direct(artifact, name, model.parameters[name].shape),
                    source.read_flat(field).reshape(model.parameters[name].shape),
                )
            conv = "text/layers/0/kda/convolution"
            expected = torch.cat(
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
            assert obj(conv).format == "bf16"
            assert torch.equal(direct(artifact, conv, (4, 384)), expected)

            for prefix, packed, separate in (
                (
                    "text/layers/1/attention/",
                    ("latent", "query_a"),
                    ("index_key", "pool_gate", "index_weight"),
                ),
                (
                    "mtp/layers/0/attention/",
                    ("latent", "index_key", "query_a", "index_weight"),
                    ("pool_gate",),
                ),
            ):
                a = [parts(artifact, prefix + role)[0] for role in packed]
                b = [parts(artifact, prefix + role)[0] for role in separate]
                assert len({item[0] for item in a}) == len({item[0] for item in b}) == 1
                assert a[0][0] != b[0][0]
                assert a[0][1] == b[0][1] == 0
                assert all(left[2] == right[1] for left, right in zip(a, a[1:]))
                assert all(left[2] == right[1] for left, right in zip(b, b[1:]))
            assert obj("mtp/layers/0/attention/key_up").format == "mxfp8"
            assert obj("mtp/layers/0/attention/value_up").format == "mxfp8"

            bank = obj("text/layers/1/moe/experts/gate")
            assert bank.format == "nvfp4_mul" and bank.layout == "csf_32x2_v1"
            assert bank.shape == (2, 256, 128) and bank.layout_parameters == {
                "exception_count": 4
            }
            geometry = csf32_geometry(bank.shape, 4)
            payload = artifact.read_object(bank.id)
            planes = {
                name: payload[offset : offset + size]
                for name, offset, size in geometry.planes
            }
            assert set(planes["codes"]) == {0x21}
            assert struct.unpack("<2f", planes["matrix_scale"]) == (1.25, 2.25)
            assert np.frombuffer(planes["page_bases"], dtype="<u4").tolist() == [0, 4]
            scales = decode_csf32_scales(planes, bank.shape, 4)
            reference = torch.full((2, 256, 8), 0x38, dtype=torch.uint8)
            reference[:, (0, 128), 0] = 0x40
            assert torch.equal(scales, reference)
            uses = {item["parameter"]: item for item in artifact.directory.uses}
            for role, expected in (
                ("gate", (1.5, 2.5)),
                ("up", (1.5, 2.5)),
                ("down", (2.5, 3.5)),
            ):
                use = uses["text/layers/1/moe/experts/" + role]
                ref = binding_parts(
                    use["auxiliaries"]["activation_input_multiplier"], artifact.by_id
                )[0]
                assert ref[2] - ref[1] == 2
                assert (
                    struct.unpack(
                        "<2f", artifact.read_object(ref[0])[ref[1] * 4 : ref[2] * 4]
                    )
                    == expected
                )
            assert uses["text/output_head"]["activation_policy"] == "AllowA8"
            assert (
                uses["text/layers/1/attention/latent"]["activation_policy"] == "A16Only"
            )
            assert (
                uses["mtp/layers/0/moe/experts/gate"]["activation_policy"] == "A16Only"
            )
            assert uses["vision/layers/0/mlp/gate"]["activation_policy"] == "A16Only"
            assert obj("vision/layers/0/mlp/gate").format == "nvfp4_mul"


def test_native_values_preserve_source_multiplier_and_fp32_scales(tmp_path):
    with build_native_checkpoint(tmp_path / "source") as source:
        adapter = ModelOptSource(source)
        p = "model.language_model.layers.1.mlp.experts.0.gate_proj"
        values = adapter.matrix(p, (128, 128)).rows(0, 1)
        assert values[0, 0] == 0.5 * 2 * 1.25
        assert values[0, 1] == 1 * 2 * 1.25
        assert values[0, 16] == 0.5 * 1.25
        p = "model.language_model.layers.1.self_attn.q_a_proj"
        expected = source.read_flat(p + ".weight_scale")[0]
        assert adapter.matrix(p, (128, 128)).values(0, 1)[0] == expected
        assert expected != expected.bfloat16().float()
