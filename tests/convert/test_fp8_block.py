from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.fp8_block import decode_fp8_block_scaled_words
from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.sources.fp8_block import fp8_block_source
from tools.convert.sources.logical import array_source
from tools.convert.sources.safetensors import SafetensorsSource


def test_fp8_block_source_and_streamed_gate_up_bank(tmp_path):
    tensors = {}
    for expert in range(2):
        for role, value in (("gate", 1.0), ("up", -2.0)):
            prefix = f"expert.{expert}.{role}"
            tensors[prefix + ".weight"] = torch.full((128, 256), value).to(
                torch.float8_e4m3fn
            )
            tensors[prefix + ".weight_scale_inv"] = torch.tensor(
                [[(expert + 1) / 32, (expert + 1) / (16 if role == "gate" else 8)]],
                dtype=torch.bfloat16,
            )
    source_path = tmp_path / "model.safetensors"
    save_file(tensors, str(source_path))
    model = Model({"text": {"config": {}}})
    for role in ("gate", "up"):
        source = array_source(torch.zeros((2, 128, 256)), "unused bf16 bank")
        model.add(Parameter(role, source.shape, source, inputs=("input",)))
    recipe = Recipe(model)

    def no_values(begin, end):
        raise AssertionError(
            "FP8 import must preserve encoded words without requantization"
        )

    with SafetensorsSource(source_path) as store:
        parts = []
        for expert in range(2):
            for role in ("gate", "up"):
                prefix = f"expert.{expert}.{role}"
                source = fp8_block_source(store, prefix, (128, 256))
                expected = tensors[prefix + ".weight"].float() * tensors[
                    prefix + ".weight_scale_inv"
                ].float().repeat_interleave(128, 0).repeat_interleave(128, 1)
                assert torch.equal(
                    source.values(129, 17001), expected.flatten()[129:17001]
                )
                with pytest.raises(ValueError, match="128-row"):
                    source.read_encoded(1, 128)
                parts.append(
                    (
                        role,
                        expert * 32768,
                        (expert + 1) * 32768,
                        replace(source, read_values=no_values),
                    )
                )
        recipe.parent(
            parts,
            shape=(2, 256, 256),
            format="fp8_e4m3fn_block128x128_bf16",
            method=import_encoded,
        )
        prepared = recipe.prepare(device="cpu", rows_per_chunk=129)
        path = tmp_path / "fp8.ninfer"
        with ArtifactWriter(
            path,
            [w.spec for w in prepared.weights],
            components=model.components,
            bindings=prepared.bindings,
            uses=prepared.uses,
        ) as writer:
            for job in prepared.weights:
                job.prepared.produce(TensorOutput(writer, job.spec.id))
        assert all(use["activation_policy"] == "A16Only" for use in prepared.uses)
    with Artifact(path) as artifact:
        codes, scales = decode_fp8_block_scaled_words(
            artifact.read_object(prepared.weights[0].spec.id), (2, 256, 256)
        )
    for expert in range(2):
        for row, role in enumerate(("gate", "up")):
            prefix = f"expert.{expert}.{role}"
            assert torch.equal(
                codes[expert, row * 128 : (row + 1) * 128],
                tensors[prefix + ".weight"].view(torch.uint8),
            )
            assert torch.equal(
                scales[expert, row : row + 1].view(torch.int16),
                tensors[prefix + ".weight_scale_inv"].view(torch.int16),
            )
