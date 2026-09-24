from __future__ import annotations

import struct
from dataclasses import replace

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.artifact.layouts import block_scale_geometry, encoded_size
from tools.artifact.schema import binding_parts
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import grouped_absmax, import_encoded
from tools.convert.model import Model, Parameter
from tools.artifact.tensor_output import TensorOutput
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import EncodedRows, LogicalSource, array_source


def _model(names=("query", "key", "gate", "value")):
    model = Model({"text": {"config": {}}})
    for index, name in enumerate(names):
        values = (
            ((index + 1) * torch.tensor([1, 2, 4, 8], dtype=torch.bfloat16))[:, None]
            .expand(4, 128)
            .contiguous()
        )
        model.add(
            Parameter(name, (4, 128), array_source(values, name), inputs=("input",))
        )
    model.packing_groups = [
        group for group in (tuple(names), tuple(names[:2]), tuple(names[2:])) if group
    ]
    return model


def _write(path, model, prepared):
    specs = [job.spec for job in prepared.weights] + [
        spec for spec, _ in prepared.auxiliaries
    ]
    with ArtifactWriter(
        path,
        specs,
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
        for spec, data in prepared.auxiliaries:
            writer.write_object(spec.id, data)


def _assert_quantized_rows(artifact, part, amplitudes, format):
    object_id, begin, end = part
    obj = artifact.object(object_id)
    assert obj.format == format and end - begin == len(amplitudes) * 128
    scales, codes = decode_row_split_codes(
        artifact.read_object(object_id), format, obj.shape
    )
    first, last = begin // 128, end // 128
    maximum = {"q4_g64_fp16": 7, "q5_g64_fp16": 15}[format]
    assert bool((codes[first:last] == maximum).all())
    expected = [
        struct.unpack("<H", struct.pack("<e", value / maximum))[0]
        for value in amplitudes
    ]
    assert scales[first:last].view(torch.int16).tolist() == [
        [word, word] for word in expected
    ]


def test_mixed_attention_uses_two_parents_and_distinct_permissions(tmp_path):
    model = _model()
    recipe = Recipe(model)
    recipe.assign(
        ("query", "key"),
        format="q4_g64_fp16",
        method=grouped_absmax,
        activation_policy="AllowA4",
    )
    recipe.assign(
        ("gate", "value"),
        format="q5_g64_fp16",
        method=grouped_absmax,
        activation_policy="AllowA8",
    )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=3)
    path = tmp_path / "mixed.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        parts = {
            name: binding_parts(binding, artifact.by_id)[0]
            for name, binding in artifact.directory.bindings.items()
        }
        assert parts["query"][0] == parts["key"][0]
        assert parts["gate"][0] == parts["value"][0]
        assert parts["query"][0] != parts["gate"][0]
        assert parts["query"][1:] == parts["gate"][1:] == (0, 512)
        assert parts["key"][1:] == parts["value"][1:] == (512, 1024)
        for index, name in enumerate(("query", "key", "gate", "value")):
            format = "q4_g64_fp16" if index < 2 else "q5_g64_fp16"
            _assert_quantized_rows(
                artifact,
                parts[name],
                [(index + 1) * row for row in (1, 2, 4, 8)],
                format,
            )
        assert {
            use["parameter"]: use["activation_policy"]
            for use in artifact.directory.uses
        } == {
            "query": "AllowA4",
            "key": "AllowA4",
            "gate": "AllowA8",
            "value": "AllowA8",
        }


def test_row_overrides_preserve_binding_order_and_coverage(tmp_path):
    model = _model(("weight",))
    model.packing_groups = []
    recipe = Recipe(model)
    recipe.assign("weight", format="q4_g64_fp16", method=grouped_absmax)
    recipe.assign("weight", rows=(1, 3), format="q5_g64_fp16")
    prepared = recipe.prepare(device="cpu", rows_per_chunk=1)
    path = tmp_path / "parts.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        parts = binding_parts(artifact.directory.bindings["weight"], artifact.by_id)
        assert len(parts) == 3
        for part, amplitudes, format in zip(
            parts, ([1], [2, 4], [8]), ("q4_g64_fp16", "q5_g64_fp16", "q4_g64_fp16")
        ):
            _assert_quantized_rows(artifact, part, amplitudes, format)


def test_shared_weight_keeps_use_independent_and_can_be_overridden():
    model = _model(("key", "context_key"))
    model.packing_groups = []
    recipe = Recipe(model)
    recipe.assign("*", format="q8_g32_fp16", method=grouped_absmax)
    recipe.share("context_key", "key")
    recipe.use("context_key", "input", activation_policy="AllowA8")
    shared = recipe.prepare(device="cpu")
    assert len(shared.weights) == 1
    assert shared.bindings["key"] == shared.bindings["context_key"]
    assert shared.uses[0]["activation_policy"] == "A16Only"
    assert shared.uses[1]["activation_policy"] == "AllowA8"
    recipe.assign("context_key", format="q5_g64_fp16")
    independent = recipe.prepare(device="cpu")
    assert len(independent.weights) == 2
    assert independent.bindings["key"] != independent.bindings["context_key"]


def _encoded_source(name, divisor, shift=0, rows=128, activation=None):
    codes = (
        (torch.arange(rows * 32) + shift)
        .remainder(256)
        .to(torch.uint8)
        .reshape(rows, 32)
    )
    scales = (
        (torch.arange(rows * 4) + shift).remainder(127).to(torch.uint8).reshape(rows, 4)
    )
    word = struct.pack("<f", divisor)

    def no_values(begin, end):
        raise AssertionError("encoded import must not require decoded source values")

    return (
        LogicalSource(
            (rows, 64),
            name,
            no_values,
            lambda begin, end: EncodedRows(
                "nvfp4", codes[begin:end], scales[begin:end], word
            ),
            lambda: word,
            None if activation is None else (lambda: struct.pack("<f", activation)),
        ),
        codes,
        scales,
    )


def test_nvfp4_encoded_only_import_streams_complete_parent_tiles(tmp_path):
    model = Model({"text": {"config": {}}})
    first, codes, scales = _encoded_source("first", 2.0)
    second, other_codes, other_scales = _encoded_source("second", 2.0, shift=17)
    for name, source in (("gate", first), ("up", second)):
        model.add(Parameter(name, source.shape, source, inputs=("input",)))
    model.packing_groups = [("gate", "up")]
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.use(
        "gate",
        "input",
        activation_policy="AllowA4",
        auxiliaries={"activation_input_divisor": 1.5},
    )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert len(prepared.weights) == 1 and len(prepared.auxiliaries) == 1
    path = tmp_path / "nvfp4.ninfer"
    _write(path, model, prepared)
    expected = encode_nvfp4(
        torch.cat((codes, other_codes)),
        torch.cat((scales, other_scales)),
        struct.pack("<f", 2.0),
        (256, 64),
    )
    with Artifact(path) as artifact:
        assert artifact.read_object(prepared.weights[0].spec.id) == expected
        assert artifact.read_object(prepared.auxiliaries[0][0].id) == struct.pack(
            "<f", 1.5
        )


def test_differing_nvfp4_divisors_stay_apart_by_default_and_stack_when_grouped(tmp_path):
    """Sources quantised apart are not merged unasked, and keep both divisors when they are.

    Default packing leaves them as separate parents because nothing said they belong together.
    An explicit group does say so, and the parent then stores one divisor per source instead of
    rewriting either source's block scales onto the other's divisor.
    """

    model = Model({"text": {"config": {}}})
    words = []
    for name, divisor in (("gate", 2.0), ("up", 4.0)):
        source, _, _ = _encoded_source(name, divisor)
        model.add(Parameter(name, source.shape, source))
        words.append(struct.pack("<f", divisor))
    model.packing_groups = [("gate", "up")]
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    assert len(recipe.prepare(device="cpu").weights) == 2

    recipe.group(("gate", "up"))
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert len(prepared.weights) == 1
    spec = prepared.weights[0].spec
    assert spec.divisors == 2
    path = tmp_path / "stacked.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        payload = artifact.read_object(spec.id)
    geometry = block_scale_geometry("nvfp4", spec.shape, spec.divisors)
    assert payload[geometry.weight_divisor_offset :] == words[0] + words[1]


def test_sources_sharing_one_nvfp4_divisor_stay_at_one_divisor():
    """Stacking is not what makes a parent hold several divisors -- disagreement is."""

    model = Model({"text": {"config": {}}})
    for name in ("gate", "up"):
        source, _, _ = _encoded_source(name, 2.0)
        model.add(Parameter(name, source.shape, source))
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.group(("gate", "up"))
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert prepared.weights[0].spec.divisors == 1


def test_unequal_nvfp4_sources_with_different_divisors_are_refused_at_plan_time():
    """A divisor per source addresses an equal share of the rows, so the sources must be equal.

    The refusal belongs where the parent's shape is chosen: every method has already run by the
    time the writer would notice.
    """

    model = Model({"text": {"config": {}}})
    for name, divisor, rows in (("gate", 2.0, 256), ("up", 4.0, 128)):
        source, _, _ = _encoded_source(name, divisor, rows=rows)
        model.add(Parameter(name, source.shape, source))
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.group(("gate", "up"))
    with pytest.raises(ValueError, match="equal row counts"):
        recipe.prepare(device="cpu")


def test_a_divisor_may_not_cover_part_of_a_scale_tile():
    """The scale plane is addressed in whole 128-row tiles, so a source shorter than one is not
    representable however its rows divide the parent."""

    with pytest.raises(ValueError, match="128-row scale tiles"):
        block_scale_geometry("nvfp4", (256, 64), 4)


def test_only_the_block_scale_layout_stores_several_divisors():
    """Other layouts have nowhere to put them, and the engine refuses such an object outright."""

    with pytest.raises(ValueError, match="stores one divisor"):
        encoded_size("contiguous_le_v1", "bf16", (4, 128), 3)


def test_alias_does_not_inherit_another_uses_calibration():
    model = Model({"text": {"config": {}}})
    source, _, _ = _encoded_source("key", 2.0)
    for name in ("key", "context_key"):
        model.add(Parameter(name, source.shape, source, inputs=(name + "/input",)))
    recipe = Recipe(model)
    recipe.share("context_key", "key")
    recipe.assign("key", format="nvfp4", method=import_encoded)
    recipe.use("context_key", "context_key/input", activation_policy="AllowA4")
    with pytest.raises(ValueError, match="independent activation divisor"):
        recipe.prepare(device="cpu")
    with pytest.raises(ValueError, match="positive finite"):
        recipe.use(
            "context_key",
            "context_key/input",
            auxiliaries={"activation_input_divisor": 0.0},
        )
    recipe.use(
        "context_key",
        "context_key/input",
        auxiliaries={"activation_input_divisor": 3.0},
    )
    prepared = recipe.prepare(device="cpu")
    assert len(prepared.weights) == 1
    assert prepared.bindings["key"] == prepared.bindings["context_key"]
    assert prepared.auxiliaries[0][1] == struct.pack("<f", 3.0)


def test_private_component_storage_cannot_be_packed_with_target_weights():
    model = _model(("target", "draft"))
    model.packing_groups = []
    model.parameters["draft"] = replace(model.parameters["draft"], residency="dflash2")
    recipe = Recipe(model)
    recipe.group(("target", "draft"))
    with pytest.raises(ValueError, match="independently selected"):
        recipe.prepare(device="cpu")
    shared = Recipe(model)
    shared.share("draft", "target")
    prepared = shared.prepare(device="cpu")
    assert len(prepared.weights) == 1
    assert prepared.bindings["draft"] == prepared.bindings["target"]


def test_allow_a4_without_an_override_takes_the_source_activation_divisor():
    """An AllowA4 input with no override has to carry the source's own calibrated word.

    `prepare_sparse_moe_weights` refuses an AllowA4 input that has no activation divisor, so a
    conversion that emits none produces an artifact that fails at bind rather than at build.
    """

    model = Model({"text": {"config": {}}})
    source, _, _ = _encoded_source("gate", 2.0, activation=3.0)
    model.add(Parameter("gate", source.shape, source, inputs=("input",)))
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.use("gate", "input", activation_policy="AllowA4")
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert len(prepared.auxiliaries) == 1
    assert prepared.auxiliaries[0][1] == struct.pack("<f", 3.0)


def test_a_stacked_parent_takes_the_smallest_source_activation_divisor():
    """One plane, one quantised activation, so exactly one of the sources' words can survive.

    It cancels in the GEMM's alpha, so the choice only decides where a block scale lands on the
    e4m3 grid; the smallest is the one direction that cannot saturate another source's blocks
    upward.
    """

    model = Model({"text": {"config": {}}})
    # The smallest is neither the first nor the last source, so a lookup that takes either is
    # caught rather than agreeing with the answer by accident.
    for name, divisor, activation in (("gate", 4.0, 7.0), ("up", 2.0, 5.0), ("extra", 3.0, 9.0)):
        source, _, _ = _encoded_source(name, divisor, activation=activation)
        model.add(Parameter(name, source.shape, source, inputs=("input",)))
    model.packing_groups = [("gate", "up", "extra")]
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.group(("gate", "up", "extra"))
    recipe.use("gate", "input", activation_policy="AllowA4")
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert prepared.weights[0].spec.divisors == 3
    assert len(prepared.auxiliaries) == 1
    assert prepared.auxiliaries[0][1] == struct.pack("<f", 5.0)


def test_one_weight_divisor_still_shares_one_activation_divisor():
    """Agreeing on the weight divisor does not mean agreeing on the activation one.

    The two are calibrated apart, so a parent can collapse to a single weight divisor and still
    hold sources with different activation words. Both words reaching the artifact has the bank
    refused at bind, so the plane has to settle on one here.
    """

    model = Model({"text": {"config": {}}})
    for name, activation in (("gate", 7.0), ("up", 5.0)):
        source, _, _ = _encoded_source(name, 2.0, activation=activation)
        model.add(Parameter(name, source.shape, source, inputs=("input",)))
    model.packing_groups = [("gate", "up")]
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.group(("gate", "up"))
    recipe.use("gate", "input", activation_policy="AllowA4")
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert prepared.weights[0].spec.divisors == 1
    assert len(prepared.auxiliaries) == 1
    assert prepared.auxiliaries[0][1] == struct.pack("<f", 5.0)

