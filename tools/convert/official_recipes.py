"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    IMPORT_CHUNK_BYTES,
    PreparedMethod,
    _preflight,
    PrepareRequest,
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    nvfp4_mse,
    nvfp4_mse_stacked,
)

Q3 = "q3_g128_fp16"
Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    if "dflash2" in model.components:
        layers = model.components["dflash2"]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"dflash2/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dflash2_nvfp4_gate_up(model, recipe):
    """Store each DFlash2 layer's MLP gate/up parent as NVFP4 with 16-bit activations."""
    if "dflash2" not in model.components:
        return
    layers = model.components["dflash2"]["config"]["num_hidden_layers"]
    for layer in range(layers):
        recipe.assign(
            [f"dflash2/layers/{layer}/mlp/gate", f"dflash2/layers/{layer}/mlp/up"],
            format="nvfp4",
            method=nvfp4_mse,
            activation_policy="A16Only",
        )


def _mtp_nvfp4_layer(model, recipe):
    """Store the MTP layer's MLP gate/up, MLP down and attention output as NVFP4 with 16-bit activations.

    Quantized from the source's MTP weights by nvfp4_mse. At tp 2 their halves ([17408,5120],
    [5120,8704], [5120,3072]) run the NVFP4 linear routes the target's MLP and attention output use.
    The packed attention input projection (Q|K|Gate|V) and mtp/input_projection stay Q8: the MTP
    layer runs its attention input as a plain linear, and NVFP4 has no [7168,5120] linear half."""
    if "mtp" not in model.components:
        return
    for names in (
        ["mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up"],
        "mtp/layers/0/mlp/down",
        "mtp/layers/0/attention/output",
    ):
        recipe.assign(names, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only")


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


# Dense projections the NVFP4-dense Flash-Next recipe stores as NVFP4: the attention and GDN
# input/output projections, the shared expert and the output head. Their tensor-parallel halves are
# the per-rank GEMV shapes measured for F0-B ([6656,2560], [8192,2560], [2560,3072], [640,2560],
# [124160,2560]). Hyper-connection mixers, the QSA indexer, the PLE projections and the MTP stem
# projections stay Q8: their format is not decided, and the hyper-connection down projection
# ([320|4, 10240]) cannot be NVFP4 at all (N % 128 != 0).
_FLASH_NEXT_NVFP4_GROUPS = {
    "attention": (
        "/attention/query",
        "/attention/key",
        "/attention/gate",
        "/attention/value",
        "/attention/output",
    ),
    "gdn": ("/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z", "/gdn/output"),
    "shared": ("/moe/shared/gate", "/moe/shared/up", "/moe/shared/down"),
}
_FLASH_NEXT_NVFP4_DENSE = tuple(s for group in _FLASH_NEXT_NVFP4_GROUPS.values() for s in group)

# T103: dense projections a Flash-Next recipe may keep in BF16 (as the checkpoint, GB300 and Strata):
# the QSA indexer q|k, every hyper-connection mixer (block attn_hc/ffn_hc and the text/MTP hc_head,
# Down|inject and Up) and the GDN q|k|v|z|out projections. The engine runs them through the generic
# BF16 linear (ops/linear/bf16/bf16_generic.cu) and the hyper-connection general route.
_FLASH_NEXT_BF16_GROUPS = {
    "indexer": ("/attention/indexer_query", "/attention/indexer_key"),
    "hc": tuple(f"{mixer}/{role}" for mixer in ("attn_hc", "ffn_hc", "hc_head") for role in ("down", "inject", "up")),
    "gdn": _FLASH_NEXT_NVFP4_GROUPS["gdn"],
}


def _flash_next(model, recipe, *, nvfp4_dense: bool, nvfp4_groups=None, nvfp4_head=None, bf16_groups=()):
    """Qwen3.8-Flash-Next (Qwen4Exp) from the ModelOpt NVFP4 checkpoint.

    Routed Text experts keep their ModelOpt NVFP4 codes, block scales, weight divisors and
    activation divisors (``import_encoded``, AllowA4). Each layer's routed gate/up and down
    projections become one parent each, a stack with one divisor per source matrix, so an expert
    is a fixed row range of its bank: in the code plane, in the swizzled scale plane (each source
    covers whole 128-row tiles) and one FP32 divisor word per projection. The MTP layer's block-FP8
    experts are re-encoded to NVFP4 by nvfp4_mse_stacked, so they have the same per-matrix divisor
    layout (A16Only: the checkpoint has no activation calibration for them; ninfer-ext instead
    stores one absmax divisor for the whole MTP bank). The n-gram table keeps its FP8 codes under the shared
    multiplier. Routers, shared-expert gates, norms and small vectors stay direct; the token
    embedding is Q8.

    Dense projections are Q8 with a Q6 output head, exactly as ninfer-ext's recipe of the same
    name, unless ``nvfp4_dense``: then the projections in _FLASH_NEXT_NVFP4_DENSE and the output
    head are NVFP4 (nvfp4_mse, A16Only) and the GDN a/b projections stay Q8 in their own parent.
    ``nvfp4_groups`` (names of _FLASH_NEXT_NVFP4_GROUPS) and ``nvfp4_head`` select a subset instead
    (mixed recipes); the selection applies to the Text and MTP layers alike. ``bf16_groups``
    (names of _FLASH_NEXT_BF16_GROUPS, T103) keep those projections in BF16 (cast_direct), Text
    and MTP alike; they take precedence over ``nvfp4_groups``.
    """
    if nvfp4_groups is None:
        nvfp4_groups = tuple(_FLASH_NEXT_NVFP4_GROUPS) if nvfp4_dense else ()
    if nvfp4_head is None:
        nvfp4_head = nvfp4_dense
    nvfp4_suffixes = tuple(s for group in nvfp4_groups for s in _FLASH_NEXT_NVFP4_GROUPS[group])
    bf16_suffixes = tuple(s for group in bf16_groups for s in _FLASH_NEXT_BF16_GROUPS[group])
    if model.config.get("model_type") != "qwen4_exp_text" or "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen4Exp MoE mathematics")
    # The MTP choices of _optional; the loop below restates every Text and MTP choice.
    _optional(model, recipe)
    # The Vision tower Q8 (v0.6): the 27B's mixed Q4/Q5/Q6 tower departs from the BF16 reference
    # by 0.32 (relative, merged embeddings), Q8 by 0.09, as much as a BF16 reference run departs
    # from FP32; it costs one expert slot per layer more than Q4/Q5 at tp 2.
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") and parameter.projection:
            _assign(recipe, name, Q8)
    _assign(recipe, "text/token_embedding", Q8)
    if nvfp4_head:
        recipe.assign("text/output_head", format="nvfp4", method=nvfp4_mse, activation_policy="A16Only")
    else:
        _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") or not (parameter.projection or name.endswith("/ple/table")):
            continue
        if name.endswith("/ple/table"):
            recipe.assign(name, format=FP8, method=import_encoded, source=parameter.source)
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")) or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if "/moe/experts/" in name:
            if name.startswith("text/"):
                recipe.assign(
                    name,
                    format="nvfp4",
                    method=import_encoded,
                    source=parameter.source,
                    activation_policy="AllowA4",
                )
            else:
                recipe.assign(name, format="nvfp4", method=nvfp4_mse_stacked, activation_policy="A16Only")
            continue
        if bf16_suffixes and name.endswith(bf16_suffixes):
            _assign(recipe, name, "bf16")
            continue
        if nvfp4_suffixes and name.endswith(nvfp4_suffixes):
            recipe.assign(name, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only")
            continue
        _assign(recipe, name, Q8)
    # Each layer's gate/up and down banks become one parent so an expert is a fixed row range of
    # its bank, addressed by the expert cache as a few contiguous spans.
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """Flash-Next, NVFP4 experts, Q8 dense projections, Q6 head (ninfer-ext's choices)."""
    _flash_next(model, recipe, nvfp4_dense=False)


def qwen3_8_flash_next_nvfp4_dense(model, recipe, sources):
    """Flash-Next, NVFP4 experts, NVFP4 attention/GDN/shared-expert projections and head."""
    _flash_next(model, recipe, nvfp4_dense=True)


def qwen3_8_flash_next_nvfp4_mix(model, recipe, sources):
    """Flash-Next, NVFP4 experts, NVFP4 attention/GDN projections and head, Q8 shared expert (T26b)."""
    _flash_next(model, recipe, nvfp4_dense=False, nvfp4_groups=("attention", "gdn"), nvfp4_head=True)


def qwen3_8_flash_next_nvfp4_gdn(model, recipe, sources):
    """Flash-Next, NVFP4 experts, NVFP4 GDN projections (q/k/v/z/out), everything else as the Q8 recipe (T26c)."""
    _flash_next(model, recipe, nvfp4_dense=False, nvfp4_groups=("gdn",), nvfp4_head=False)


def _flash_next_g_bf16(*groups, gdn_q8=False):
    """G (NVFP4 GDN, Q8 dense, Q6 head) with the T103 BF16 groups; BF16 GDN replaces NVFP4 GDN,
    and ``gdn_q8`` keeps the GDN projections Q8 (as the rest of the dense) instead of NVFP4."""

    def recipe_function(model, recipe, sources):
        _flash_next(
            model,
            recipe,
            nvfp4_dense=False,
            nvfp4_groups=() if ("gdn" in groups or gdn_q8) else ("gdn",),
            nvfp4_head=False,
            bf16_groups=groups,
        )

    gdn = " and Q8 GDN" if gdn_q8 else ""
    recipe_function.__doc__ = f"Flash-Next G with {', '.join(groups)} projections in BF16{gdn} (T103)."
    return recipe_function


qwen3_8_flash_next_nvfp4_gdn_idxbf16 = _flash_next_g_bf16("indexer")
qwen3_8_flash_next_nvfp4_gdn_hcbf16 = _flash_next_g_bf16("hc")
qwen3_8_flash_next_nvfp4_gdnbf16 = _flash_next_g_bf16("gdn")
qwen3_8_flash_next_nvfp4_gdn_idxhcbf16 = _flash_next_g_bf16("indexer", "hc")
qwen3_8_flash_next_nvfp4_allbf16 = _flash_next_g_bf16("indexer", "hc", "gdn")
# Valerio's T103 candidate: indexer and hyper-connections BF16, GDN Q8, everything else as G.
qwen3_8_flash_next_nvfp4_idxhcbf16_gdnq8 = _flash_next_g_bf16("indexer", "hc", gdn_q8=True)
# T104 (Valerio): indexer BF16 + GDN Q8, hyper-connections Q8 as G (cheapest candidate).
qwen3_8_flash_next_nvfp4_idxbf16_gdnq8 = _flash_next_g_bf16("indexer", gdn_q8=True)


def qwen4_bf16(model, recipe, sources):
    """A dense Qwen4 checkpoint (Qwen4Exp mathematics, dense FFN, with or without PLE) in BF16.

    Every matrix and vector keeps its BF16 values (cast_direct; the BF16 linear accepts any shape,
    ops/linear/bf16/bf16_dispatch.cpp), the fused groups (q|k|gate|v, q|k|v|z, a|b, indexer q|k,
    hyper-connection down|inject, PLE key|value, MLP gate|up) become one parent each. The n-gram
    table, which the engine reads row-scaled FP8 from the file, keeps its FP8 codes when the
    checkpoint stores them (import_encoded) and is quantized by fp8_row_maxabs otherwise. Routed
    experts are not covered: their banks have NVFP4 routes only (qwen3_8_flash_next_* recipes).
    Meant for correctness work on new geometries before their quantized routes exist."""
    if model.config.get("model_type") != "qwen4_exp_text":
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    if "num_experts" in model.config:
        raise ValueError("the qwen4_* recipes convert dense Qwen4 FFNs; routed experts need an NVFP4 recipe")
    for name, parameter in model.parameters.items():
        if name.endswith("/ple/table"):
            if _encoded_format(parameter.source) == FP8:
                recipe.assign(name, format=FP8, method=import_encoded, source=parameter.source)
            else:
                recipe.assign(name, format=FP8, method=fp8_row_maxabs)
        elif parameter.projection:
            _assign(recipe, name, "bf16")


def qwen4_bf16_hcq8(model, recipe, sources):
    """qwen4_bf16 with every hyper-connection mixer (Down|inject and Up) in Q8 by grouped max-abs,
    the format of the production Flash-Next mixers: a synthetic checkpoint converted with it runs the
    Q8 small-T route of the hyper-connection mixer end to end (at hidden 2560, and at 5120 where
    only Q8 has that route), comparable with the general route on the same weights
    (NINFER_QWEN4_HC_GENERAL=1)."""
    qwen4_bf16(model, recipe, sources)
    hc = _FLASH_NEXT_BF16_GROUPS["hc"]
    for name, parameter in model.parameters.items():
        if parameter.projection and name.endswith(hc):
            _assign(recipe, name, Q8)


# The dense Qwen4 projections qwen4_nvfp4 stores as NVFP4: the Qwen3.8-27B NVFP4 projection set
# (attention q|k|gate|v and output, GDN q|k|v|z and output, MLP gate|up and down).
_QWEN4_NVFP4_DENSE = (
    _FLASH_NEXT_NVFP4_GROUPS["attention"] + _FLASH_NEXT_NVFP4_GROUPS["gdn"] + ("/mlp/gate", "/mlp/up", "/mlp/down")
)


def qwen4_nvfp4(model, recipe, sources):
    """A dense Qwen4 checkpoint with the Qwen3.8-27B NVFP4 projection set, for the fused routes.

    As qwen4_bf16, except the attention, GDN and MLP projections of the Text and MTP layers:
    NVFP4 by nvfp4_mse, A16Only (a checkpoint without activation calibration has no A4 divisors).
    The GDN a|b projection, hyper-connection mixers, QSA indexer, PLE projections, embedding and
    output head stay BF16. At hidden 5120 these are the parents
    the Text forward's fused Ops register (docs/maintainer/qwen4-exp-model.md); a synthetic checkpoint
    converted with it exercises them against the generic chain (NINFER_QWEN4_FUSED=0)."""
    qwen4_bf16(model, recipe, sources)
    for name, parameter in model.parameters.items():
        if parameter.projection and name.endswith(_QWEN4_NVFP4_DENSE):
            recipe.assign(name, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only")


MXFP8 = "mxfp8_e4m3fn_g32_ue8m0"
NVFP4_ROW = "nvfp4_row_e4m3_g16"
_VERBATIM_LAYOUTS = {MXFP8: "block_scale_k32_v1", NVFP4_ROW: "row_interleaved_v1"}
_VERBATIM_CHUNK_BYTES = 64 * 1024 * 1024


def _verbatim_row_bytes(format: str, k: int) -> int:
    return k + k // 32 if format == MXFP8 else k // 2 + k // 16


def import_verbatim(request: PrepareRequest) -> PreparedMethod:
    """Copy MXFP8 or row-NVFP4 checkpoint codes and scales into the same artifact format.

    Every input must provide encoded rows in the target format; the parent is their row
    concatenation (a fused q|k|v or gate|up group is the concatenation of both planes along N).
    Nothing is computed: the stored words are the checkpoint words. The activations stay 16-bit.
    """
    format, shape = request.target.format, request.target.shape
    if format not in _VERBATIM_LAYOUTS or len(shape) != 2:
        raise ValueError("import_verbatim requires an MXFP8 or row-NVFP4 matrix target")
    if request.parameters:
        raise ValueError(f"{request.target.id}: import_verbatim accepts no numerical parameters")
    if request.row_offsets is None or request.row_offsets[-1] != shape[0]:
        raise ValueError(f"{request.target.id}: import_verbatim inputs must be complete rows of the parent")
    for item in request.inputs:
        if item.source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = item.source.read_encoded(0, 1)
        if first.format != format:
            raise ValueError(f"{item.parameter}: source {first.format} differs from target {format}")
        for parameter, input_name in item.uses:
            if request.policies[(parameter, input_name)] != "A16Only":
                raise ValueError(f"{parameter}: {format} weights take 16-bit activations (A16Only)")
    n, k = shape
    chunk = max(request.rows_per_chunk, _VERBATIM_CHUNK_BYTES // _verbatim_row_bytes(format, k))

    def produce(output):
        for begin in range(0, n, chunk):
            words = request.encoded_rows(begin, min(n, begin + chunk))
            output.write_codes(begin, words.codes, words.scales)

    return request.job(produce=produce)


def _encoded_format(source) -> str | None:
    """The checkpoint encoding of a matrix source (None: plain values such as BF16)."""
    if source.read_encoded is None or len(source.shape) != 2:
        return None
    try:
        return source.read_encoded(0, 1).format
    except ValueError as error:
        # A plain matrix resolves to values only; any other error is a real source problem.
        if "does not provide encoded rows" in str(error):
            return None
        raise


def qwen3_8_flash_next_fp4_local(model, recipe, sources):
    """Flash-Next from ``nvidia/Qwen3.8-Flash-Next-FP4-Local`` in the checkpoint's own formats.

    No value is re-quantised. Routed Text experts keep their NVFP4 codes, CSF-decoded E4M3 block
    scales, weight divisors and activation divisors (import_encoded, AllowA4); the MTP experts'
    NVFP4 words are imported the same way with 16-bit activations (no activation calibration in
    the checkpoint). Every MXFP8 matrix (attention, QSA indexer, GDN, shared expert) is copied into
    ``mxfp8_e4m3fn_g32_ue8m0`` and the n-gram table's row-NVFP4 shards into ``nvfp4_row_e4m3_g16``
    with its multiplier as ``ple/table_global_scale``. BF16 tensors (embedding, output head,
    hyper-connections, routers, PLE projections, the MTP dense layer) stay BF16; the FP32 norm
    vectors become BF16, the only rounding. Fused projection groups and expert banks are one parent
    each. Text and MTP only: Vision is not served from this checkpoint.
    """
    if model.config.get("model_type") != "qwen4_exp_text" or "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen4Exp MoE mathematics")
    if "vision" in model.components:
        raise ValueError("qwen3_8_flash_next_fp4_local serves text and mtp only; use --components text,mtp")
    verbatim = set()
    for name, parameter in model.parameters.items():
        if name.endswith("/ple/table"):
            format = _encoded_format(parameter.source)
            if format != NVFP4_ROW:
                raise ValueError(f"{name}: expected a row-NVFP4 n-gram table, got {format}")
        elif not parameter.projection:
            continue
        elif "/moe/experts/" in name:
            if _encoded_format(parameter.source) != "nvfp4":
                raise ValueError(f"{name}: expected NVFP4 expert words")
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=parameter.source,
                activation_policy="AllowA4" if name.startswith("text/") else "A16Only",
            )
            continue
        else:
            format = _encoded_format(parameter.source)
            if format is None:
                continue  # BF16 in the checkpoint: the default BF16 selection keeps it.
            if format != MXFP8:
                raise ValueError(f"{name}: unexpected checkpoint encoding {format}")
        recipe.assign(name, format=format, layout=_VERBATIM_LAYOUTS[format], method=import_verbatim)
        verbatim.add(name)
    # Fused groups (q|k|gate|v, q|k|v|z, a|b, indexer q|k, shared gate|up) and the routed-expert banks
    # become one parent each. Packing groups are listed outermost first; the first complete one wins.
    used = set()
    for names in model.packing_groups:
        experts = all("/moe/experts/" in name for name in names)
        if (experts or all(name in verbatim for name in names)) and not used & set(names):
            recipe.group(names)
            used.update(names)


def import_q4_as_q8(request: PrepareRequest) -> PreparedMethod:
    """Re-express imported q4_g64_fp16 rows as q8_g32_fp16 without changing a value.

    Every Q4 group of 64 becomes two Q8 groups of 32 with the same binary16 multiplier and the
    same signed codes ([-8, 7] lies inside the Q8 code domain), so each represented weight is
    identical. Used where an Op has a Q8 route and no Q4 one (the token-embedding gather).
    """
    if request.target.format != Q8 or len(request.target.shape) != 2:
        raise ValueError("import_q4_as_q8 requires a q8_g32_fp16 matrix target")
    _preflight(request, values=False)
    n, k = request.target.shape
    if k % 64:
        raise ValueError("import_q4_as_q8 requires whole Q4 groups")
    chunk = max(request.rows_per_chunk, IMPORT_CHUNK_BYTES // (k + k // 16))

    def produce(output):
        for begin in range(0, n, chunk):
            words = request.encoded_rows(begin, min(n, begin + chunk))
            if words.format != Q4:
                raise ValueError(f"import_q4_as_q8: source rows are {words.format}, expected {Q4}")
            rows = words.codes.shape[0]
            codes = words.codes.reshape(rows, k // 32, 32)
            scales = words.scales.repeat_interleave(2, dim=1)
            output.write_codes(begin, codes, scales)

    return request.job(produce=produce)


def qwen3_8_27b_gsq3(model, recipe, sources):
    """ISTA-DASLab Qwen3.8-27B 3-bit GSQ -> one 16 GB GPU (v0.7 27b-16gb).

    ``--model`` is the GSQ checkpoint (ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ, compressed-tensors
    pack-quantized) and ``--source mtp=`` any checkpoint carrying the official BF16 MTP layer.

    - Text body (320 projections): the GSQ 3-bit G128 codes imported verbatim as q3_g128_fp16
      (signed codes unchanged; BF16 multipliers rounded to binary16, rejected above 2^-25).
    - Output head: the GSQ 4-bit RTN G64 words imported verbatim as q4_g64_fp16; the proposal
      head is a row gather of the same words (no requantization).
    - Token embedding: the same Q4 words re-expressed losslessly as q8_g32_fp16 (the gather has
      a Q8 route, no Q4 one); meant to stay in host RAM (--embedding-host).
    - GDN a/b projections, norms, convolution: BF16 as published.
    - Vision (BF16 in the GSQ release): the official optional formats (Q4/Q5/Q6/Q8).
    - MTP (from ``mtp``): Q8 stem and attention input, NVFP4 MLP and attention output by
      nvfp4_mse with A16 activations, as the production MTP3 artifact (quasar_mtpnv).
    """
    from .proposal import DEFAULT_RANKING, add_proposal

    if "num_experts" in model.config:
        raise ValueError("this recipe requires Qwen3.5 Dense mathematics")
    base = sources["base"]
    _optional(model, recipe)
    if "mtp" in model.components:
        _mtp_nvfp4_layer(model, recipe)
        mtp_store = sources["mtp"]
        for name in model.parameters:
            if name.startswith("mtp/"):
                recipe.assign(name, source=model.source(name, mtp_store))
    recipe.assign(
        "text/token_embedding",
        format=Q8,
        method=import_q4_as_q8,
        source=model.source("text/token_embedding", base, Q4),
    )
    recipe.assign(
        "text/output_head",
        format=Q4,
        method=import_encoded,
        source=model.source("text/output_head", base, Q4),
        activation_policy="A16Only",
    )
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            continue  # BF16 in the checkpoint (compressed-tensors ignore list)
        recipe.assign(
            name,
            format=Q3,
            method=import_encoded,
            source=model.source(name, base, Q3),
            activation_policy="A16Only",
        )
    add_proposal(recipe, ranking=DEFAULT_RANKING, source=recipe.selections["text/output_head"][0].source)
    recipe.assign("proposal/head", format=Q4, method=import_encoded)


RECIPES = {
    "qwen3_8_27b_gsq3": qwen3_8_27b_gsq3,
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
    "qwen3_8_flash_next_nvfp4_dense": qwen3_8_flash_next_nvfp4_dense,
    "qwen3_8_flash_next_nvfp4_mix": qwen3_8_flash_next_nvfp4_mix,
    "qwen3_8_flash_next_nvfp4_gdn": qwen3_8_flash_next_nvfp4_gdn,
    "qwen3_8_flash_next_nvfp4_gdn_idxbf16": qwen3_8_flash_next_nvfp4_gdn_idxbf16,
    "qwen3_8_flash_next_nvfp4_gdn_hcbf16": qwen3_8_flash_next_nvfp4_gdn_hcbf16,
    "qwen3_8_flash_next_nvfp4_gdnbf16": qwen3_8_flash_next_nvfp4_gdnbf16,
    "qwen3_8_flash_next_nvfp4_gdn_idxhcbf16": qwen3_8_flash_next_nvfp4_gdn_idxhcbf16,
    "qwen3_8_flash_next_nvfp4_allbf16": qwen3_8_flash_next_nvfp4_allbf16,
    "qwen3_8_flash_next_nvfp4_idxhcbf16_gdnq8": qwen3_8_flash_next_nvfp4_idxhcbf16_gdnq8,
    "qwen3_8_flash_next_nvfp4_idxbf16_gdnq8": qwen3_8_flash_next_nvfp4_idxbf16_gdnq8,
    "qwen3_8_flash_next_fp4_local": qwen3_8_flash_next_fp4_local,
    "qwen4_bf16": qwen4_bf16,
    "qwen4_bf16_hcq8": qwen4_bf16_hcq8,
    "qwen4_nvfp4": qwen4_nvfp4,
}
