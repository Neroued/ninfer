"""Read GGUF tensors (F16 or any ggml quant level) as a named conversion source.

`tools.convert` normally reads the base checkpoint straight from Safetensors
(see sources/safetensors.py). This module lets a recipe pull some or all
logical parameters from a GGUF file instead -- typically because that is the
only local copy of the weights (e.g. an F16 or Q4_K/Q5_K/Q6_K/Q8_0 export),
and downloading the full BF16/FP32 Safetensors checkpoint again is wasteful.

Parsing and dequantization are delegated to the `gguf` package (the ggml-org
reference reader used by llama.cpp itself) rather than reimplemented here --
GGUF's block-quant byte layouts are intricate and easy to get subtly wrong,
and `gguf.dequantize` is the maintained, tested implementation. `pip install
gguf` if it is not already available.

GGUF tensor names never match the HF Safetensors names a recipe's source
tensors are keyed by (e.g. "model.language_model.layers.0.self_attn.q_proj.weight"
vs. "blk.0.attn_q.weight"), and Qwen3.5's hybrid linear-attention layers
reorder and rescale some tensors on export. `HFAliasSource` bridges that gap
with an explicit {hf_name: gguf_name} map plus optional per-tensor value
transforms, so it satisfies the same has()/describe()/read_flat() surface as
SafetensorsSource and can be dropped in wherever a recipe accepts an
alternate source (see the "Read another source" section of
docs/weight-conversion.md).

`standard_dense_name_map`, `qwen35_linear_attention_name_map`,
`qwen35_mtp_name_map`, and `qwen35_vision_name_map` were each checked against
real checkpoint/GGUF pairs on this machine -- the Swift-Qwen3.8-27B HF
checkpoint (/mnt/storage/models/swift-qwen3.8-27b, byte-identical tensor
names/shapes to official Qwen/Qwen3.8-27B), a standalone MTP checkpoint and
its matching MTP-only GGUF, and the matching mmproj vision GGUF, against Q5_0-
and Q8_0-quantized GGUF exports of the same architecture -- comparing every
mapped tensor's dequantized values back to the HF original. Unquantized
companions (F32 norms, A_log, dt_bias, conv1d, vision LayerNorms/biases)
matched exactly; quantized ones matched to within that quant level's own
noise (a few percent relative error, uniform across tensors, confirming the
mapping and any reorder/shift are structurally correct rather than merely
close by chance). Together they cover: dense attention/MLP/norms for
full-attention layers, Qwen3.5's fused-QKV/Z/alpha/beta GDN layers, the MTP
extra layer (a separate GGUF file in every case seen), and the vision
tower + qwen3vl_merger projector (also a separate "mmproj-*.gguf" file, GGUF
architecture `clip`). None of them cover mixture-of-experts routing/shared-
expert tensors, and the vision map doesn't cover deepstack layers -- extend
the maps for those using list_tensors() below to see exactly what a given
file calls things, and verify any addition the same way (an unquantized
companion tensor if you can find one, or a Q8_0/F16 GGUF for a tighter
statistical check). DFlash2 needs no GGUF-specific code in the cases seen so
far -- see docs/weight-conversion.md.

Nothing here changes what --model supplies: point it at a directory that
still has the checkpoint's config.json and frontend resources (tokenizer.json
etc. -- a few KB, no need to keep the multi-GB weight files) and use
`--source NAME=quantized.gguf` to supply weight values from GGUF instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    import gguf
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError(
        "reading a .gguf source requires the 'gguf' package: pip install gguf"
    ) from error


@dataclass(frozen=True, slots=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str  # ggml quantization type name, informational only


class GGUFSource:
    """Bounded flat reads over a GGUF file's own tensor names, dequantized to FP32.

    Exposes the same has()/describe()/read_flat()/close() surface as
    SafetensorsSource, keyed by the GGUF file's native tensor names (e.g.
    "blk.0.attn_q.weight"), so it can be used directly wherever a recipe
    expects that interface.

    A GGUF tensor's raw `ReaderTensor.data` is itself row-structured: its
    first axis always has exactly as many entries as the tensor's outer (HF-
    order) dimension, whether the tensor is quantized (raw per-row byte
    blocks) or plain float (already viewed to the right dtype) -- confirmed
    on real Q5_0/Q8_0 files on this machine (`gguf.dequantize` on a leading
    row-slice of `.data` reproduces exactly the same leading rows as
    dequantizing the whole tensor, byte for byte). read_flat() uses that to
    decode only the covering row range for the requested [begin, end)
    element range, rather than dequantizing the whole tensor on every call:
    this matters because tools.convert reads each tensor twice per
    parameter (a 1-element sample during recipe.prepare()'s preflight
    validation, then the real data row-chunk by row-chunk while writing),
    and GGUF's block quantization has no cheaper way to serve a narrow read
    than decoding whichever blocks it falls in -- so avoiding a wasted full
    decode on the 1-element preflight probe alone was roughly halving
    conversion time before this was fixed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader = gguf.GGUFReader(self.path)
        self._tensors = {tensor.name: tensor for tensor in self._reader.tensors}
        self.config: dict = {}
        self.bytes_read = 0

    def has(self, name: str) -> bool:
        return name in self._tensors

    def _tensor(self, name: str):
        try:
            return self._tensors[name]
        except KeyError as error:
            raise ValueError(f"{self.path}: missing GGUF tensor {name!r}") from error

    def describe(self, name: str) -> TensorInfo:
        tensor = self._tensor(name)
        # GGUF's ne[] lists dimensions fastest-first; reverse to match the
        # row-major (fastest-last) shape convention Safetensors/PyTorch use.
        # The underlying bytes are in the same flat element order either way.
        shape = tuple(int(dim) for dim in reversed(tensor.shape.tolist()))
        return TensorInfo(shape, tensor.tensor_type.name)

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        tensor = self._tensor(name)
        shape = self.describe(name).shape
        elements = 1
        for dim in shape:
            elements *= dim
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        row_width = elements // shape[0] if shape else 1
        row_begin, row_end = begin // row_width, -(-end // row_width)
        decoded = gguf.dequantize(tensor.data[row_begin:row_end], tensor.tensor_type)
        flat = np.ascontiguousarray(decoded.reshape(-1), dtype=np.float32)
        offset = begin - row_begin * row_width
        result = flat[offset : offset + (end - begin)].copy()
        self.bytes_read += result.nbytes
        return torch.from_numpy(result)

    def list_tensors(self):
        """Yield (name, shape, ggml_type) for every tensor, for building a name_map."""
        for name in self._tensors:
            info = self.describe(name)
            yield name, info.shape, info.dtype

    def metadata(self) -> dict:
        """Raw GGUF key/value metadata (architecture, hyperparameters, tokenizer, ...)."""
        result = {}
        for key, field in self._reader.fields.items():
            try:
                result[key] = field.contents()
            except Exception:
                result[key] = None
        return result

    def close(self) -> None:
        pass

    def __enter__(self) -> "GGUFSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


Transform = Callable[[torch.Tensor, tuple[int, ...]], tuple[torch.Tensor, tuple[int, ...]]]


class HFAliasSource:
    """Expose a GGUFSource's tensors under caller-supplied HF-style names.

    tools.convert recipes capture the exact HF Safetensors tensor name at
    model-build time and later call describe()/read_flat() with that same
    string against whichever store a recipe assigns (see model.source() /
    docs/weight-conversion.md's "Read another source"). A GGUF file has no
    such names, and some tensors are reordered relative to their HF layout;
    this wrapper looks up an explicit map instead of guessing at either.

    `transforms` is an optional {hf_name: fn(flat_values, shape) -> (flat_values,
    shape)} map for tensors GGUF stores in a different element order or scale
    than HF (see qwen35_linear_attention_name_map / standard_dense_name_map).
    Most tensors need no transform: GGUF's ne[] is simply HF's shape reversed,
    with the same flat element order and value.

    `shapes` is an optional {hf_name: fn(gguf_shape) -> hf_shape} map for
    tensors whose *rank* also differs -- e.g. llama.cpp's exporter squeezes
    a PyTorch Conv1d weight's singleton (out_channels, 1, kernel_size) shape
    down to (out_channels, kernel_size) before writing it, so GGUF reports a
    2D tensor where the HF-side caller's captured shape is 3D even though the
    flat element order and count are identical. Most tensors need no entry
    here either.

    `composites` is an optional {hf_name: fn(GGUFSource) -> (flat_values, hf_shape)}
    map for the rarer case where ONE HF tensor was split across MULTIPLE GGUF
    tensors on export -- e.g. Qwen3-VL's Conv3d patch-embed kernel (temporal
    depth 2) is written as two separate 2D kernels ("v.patch_embd.weight" and
    "v.patch_embd.weight.1"); the composite function reads both from the raw
    GGUFSource and stacks them back into the original 5D layout. A composite
    entry takes priority over `name_map`/`transforms`/`shapes` for that name.
    """

    def __init__(
        self,
        source: GGUFSource,
        name_map: dict[str, str],
        *,
        transforms: dict[str, Transform] | None = None,
        shapes: dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]] | None = None,
        composites: dict[str, Callable[[GGUFSource], tuple[torch.Tensor, tuple[int, ...]]]]
        | None = None,
    ) -> None:
        self._source = source
        self._names = dict(name_map)
        self._transforms = dict(transforms) if transforms else {}
        self._shapes = dict(shapes) if shapes else {}
        self._composites = dict(composites) if composites else {}
        self.path = source.path
        self.config: dict = {}
        # Transformed tensors (GDN's fused/reordered ones) need their whole
        # row group decoded regardless of the requested [begin, end) range,
        # since the reorder isn't expressible on a row subset alone. Cache
        # the last one so a tensor's own sequential produce() chunk reads
        # don't repeat that decode+transform per chunk; a single slot is
        # enough because the writer processes one parameter at a time and
        # this is never the same tensor recipe.prepare()'s preflight already
        # decoded (that pass and the write pass are far enough apart in
        # tensor order that nothing would still be cached anyway).
        self._transform_cache_name: str | None = None
        self._transform_cache_values: torch.Tensor | None = None

    def _resolve(self, hf_name: str) -> str:
        try:
            return self._names[hf_name]
        except KeyError as error:
            raise ValueError(
                f"no GGUF tensor mapped for {hf_name!r}; add it to the name_map "
                f"passed to HFAliasSource (inspect {self._source.path} with "
                "GGUFSource.list_tensors() to see the tensor names it actually has)"
            ) from error

    def has(self, name: str) -> bool:
        if name in self._composites:
            return True
        return name in self._names and self._source.has(self._names[name])

    def describe(self, name: str) -> TensorInfo:
        composite = self._composites.get(name)
        if composite is not None:
            values, shape = composite(self._source)
            return TensorInfo(shape, "composite")
        # Every transform here reorders or rescales values in place; none of
        # them change shape, so describe() never needs to run one -- only
        # consult `shapes` for the (rare) rank mismatch.
        info = self._source.describe(self._resolve(name))
        reshape = self._shapes.get(name)
        return info if reshape is None else TensorInfo(reshape(info.shape), info.dtype)

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        composite = self._composites.get(name)
        if composite is not None:
            values, shape = composite(self._source)
            elements = values.numel()
            end = elements if end is None else end
            if not 0 <= begin <= end <= elements:
                raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
            return values[begin:end].clone()
        gguf_name = self._resolve(name)
        transform = self._transforms.get(name)
        if transform is None:
            return self._source.read_flat(gguf_name, begin, end)
        shape = self._source.describe(gguf_name).shape
        if self._transform_cache_name == name and self._transform_cache_values is not None:
            values = self._transform_cache_values
        else:
            values, _ = transform(self._source.read_flat(gguf_name), shape)
            self._transform_cache_name, self._transform_cache_values = name, values
        elements = values.numel()
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
        return values[begin:end].clone()

    def close(self) -> None:
        self._transform_cache_name, self._transform_cache_values = None, None

    def __enter__(self) -> "HFAliasSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def _shift_norm(values: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    # Verified against /mnt/storage/models/swift-qwen3.8-27b +
    # .../qwen3.8/Qwen3.8-27B-Q5_0.gguf: every "*.norm.weight" tensor except
    # linear_attn.norm.weight is stored on export as (HF weight + 1), an
    # exact match confirmed on input_layernorm, post_attention_layernorm,
    # q_norm, k_norm and the final model.norm.
    return values - 1.0, shape


def standard_dense_name_map(
    num_hidden_layers: int,
    *,
    tie_word_embeddings: bool = False,
    text_prefix: str = "model.language_model.",
) -> tuple[dict[str, str], dict[str, Transform], dict]:
    """HF -> GGUF (name_map, transforms, shapes) for dense attention/MLP/norm
    tensors -- see HFAliasSource for what each dict is for (`shapes` is
    always empty here: none of these tensors change rank).

    Matches the standard llama.cpp tensor roles used for Llama/Qwen2/Qwen3
    -family dense decoders, verified end-to-end (real values, not just
    names) against a real Qwen3.8-27B checkpoint pair on this machine -- see
    the module docstring. Covers only full-attention layers with a plain
    gate/up/down MLP; extend the returned maps for GDN (see
    qwen35_linear_attention_name_map), MoE routing/shared experts, or MTP
    tensors.

    `text_prefix` defaults to "model.language_model." because that is what
    Qwen3_5ForConditionalGeneration (vision-enabled) checkpoints use, verified
    against the real checkpoint above. A text-only Qwen3_5ForCausalLM
    checkpoint may use plain "model." instead -- check your checkpoint's
    model.safetensors.index.json weight_map (or GGUFSource.list_tensors())
    before trusting this default for that variant.
    """
    name_map = {
        text_prefix + "embed_tokens.weight": "token_embd.weight",
        text_prefix + "norm.weight": "output_norm.weight",
    }
    transforms: dict[str, Transform] = {text_prefix + "norm.weight": _shift_norm}
    if not tie_word_embeddings:
        name_map["lm_head.weight"] = "output.weight"
    for i in range(num_hidden_layers):
        hf = f"{text_prefix}layers.{i}."
        gg = f"blk.{i}."
        name_map.update(
            {
                hf + "input_layernorm.weight": gg + "attn_norm.weight",
                hf + "self_attn.q_proj.weight": gg + "attn_q.weight",
                hf + "self_attn.q_norm.weight": gg + "attn_q_norm.weight",
                hf + "self_attn.k_proj.weight": gg + "attn_k.weight",
                hf + "self_attn.k_norm.weight": gg + "attn_k_norm.weight",
                hf + "self_attn.v_proj.weight": gg + "attn_v.weight",
                hf + "self_attn.o_proj.weight": gg + "attn_output.weight",
                hf + "post_attention_layernorm.weight": gg + "post_attention_norm.weight",
                hf + "mlp.gate_proj.weight": gg + "ffn_gate.weight",
                hf + "mlp.up_proj.weight": gg + "ffn_up.weight",
                hf + "mlp.down_proj.weight": gg + "ffn_down.weight",
            }
        )
        for norm_name in (
            hf + "input_layernorm.weight",
            hf + "self_attn.q_norm.weight",
            hf + "self_attn.k_norm.weight",
            hf + "post_attention_layernorm.weight",
        ):
            transforms[norm_name] = _shift_norm
    return name_map, transforms, {}


def reorder_linear_attention_v_heads(
    values: torch.Tensor,
    shape: tuple[int, ...],
    dim: int,
    num_row_groups: int,
    group_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Swap V-head grouping between HF's "grouped by K head" order and
    ggml's tiled broadcast order.

    Ports `_LinearAttentionVReorderBase._reorder_v_heads` from
    /mnt/storage/llama.cpp/conversion/qwen.py: reshape to
    [..., num_row_groups, group_size, head_dim, ...], swap the two head
    axes, reshape back. This is a plain transpose of a (num_row_groups,
    group_size) matrix of head-blocks, so it is only its own inverse when
    num_row_groups == group_size. In general (Qwen3.8-27B has 16 K-heads and
    3 V-heads per K-head, so it is not) the inverse call must swap the two
    count arguments: to undo `reorder(x, dim, K, V, d)`, call
    `reorder(y, dim, V, K, d)`. Verified exactly (down to float32 rounding)
    against real HF vs. GGUF tensor pairs for both the forward (export) and
    swapped-argument inverse (import) directions -- see the module docstring.
    """
    tensor = values.reshape(shape)
    if dim < 0:
        dim += len(shape)
    new_shape = (
        list(shape[:dim]) + [num_row_groups, group_size, head_dim] + list(shape[dim + 1 :])
    )
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def qwen35_linear_attention_name_map(
    layer_index: int,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    text_prefix: str = "model.language_model.",
) -> tuple[dict[str, str], dict[str, Transform], dict[str, Callable[[tuple[int, ...]], tuple[int, ...]]]]:
    """HF -> GGUF (name_map, transforms, shapes) for one Qwen3.5 GDN (linear
    attention) layer -- see HFAliasSource for what each dict is for.

    Ported from /mnt/storage/llama.cpp/conversion/qwen.py's
    _LinearAttentionVReorderBase/Qwen3NextModel and gguf-py/gguf/constants.py's
    QWEN35 TENSOR_NAMES, then verified end-to-end against a real Qwen3.8-27B
    checkpoint pair on this machine (see the module docstring): every name,
    the A_log `-exp()` transform, the norm.weight exclusion from the +1 shift
    applied elsewhere, and the V-head reorder (with swapped arguments for
    this import direction -- see reorder_linear_attention_v_heads) all
    reproduced the original HF tensor exactly for unquantized companions
    (A_log, dt_bias, conv1d are F32 in the GGUF checked) and within the
    source GGUF's own quantization noise for quantized ones (in_proj_qkv,
    in_proj_z, in_proj_a, in_proj_b, out_proj).
    """
    hf = f"{text_prefix}layers.{layer_index}.linear_attn."
    gg = f"blk.{layer_index}."
    name_map = {
        hf + "in_proj_qkv.weight": gg + "attn_qkv.weight",
        hf + "in_proj_z.weight": gg + "attn_gate.weight",
        hf + "in_proj_a.weight": gg + "ssm_alpha.weight",
        hf + "in_proj_b.weight": gg + "ssm_beta.weight",
        hf + "conv1d.weight": gg + "ssm_conv1d.weight",
        hf + "A_log": gg + "ssm_a",
        hf + "dt_bias": gg + "ssm_dt.bias",
        hf + "norm.weight": gg + "ssm_norm.weight",  # no +1 shift: excluded by the exporter
        hf + "out_proj.weight": gg + "ssm_out.weight",
    }

    # llama.cpp's exporter squeezes the singleton middle dim off Conv1d's
    # PyTorch (out_channels, 1, kernel_size) weight before writing it; insert
    # it back so describe() matches the shape the recipe captured from HF.
    shapes = {hf + "conv1d.weight": lambda native: (native[0], 1, native[1])}

    num_v_per_k = num_v_heads // num_k_heads
    q_dim = head_k_dim * num_k_heads
    k_dim = head_k_dim * num_k_heads

    def reorder_v(rows: torch.Tensor, head_dim: int, dim: int = 0) -> torch.Tensor:
        return reorder_linear_attention_v_heads(rows, rows.shape, dim, num_v_per_k, num_k_heads, head_dim)

    if num_v_per_k <= 1:
        return name_map, {hf + "A_log": _inverse_a_log}, shapes

    def qkv_transform(values, shape):
        n, k = shape
        matrix = values.reshape(n, k)
        q, kk, v = matrix[:q_dim], matrix[q_dim : q_dim + k_dim], matrix[q_dim + k_dim :]
        v = reorder_v(v, head_v_dim)
        result = torch.cat([q, kk, v], dim=0)
        return result.reshape(-1), shape

    def rows_transform(head_dim):
        def apply(values, shape):
            return reorder_v(values.reshape(shape), head_dim).reshape(-1), shape

        return apply

    def out_proj_transform(values, shape):
        return reorder_v(values.reshape(shape), head_v_dim, dim=1).reshape(-1), shape

    def conv1d_transform(values, shape):
        n, k = shape
        matrix = values.reshape(n, k)
        qk_channels = head_k_dim * num_k_heads * 2
        qk_part, v_part = matrix[:qk_channels], matrix[qk_channels:]
        v_part = reorder_v(v_part, head_v_dim)
        return torch.cat([qk_part, v_part], dim=0).reshape(-1), shape

    transforms = {
        hf + "in_proj_qkv.weight": qkv_transform,
        hf + "in_proj_z.weight": rows_transform(head_v_dim),
        hf + "in_proj_a.weight": rows_transform(1),
        hf + "in_proj_b.weight": rows_transform(1),
        hf + "out_proj.weight": out_proj_transform,
        hf + "conv1d.weight": conv1d_transform,
        hf + "dt_bias": rows_transform(1),
        hf + "A_log": _inverse_a_log_reordered(reorder_v),
    }
    return name_map, transforms, shapes


def _inverse_a_log(values: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    # GGUF stores -exp(A_log); undo it (no V-head reorder when num_v_per_k == 1).
    return torch.log(-values), shape


def _inverse_a_log_reordered(reorder_v):
    def apply(values: torch.Tensor, shape: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
        reordered = reorder_v(values.reshape(shape), 1)
        return torch.log(-reordered).reshape(-1), shape

    return apply


def qwen35_mtp_name_map(
    gguf_layer_index: int, *, hf_prefix: str = "mtp."
) -> tuple[dict[str, str], dict[str, Transform], dict]:
    """HF -> GGUF (name_map, transforms, shapes) for the Qwen3.5 MTP (next-token
    prediction) extra layer.

    llama.cpp exports the MTP block as one more transformer layer appended
    after the base model's real layers (here layer 64, since num_hidden_layers
    is 64), reusing the ordinary full-attention/MLP/norm tensor roles
    standard_dense_name_map already covers, plus four MTP-specific tensors
    (nextn.eh_proj/enorm/hnorm/shared_head_norm) renamed from HF's
    fc/pre_fc_norm_embedding/pre_fc_norm_hidden/norm.

    `hf_prefix` is "mtp." when the MTP tensors are fused into the main
    checkpoint's state dict (e.g. swift-qwen3.8-27b's "mtp.fc.weight", ...)
    and "" for a standalone MTP-only checkpoint whose tensors have no prefix
    at all (e.g. a repo containing only "fc.weight", "layers.0....").

    Verified end-to-end against a real standalone MTP checkpoint and its
    matching MTP-only GGUF export on this machine (see the module
    docstring's sibling text/GDN validation): unlike the base model's
    layers, NONE of the MTP layer's norm.weight tensors get the +1 shift
    standard_dense_name_map applies elsewhere -- input_layernorm, q_norm,
    k_norm, post_attention_layernorm, and the three nextn norm-ish tensors
    all matched their HF originals exactly with no transform at all. (The
    exporter appears to apply that shift only while iterating the base
    model's own layer loop, before the MTP mixin's tensors are renamed and
    appended -- but that is inference from the observed behavior, not from
    reading that exact code path; trust the validated "no shift" result over
    the guess at why.)
    """
    gg = f"blk.{gguf_layer_index}."
    hf = hf_prefix
    name_map = {
        hf + "fc.weight": gg + "nextn.eh_proj.weight",
        hf + "pre_fc_norm_embedding.weight": gg + "nextn.enorm.weight",
        hf + "pre_fc_norm_hidden.weight": gg + "nextn.hnorm.weight",
        hf + "norm.weight": gg + "nextn.shared_head_norm.weight",
        hf + "layers.0.input_layernorm.weight": gg + "attn_norm.weight",
        hf + "layers.0.self_attn.q_proj.weight": gg + "attn_q.weight",
        hf + "layers.0.self_attn.q_norm.weight": gg + "attn_q_norm.weight",
        hf + "layers.0.self_attn.k_proj.weight": gg + "attn_k.weight",
        hf + "layers.0.self_attn.k_norm.weight": gg + "attn_k_norm.weight",
        hf + "layers.0.self_attn.v_proj.weight": gg + "attn_v.weight",
        hf + "layers.0.self_attn.o_proj.weight": gg + "attn_output.weight",
        hf + "layers.0.post_attention_layernorm.weight": gg + "post_attention_norm.weight",
        hf + "layers.0.mlp.gate_proj.weight": gg + "ffn_gate.weight",
        hf + "layers.0.mlp.up_proj.weight": gg + "ffn_up.weight",
        hf + "layers.0.mlp.down_proj.weight": gg + "ffn_down.weight",
    }
    return name_map, {}, {}


def _patch_embed_composite(hf_name0: str, hf_name1: str, hf_shape: tuple[int, ...]):
    def build(source: GGUFSource) -> tuple[torch.Tensor, tuple[int, ...]]:
        out_channels, in_channels, temporal, patch_h, patch_w = hf_shape
        slice0 = source.read_flat(hf_name0).reshape(out_channels, in_channels, patch_h, patch_w)
        slice1 = source.read_flat(hf_name1).reshape(out_channels, in_channels, patch_h, patch_w)
        stacked = torch.stack([slice0, slice1], dim=2)
        return stacked.reshape(-1), hf_shape

    return build


def qwen35_vision_name_map(
    num_blocks: int,
    *,
    hidden_size: int,
    in_channels: int = 3,
    temporal_patch_size: int = 2,
    patch_size: int,
    hf_prefix: str = "model.visual.",
) -> tuple[dict[str, str], dict[str, Transform], dict, dict]:
    """HF -> GGUF (name_map, transforms, shapes, composites) for the Qwen3.5
    vision tower + qwen3vl_merger projector, as exported by llama.cpp into a
    separate "mmproj-*.gguf" file (architecture "clip", not the text model's
    "qwen35"/"qwen35moe").

    Verified end-to-end against a real checkpoint/mmproj pair on this
    machine: block norms (LayerNorm, not RMSNorm -- unlike the text model,
    none of these get a +1 shift), the merger's post-tower LayerNorm
    ("merger.norm" -> "v.post_ln", not obviously the same tensor by name
    alone), and the patch-embed Conv3d kernel's temporal-depth split (HF
    stores one (out, in, temporal_patch_size, patch, patch) kernel; the
    exporter splits it into "v.patch_embd.weight" and ".weight.1", one 2D
    kernel per temporal slice, reassembled here via a `composites` entry)
    all reproduced the original tensor exactly (F32/F16 companions) or
    within Q8_0's own quantization noise (matrix weights).

    Does not cover deepstack layers (this checkpoint's
    clip.vision.is_deepstack_layers were all false) -- extend for those if
    your model uses them.
    """
    hf = hf_prefix
    name_map = {
        hf + "patch_embed.proj.bias": "v.patch_embd.bias",
        hf + "pos_embed.weight": "v.position_embd.weight",
        hf + "merger.norm.weight": "v.post_ln.weight",
        hf + "merger.norm.bias": "v.post_ln.bias",
        hf + "merger.linear_fc1.weight": "mm.0.weight",
        hf + "merger.linear_fc1.bias": "mm.0.bias",
        hf + "merger.linear_fc2.weight": "mm.2.weight",
        hf + "merger.linear_fc2.bias": "mm.2.bias",
    }
    for i in range(num_blocks):
        b = f"{hf}blocks.{i}."
        v = f"v.blk.{i}."
        name_map.update(
            {
                b + "norm1.weight": v + "ln1.weight",
                b + "norm1.bias": v + "ln1.bias",
                b + "norm2.weight": v + "ln2.weight",
                b + "norm2.bias": v + "ln2.bias",
                b + "attn.qkv.weight": v + "attn_qkv.weight",
                b + "attn.qkv.bias": v + "attn_qkv.bias",
                b + "attn.proj.weight": v + "attn_out.weight",
                b + "attn.proj.bias": v + "attn_out.bias",
                b + "mlp.linear_fc1.weight": v + "ffn_up.weight",
                b + "mlp.linear_fc1.bias": v + "ffn_up.bias",
                b + "mlp.linear_fc2.weight": v + "ffn_down.weight",
                b + "mlp.linear_fc2.bias": v + "ffn_down.bias",
            }
        )
    # HF stores one (hidden_size, in_channels, temporal_patch_size, patch,
    # patch) Conv3d kernel; the exporter splits it into one 2D
    # (hidden_size, in_channels, patch, patch) kernel per temporal slice
    # ("v.patch_embd.weight" / ".weight.1"). Only temporal_patch_size == 2
    # is implemented, matching every Qwen3-VL checkpoint seen so far.
    if temporal_patch_size != 2:
        raise ValueError(
            f"qwen35_vision_name_map only supports temporal_patch_size=2, "
            f"got {temporal_patch_size}"
        )
    patch_shape = (hidden_size, in_channels, temporal_patch_size, patch_size, patch_size)
    composites = {
        hf
        + "patch_embed.proj.weight": _patch_embed_composite(
            "v.patch_embd.weight", "v.patch_embd.weight.1", patch_shape
        )
    }
    return name_map, {}, {}, composites


def _main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List a GGUF file's tensor names, shapes and ggml types."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    with GGUFSource(args.path) as source:
        for name, shape, dtype in sorted(source.list_tensors()):
            print(f"{name}\t{tuple(shape)}\t{dtype}")


if __name__ == "__main__":
    _main()
