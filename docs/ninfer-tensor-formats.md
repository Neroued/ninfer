# NInfer Persistent Tensor Numeric Formats

> Status: accepted and implemented by the `.ninfer` writer/reader, quantizer,
> converter, verifier, Python reference, production C++ binder/materializer, and Engine.
>
> Authority: this document defines the closed set of persistent numeric tensor formats accepted by
> NInfer. It defines the logical words of the direct scalar formats, the numerical meaning and
> canonical reference encoder of the grouped quantized-weight formats, conformance rules, and the
> boundary between a numeric format and the container, storage layout, checkpoint recipe, operator,
> kernel, and runtime state. It does not define the `.ninfer` container, JSON schema, byte
> packing, tensor assignment for any checkpoint, kernel implementation, activation quantization, or
> runtime-state codecs.
>
> This document governs the seven logical formats used by product `.ninfer` artifacts. Removed
> legacy formats are not revised, aliased, or accepted by this registry.

## 1. Decision

NInfer starts with exactly seven persistent numeric tensor formats in two categories.

Direct scalar formats preserve one logical scalar word per tensor element:

| Canonical name | Category | Logical width | Logical meaning |
|---|---|---:|---|
| `BF16` | direct floating point | 16 | bfloat16 bit encoding |
| `FP32` | direct floating point | 32 | IEEE-754 binary32 |
| `I32` | direct signed integer | 32 | 32-bit two's-complement integer |

Grouped quantized-weight formats preserve signed codes plus one scale per logical group:

| Canonical name | Code width | Group size | Legal signed codes | Scale | Full-group logical bits/weight |
|---|---:|---:|---:|---|---:|
| `Q4G64_F16S` | 4 | 64 | `[-8, 7]` | one binary16 scale/group | 4.25 |
| `Q5G64_F16S` | 5 | 64 | `[-16, 15]` | one binary16 scale/group | 5.25 |
| `Q6G64_F16S` | 6 | 64 | `[-32, 31]` | one binary16 scale/group | 6.25 |
| `W8G32_F16S` | 8 | 32 | `[-127, 127]` | one binary16 scale/group | 8.50 |

This is a closed registry, not a template from which arbitrary scalar types, bit widths, and group
sizes may be constructed. In particular, `FP16`, `I64`, `Q4G32_F16S`, `Q6G128_F16S`, and
`W8G64_F16S` do not become valid merely because their components look familiar. The use of a
binary16 scale inside the four quantized formats does not register `FP16` as a direct tensor format.

NInfer does not reserve entries for hypothetical formats. A future format is added only for a real
selected checkpoint and GPU target, after its numerical contract, cited upstream quality evidence
where quantization is involved, storage path, and useful kernels are known. Registry membership
means that a logical representation is allowed; it does not imply that every tensor role, operator,
layout, shape, checkpoint, or GPU can consume it.

The design deliberately preserves the currently used group sizes. Q4, Q5, and Q6 share one G64
group geometry, allowing their converter, codec, and kernel structures to share the same grouping
model. W8 retains the single implemented G32 geometry. The registered Qwen3.6-27B target provides
the current implementation evidence for these choices. These are accepted project geometries, not
claims that G64 or G32 is universally optimal.

NInfer does not undertake an independent model-quality research campaign to qualify formats or
checkpoint recipes. Quantization quality decisions rely on credible evidence from model publishers,
upstream implementations, or existing public evaluations. NInfer remains responsible for local
artifact validation, independent codec and operator correctness, actual loading and execution,
runtime stability, memory use, and end-to-end prefill and decode performance. Those local checks do
not substitute for, or claim to be, an independent model-quality study.

## 2. Terms and ownership

The registry keeps the following concerns separate.

### 2.1 Persistent numeric format

A **persistent numeric format** defines the logical words needed to recover a numeric tensor from
an artifact. The closed registry contains both direct scalar formats and grouped quantized-weight
formats. It does not identify a tensor's model role, physical byte layout, or supported consumer.

### 2.2 Direct scalar format

A **direct scalar format** assigns one fixed-width logical word to every logical tensor coordinate.
It defines the value of that word without a group, scale, zero point, or reconstruction step. `BF16`,
`FP32`, and `I32` are direct scalar formats.

The adjective “direct” does not require a particular physical layout or prohibit a registered
lossless storage transformation. It means only that layout decoding recovers the original logical
word rather than a quantized approximation of another word.

### 2.3 Quantization scheme

A **quantization scheme** defines only the persistent logical representation of a quantized weight:

- the signed integer code domain;
- the group axis and group size;
- the scale type and scale granularity;
- the validity rules for codes and scales;
- the mathematical reconstruction of each represented weight.

The four canonical names above identify schemes in this sense. Their meanings are immutable: a
consumer must not infer a different zero point, scale geometry, code range, or reconstruction rule
from context.

### 2.4 Encoder profile

An **encoder profile** defines how source floating-point values are converted into the codes and
scales of a scheme. Scale selection, rounding order, calibration, clipping, and error optimization
belong here.

NInfer defines one canonical quantization encoder profile in Section 7. That encoder provides a
defined baseline and an independent artifact oracle. A checkpoint-specific process may use another
documented encoder to produce the same scheme—for example, an upstream error-optimized recipe—but
that does not create a new quantization scheme if the resulting persistent codes and scales obey the
same logical contract. Encoder provenance must not be confused with decoder semantics.

Direct formats also separate representation from conversion. For example, the `BF16` format does
not decide whether an FP32 source is rounded, truncated, or rejected. Any conversion from a source
type to a different direct format belongs to the checkpoint recipe.

### 2.5 Checkpoint numeric recipe

A **checkpoint numeric recipe** decides which source tensors use which registered direct formats or
quantization schemes and encoder profiles. It also owns source-type conversion, cited upstream
quality evidence, sensitive-tensor selection, mixed-precision policy, tensor-specific value
restrictions, and source-checkpoint identity.

The recipe is checkpoint-specific. Selecting `Q4G64_F16S` for one tensor says nothing about another
tensor, another checkpoint in the same model family, or a differently shaped variant. Choosing
`BF16` for one norm or `I32` for one index tensor likewise says nothing about any other tensor. Role
names such as control, norm, expert map, or recurrent parameter are recipe metadata, not numeric
format variants.

### 2.6 Storage layout

A **storage layout** maps direct logical words or quantized codes and scales to bytes. It owns,
among other things:

- bit and byte packing;
- plane ordering and interleaving;
- row, tile, expert, or kernel-native ordering;
- physical padding and alignment;
- endianness and any layout-specific validation metadata.

One format may have more than one deliberately supported layout, but every layout must decode to
exactly the same direct words or logical codes and scales. The currently registered layouts are
`contiguous-le-v1` for direct words and `row-split-k128-v1` for grouped formats. Their byte order,
plane packing, 128-element K padding, and alignment rules belong to the layout registry, not to these
seven numeric formats.

### 2.7 Compute profile and kernel support

A **compute profile** owns observable input and output types, accumulation precision, required
rounding or fusion boundaries, determinism requirements, and operator-level numerical tolerances.
Kernel tiling, Tensor Core use, internal reduction trees, split strategy, fusion schedule,
dequantization staging, launch geometry, and hardware dispatch remain implementation choices when
they satisfy those observable boundaries. Only an explicitly bitwise compute profile freezes an
exact internal order.

Consequently, `W8G32_F16S` means neither W8A8 nor FP16 output, and `F16S` says nothing about the
activation type. A kernel is usable only for an explicitly supported combination of numeric format,
storage layout, operator semantics, shape regime, compute profile, and selected GPU. This support
matrix is a closed whitelist, not the Cartesian product of separately known components. There is no
generic fallback obligation for another combination.

### 2.8 Runtime-state codec

KV-cache quantization, activation quantization, temporary kernel compression, recurrent state, and
communication formats are runtime-state codecs. They are outside this persistent-tensor registry
even if they also use signed integers and grouped scales. In particular, an INT8 KV-cache format
must not be labeled `W8G32_F16S` merely because some of its fields look similar.

## 3. Canonical identities and direct-format semantics

### 3.1 Direct scalar formats

#### BF16

`BF16` is the bfloat16 bit encoding. Its abstract 16-bit word has one sign bit at bit 15, an 8-bit
exponent at bits 14 through 7 with bias 127, and a 7-bit trailing fraction at bits 6 through 0.

Its value classes are:

- exponent zero and fraction zero: positive or negative zero according to the sign bit;
- exponent zero and nonzero fraction: a subnormal value
  `(-1)^sign * fraction * 2^-133`;
- exponent 1 through 254: a normal value
  `(-1)^sign * 2^(exponent-127) * (1 + fraction/2^7)`;
- exponent 255 and fraction zero: positive or negative infinity;
- exponent 255 and nonzero fraction: NaN, including its sign, quiet/signaling bit, and payload.

Equivalently, its exact expansion to binary32 places the 16-bit logical word in the high 16 bits of
a binary32 word and appends 16 zero low bits. `BF16` is not IEEE binary16/FP16.

All 65,536 logical BF16 words are valid at the format layer. Positive and negative zero remain
distinct words. Infinity and NaN are also representable, and a direct layout must recover their
original logical words without canonicalizing a NaN payload or changing its quiet/signaling bit.
Whether a particular model tensor permits non-finite values is a checkpoint-recipe constraint, not
a different BF16 format. A compute operation is not required to preserve a NaN payload unless its
compute profile says so.

#### FP32

`FP32` is IEEE-754 binary32: one sign bit at bit 31, an 8-bit exponent at bits 30 through 23 with
bias 127, and a 23-bit trailing fraction at bits 22 through 0. Its zeros, subnormals, normal values,
infinities, and NaNs have the standard binary32 interpretation; the minimum positive subnormal is
`2^-149`.

All 2^32 logical words are valid at the format layer. As with BF16, signed zeros, infinities, and
NaN payloads remain distinct persistent words. Tensor-specific restrictions on non-finite values
belong to the checkpoint recipe, while compute-time propagation belongs to the compute profile.

#### I32

`I32` is a 32-bit two's-complement signed integer. For an abstract unsigned logical word `u`:

```text
value = u                         if u < 2^31
value = u - 2^32                  otherwise
```

Every 32-bit word is valid, and the value interval is `[-2147483648, 2147483647]`. `I32` has no
scale, zero point, saturation behavior, sentinel convention, or dependency on the host C++ `int`
type. Requirements such as nonnegativity, vocabulary bounds, or the meaning of `-1` belong to the
specific tensor role.

The canonical identifier is `I32`; “INT32” is explanatory prose, not a second accepted name or a
compatibility alias.

#### Common direct-format rules

For a direct tensor, every logical coordinate owns one independent word of the selected format.
Direct formats are shape-agnostic: rank, dimension validity, coordinate traversal, strides,
padding, and alignment are defined by the container and selected layout, not by the scalar format.
Byte order is likewise a layout property and must never be inferred from host-native representation.

Persistence from the same source type is bitwise identity at the logical-word boundary. A conversion
between different source and target types is not implicit. The checkpoint recipe must define it,
including rounding for FP32-to-BF16 and range/integrality checks for any conversion to I32. A
producer must not silently wrap, truncate, or saturate a value merely because the destination word
has a fixed width.

The direct format does not encode a model role. `BF16_CTRL`, `FP32_CTRL`, and `I32_CTRL` are internal
execution `QType` names, not persistent NInfer format identities, and receive no aliases in the
artifact registry. Control parameters, ordinary weights, norms, indexes, and maps use a direct
format plus a separate model-assignment role.

### 3.2 Grouped quantized-weight identities

The canonical names are closed identifiers with the following readable components:

```text
Q4G64_F16S
│ │   └── one IEEE-754 binary16 scale per logical group
│ └────── 64 logical weights per full group
└──────── 4-bit signed integer weight codes
```

`Q5G64_F16S` and `Q6G64_F16S` follow the same convention. `W8G32_F16S` retains the established W8
spelling for the signed 8-bit weight path. The different leading letter does not imply activation
quantization or a generic family distinction beyond the exact rules in this document.

All seven names are identifiers, not a grammar. A parser must compare a name against this closed
registry; it must not accept an unknown combination by splitting a name into components.
Abbreviations such as `Q4`, `Q5`, `Q6`, `W8G32`, and `INT32` may be used in explanatory prose only.

A container representation must resolve its stored identity to exactly one of these canonical names
without constructing or reinterpreting a format. The container contract owns how that identity is
serialized.

## 4. Grouped quantized tensor model

### 4.1 Shape and group axis

Let a logical quantized weight tensor have rank `r >= 1`, a positive `K`, and shape:

```text
[D0, D1, ..., D(r-2), K]
```

For `r = 1`, this notation reduces to `[K]` with no leading coordinate.

The last dimension `K` is always the quantization axis. Every coordinate in the leading dimensions
identifies an independent logical row. For a leading coordinate vector `p` and element index `k`:

```text
group(p, k) = (p, floor(k / G))
```

where `G` is 64 for Q4/Q5/Q6 and 32 for W8. A group never crosses a row boundary or any leading
dimension. For a rank-3 expert bank `[E, N, K]`, for example, each `(expert, row)` pair is quantized
independently; a group cannot cross from one expert to another.

The logical scale tensor therefore has shape:

```text
[D0, D1, ..., D(r-2), ceil_div(K, G)]
```

This definition does not assign meanings such as output channel, expert, adapter, or convolution
axis to the leading dimensions. A checkpoint adapter owns any reshape or transpose needed to place
the intended quantization axis last before encoding. It is an abstract scheme definition. The
currently implemented `row-split-k128-v1` layout and canonical producer accept only positive rank-two
matrices `[N,K]`; supporting a higher-rank physical tensor requires an explicitly registered layout
and implementation, or a model recipe that defines a semantics-preserving rank-two reshape.

### 4.2 Final partial group

`K` need not be divisible by `G` at the scheme level. The final group of each row contains
`K - floor(K / G) * G` logical values when that number is nonzero. Its scale and codes are computed
from those logical values only.

A storage layout may physically extend that group or pad the tensor further, subject to all of the
following:

- padding is not part of the logical tensor shape;
- padding must not change any logical scale or code;
- padding must not change any decoded logical value or observable operator result;
- a reader obtains logical and physical extents from its artifact contract and never guesses them
  from payload length.

The scheme does not choose a padded extent, alignment multiple, whether padding is stored, or the
canonical contents of materialized padding. Each storage layout defines those matters and validates
them at its own boundary. A logical encoder produces no physical padding.

## 5. Code domains

All four schemes are zero-point-free, weight-only signed-integer schemes. “Symmetric” in this
document means that reconstruction is `scale * signed_code` with zero point zero. It does not mean
that every legal interval has equal positive and negative magnitude.

### 5.1 Q4, Q5, and Q6

The logical code is a two's-complement signed integer of the stated width:

| Scheme | Width | Sign bit | Legal code interval | `qmax` used by reference encoder |
|---|---:|---:|---:|---:|
| `Q4G64_F16S` | 4 | `0x8` | `[-8, 7]` | 7 |
| `Q5G64_F16S` | 5 | `0x10` | `[-16, 15]` | 15 |
| `Q6G64_F16S` | 6 | `0x20` | `[-32, 31]` | 31 |

Every bit pattern of the stated logical width is valid. Conversion between an unsigned logical word
`u` and signed code `q` is:

```text
q = u                         if u < 2^(b - 1)
q = u - 2^b                   otherwise
```

This is not offset binary, sign-magnitude encoding, a codebook index, or an unsigned GPTQ/AWQ
zero-point convention. A storage layout may split or reorder the bits, but after layout decoding the
logical word and signed value must be exactly those defined here.

### 5.2 W8

`W8G32_F16S` uses an 8-bit two's-complement signed code with the deliberately restricted interval
`[-127, 127]`. The byte pattern `0x80`, which would represent `-128`, is outside the valid artifact
language. The project-owned quantizer must never emit it; decoding semantics are defined only for a
valid code stream.

This restriction is part of the scheme, not an encoder preference. A kernel may use a native
signed-byte load because the registered conversion path establishes the code invariant; that
implementation convenience does not make `-128` legal. The trusted local runtime does not rescan
the complete W8 payload solely to prove an invariant already established by its producer.

## 6. Scale and reconstruction semantics

### 6.1 Scale representation

Each logical group has exactly one IEEE-754 binary16 scale. It is a dequantization multiplier, not
an inverse scale. There is no zero point, offset, minimum, secondary scale, block exponent, or
codebook.

A valid scale is either:

- positive zero, binary16 bit pattern `0x0000`; or
- a finite positive binary16 normal or subnormal number.

Negative finite values, negative zero, positive or negative infinity, and every NaN are invalid.
Positive binary16 subnormals are valid; consumers must preserve their defined value. In particular,
the minimum positive binary16 scale is `2^-24`, bit pattern `0x0001`.

If a stored scale is positive zero, every logical code in that group must be zero. The converse is
not required: a valid representation may have a positive scale and all-zero codes. The canonical
reference encoder nevertheless emits the unique `scale = +0, codes = 0` representation for an
all-zero source group.

### 6.2 Abstract reconstruction

Let `q[p, k]` be the signed code and `s[p, g]` the binary16 scale for group `g`. The represented
weight is:

```text
g         = floor(k / G)
s32       = exact_binary16_to_binary32(s[p, g])
w_hat[p,k] = binary32(s32 * binary32(q[p, k]))
```

This binary32 expression is the mathematical conformance oracle. Every legal signed code is exactly
representable in binary32, every binary16 value expands exactly to binary32, and their product is
the scheme's represented value. A kernel need not materialize that product or use binary32 at every
stage, but its operator-level result must satisfy the tolerance and rounding boundary of its compute
profile relative to this oracle.

The formula `code / scale` is wrong: the stored scale is a multiplier. An independent scheme decoder
or exact oracle that introduces a zero point, recenters the code interval, or converts the scale
through BF16 implements a different numerical contract. A production kernel may use lower-precision
internal staging only when its compute profile explicitly permits it and its observable operator
result passes that profile's tolerance.

### 6.3 Logical storage cost

For code width `b`, group size `G`, and one 16-bit scale per group, the ideal logical cost for full
groups is:

```text
logical_bits_per_weight = b + 16 / G
```

This produces the values in Section 1. It excludes partial-group overhead, tensor metadata, physical
padding, alignment, indexes, integrity data, and layout-specific duplication. It must not be quoted
as the exact `.ninfer` artifact size.

The apparently finer W8 group is intentional. Q4, Q5, and Q6 retain one shared G64 grouping model
across their existing converter, codec, and kernel paths, while W8G32 is the one current W8
geometry used by the registered Qwen3.6-27B recipe. The assignment of that geometry to model roles
belongs to checkpoint recipes, not to the scheme definition. Its 16-bit scale contributes 0.5 bit
per weight at G32, while the scale contributes 0.25 bit per weight to Q4 at G64; both are 6.25% of
their code payload. This cost calculation explains why the finer W8 grouping is affordable. It is
not independent quality evidence or a claim that G32 is universally optimal.

## 7. Canonical reference encoder

### 7.1 Purpose and input boundary

The canonical profile identity is `MAXABS_F16_RECIP_RNE_V1`. NInfer owns exactly one current
canonical profile for conversion consistency, converter parity, and artifact verification. It is a
per-row, per-group symmetric maximum-absolute-value encoder. It fixes the arithmetic order used by
the registered converter. Direct division during code selection is not interchangeable with its
reciprocal-multiply order at rounding boundaries.

The current implementation applies this profile to positive rank-two matrices, rejects non-finite
source values and unrepresentable scales, and defines physical tail padding through
`row-split-k128-v1`.

The checkpoint adapter supplies a logical tensor in the shape and axis convention of Section 4.
Each source value is converted to IEEE-754 binary32 using round-to-nearest, ties-to-even before group
processing. Finite BF16 and binary16 values convert exactly; finite binary32 values are unchanged.
If a source value cannot be represented as finite binary32, the encoder fails. NaN and infinity are
never quantized silently.

All binary32 and binary16 operations below use round-to-nearest, ties-to-even. No flush-to-zero is
permitted where it would change a specified binary16 scale. A bit-level host/software implementation
of this ordered algorithm is the encoder oracle. A production CPU or GPU converter may use faster
arithmetic only after parity with that oracle is established on the conformance boundaries in
Section 12.

### 7.2 Per-group algorithm

For one logical group of finite binary32 values `x[i]`, let `(qmin, qmax)` be the interval for the
selected scheme. The canonical result is:

```text
amax = max_i(abs(x[i]))                         # binary32 values

if amax == +0:
    scale16 = binary16(+0)
    code[i] = 0 for every i
else:
    raw_scale32 = round_f32(amax / binary32(qmax))
    scale16     = round_f16(raw_scale32)

    if scale16 == +0:
        scale16 = binary16_from_bits(0x0001)    # 2^-24 underflow rescue

    if scale16 is not finite and positive:
        fail

    scale32 = exact_binary16_to_binary32(scale16)
    inv32   = round_f32(binary32(1.0) / scale32)

    for every i:
        normalized32 = round_f32(x[i] * inv32)
        rounded       = round_to_integral_ties_to_even(normalized32)
        code[i]       = clamp(rounded, qmin, qmax)
```

`qmax`, not `abs(qmin)`, is the scale denominator. Thus Q4 uses 7, Q5 uses 15, Q6 uses 31, and W8
uses 127. The clamp happens after integral rounding. The W8 result therefore never emits `-128`.

The integral rounding examples are:

```text
+0.5 ->  0       -0.5 ->  0
+1.5 ->  2       -1.5 -> -2
+2.5 ->  2       -2.5 -> -2
```

The sequence `inv32 = round_f32(1 / scale32)` followed by
`normalized32 = round_f32(x * inv32)` is normative for this encoder. Replacing it with
`round_f32(x / scale32)` can select a different integer code even though the real-number
expressions are algebraically equivalent.

If conversion of `raw_scale32` to binary16 produces infinity, encoding fails rather than storing it
or silently clipping the scale. If a nonzero `raw_scale32` rounds to binary16 zero, the explicit
`0x0001` rescue preserves a positive dequantization scale. An all-zero group is handled before the
reciprocal and always produces positive-zero scale and zero codes.

### 7.3 Alternative encoders

The scheme identity is determined by the persistent representation and reconstruction contract,
not by how a converter found its codes. A documented upstream, calibrated, or error-optimized
encoder may emit the same four schemes if all output codes and scales satisfy Sections 4 through 6.

Such an encoder is not an implicit NInfer user option and cannot silently replace the canonical
reference path. Its exact recipe and provenance belong to the selected checkpoint's quantization
policy, and its cited upstream quality evidence applies only to that checkpoint and recipe. If it
needs a zero point, different group geometry, different scale type, codebook, shared exponent, or
different reconstruction equation, it is a different scheme and must pass the admission process in
Section 11.

Any change to the canonical algorithm that can alter a code or scale creates a new encoder-profile
version. It does not change the quantization scheme when the resulting representation still obeys
Sections 4 through 6, but it must be distinguishable in checkpoint provenance and must never replace
`MAXABS_F16_RECIP_RNE_V1` silently. NInfer still exposes only the one profile selected by each
current checkpoint recipe; profile versioning is provenance, not a user-facing quality menu.

## 8. Conformance responsibilities

### 8.1 Producer and encoder

A conforming producer must:

- emit only a registered numeric format;
- preserve each direct logical word exactly when no source-type conversion is requested;
- apply only a source-type conversion explicitly defined by the checkpoint recipe;
- for a quantized format, preserve the logical shape and last-axis group rule;
- for a quantized format, emit one valid scale per logical group and only legal signed codes,
  including never emitting W8 `-128`;
- record enough checkpoint-recipe provenance for the artifact producer to identify how the values
  were derived;
- for quantization, fail rather than silently encode non-finite source data or an unrepresentable
  scale.

Direct BF16 and FP32 formats can represent non-finite words, so their presence is not a producer
error by itself. Whether such a word is allowed in one tensor is a checkpoint-recipe constraint.
Physical padding is produced and validated by the selected storage layout, not by the logical direct
or quantization encoder.

Only an encoder claiming canonical-reference parity must reproduce Section 7 bit for bit. Another
approved checkpoint recipe may choose different valid scales and codes, but it cannot change their
meaning.

### 8.2 Container and layout

The `.ninfer` container and each registered storage layout must:

- identify the numeric format unambiguously;
- preserve logical shape separately from any physical padded extent;
- reconstruct every direct logical word exactly;
- for grouped formats, make the number and ownership of logical groups unambiguous and reconstruct
  every logical code and binary16 scale without inference from a kernel implementation;
- define its canonical physical-padding contents and producer responsibilities, if it materializes
  padding;
- reject unknown formats and unsupported format/layout combinations;
- define the structural checks needed to locate and decode the payload.

The container must not embed an open-ended `(bits, group_size, scale_dtype)` constructor that makes
unregistered combinations valid. Its representation resolves to one closed canonical identity;
that representation belongs to the container contract.

### 8.3 Verifier, loader, and binder

The project-owned converter and quantizer establish the value invariants in Sections 3 through 7
while writing the artifact: direct-word preservation, legal codes and scales, and canonical physical
padding. The layout codec preserves those already selected words. The offline checkpoint verifier
checks the complete target inventory and representative source-to-artifact values.

The runtime reader establishes numeric-format identity, layout compatibility, logical and physical
extents, encoded size, alignment, and payload bounds. The model binder then establishes the exact
format/layout/consumer/shape/compute-profile/GPU whitelist and any cheap role-specific invariant it
needs directly. The generic parser does not own that target support matrix.

The runtime intentionally does not scan every code, scale, direct word, or padding byte of a locally
generated artifact before upload. Consumers may rely on the registered producer's value contract;
codec and operator tests independently protect the representation and numerical interpretation.

### 8.4 Kernel and operator

A consuming kernel or model component must interpret direct logical words according to Section 3 or
quantized codes and scales according to Sections 5 and 6. It may fuse, reorder, or approximate
intermediate work only as allowed by its operator compute profile. Kernel implementation details do
not alter the persistent format and must not be needed to decode an artifact independently.

There is no mandatory generic unpack-to-dense fallback. If NInfer has not implemented the exact
combination required by a selected target, conversion or loading fails explicitly rather than
changing formats or selecting a slower compatibility path. Whether a container/runtime permits
load-time persistent repacking is outside this numeric-format contract and cannot change the numeric
meanings defined here.

## 9. Checkpoint and model boundary

This registry does not say where any of the seven formats are used. A checkpoint numeric-format
document must separately define, for every persisted source tensor or derived tensor:

- its source checkpoint identity and source tensor or derivation;
- its logical shape and axis interpretation;
- its selected registered direct format or quantization scheme;
- any source-type conversion and, for quantization, its encoder-profile provenance;
- any fusion, concatenation, permutation, flattening, or transpose performed before encoding;
- the cited upstream/public evidence and engineering rationale used to select the complete lossy
  checkpoint recipe.

Those facts may differ between exact checkpoints in the same family. Dense attention, MoE experts,
MTP predictors, vision towers, embeddings, and output heads receive no automatic assignment from
their role names. Section 4 defines how a higher-rank expert bank would be grouped numerically, but
the current row-split storage implementation remains rank two; a future MoE recipe must explicitly
register its physical layout or define the reshape used by its compiled target.

Similarly, a format's presence in a `.ninfer` artifact does not prove runtime support. The selected
model implementation must bind that tensor to a supported model consumer or operator path on the
exact GPU target.

## 10. Explicit exclusions from the first registry

The first registry contains no implicit or reserved support for:

- other direct scalar types, including `FP16`, `FP64`, signed widths other than `I32`, and unsigned
  integers;
- other integer widths, including Q2 and Q3;
- alternate group sizes for the four accepted code widths;
- asymmetric or affine quantization with zero points;
- per-channel schemes disguised as an arbitrary group size;
- codebook formats such as NF4;
- GGUF K-quant, I-quant, or block layouts as scheme aliases;
- GPTQ or AWQ serialization dialects as scheme aliases;
- any persistent FP8 weight format, including per-tensor, per-row, per-channel, and block-scaled
  variants;
- MXFP4, NVFP4, or other microscaling/shared-exponent formats;
- activation, KV-cache, or recurrent-state quantization.

These are exclusions, not judgments that the methods are poor. They have materially different
numeric or storage contracts, or no demonstrated need in a selected NInfer target. Their existence
in llama.cpp, TensorRT-LLM, vLLM, a model release, or a hardware library is research evidence only;
it does not create a NInfer registry entry, loader branch, conversion option, or kernel obligation.

Binary16 remains the scale component denoted by `F16S`; that use does not imply a direct `FP16`
format. The internal execution names `BF16_CTRL`, `FP32_CTRL`, and `I32_CTRL` are not persistent
registry identities.

## 11. Admission and retirement

### 11.1 Admission trigger

A new numeric format is considered only when an accepted, measurable requirement of a real selected
checkpoint and GPU target cannot be met by the current registry and the candidate has a concrete
benefit. Cartesian completeness, upstream popularity, theoretical elegance, or a desire to reserve
future IDs is insufficient.

The proposal must provide:

- the exact checkpoint tensors that need the format;
- the exact logical-word or code/scale representation and reconstruction oracle;
- any required source conversion or a defined quantization encoder;
- for a lossy format, cited credible upstream/public evidence or implementation precedent supporting
  acceptable quality for that checkpoint;
- the intended persistent layout and validation strategy;
- useful operator and kernel paths for the selected GPU;
- measured local value in model residency, verified context, decode, or prefill, together with the
  cited upstream basis for acceptable quality;
- an explanation of why an existing format cannot provide that value and, if the proposal is
  additive, the unique active tensor role that prevents replacement of an existing format;
- the ongoing converter, verifier, loader, kernel, and documentation cost.

NInfer relies on credible publisher, upstream, or public quality evaluation to select a lossy
candidate; independent model-quality qualification is not a local admission deliverable. NInfer
must still verify the exact artifact's structural correctness, codec reconstruction, operator
numerics, runtime behavior, memory use, and end-to-end performance on its own target.

### 11.2 Review outcome

Admission is an explicit project decision. If accepted, the format receives a new canonical name
whose semantics are fully defined before container and consumer support are merged. No existing name
may be silently reinterpreted.

NInfer prefers replacement to permanent accumulation. A format is retired when a new format
supersedes it for all selected targets, when no active checkpoint recipe uses it, when its sole
target is removed, or when its maintenance cost is no longer justified by an active role. The old
converter, loader, dispatch, kernels, and active requirements are removed together after affected
artifacts are regenerated. The project does not keep aliases, fallback readers, deprecated options,
or compatibility shims solely to preserve a project-owned historical format.

Removing implementation support does not permit old artifacts to be decoded under a different
meaning. A retired canonical name is not reused or reinterpreted for another meaning. Historical
evidence may retain the old definition, while the current registry lists only what the current
NInfer product owns.

## 12. Required conformance evidence

Implementation of this document is protected at the representation boundary, not by tests that scan
enum spellings or private kernel layout. The retained codec and encoder evidence covers:

- exact representative BF16, FP32, and I32 word round trips, including signed zeros, subnormals, NaN
  payload bits, and integer extrema, plus rejection of implicit cross-type encoding;
- Q4, Q5, Q6, and W8 plane bit order, legal interval endpoints, encoded-size geometry, partial-K zero
  padding, consecutive row views, and arbitrary row gathers;
- canonical binary16 scale rounding, reciprocal-multiply rather than direct division, positive and
  negative ties-to-even, minimum-subnormal rescue, and rejection of non-finite or overflowing source
  groups;
- canonical quantization followed by exact stored code/scale decode for a partially populated group.

For direct formats, the independent decode oracle is the abstract logical word in Section 3. For
grouped formats, it is the binary32 reconstruction in Section 6.2. The canonical quantization encode
oracle is a bit-level host/software implementation of the ordered algorithm in Section 7; production
converters require parity against it rather than defining the oracle through their own arithmetic.
Numerical operator tests separately protect the combinations used by the registered target, including
their activation rounding boundary, accumulation precision, output tolerance, and real target shapes.

## 13. Consequences for container and model design

This decision leaves directory, metadata encoding, integrity, sharding, and physical-layout choices
to the container and layout contracts. This numeric-format decision does not leave the following
questions open:

- persistent low-bit weights use only the four registered scheme identities until an explicit
  admission changes the registry;
- direct persistent tensors use only `BF16`, `FP32`, and `I32`, with the exact logical words in
  Section 3;
- groups run along the final logical dimension and never cross a leading coordinate;
- scales are one finite nonnegative binary16 multiplier per group;
- codes and reconstruction have the exact meanings in Sections 5 and 6;
- the canonical `MAXABS_F16_RECIP_RNE_V1` encoder uses FP16-rounded scale followed by binary32
  reciprocal-multiply;
- the container, checkpoint recipe, compute profile, and runtime-state codecs remain separate
  contracts.

The `.ninfer` container specification references these definitions rather than restating or
specializing them for one model. A model-specific numeric recipe may select and combine the formats,
but it cannot redefine them. This is the boundary that allows NInfer to add deeply optimized
checkpoints without turning the core tensor representation into either a Qwen3.6-27B artifact or an
unbounded compatibility framework.
