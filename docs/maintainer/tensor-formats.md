# NInfer Persistent Tensor Numeric Formats

This reference defines the nine persistent numeric tensor formats accepted by current `.ninfer`
artifacts: their logical words, quantization semantics, canonical reference encoders where
applicable, and conformance boundaries. [Container framing](artifact-container.md),
[physical layouts](storage-layouts.md), weight recipes and runtime-state codecs are defined
separately.

## 1. Registered formats

NInfer has exactly nine persistent numeric tensor formats in four categories.

Direct scalar formats preserve one logical scalar word per tensor element:

| Canonical name | Category | Logical width | Logical meaning |
|---|---|---:|---|
| `bf16` | direct floating point | 16 | bfloat16 bit encoding |
| `fp32` | direct floating point | 32 | IEEE-754 binary32 |
| `int32` | direct signed integer | 32 | 32-bit two's-complement integer |

Grouped quantized-weight formats preserve signed codes plus one scale per logical group:

| Canonical name | Code width | Group size | Legal signed codes | Scale | Full-group logical bits/weight |
|---|---:|---:|---:|---|---:|
| `q4_g64_fp16` | 4 | 64 | `[-8, 7]` | one binary16 scale/group | 4.25 |
| `q5_g64_fp16` | 5 | 64 | `[-16, 15]` | one binary16 scale/group | 5.25 |
| `q6_g64_fp16` | 6 | 64 | `[-32, 31]` | one binary16 scale/group | 6.25 |
| `q8_g32_fp16` | 8 | 32 | `[-127, 127]` | one binary16 scale/group | 8.50 |

The block-scaled floating-point weight format is:

| Canonical name | Code | K group | Block scale | Global field |
|---|---|---:|---|---|
| `nvfp4` | E2M1, 4 bits/weight | 16 | one E4M3FN word/group | one positive FP32 weight divisor |

The row-scaled floating-point weight format is:

| Canonical name | Code | Scale granularity | Scale |
|---|---|---|---|
| `fp8_e4m3fn_row_bf16` | E4M3FN, 8 bits/weight | one multiplier per logical row | BF16 |

Each name fixes a code and scale contract. The format registry is implemented in
[`tools/artifact/formats.py`](../../tools/artifact/formats.py) and
[`src/artifact/formats.cpp`](../../src/artifact/formats.cpp). Additional formats need an explicit
numeric definition and codec implementation. Layout and Op support are separate: a known format
can be stored without every model consumer supporting it.

## 2. Terms and ownership

The registry keeps the following concerns separate.

### 2.1 Persistent numeric format

A **persistent numeric format** defines the logical words needed to recover a numeric tensor from
an artifact. The closed registry contains direct scalar formats, grouped signed-integer formats,
the block-scaled `nvfp4` format, and the row-scaled `fp8_e4m3fn_row_bf16` format. It does not
identify a tensor's model role, physical byte layout, or supported consumer.

### 2.2 Direct scalar format

A **direct scalar format** assigns one fixed-width logical word to every logical tensor coordinate.
It defines the value of that word without a group, scale, zero point, or reconstruction step. `bf16`,
`fp32`, and `int32` are direct scalar formats.

The adjective “direct” does not require a particular physical layout or prohibit a registered
lossless storage transformation. It means only that layout decoding recovers the original logical
word rather than a quantized approximation of another word.

### 2.3 Quantization scheme

A **quantization scheme** defines only the persistent logical representation of a quantized weight:

- the code domain;
- the group axis and group size;
- the scale type and scale granularity;
- the validity rules for codes and scales;
- the mathematical reconstruction of each represented weight.

The six quantized names above identify schemes in this sense. Their meanings are immutable: a
consumer must not infer a different zero point, scale geometry, code range, or reconstruction rule
from context.

### 2.4 Conversion method

A **conversion method** produces the words of a selected format from a logical source. Scale
selection, rounding order, calibration, clipping, and error optimization belong here. Methods
may preserve an already encoded source or quantize floating-point values.

The built-in `grouped_absmax` method implements the reference encoder in Section 7 for all four
grouped integer formats. `fp8_row_maxabs` rounds source values to BF16 and quantizes each row to
E4M3FN codes with a BF16 multiplier. `import_encoded` preserves compatible FP8 or NVFP4 codes,
scales, and, for NVFP4, the matrix weight divisor. NInfer currently provides no built-in
floating-point-to-NVFP4 quantizer.

A recipe can supply a Python callable as its method. Different methods can produce different
valid codes and scales for the same format; they share the format's decoding contract. Method
identity and numerical parameters belong to conversion provenance.

Direct formats also separate representation from conversion. For example, `bf16` defines stored
bits; the conversion method decides how an FP32 source becomes those bits.

### 2.5 Weight recipe

A **weight recipe** selects the format and conversion method for each logical parameter or
region. Parameter meaning and shape come from the architecture. Source selection, grouping and
per-input activation permissions are described in the [conversion guide](../weight-conversion.md).

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
`contiguous_le_v1` for direct words, `row_split_k128_v1` for grouped signed-integer formats, and
`block_scale_k16_m128x4_v1` for `nvfp4`, and `row_scale_v1` for
`fp8_e4m3fn_row_bf16`. Their byte order, plane packing, padding, swizzle, divisor placement, and
alignment rules belong to the layout registry, not to these nine numeric formats.

### 2.7 Compute profile and kernel support

A **compute profile** owns observable input and output types, explicitly semantic casts or
quantization boundaries, determinism requirements, and operator-level numerical tolerances. Kernel
operand and accumulator precision, tiling, Tensor Core use, internal reduction trees, private
staging or materialization, split strategy, fusion schedule, launch geometry, and hardware dispatch
remain implementation choices. Exact or bitwise operations may require an exact observable result;
that requirement does not turn a floating-point oracle into a prescribed internal evaluation order.

In `q8_g32_fp16`, the `fp16` suffix describes stored scales. Activation and output precision are
chosen by the consumer. A kernel supports specific combinations of numeric format, storage layout,
operator semantics, shape, compute profile, and execution platform.
Native Op implementations define these combinations and check them when preparing or consuming
operands. A recipe's ability to encode a combination is independent of native execution support.

### 2.8 Runtime-state codec

KV-cache quantization, activation quantization, temporary kernel compression, recurrent state, and
communication formats are runtime-state codecs. They are outside this persistent-tensor registry
even if they also use signed integers and grouped scales. In particular, an INT8 KV-cache format
must not be labeled `q8_g32_fp16` merely because some of its fields look similar.

## 3. Canonical identities and format semantics

### 3.1 Direct scalar formats

#### `bf16`

`bf16` is the bfloat16 bit encoding. Its abstract 16-bit word has one sign bit at bit 15, an 8-bit
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
a binary32 word and appends 16 zero low bits. `bf16` is not IEEE binary16/FP16.

All 65,536 logical BF16 words are valid at the format layer. Positive and negative zero remain
distinct words. Infinity and NaN are also representable, and a direct layout must recover their
original logical words without canonicalizing a NaN payload or changing its quiet/signaling bit.
Whether a particular model tensor permits non-finite values is a parameter or conversion constraint, not
a different BF16 format. A compute operation is not required to preserve a NaN payload unless its
compute profile says so.

#### `fp32`

`fp32` is IEEE-754 binary32: one sign bit at bit 31, an 8-bit exponent at bits 30 through 23 with
bias 127, and a 23-bit trailing fraction at bits 22 through 0. Its zeros, subnormals, normal values,
infinities, and NaNs have the standard binary32 interpretation; the minimum positive subnormal is
`2^-149`.

All 2^32 logical words are valid at the format layer. As with BF16, signed zeros, infinities, and
NaN payloads remain distinct persistent words. Tensor-specific restrictions on non-finite values
belong to the recipe, while compute-time propagation belongs to the compute profile.

#### `int32`

`int32` is a 32-bit two's-complement signed integer. For an abstract unsigned logical word `u`:

```text
value = u                         if u < 2^31
value = u - 2^32                  otherwise
```

Every 32-bit word is valid, and the value interval is `[-2147483648, 2147483647]`. `int32` has no
scale, zero point, saturation behavior, sentinel convention, or dependency on the host C++ `int`
type. Requirements such as nonnegativity, vocabulary bounds, or the meaning of `-1` belong to the
specific tensor role.

#### Common direct-format rules

For a direct tensor, every logical coordinate owns one independent word of the selected format.
Direct formats are shape-agnostic: rank, dimension validity, coordinate traversal, strides,
padding, and alignment are defined by the container and selected layout, not by the scalar format.
Byte order is likewise a layout property and must never be inferred from host-native representation.

Persistence from the same source type is bitwise identity at the logical-word boundary. A conversion
between different source and target types is not implicit. The recipe must define it,
including rounding for FP32-to-BF16 and the value conversion and range policy for INT32.

Control parameters, ordinary weights, norms, indexes, and maps use a direct format plus a separate
logical parameter role.

### 3.2 Grouped signed-integer weight identities

The identifiers record code width, group size and scale type. For example, `q4_g64_fp16` uses
4-bit signed codes and one IEEE-754 binary16 scale per group of 64 weights. `q5_g64_fp16`,
`q6_g64_fp16`, and `q8_g32_fp16` follow the same naming convention.

The parser resolves names through the format registry. `Q4`, `Q5`, `Q6`, and `Q8` are prose
abbreviations; artifacts store the complete canonical names.

### 3.3 `nvfp4`

`nvfp4` is a block-scaled floating-point weight representation, not a signed-integer
`QuantFormat`. For a logical matrix `[N,K]`, every K-axis group contains 16 E2M1 code words and one
E4M3FN scale word. The representation also contains one FP32 serialized weight divisor `d_w` for
the complete matrix.

An E2M1 word has sign bit 3, exponent bits 2:1, and mantissa bit 0. Positive code words `0..7`
decode to:

```text
0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```

Bit 3 negates that value. Words `0x0` and `0x8` are distinct positive and negative zero words; all
16 code words are valid and must remain bit-exact.

An E4M3FN word has sign bit 7, exponent `e` in bits 6:3, fraction `m` in bits 2:0, and bias 7:

```text
e == 0, m == 0 : signed zero
e == 0, m != 0 : (-1)^sign * m * 2^-9
1 <= e <= 14   : (-1)^sign * (1 + m/8) * 2^(e-7)
e == 15, m < 7 : (-1)^sign * (1 + m/8) * 2^8
e == 15, m == 7: NaN
```

Stored NVFP4 weight scales admit only sign-zero finite words, including positive zero. Negative
values, negative zero, and both NaN words are invalid. The serialized binary32 word `d_w` must be
finite and strictly positive.

For code `c[n,k]`, scale word `s[n,g]`, and `g=floor(k/16)`, the exact represented weight is:

```text
W[n,k] = decode_e2m1(c[n,k]) * decode_e4m3fn(s[n,g]) / d_w
```

`import_encoded` copies all three fields without requantizing or canonicalizing them. Activation
calibration is not part of this weight format. In particular, a
site-level input divisor used by an NVFP4 execution path is a separate model-role tensor and cannot
be inferred from `nvfp4`, its block scales, or `d_w`.

### 3.4 `fp8_e4m3fn_row_bf16`

`fp8_e4m3fn_row_bf16` is a rank-two weight matrix `[N,K]` with positive dimensions. Every logical
weight owns one E4M3FN code word, and every logical row owns one BF16 dequantization multiplier. The
row is the first matrix coordinate `n`; “row” is structural and does not infer a model-specific
channel role.

The E4M3FN code meaning is the one defined in Section 3.3. Both signed-zero words and every finite
subnormal or normal word are valid. The positive and negative NaN words `0x7f` and `0xff` are
invalid; E4M3FN has no infinity words.

The BF16 scale meaning is the one defined in Section 3.1. A valid row scale has sign bit zero and is
finite, including positive zero and positive subnormals. Negative values, negative zero, infinity,
and NaN are invalid. If a row scale is positive zero, every code in that row must be either positive
zero `0x00` or negative zero `0x80`. The converse is not required.

For code word `c[n,k]` and row-scale word `s[n]`, the exact represented weight is:

```text
c32          = exact_e4m3fn_to_binary32(c[n,k])
s32          = exact_bfloat16_to_binary32(s[n])
w_hat[n,k]   = binary32(c32 * s32)
```

The scale is a multiplier. Division by `s[n]`, a zero point, a matrix-level divisor, or another
implicit coefficient implements a different format. The code plane and row-scale plane together
form one persistent weight; neither a bare E4M3FN tensor nor an independently named scale tensor is
an alias for this format.

The format does not define how a floating-point source is assigned a scale or rounded to E4M3FN.
A recipe either preserves already selected code and scale words exactly or names its
conversion method. Activation quantization and activation scales are separate compute or runtime-state
concerns and are not persistent fields of this format.

## 4. Grouped signed-integer tensor model

### 4.1 Shape and group axis

For the four grouped signed-integer schemes, let a logical quantized weight tensor have rank
`r >= 1`, a positive `K`, and shape:

```text
[D0, D1, ..., D(r-2), K]
```

For `r = 1`, this notation reduces to `[K]` with no leading coordinate.

The last dimension `K` is always the quantization axis. Every coordinate in the leading dimensions
identifies an independent logical row. For a leading coordinate vector `p` and element index `k`:

```text
group(p, k) = (p, floor(k / G))
```

where `G` is 64 for Q4/Q5/Q6 and 32 for Q8. A group never crosses a row boundary or any leading
dimension. For a rank-3 expert bank `[E, N, K]`, for example, each `(expert, row)` pair is quantized
independently; a group cannot cross from one expert to another.

The logical scale tensor therefore has shape:

```text
[D0, D1, ..., D(r-2), ceil_div(K, G)]
```

This definition does not assign meanings such as output channel, expert, adapter, or convolution
axis to the leading dimensions. A checkpoint adapter owns any reshape or transpose needed to place
the intended quantization axis last before encoding. It is an abstract scheme definition. The
currently implemented `row_split_k128_v1` layout and canonical producer accept only positive rank-two
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

## 5. Grouped signed-integer code domains

All four schemes are zero-point-free, weight-only signed-integer schemes. “Symmetric” in this
document means that reconstruction is `scale * signed_code` with zero point zero. It does not mean
that every legal interval has equal positive and negative magnitude.

### 5.1 Q4, Q5, and Q6

The logical code is a two's-complement signed integer of the stated width:

| Scheme | Width | Sign bit | Legal code interval | `qmax` used by reference encoder |
|---|---:|---:|---:|---:|
| `q4_g64_fp16` | 4 | `0x8` | `[-8, 7]` | 7 |
| `q5_g64_fp16` | 5 | `0x10` | `[-16, 15]` | 15 |
| `q6_g64_fp16` | 6 | `0x20` | `[-32, 31]` | 31 |

Every bit pattern of the stated logical width is valid. Conversion between an unsigned logical word
`u` and signed code `q` is:

```text
q = u                         if u < 2^(b - 1)
q = u - 2^b                   otherwise
```

This is not offset binary, sign-magnitude encoding, a codebook index, or an unsigned GPTQ/AWQ
zero-point convention. A storage layout may split or reorder the bits, but after layout decoding the
logical word and signed value must be exactly those defined here.

### 5.2 Q8

`q8_g32_fp16` uses an 8-bit two's-complement signed code with the deliberately restricted interval
`[-127, 127]`. The byte pattern `0x80`, which would represent `-128`, is outside the valid artifact
language. The project-owned quantizer must never emit it; decoding semantics are defined only for a
valid code stream.

This restriction is part of the scheme, not an encoder preference. A kernel may use a native
signed-byte load because the registered conversion path establishes the code invariant; that
implementation convenience does not make `-128` legal. The trusted local runtime does not rescan
the complete Q8 payload solely to prove an invariant already established by its producer.

## 6. Grouped signed-integer scale and reconstruction semantics

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
stage; its operator-level result is qualified against this oracle with the Op's named criterion for
that implementation profile.

The formula `code / scale` is wrong: the stored scale is a multiplier. An independent scheme decoder
or exact oracle that introduces a zero point, recenters the code interval, or converts the scale
through BF16 implements a different numerical contract. A production kernel may use lower-precision
internal staging, different accumulator precision, or another reduction association; these are
implementation choices, and the observable operator result must pass the Op's named criterion
against the same oracle.

### 6.3 Logical storage cost

For code width `b`, group size `G`, and one 16-bit scale per group, the ideal logical cost for full
groups is:

```text
logical_bits_per_weight = b + 16 / G
```

This produces the values in Section 1. It excludes partial-group overhead, tensor metadata, physical
padding, alignment, indexes, integrity data, and layout-specific duplication. It must not be quoted
as the exact `.ninfer` artifact size.

The 16-bit scale contributes 0.5 bit per weight at G32 and 0.25 bit per weight at G64. The choice
of format for a model parameter belongs to its recipe.

## 7. Grouped signed-integer reference encoder

### 7.1 Purpose and input boundary

The built-in `grouped_absmax` method uses per-row, per-group maximum absolute values. Its
implementation is in
[`groupwise.py`](../../tools/convert/quantization/groupwise.py). The arithmetic below defines its
exact output, including FP16 scale rounding and reciprocal-multiply code selection. It applies to
the four grouped integer formats.

The implementation accepts positive rank-two matrices, rejects non-finite source values and
unrepresentable scales, and supplies zero tail padding for `row_split_k128_v1`.

The logical source supplies values in the shape and axis convention of Section 4. Each source value
is converted to IEEE-754 binary32 using round-to-nearest, ties-to-even before group processing.
Finite BF16 and binary16 values convert exactly; finite binary32 values are unchanged. If a source
value cannot be represented as finite binary32, the encoder fails.

All binary32 and binary16 operations below use round-to-nearest, ties-to-even. No flush-to-zero is
permitted where it would change a specified binary16 scale. An independent implementation of this
ordered algorithm is the encoder oracle; Section 9 describes the relevant conformance evidence.

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

`qmax`, not `abs(qmin)`, is the scale denominator. Thus Q4 uses 7, Q5 uses 15, Q6 uses 31, and Q8
uses 127. The clamp happens after integral rounding. The Q8 result therefore never emits `-128`.

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

User recipes select these methods explicitly and record their provenance. An alternative method
need not match `grouped_absmax` bit for bit. It must produce valid words of the selected format.
A change to group geometry, scale type, code domain or reconstruction equation requires a different
numeric format, with its own codec and consumer support.

## 8. Conformance responsibilities

### 8.1 Producer and encoder

A conforming producer must:

- emit only a registered numeric format;
- preserve each direct logical word exactly when no source-type conversion is requested;
- apply only a source-type conversion explicitly defined by the recipe;
- for a quantized format, preserve the logical shape and last-axis group rule;
- for a grouped signed-integer format, emit one valid binary16 scale per logical group and only
  legal signed codes, including never emitting Q8 `-128`;
- for `nvfp4`, emit only valid E2M1 code words, nonnegative finite E4M3FN scale words, and one finite
  positive FP32 weight divisor under Section 3.3;
- for `fp8_e4m3fn_row_bf16`, emit only finite E4M3FN code words and valid BF16 row multipliers,
  with signed-zero codes as the only legal codes in a positive-zero-scale row under Section 3.4;
- record enough conversion provenance for the artifact producer to identify how the values
  were derived;
- when an encoder converts floating-point source values, fail rather than silently quantize
  non-finite source data or emit an unrepresentable scale.

Direct BF16 and FP32 formats can represent non-finite words, so their presence is not a producer
error by itself. Whether such a word is allowed in one tensor is a parameter or conversion constraint.
Physical padding is produced and validated by the selected storage layout, not by the logical direct
or quantization encoder.

Only a grouped signed-integer encoder claiming canonical-reference parity must reproduce Section 7
bit for bit. Another recipe may choose different valid words for any
quantization scheme, but it cannot change their registered meaning.

### 8.2 Container and layout

The `.ninfer` container and each registered storage layout must:

- identify the numeric format unambiguously;
- preserve logical shape separately from any physical padded extent;
- reconstruct every direct logical word exactly;
- for grouped signed-integer formats, make the number and ownership of logical groups unambiguous
  and reconstruct every signed code and binary16 scale without inference from a kernel
  implementation;
- for `nvfp4`, reconstruct every E2M1 code word, natural E4M3FN scale word, and the matrix FP32
  divisor under Section 3.3;
- for `fp8_e4m3fn_row_bf16`, reconstruct every E4M3FN code word and its owning BF16 row multiplier
  under Section 3.4;
- define its canonical physical-padding contents and producer responsibilities, if it materializes
  padding;
- reject unknown formats and unsupported format/layout combinations;
- define the structural checks needed to locate and decode the payload.

The container must not embed an open-ended `(bits, group_size, scale_dtype)` constructor that makes
unregistered combinations valid. Its representation resolves to one closed canonical identity;
that representation belongs to the container contract.

### 8.3 Reader, loader, and binder

Producers establish the numeric invariants above. Layout codecs preserve the selected words and
supply canonical physical padding. The generic reader validates directory structure and payload
bounds; when an object is selected, it resolves its format/layout and checks encoded geometry and
size. Unused optional components do not require decoding their weights.

The semantic binder checks parameter roles, shapes, views and Uses. It reads cheap role-specific
values such as indexes and activation divisors where necessary. Native Op preparation, resource
queries, warmup or execution check actual operand support. There is no separate whole-model
capability registry.

The runtime uploads encoded payloads and keeps their physical representation. It does not scan
every code, scale or padding byte before upload. Codec and Op tests independently protect the
producer contract and the represented values.

### 8.4 Kernel and operator

A consuming kernel or model component must interpret direct logical words according to Section 3.1,
grouped signed-integer identities, codes, and scales according to Section 3.2 and Sections 5 and 6,
`nvfp4` words and divisor according to Section 3.3, and row-scaled FP8 words according to Section
3.4. It may choose its private fusion, reduction, staging, and intermediate precision; the
observable Op result is qualified against the independent oracle with the Op's named criterion for
that implementation profile. Kernel implementation details do not alter the persistent format and
must not be needed to decode an artifact independently.

An unsupported native combination fails at its consumer. Loading does not repack or requantize
weights to obtain another execution path.

## 9. Conformance evidence

Implementation of this document is protected at the representation boundary, not by tests that scan
enum spellings or private kernel layout. The retained codec and encoder evidence covers:

- exact representative BF16, FP32, and I32 word round trips, including signed zeros, subnormals, NaN
  payload bits, and integer extrema, plus rejection of implicit cross-type encoding;
- Q4, Q5, Q6, and Q8 plane bit order, legal interval endpoints, encoded-size geometry, partial-K zero
  padding, consecutive row views, and arbitrary row gathers;
- all 16 E2M1 words, all 256 E4M3FN words, NVFP4 scale/divisor validity, the exact divisor-based
  reconstruction equation, and known block-scale swizzle offsets;
- finite E4M3FN weight-code validity, BF16 row-scale validity, signed-zero rows, exact code/scale
  plane round trips, and the row-multiplier reconstruction equation for
  `fp8_e4m3fn_row_bf16`;
- canonical binary16 scale rounding, reciprocal-multiply rather than direct division, positive and
  negative ties-to-even, minimum-subnormal rescue, and rejection of non-finite or overflowing source
  groups;
- canonical quantization followed by exact stored code/scale decode for a partially populated group.

For direct formats, the independent decode oracle is the abstract logical word in Section 3.1. For
grouped signed-integer formats, it is the binary32 reconstruction in Section 6.2. Their canonical
encode oracle is a bit-level host/software implementation of the ordered algorithm in Section 7;
production converters claiming that profile require parity against it rather than defining the
oracle through their own arithmetic. The `nvfp4` decode oracle is the E2M1/E4M3FN/FP32 divisor
reconstruction in Section 3.3; its current producer is protected by exact source-word comparison
rather than Section 7. An NVFP4 Op oracle starts from the represented public activation and this
exact-decoded persistent weight. A site-level activation divisor and any private activation
quantization do not alter the ideal Op formula; their numerical effects are covered by the
production route's output criterion rather than reproduced inside the oracle. Numerical operator
tests separately protect the combinations used by the registered target, including their public
input/output formats, output tolerance, and real target shapes. The
`fp8_e4m3fn_row_bf16` decode oracle is the E4M3FN/BF16 row-multiplier reconstruction in Section
3.4; a producer that copies an upstream quantized tensor is protected by exact source-word
comparison, while any source-to-FP8 encoder is protected under its own named profile. Private
activation quantization, staging, and accumulation remain implementation choices.
