# Weight conversion

NInfer's converter creates `.ninfer` artifacts from local weights and a Python recipe. A recipe
can reuse an official conversion, change selected layers or projections, combine sources, or call
your own conversion method. The artifact contains the resulting configuration, encoded weights,
logical bindings and frontend resources.

Run the commands below from the repository root.

## Upgrade an existing v2 artifact

The offline upgrade tool supports the official Qwen3.6/3.8-27B groupwise-int and NVFP4 artifacts,
and Qwen3.6-35B-A3B groupwise-int. Update your checkout to the current `master` and
[rebuild NInfer](../README.md#quick-start), then run with Python 3.11:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```

The output must use a new path. After upgrading, use it directly or rename it to replace the
original file. Stored weight values and formats are preserved. The upgrade also installs the
matching template from `tools/chat_templates/`. Published SHA-256 checksums apply only to
downloaded files.

## Start with an official recipe

Source-weight conversion requires a Python 3.11 environment with PyTorch and NumPy. It uses CUDA
by default; `--device cpu` selects CPU conversion. The input paths below are placeholders for your
local checkpoint directories.

For Qwen3.6-27B floating-point source weights:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```

`--components` defaults to `text`. Include only the optional components you want to distribute.
`--proposal` adds the architecture's optimized proposal head. Qwen3.5-based models use the
repository's token ranking and default to a 131,072-row shortlist; Qwen4Exp keeps the complete
vocabulary and quantizes an additional head. The ordinary target output head is retained.

The built-in recipes are ordinary Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py):

| Recipe | Main representation choices | Additional source |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary weights | None |
| `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary weights | None |
| `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection weights | None |
| `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary weights | `quantized` |
| `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16 | `quantized` |
| `qwen3_8_flash_next` | Q8 main projections, NVIDIA NVFP4 expert banks, NVIDIA block-FP8 MTP experts, paged NF4 PLE, BF16 auxiliary projections/Vision | `nvidia` |

These names select conversion choices. Runtime execution is selected from the architecture,
configuration and actual bindings stored in the artifact. `--name` sets the public model name;
it does not select kernels.

For a Qwen3.8-27B NVFP4/FP8 artifact with DFlash2:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```

MTP and Vision use the main source. DFlash and DFlash2 use the corresponding named source, supplied
as `--source dflash=PATH` or `--source dflash2=PATH`. An artifact may contain several optional
components; the Engine loads only the ones selected at startup, including at most one speculative
backend. Component availability and startup selection are independent.

For Qwen3.8-Flash-Next, select the complete NVIDIA checkpoint for Main experts and MTP:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-Flash-Next/base-hf-bf16 \
  --recipe qwen3_8_flash_next \
  --source nvidia=/path/to/Qwen3.8-Flash-Next/nvidia-nvfp4 \
  --components text,vision,mtp \
  --proposal \
  --name qwen3.8-flash-next \
  --out out/qwen3_8_flash_next.ninfer
```

The Qwen4Exp adapter selects the dedicated
[`qwen3_8_flash_next.jinja`](../tools/chat_templates/qwen3_8_flash_next.jinja) by default;
`--resource chat_template.jinja=PATH` explicitly replaces it. The 27B template is independent.

The Qwen4Exp adapter and Engine support Text, Vision and native MTP. For this architecture,
`--proposal` converts an additional full-vocabulary Q4 head from the BF16 main head, with
`A16Only` activation policy. It does not read the ranking or use `--proposal-rows`.
Select it at startup with `--spec mtp --draft-tokens 3 --proposal-head optimized`.
The default `--proposal-head full` shares the main BF16 head and leaves the Q4 object unloaded.
DFlash/DFlash2 and indexed proposal execution are unsupported for Qwen4Exp.

CUDA self-quantization compiles the encoder for the selected CUDA build architecture on first use
and requires a CUDA toolkit, C++ compiler and Ninja. PLE uses two passes per shard, with a bounded
source cache and no full-table materialization. The complete recipe including the optimized head
contains about 100.98 GiB of weight/control payload, split by the V3 writer.
Main experts use two NVFP4 banks per layer with independent matrix divisors. MTP uses two
128×128 block-FP8 banks, preserving NVIDIA E4M3FN codes and BF16 multipliers; its other private
parameters also come from the NVIDIA source. PLE alone uses the BF16-to-NF4 stored-scale method.
See the [model and weight contract](maintainer/qwen4_exp-model.md) for layouts and precision, and
the [execution reference](maintainer/qwen4_exp-execution.md) for runtime support and memory use.

### Conversion resources

Built-in methods size batches by row width to target about 16 MiB of working input or output,
with alignment where the encoding requires it. One reader prefetches the next source batch while
the current batch is converted and written. CUDA conversion performs grouped-integer and
row-FP8 quantization on the GPU; encoded NVFP4/block-FP8 imports retain their original codes/scales.

| Option | Default | Behavior |
|---|---:|---|
| `--rows-per-chunk` | Automatic | Override row batching; methods align rows to their encoding boundaries. |
| `--cpu-threads` | `1` | Set PyTorch CPU intra-op threads; the source reader is a separate worker. |
| `--source-cache-mib` | `2048` | Maximum FP32 source size retained on the conversion device for one multi-pass object. Larger objects stream both passes; `0` disables this cache. |

`stored_scale_mse` first finds the whole object's maximum, then selects stored scales/codes.
Its cache avoids rereading eligible objects between these passes. CUDA candidate pruning uses
conservative error bounds and retains the FP64 selection and tie rules defined in the
[numeric contract](maintainer/tensor-formats.md#36-stored_scale_mse-conversion-method).
The conversion report records resource settings and per-object conversion times.

## Change part of a recipe

Save the following as `my_recipe.py`:

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```

The default entry function is `configure`; `--recipe my_recipe.py:customize` selects another
function. Alternatively, use an official `--recipe` and put only the changes in an `--override`
file. Overrides run after the base recipe and optional proposal-head setup.

`model.parameters` maps logical names to their shape, source and mathematical inputs. To see the
available names for your selected components, a recipe can print them:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` accepts one name, a list of names, or shell-style patterns such as
`text/layers/*/mlp/down`. A selector that matches nothing fails. Assignments run in Python order;
later assignments replace only the explicitly supplied choices. Changing `format` does not
automatically change `method`. Use `layout="auto"` to select the registered layout for the format
when overriding an earlier explicit layout choice.

The principal choices are:

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

For example, `rows=(0, 128)` can give the first 128 rows a different format. This creates multiple
physical parts when necessary. The container can represent that result; the intended Op must also
support consuming those parts. Current native projections generally require a contiguous parent
region, so arbitrary splits of one projection are not automatically executable.

## Formats, methods and activation precision

The converter currently writes these formats:

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `fp8_e4m3fn_block128x128_bf16` | None | `import_encoded` |
| `nvfp4` | `stored_scale_mse` | `import_encoded` |
| `nf4_g16_e4m3fn` | `stored_scale_mse` | Supply encoded words through `TensorOutput` |

`grouped_absmax` stores one FP16 scale per group and signed integer codes. `fp8_row_maxabs` first
rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier per row.
`import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight divisor.
It does not dequantize and requantize them.
Rank-three NVFP4 targets select `batched_block_scale_k16_m128x4_v1` explicitly; each matrix owns
its divisor. Block-FP8 uses `block_scale_m128_k128_v1` for matrices and banks.

`stored_scale_mse` computes one FP32 divisor from the complete parent amax, then searches stored
E4M3FN scales per G16 group using FP64 reconstruction error. NF4 uses the fixed codebook;
NVFP4 uses E2M1. The method provides no activation calibration. ModelOpt import preserves
packed codes and scale bytes while converting FP32 global multipliers to stored reciprocals.

The exact numeric and packing rules are in [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Source format names alone do not establish
compatibility: scale direction, granularity, code meaning and axis order must also match.

Activation permissions are independent of the stored weight format:

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

They permit choices; they do not force a kernel to use the lowest precision. A fused operation that
shares one activation across several projections must respect the intersection of their
permissions. `recipe.use(parameter, input_name, ...)` can set one mathematical input independently;
the names are available in `parameter.inputs`.

An NVFP4 A4 input requires a positive finite activation divisor. `import_encoded` obtains it from
the selected source, or a recipe supplies it through
`recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights retain separate
Use records; sharing weights does not share calibration implicitly.

## Fused parents and logical projections

The Qwen adapter exposes Q, K, gate and V separately, even when they came from fused source tensors.
It also supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are
packed into a shared parent automatically by the built-in methods.

For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V form one Q5 parent.
The native fused Op receives two weights. To use the supported single-parent FP8 form, an override
can assign all four projections together:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

Use this file as `--override` after `qwen3_6_27b` or `qwen3_8_27b`. It leaves the other parameters
under that recipe. The adapter handles source Q/gate row order; the override works with logical
projections. Changing their representation can change numerical results and the physical kernels.

For explicit organization, `recipe.group([names...])` concatenates compatible selections in the
given order, `recipe.separate(names)` disables automatic grouping for those parameters, and
`recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
and method parameters. NVFP4 parents also require a common weight divisor. Automatic grouping is
limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation, without repacking an inconvenient arrangement.

## Read another source

`--model` supplies the main config, default resources and the source named `base`. Add other
Safetensors sources with repeated `--source NAME=PATH`; they are opened when used. Single-file
Safetensors and indexed shards are supported. Additional tensor-only sources can omit model config;
sources carrying config are checked against the relevant model geometry.

To replace a logical parameter from another compatible checkpoint in a recipe:

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

Supply `--source alternate=/path/to/alternate-checkpoint`. `model.source` applies the architecture's
source-name and axis mapping, including Q/gate extraction. The built-in compressed-tensors reader
understands the implemented per-row FP8 and NVFP4 code/scale conventions. It can expose decoded
values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader
accepts flat C-order element bounds and returns exactly that range. For example, a recipe can read
a logical matrix from a NumPy file stored beside the recipe:

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```

Use this as an override. The NumPy file must already follow the logical row/column order. For an
unfamiliar quantized source, its reader performs the corresponding decoding before returning
values. To preserve existing compatible encoded words, also provide `encoded_format` (a format
name or metadata-only resolver), `read_encoded` returning `EncodedRows`, and the format's required
divisor accessors. Their definitions are in
[`sources/logical.py`](../tools/convert/sources/logical.py).

For a logical bank spread across several physical parents, `recipe.parent(parts, shape=...,
format=..., method=..., order=...)` accepts `(parameter, begin, end, source)` fragments.
`begin/end` are C-order logical element bounds; the supplied `LogicalSource` covers that fragment.
Explicit parents must cover every affected parameter completely without overlap. This permits
one gate/up matrix per expert while retaining separate logical gate and up banks.

`recipe.place(selectors, order)` controls the physical ordering of ordinary parents; an explicit
parent's `order` takes precedence. Binding parts remain in logical order. Shared calibration uses
`recipe.auxiliary_tensor(name, AuxiliaryValue(...))` and an `AuxiliaryReference(name, element)`
passed to `recipe.use(..., auxiliaries=...)`, so several Uses can reference one FP32 element.
These functions are used directly by the Flash-Next recipe.

## Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates
the target and determines auxiliary values. The `produce` function reads bounded source regions
and writes values or codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the existing grouped quantizer. It demonstrates the
method interface; the clipping threshold is a numerical choice made by the recipe author.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("clip must be positive and finite")
    n, k = request.target.shape
    chunk = request.chunk_rows(k * 4)

    def produce(output):
        for begin, end, values in request.read_chunks(request.values, n * k, chunk * k):
            values = values.reshape((end - begin) // k, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin // k, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

Use this as an override. `request.values` traverses the prepared logical inputs in parent order,
including explicit groups. `output.write_codes` performs the registered packing and validates
codes/scales; the method should not duplicate that byte-layout logic. Direct output uses
`output.write_values`. Keep source blocks and temporary device tensors bounded to the method's
working set. `--rows-per-chunk` defaults to 512; custom methods own how they use it.
For an encoded bank, `output.write_codes(..., matrix=e)` writes rows within matrix e. NVFP4
checks divisor consistency within that matrix; block-FP8 writes complete 128-row scale blocks.

## Resources, files and inspection

Text includes `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`. Vision adds its image and video processor configs. Resources come from
`--model`, except that Qwen4Exp defaults to the maintained Flash Next chat template.
`--resource ROLE=PATH` replaces a selected resource:

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

The official conversion examples select these maintained templates:

| Model | Template | Defaults |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | thinking on; closed-turn reasoning omitted |
| Qwen3.8 27B | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |
| Qwen3.8 Flash Next | [qwen3_8_flash_next.jinja](../tools/chat_templates/qwen3_8_flash_next.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |

The Flash Next template maps `minimal` to `low`, `high`/`max` to `xhigh`, and `none` to disabled
thinking. Final-assistant continuation preserves its content, including trailing whitespace.

Use your own Jinja file to change the artifact's default template. A startup
[`--chat-template FILE`](cli.md#text-input) overrides the stored template.
`generation_config.json` is preserved; sampling presets remain determined by the architecture
and explicit application/request settings.

The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts remain
one file. Larger artifacts use an entry such as `models/my_qwen.ninfer` plus
`my_qwen.ninfer.part-0001`, `my_qwen.ninfer.part-0002`, and so on in the same directory. Pass only the
entry path to NInfer and keep all its recorded parts together. `--max-file-bytes` changes the limit.

Conversion writes `models/my_qwen.ninfer.conversion.json` alongside the artifact, recording sources,
methods, formats, component configs, files and timing. Existing output files are not overwritten.
The report is useful for reproducing a recipe; the Engine reads the artifact itself.

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

Inspection reads directory facts without running inference. Conversion rejects missing logical
coverage, invalid source geometry, unsupported encodings and invalid method output. Actual Op
support is checked by consumers during preparation, resource queries, warmup or execution. A
valid file may need additional Op support before its chosen combination can run. Exercise the
phases and optional components you intend to use through the normal [CLI](cli.md) or
[serving](serving.md) route.
