# 权重转换

[English](weight-conversion.md) | 简体中文

NInfer 的转换器根据本地权重和一份 Python 配方生成 `.ninfer` 产物。配方可以复用官方转换、
改动选定的层或投影、组合多个来源，或者调用你自己的转换方法。产物包含最终的配置、
编码后的权重、逻辑绑定和前端资源。

现成可用的产物（包括 Flash Next 与 GLM）列在[下载表](../README.md#get-weights)中。
需要创建或自定义权重时使用本指南。以下命令均在仓库根目录执行。

## 从官方配方开始

源权重转换需要带 PyTorch 和 NumPy 的 Python 3.11 环境。默认使用 CUDA；`--device cpu` 切换到
CPU 转换。下面的输入路径只是占位符，请替换为你本地的 checkpoint 目录。

Qwen3.6-27B 浮点源权重：

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

`--components` 默认值为 `text`。只加入你打算分发的可选组件。`--proposal` 会加入该架构优化后的
proposal head。基于 Qwen3.5 的模型使用仓库内的 token ranking，默认采用 131,072 行的候选短表；
Qwen4Exp 与 GLM5Next 保留完整词表，并额外量化一个 Q4 A16 head。普通的目标输出 head 会保留。

内置配方就是
[`official_recipes.py`](../tools/convert/official_recipes.py) 中的普通 Python 函数：

| 配方 | 主要表示选择 | 额外源 |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 投影、Q6 词表权重 | 无 |
| `qwen3_8_27b` | Q4/Q5 投影、Q8 词表权重 | 无 |
| `qwen3_6_35b_a3b` | Q4 专家、Q5/Q6 expert down、Q8 共享/投影权重 | 无 |
| `qwen3_6_27b_nvfp4` | 导入 NVFP4、选定 BF16 投影、Q8 词表权重 | `quantized` |
| `qwen3_8_27b_nvfp4` | 导入 NVFP4/FP8、由 BF16 生成 FP8 embedding | `quantized` |
| `qwen3_8_flash_next` | Q8 主投影、NVIDIA NVFP4 专家、block-FP8 MTP 专家、NF4 PLE、BF16 辅助投影/Vision | `nvidia` |
| `glm5next` | NVIDIA NVFP4 主专家/稠密 MLP、Q8 KDA 与 MLA 输出/共享投影、block-FP8 MTP 专家；BF16 主 head、embedding 与 Vision | 无；NVIDIA checkpoint 即主源 |

这些名字只选择转换方案。运行时执行路径由产物中存储的架构、配置和实际绑定决定。
`--name` 设置公开模型名，不用于选择 kernel。

带 DFlash2 的 Qwen3.8-27B NVFP4/FP8 产物：

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

MTP 与 Vision 使用主源。DFlash 与 DFlash2 使用对应的命名源，通过 `--source dflash=PATH` 或
`--source dflash2=PATH` 提供。一个产物可以包含多个可选组件；Engine 只加载启动时选定的那些，
且最多一个投机后端。组件是否可用与启动时是否选中相互独立。

### Qwen3.8-Flash-Next

对于 Qwen3.8-Flash-Next，为 Main 专家和 MTP 选用完整的 NVIDIA checkpoint：

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

Qwen4Exp 适配器默认选择专用的
[`qwen3_8_flash_next.jinja`](../tools/chat_templates/qwen3_8_flash_next.jinja)；
`--resource chat_template.jinja=PATH` 可显式替换它。27B 的模板相互独立。

Qwen4Exp 适配器与 Engine 支持 Text、Vision 和原生 MTP。对该架构，`--proposal` 会从 BF16 主 head
额外转换出一个全词表 Q4 head，并带 `A16Only` 激活策略。它不读取 ranking，也不使用
`--proposal-rows`。启动时用 `--spec mtp --draft-tokens 3 --proposal-head optimized` 选用。
默认的 `--proposal-head full` 共用主 BF16 head，不加载 Q4 对象。
Qwen4Exp 不支持 DFlash/DFlash2 和带索引的 proposal 执行。

该配方保留导入的 NVFP4/block-FP8 code 与校准值。PLE 以有界批次从 BF16 源转换为 NF4。
包含优化 head 的完整配方会写成分片产物。
其表示选择见 [Qwen4Exp 架构参考](maintainer/qwen4_exp-model.md)。

### GLM-5.3-Flash

使用 NVIDIA NVFP4 checkpoint 作为主源。它提供配置、校准后的 Main 权重、其余浮点参数、
分词器/模板和可选组件：

```bash
python3 -m tools.convert \
  --model /path/to/GLM-5.3-Flash-NVFP4 \
  --recipe glm5next \
  --components text,vision,mtp \
  --proposal \
  --name glm5.3-flash \
  --out out/glm5_3_flash.ninfer
```

该配方导入 Main NVFP4 专家与稠密 MLP 权重，把 KDA Q/K/V/O、MLA O 和共享专家投影转换为 Q8，
并把 MTP 专家转换为 block-FP8。主 head、embedding、Vision 及其他 BF16 参数保持 BF16；
源 FP32 的 KDA 控制字段保留其精度。`--proposal` 加入全词表 Q4 A16 draft head，
不改变 BF16 主 head。`--ranking` 和 `--proposal-rows` 不会裁剪 GLM 或 Flash Next 的草稿词表。

GLM 前端资源来自主 checkpoint；不要替换成 Qwen 模板。转换完成后，在两台 Spark 上都保持
入口文件与每个分片在一起，并使用 [GLM 部署示例](../examples/configs/spark2-glm5_3-flash.toml)。
[GLM5Next 参考](maintainer/glm5next-model.md) 记录了该模型的表示选择。

## 转换资源

CUDA 自量化会在首次使用时编译其编码器，需要 CUDA Toolkit、C++ 编译器和 Ninja。
请为所选 GPU 架构使用支持 CUDA 的 PyTorch 构建。同一个转换器可在 RTX 5090 与 Spark 上运行；
不需要推理服务。

内置方法按行宽决定批大小，目标是约 16 MiB 的工作输入或输出，并在编码需要时做对齐。
一个读取器在当前批次转换与写入时预取下一批源数据。CUDA 转换在 GPU 上执行分组整数量化和
行 FP8 量化；已编码的 NVFP4/block-FP8 导入保留其原始 codes/scales。

| 选项 | 默认值 | 行为 |
|---|---:|---|
| `--rows-per-chunk` | 自动 | 覆盖行分批；各方法按自身编码边界对齐行。 |
| `--cpu-threads` | `1` | 设置 PyTorch CPU intra-op 线程数；源读取器是独立 worker。 |
| `--source-cache-mib` | `2048` | 单个多遍对象在转换设备上保留的最大 FP32 源大小。更大的对象两遍都流式处理；`0` 关闭该缓存。 |

`stored_scale_mse` 先求整个对象的最大值，再选择存储的 scales/codes。其缓存避免在这两遍之间
重读符合条件的对象。CUDA 候选剪枝使用保守的误差界，并保留
[stored-scale 实现](../tools/convert/quantization/stored_scale.py) 中定义的 FP64 选择规则与
平局规则。转换报告会记录资源设置和每个对象的转换耗时。

## 改动配方的一部分

把下面内容保存为 `my_recipe.py`：

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

用相同的源和组件选择运行：

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```

默认入口函数为 `configure`；`--recipe my_recipe.py:customize` 可选择其他函数。
也可以使用官方 `--recipe`，只把改动放进 `--override` 文件。override 在基础配方和可选的
proposal head 设置之后运行。

`model.parameters` 把逻辑名映射到其形状、来源和数学输入。要查看所选组件可用的名字，
配方可以打印它们：

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` 接受单个名字、名字列表，或 `text/layers/*/mlp/down` 这类 shell 风格通配符。
匹配不到任何对象的选择器会报错。赋值按 Python 顺序执行；后续赋值只替换显式给出的选项。
修改 `format` 不会自动改变 `method`。覆盖先前显式设置的 layout 时，可用 `layout="auto"`
为该格式选择已注册的 layout。

主要选项如下：

| 参数 | 含义 |
|---|---|
| `format` | 持久化数值格式 |
| `layout` | 物理编码；未显式设置时由 format 推断 |
| `method` | 内置方法名或 Python callable |
| `source` | 来自所选源的逻辑值或已编码行 |
| `parameters` | 传给方法的、可 JSON 序列化的数值参数 |
| `rows=(begin, end)` | 以半开区间覆盖前导轴上完整的行 |
| `activation_policy` | 参数数学输入处对激活精度的许可 |

例如 `rows=(0, 128)` 可让前 128 行使用不同的格式。必要时会生成多个物理分片。
容器能表示该结果，但目标 Op 也必须支持消费这些分片。当前原生投影通常要求连续的父区域，
因此对单个投影的任意切分并不会自动可执行。

## 格式、方法与激活精度

转换器目前写入以下格式：

| 格式 | 浮点输入的内置方法 | 已编码输入的导入 |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | 由源读取器直传原始字 |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | 按需提供自定义 method/source |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `fp8_e4m3fn_block128x128_bf16` | 无 | `import_encoded` |
| `nvfp4` | `stored_scale_mse` | `import_encoded` |
| `nf4_g16_e4m3fn` | `stored_scale_mse` | 通过 `TensorOutput` 提供已编码字 |

`grouped_absmax` 为每组存储一个 FP16 scale 和有符号整数 code。`fp8_row_maxabs` 先把输入值
舍入为 BF16，再生成 E4M3FN code 和每行一个 BF16 乘子。`import_encoded` 保留兼容的 code
与 scale 字，包括 NVFP4 的矩阵权重 divisor。它不做反量化和重量化。
三阶 NVFP4 目标会显式选择 `batched_block_scale_k16_m128x4_v1`；每个矩阵拥有自己的 divisor。
Block-FP8 对矩阵和 bank 使用 `block_scale_m128_k128_v1`。

`stored_scale_mse` 从完整父对象的 amax 计算一个 FP32 divisor，再按 G16 组用 FP64 重建误差
搜索存储的 E4M3FN scale。NF4 使用固定码本；NVFP4 使用 E2M1。该方法不提供激活校准。
ModelOpt 导入保留打包的 code 和 scale 字节，同时把 FP32 全局乘子转换为存储的倒数。

精确的数值与打包规则见[数值格式](../tools/artifact/formats.py)和
[存储布局](../src/core/weight_view.cpp)。仅凭源格式名无法确定兼容性：
scale 方向、粒度、code 含义和轴序也必须匹配。

激活精度许可与存储的权重格式相互独立：

| 策略 | 允许的激活路径 |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

它们只是许可，不会强制 kernel 使用最低精度。跨多个投影共用同一激活的融合操作，必须遵守
各许可的交集。`recipe.use(parameter, input_name, ...)` 可单独设置某个数学输入；名字见
`parameter.inputs`。

NVFP4 的 A4 输入需要一个正且有限的激活 divisor。`import_encoded` 从所选源获取它，
或由配方通过 `recipe.use(..., auxiliaries={"activation_input_divisor": value})` 提供。
共享权重保留各自的 Use 记录；共享权重不会隐式共享校准。

## 融合父对象与逻辑投影

Qwen 适配器把 Q、K、gate 和 V 分开暴露，即便它们来自融合的源张量。
它还为 attention、GDN、MLP 和 MoE 提供有限的打包组。内置方法会自动把兼容的选择集
打包进共享的父对象。

对 Dense groupwise 配方，attention 的 Q/K 组成一个 Q4 父对象，gate/V 组成一个 Q5 父对象。
原生融合 Op 接收两个权重。要使用受支持的单父对象 FP8 形式，可用 override 把四个投影一起赋值：

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

把该文件作为 `--override` 用在 `qwen3_6_27b` 或 `qwen3_8_27b` 之后。其余参数仍归属该配方。
适配器处理源 Q/gate 的行顺序；override 作用于逻辑投影。改动其表示会改变数值结果和物理 kernel。

若要显式组织，`recipe.group([names...])` 按给定顺序拼接兼容的选择集，
`recipe.separate(names)` 对这些参数禁用自动分组，`recipe.share(parameter, target)`
把形状相同的参数绑定到同一份物理数据。显式分组必须互不相交，并使用未切分的选择集，
且 format、layout、method 及方法参数一致。NVFP4 父对象还要求共用一个权重 divisor。
自动分组仅限内置方法；自定义方法可请求显式分组。

分组决定存储方式。模型执行代码决定使用哪种受支持的融合实现。
加载器直接上传存储的表示，不会为不便的排布重新打包。

## 读取其他源

`--model` 提供主配置、默认资源和名为 `base` 的源。用重复的 `--source NAME=PATH` 添加其他
Safetensors 源；它们在使用时才被打开。支持单文件 Safetensors 和索引分片。
纯张量源可以不含模型配置；携带配置的源会按相关模型几何做校验。

要在配方中用另一个兼容 checkpoint 替换某个逻辑参数：

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

提供 `--source alternate=/path/to/alternate-checkpoint`。`model.source` 会应用架构的源名称与
轴映射，包括 Q/gate 抽取。内置的 compressed-tensors 读取器理解已实现的逐行 FP8 和 NVFP4
code/scale 约定。它既能给出解码后的值供其他量化器使用，也能给出编码行用于精确导入。

若使用其他文件格式或量化约定，请提供 `LogicalSource`。其值读取器接受扁平 C-order 的
元素边界，并精确返回该范围。例如，配方可读取放在自身旁边的 NumPy 文件中的逻辑矩阵：

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

把它当作 override 使用。NumPy 文件必须已按逻辑行/列顺序存放。对于不熟悉的量化源，
其读取器需在返回值前完成相应解码。若要保留已有的兼容编码字，还需提供 `encoded_format`
（格式名或仅含元数据的解析器）、`read_encoded`（返回 `EncodedRows`），以及该格式所需的
divisor 访问器。其定义见
[`sources/logical.py`](../tools/convert/sources/logical.py)。

对于分散在多个物理父对象上的逻辑 bank，`recipe.parent(parts, shape=...,
format=..., method=..., order=...)` 接受 `(parameter, begin, end, source)` 片段。
`begin/end` 是 C-order 逻辑元素边界；所提供的 `LogicalSource` 覆盖该片段。显式父对象必须完整覆盖每个受影响的参数
且互不重叠。这样可以让每个专家拥有一个 gate/up 矩阵，同时保留独立的逻辑 gate 与 up bank。

`recipe.place(selectors, order)` 控制普通父对象的物理顺序；显式父对象的 `order` 优先。
绑定分片保持逻辑顺序。共享校准使用
`recipe.auxiliary_tensor(name, AuxiliaryValue(...))`，并把一个
`AuxiliaryReference(name, element)` 传给 `recipe.use(..., auxiliaries=...)`，
让多个 Use 引用同一个 FP32 元素。Flash-Next 配方直接使用了这些函数。

## 编写转换方法

方法接收一个 `PrepareRequest`，返回 `request.job(produce=...)`。准备阶段校验目标并确定辅助值。
`produce` 函数读取有界的源区域，并通过 `TensorOutput` 写入值或 codes/scales；写入器负责
放置与文件 I/O。

下面示例在既有分组量化器之前加入显式裁剪。它用于演示方法接口；裁剪阈值是由配方作者
做出的数值选择。

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

把它当作 override 使用。`request.values` 按父对象顺序遍历准备好的逻辑输入，包括显式分组。
`output.write_codes` 执行已注册的打包并校验 codes/scales；方法不应重复这套字节布局逻辑。
直接输出使用 `output.write_values`。请把源块和临时设备张量控制在该方法的工作集内。
`--rows-per-chunk` 默认为 512；自定义方法自行决定如何使用它。对已编码的 bank，
`output.write_codes(..., matrix=e)` 在矩阵 e 内写入行。NVFP4 检查该矩阵内的 divisor 一致性；
block-FP8 写入完整的 128 行 scale 块。

## 资源、文件与检查

Text 包含 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 和
`generation_config.json`。Vision 额外加入该架构使用的处理器资源。资源来自 `--model`，
但 Qwen4Exp 默认使用维护中的 Flash Next 对话模板。`--resource ROLE=PATH` 可替换指定资源：

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

官方转换示例选用以下维护中的模板：

| 模型 | 模板 | 默认值 |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | 启用 thinking；省略已结束轮次的推理内容 |
| Qwen3.8 27B | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | 启用 thinking；effort `xhigh`；保留已结束轮次的推理内容 |
| Qwen3.8 Flash Next | [qwen3_8_flash_next.jinja](../tools/chat_templates/qwen3_8_flash_next.jinja) | 启用 thinking；effort `xhigh`；保留已结束轮次的推理内容 |
| GLM-5.3-Flash | 主 checkpoint 模板 | 启用 thinking；effort 与历史默认值由该模板决定 |

Flash Next 模板把 `minimal` 映射到 `low`、`high`/`max` 映射到 `xhigh`，`none` 则关闭 thinking。
末轮 assistant 续写保留其内容，包括行尾空白。

使用自己的 Jinja 文件可改变产物的默认模板。启动时的
[`--chat-template FILE`](cli.md#text-input) 会覆盖已存储的模板。
`generation_config.json` 原样保留；采样预设仍由架构以及应用层/请求的显式设置决定。

默认最大文件为 32,000,000,000 字节（含封装开销）。更小的产物保持单文件。
更大的产物使用一个入口文件（如 `models/my_qwen.ninfer`）加上同目录下的
`my_qwen.ninfer.part-0001`、`my_qwen.ninfer.part-0002` 等分片。只把入口路径传给 NInfer，
并让其记录的所有分片保持在一起。`--max-file-bytes` 可修改该限制。

转换会在产物旁写出 `models/my_qwen.ninfer.conversion.json`，记录源、方法、格式、组件配置、
文件和耗时。已有输出文件不会被覆盖。该报告有助于复现配方；Engine 读取的是产物本身。

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

检查只读取目录信息，不运行推理。转换会拒绝缺失的逻辑覆盖、无效的源几何、不受支持的编码
和无效的方法输出。实际 Op 支持由消费方在准备、资源查询、预热或执行期间检查。
一个合法文件在其所选组合真正可运行前，可能还需要额外的 Op 支持。请通过正常的
[CLI](cli.md) 或[服务](serving.md)路径，实际演练你打算使用的各阶段和可选组件。

## 升级已有的 v2 产物

离线升级工具支持官方 Qwen3.6/3.8-27B groupwise-int 与 NVFP4 产物，
以及 Qwen3.6-35B-A3B groupwise-int。先把检出更新到当前 `master` 并
[重新构建 NInfer](../README.md#build)，然后用 Python 3.11 运行：

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```

输出必须使用新路径。升级后可直接使用，或重命名以替换原文件。存储的权重值与格式均保留。
升级还会从 `tools/chat_templates/` 安装匹配的模板。已发布的 SHA-256 校验和仅适用于下载文件。
