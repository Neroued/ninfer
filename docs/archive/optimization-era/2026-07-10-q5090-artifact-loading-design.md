# q5090 v4.1 工件格式冻结与分阶段加载优化设计

状态：**已实施（2026-07-10）**。`docs/q5090_packed_file_format_v4.md` 已直接更新为唯一的 v4.1
规范；converter/verifier、canonical 工件、C++ 单文件 tokenizer、独立 `LM_HEAD_DRAFT`、
Header-first prepare、按需 module residency、精确 arena 和 pinned staged uploader 均已落地。

本文面向 Qwen3.6-27B / RTX 5090 的唯一 q5090 路径，不设计通用模型容器，也不保留 v4.0
兼容读取。

---

## 0. 结论与关键路径

这项工作必须拆成四个有明确闸门的阶段，不能从加载器优化开始：

1. **冻结 v4.1 二进制合同。** 在现有 v4 规范上直接修订，加入内嵌 tokenizer，并把
   `LM_HEAD_DRAFT` 从 `TEXT_CORE` 中拆成独立 module tag。
2. **先改 converter/verifier 并生成新工件。** Python 侧能够独立证明 v4.1 的 header、catalog、
   tokenizer、module 边界和全部 weight payload 正确；生成新的 canonical `.qus`。
3. **再让 C++ 正确消费 v4.1。** 先完成格式适配和输出 parity，删除运行时 `--tokenizer`，不在
   同一提交中引入新的异步上传生命周期。
4. **最后优化加载。** Header-first fail-fast、按 feature 制定 residency plan、精确分配显存、
   pinned 流式 H2D，均建立在已经冻结且已有真实工件的 v4.1 catalog 上。

```text
v4.1 spec freeze
      |
      v
converter + verifier ----> canonical v4.1 artifact
      |                              |
      +------------ Gate F2 --------+
                                     |
                                     v
                         C++ correctness adoption
                                     |
                              output/token parity
                                     |
                                     v
                         staged loader optimization
```

这样可以把两类风险分开：如果新工件有问题，先在确定性的 Python converter/verifier 边界解决；
如果加载优化有问题，则有一份已经验收的 v4.1 工件作为稳定输入。

---

## 1. 当前 v4.0 基线与需要改变的合同

当前格式和运行时有三个相互关联的问题：

- tokenizer 不属于 `.qus`，`qus`/serve 启动时必须额外传 `--tokenizer <dir>`；运行时实际读取
  `tokenizer.json`、`merges.txt` 和 `generation_config.json`。
- 可选的 `lm_head_draft` 权重和 id-map 被放在 `TEXT_CORE` module range 内，仅靠
  `source_kind` 和 `DRAFT_HEAD_PRESENT` flag 区分。它不是 TEXT 正常推理的必需权重，却无法在
  module catalog 层独立选择。
- C++ 先把完整文件读入 host vector，再解析 header/catalog，导致错误版本也需要等待整文件 I/O；
  可选 MTP/VISION/draft payload 即使未启用，也已经发生 host read，显存规划也和文件总大小耦合。

v4.1 只改变容器和资源归属，不改变现有权重量化值、`ROW_SPLIT`/`CONTIGUOUS` 编码、
`TensorEntry`、`SegmentRecord`、`FusionGroupRecord` 和 kernel addressing ABI。

### 1.1 v4.0 -> v4.1 变更总表

| 项目 | v4.0 | v4.1 |
|---|---|---|
| identity | `version=4, format_minor=0` | `version=4, format_minor=1`，只接受 minor 1 |
| tokenizer | 外部目录 | `.qus` 中 3 个强制 CPU-only asset |
| draft head 归属 | `TEXT_CORE` 内两个 block | 独立 `LM_HEAD_DRAFT` module tag |
| module kinds | TEXT/MTP/VISION | TEXT/MTP/VISION/LM_HEAD_DRAFT |
| module load policy | 文件中声明 | 字段废弃为 reserved zero，运行时请求决定 |
| weight 编码 | v4 bit-plane/row-split | 不变 |
| runtime 兼容 | 读取 v4.0 | 不兼容；v4.0 明确拒绝 |
| 工件生成 | 现有 v4 converter | 必须重新从 HF 源生成 v4.1，不提供注入/升级工具 |

---

## 2. v4.1 身份与规范修改方式

### 2.1 直接更新现有 v4 规范

格式冻结时直接修改 `docs/q5090_packed_file_format_v4.md`，不新建并行的 v4.1/v5 规范：

- 标题和正文 identity 改为 `q5090_w4g64_mixed_v4_1`；
- `magic` 保持 `Q5090MIXEDV4`，`version` 保持 4，`format_minor` 唯一合法值改为 1；
- 文件名 marker 使用 `_v4_1.qus`，避免运维上把 minor 0 和 minor 1 混在一起；
- converter、Python verifier、C++ parser、structural dump、manifest 和所有 active docs 同步只写/读
  v4.1；
- 删除 v4.0 分支、旧 fixture、旧 CLI 示例和兼容提示，不保留双格式 parser。

`version=4` 表示 weight layout family 未变；`format_minor=1` 表示 container catalog 发生不兼容变更。
本项目没有向后兼容要求，因此 runtime 不得把 minor 当作“可忽略的新字段”。

### 2.2 固定 identity

| 字段 | v4.1 值 |
|---|---|
| magic | `Q5090MIXEDV4\0\0\0\0` |
| version | `4` |
| format_minor | `1` |
| manifest `format` | `q5090_w4g64_mixed_v4_1` |
| canonical suffix | `.q5090_w4g64_mixed_v4_1.qus` |
| header_size | `4096` |

`format_minor=0`、未知 minor、旧 manifest marker 均直接拒绝；不根据文件名猜格式。

---

## 3. v4.1 文件布局和 Header 合同

### 3.1 文件区域

```text
+--------------------------------------------------+ 0
| FileHeader (4096 bytes)                          |
+--------------------------------------------------+ module_index_offset
| ModuleRecord[module_count]        (64 bytes each)|
+--------------------------------------------------+ tensor_index_offset
| TensorEntry[tensor_count]        (128 bytes each)|
+--------------------------------------------------+ segment_index_offset
| SegmentRecord[segment_count]      (32 bytes each)|
+--------------------------------------------------+ fusion_group_index_offset
| FusionGroupRecord[...]            (64 bytes each)|
+--------------------------------------------------+ string_table_offset
| String table                                     |
+--------------------------------------------------+ align 64
| TokenizerRecord[3]                (64 bytes each)|
+--------------------------------------------------+ align 64
| tokenizer asset bytes + 64-byte zero padding     |
+--------------------------------------------------+ align 4096
| Weight block payloads (each 256-byte aligned)    |
+--------------------------------------------------+ file_size
```

原有四张 weight catalog table 的尺寸和记录布局不变。Tokenizer catalog 位于 string table 和 weight
payload 之间，不占用 `TensorEntry`，也不参与 tensor/module payload byte 统计。

### 3.2 FileHeader 扩展

4096-byte Header 中 0..231 的 v4 字段保持原偏移；从原 `format_minor` 开始冻结如下：

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 232 | 4 | u32 | `format_minor = 1` |
| 236 | 4 | u32 | `tokenizer_record_count = 3` |
| 240 | 4 | u32 | `tokenizer_record_size = 64` |
| 244 | 4 | u32 | `tokenizer_flags = 0` |
| 248 | 8 | u64 | `tokenizer_index_offset` |
| 256 | 8 | u64 | `tokenizer_index_bytes` |
| 264 | 8 | u64 | `tokenizer_data_offset` |
| 272 | 8 | u64 | `tokenizer_data_bytes` |
| 280 | 3816 | u8 | reserved zero |

`payload_offset`/`payload_bytes` 仍只描述 weight payload。严格位置公式：

```text
module_index_offset       = 4096
tensor_index_offset       = module_index_offset + module_count * 64
segment_index_offset      = tensor_index_offset + tensor_count * 128
fusion_group_index_offset = segment_index_offset + segment_count * 32
string_table_offset       = fusion_group_index_offset + fusion_group_count * 64

string_end                = string_table_offset + string_table_bytes
tokenizer_index_offset    = align_up(string_end, 64)
tokenizer_index_bytes     = 3 * 64
tokenizer_data_offset     = align_up(tokenizer_index_offset + tokenizer_index_bytes, 64)
tokenizer_data_end        = tokenizer_data_offset + tokenizer_data_bytes
payload_offset            = align_up(tokenizer_data_end, 4096)
payload_bytes             = file_size - payload_offset
```

每个加法、乘法和 `align_up` 必须用 checked u64 arithmetic；任何 region 越界、重叠、倒序或
不满足 alignment 都是结构错误。`string_end..tokenizer_index_offset`、tokenizer asset 间 padding、
`tokenizer_data_end..payload_offset` 全部为零。

### 3.3 Header flags

保留现有 bit 值，重新把 bit4 定义为独立 module presence：

| bit | 名称 | v4.1 语义 |
|---:|---|---|
| 0 | `TEXT_PRESENT` | `TEXT_CORE` ModuleRecord 存在；必须为 1 |
| 1 | `MTP_PRESENT` | `MTP_DRAFT` ModuleRecord 存在 |
| 2 | `VISION_PRESENT` | `VISION_ENCODER` ModuleRecord 存在 |
| 3 | `CALIBRATED` | 任一 weight 使用 calibrated quantization |
| 4 | `LM_HEAD_DRAFT_PRESENT` | `LM_HEAD_DRAFT` ModuleRecord 存在 |
| 5..31 | reserved | 必须为 0 |

每个 presence bit 必须和 module table 双向一致。v4.0 的 flag 名称
`DRAFT_HEAD_PRESENT` 在代码和文档中统一替换为 `LM_HEAD_DRAFT_PRESENT`，不保留别名。

---

## 4. Tokenizer 的规范化存储合同

### 4.1 为什么存原始 asset

v4.1 存储当前 C++ tokenizer 真正消费的三份原始 UTF-8 asset：

1. `tokenizer.json`：vocab、added tokens 和 tokenizer model metadata；
2. `merges.txt`：BPE merge ranks；
3. `generation_config.json`：默认 `eos_token_id`/stop token 配置。

不把 C++ 内部哈希表、BPE trie 或其他派生内存布局序列化进工件。派生结构与编译器/STL 实现耦合，
会把 tokenizer 实现细节变成文件 ABI。v4.1 只冻结输入 bytes 和语义，C++ 从内存 view 构造派生表。

`tokenizer_config.json` 当前只被 converter 的 draft-head shortlist 逻辑用于收集特殊 token，不被
runtime tokenizer 消费，因此它仍是**转换输入**而不是 runtime asset。若将来 runtime 真正需要其中
的 chat template 或字段，应通过新的明确 minor 版本加入，不能悄悄塞进 reserved region。

### 4.2 TokenizerRecord（64 bytes）

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `kind` |
| 4 | 4 | u32 | `encoding = 0 (RAW_UTF8)` |
| 8 | 8 | u64 | `data_offset`（absolute） |
| 16 | 8 | u64 | `data_bytes` |
| 24 | 4 | u32 | `crc32`（仅 asset bytes，不含 padding） |
| 28 | 4 | u32 | reserved zero |
| 32 | 32 | u8 | `sha256`（仅 asset bytes） |

`kind` 和 record 顺序固定：

| 顺序 | kind | 名称 | 最大 bytes |
|---:|---:|---|---:|
| 0 | 1 | `TOKENIZER_JSON` | 256 MiB |
| 1 | 2 | `MERGES_TXT` | 64 MiB |
| 2 | 3 | `GENERATION_CONFIG_JSON` | 1 MiB |

三条记录必须各出现一次、非空、合法 UTF-8；不允许未知 kind/encoding。总 tokenizer data 上限为
321 MiB，检查必须发生在 allocation 前。

第一个 asset 从 `tokenizer_data_offset` 开始，后续 asset 的
`data_offset = align_up(previous_asset_end, 64)`；`tokenizer_data_bytes` 覆盖从 data region 起点到
最后一个 asset 末尾，包含 asset 间 zero padding，不包含 weight payload 前的 4096-byte padding。

### 4.3 内容完整性和解析语义

- converter 按二进制原样复制三份源文件，不重排 JSON、不改换行、不添加终止 NUL；
- converter 写入 CRC32 和 SHA-256；Python verifier 两者都强制验证；
- runtime 对实际 asset bytes 重算 CRC32，并验证 UTF-8 和全部 tokenizer padding；SHA-256 与
  JSON/merges 语义由 converter/offline verifier 校验，文本入口最终由 `QwenTokenizer` 消费；
- `generation_config.json` 在 v4.1 中是必需 asset，不再允许缺失时硬编码 `{248046, 248044}`；
- `merges.txt` 在 v4.1 中是必需 asset，不再允许先构造成功、首次 encode 时才报错；
- `model.vocab`、`added_tokens` 和 `eos_token_id` 的所有 id 必须是
  `[0, FileHeader.vocab_size)` 内的唯一/有效 id，EOS 必须实际存在于 tokenizer；
- added token 必需字段必须完整且类型正确，当前 runtime 不支持的 `single_word/lstrip/rstrip/normalized`
  必须全部为 false；merge 行必须恰好包含两个 symbol 且 pair 不重复；
- tokenizer 只驻留 CPU，不进入 GPU load plan、DeviceArena 或 H2D byte 统计；Engine 上传完成后把 raw
  bundle 交给真正需要文本前端的 CLI/serve 构造派生表。

加载 API 的目标形态是内存输入而非伪造临时目录：

```cpp
struct Q5090TokenizerBundle {
    std::string tokenizer_json;
    std::string merges_txt;
    std::string generation_config_json;
};

explicit QwenTokenizer(Q5090TokenizerBundle bundle);
```

禁止把内嵌 bytes 落到临时文件后复用 filesystem constructor。

---

## 5. `LM_HEAD_DRAFT` 独立 module tag

### 5.1 ModuleKind

保留三个既有数值，新增一个值：

| value | module_kind | 内容 |
|---:|---|---|
| 0 | `TEXT_CORE` | 正常 text decode/prefill 必需权重，不含 draft head |
| 1 | `MTP_DRAFT` | MTP draft layer 权重 |
| 2 | `VISION_ENCODER` | vision encoder/merger 权重 |
| 3 | `LM_HEAD_DRAFT` | shortlisted draft head 权重及 id-map |

canonical module/payload 顺序固定为：

```text
TEXT_CORE -> LM_HEAD_DRAFT (optional) -> MTP_DRAFT (optional) -> VISION_ENCODER (optional)
```

顺序按消费依赖和当前 block 生成位置定义，不按 enum 数值排序。`TEXT_CORE` 必须存在，其他三个 module
可独立缺失；不过 runtime 请求 `LM_HEAD_DRAFT` 时还必须同时请求且存在 `MTP_DRAFT`。

把 draft module 放在 TEXT 后也能最大限度保持 v4.0 weight block 的相对顺序：原先追加在 TEXT range
尾部的两个 draft block 变为自己的连续 range，其 weight payload bytes 本身不需要重新量化。

### 5.2 ModuleRecord

`ModuleRecord` 仍为 64 bytes，offset 0..39 含义不变：

| Offset | Size | v4.1 语义 |
|---:|---:|---|
| 0 | 4 | `module_kind`，允许 0..3 中已定义值 |
| 4 | 4 | `module_version = 4` |
| 8 | 8 | `tensor_index_begin` |
| 16 | 8 | `tensor_index_count` |
| 24 | 8 | `payload_offset` |
| 32 | 8 | `payload_bytes` |
| 40 | 4 | reserved zero；删除旧 `load_policy` |
| 44 | 4 | reserved zero |
| 48 | 16 | reserved zero |

文件不能决定某个 module 应驻留 CPU/GPU。residency 是本次启动的 runtime feature request，写入工件的
`RESIDENT/LAZY_GPU` 很快会与真实调用场景冲突。因此 converter 固定写零，parser 非零拒绝。

### 5.3 Draft module 精确内容

`LM_HEAD_DRAFT` module 存在时必须恰好包含两个、且仅两个连续 block：

1. `lm_head_draft`
   - `module_kind = LM_HEAD_DRAFT`
   - `source_kind = LM_HEAD_DRAFT`（保留现有数值 6）
   - `source_layer = 0xFFFFFFFF`
   - `Q4G64_F16S`, `ROW_SPLIT`, shape `[N, 5120]`
   - 一个覆盖 `[0,N)` 的同名 segment
2. `lm_head_draft.idmap`
   - `module_kind = LM_HEAD_DRAFT`
   - `source_kind = LM_HEAD_DRAFT_IDMAP`（保留现有数值 7）
   - `source_layer = 0xFFFFFFFF`
   - `I32_CTRL`, `CONTIGUOUS`, shape `[N]`
   - 一个覆盖 `[0,N)` 的同名 segment

额外结构约束：

- `N` 必须相同且属于 converter 支持的 canonical shortlist size；
- id-map 中每个 id 唯一且 `< FileHeader.vocab_size`；
- `LM_HEAD_DRAFT_PRESENT` iff module 存在；
- 两个 draft `source_kind` 在其他 module 中非法；`TEXT_CORE` 不得再包含这两个 block；
- draft module 不属于 `MTP_DRAFT`。二者是不同资源：MTP layer 决定如何产生 draft hidden state，
  draft head 决定用较小词表如何投影；runtime 可使用 MTP + full `LM_HEAD`，但不能只加载 draft head。

### 5.4 Residency 真值表

| Runtime request | 必须 resident | 明确不 resident |
|---|---|---|
| text default | TEXT_CORE | MTP, LM_HEAD_DRAFT, VISION |
| MTP | TEXT_CORE, MTP_DRAFT | LM_HEAD_DRAFT, VISION |
| MTP + draft head | TEXT_CORE, MTP_DRAFT, LM_HEAD_DRAFT | VISION |
| vision | TEXT_CORE, VISION_ENCODER | MTP, LM_HEAD_DRAFT |

这是 runtime policy，不编码进 ModuleRecord。请求的 module 缺失时必须失败，不得静默降级或回退 full
artifact 假设。

---

## 6. v4.1 结构校验增量

原 v4 weight block、plane、segment、fusion、CRC 和数值规则继续成立。v4.1 需要在现有规范的
“Structural validation”章节前置并补充以下 MUST：

1. 只读 4096 bytes 即可验证 magic/version/minor/endian/header size 和所有 header caps；任何错误在
   metadata allocation 和 CUDA 初始化之前失败。
2. `module_count in [1,4]`；module kinds 唯一，按 §5.1 canonical 顺序，range 连续分割全部 tensor；
   presence flags 双向一致。
3. `TEXT_CORE` 不含 draft source kinds；`LM_HEAD_DRAFT` 严格满足 §5.3 两-block 合同。
4. tokenizer header 字段必须等于固定 count/record size/flags；三条记录严格排序、无缺失/重复；所有
   offset/size 在文件内且满足 §3/§4 的 adjacency 和 alignment。
5. runtime 读取 tokenizer 后强制 CRC/UTF-8 和 tokenizer-region padding；offline verifier 额外强制
   SHA-256、内容语义和全部 payload padding zero-fill。
6. weight `payload_offset` 必须位于 tokenizer region 之后；tokenizer 绝不能与任何 catalog、string 或
   weight payload 重叠。
7. ModuleRecord 两个旧 policy/flags 字段和其余 reserved bytes 全零。
8. v4.0 `format_minor=0` 明确返回 unsupported format，不进入 catalog parser。
9. tensor/segment name 长度限制为 1..4096 bytes，必须是无内嵌 NUL 的合法 UTF-8；segment/fusion
   row 累加始终保持 u64，比较通过前不得窄化为 u32。

Parser 必须区分：

- **runtime structural validation**：只依赖 header/catalog/tokenizer，防止错误寻址；
- **selected payload validation**：只对本次需要的 weight ranges 做完整 `pread`，可选择增量 CRC；
- **offline artifact verification**：扫描全文件 zero padding、所有 block CRC/SHA 和数值 oracle。

runtime 不应为了验证未请求 MTP/VISION/draft 的 padding/CRC 而读取其 weight payload。

---

## 7. Converter、manifest 与新权重生成

### 7.1 Converter 修改边界

先修改 Python 路径，C++ 暂不参与：

- `format.py`：冻结 minor 1 Header/TokenizerRecord pack/unpack，新增 module kind 3，ModuleRecord
  reserved-zero；
- `tensor_plan.py`：TEXT manifest 不再追加 draft blocks，单独生成 `LM_HEAD_DRAFT` manifest；
- `convert.py`：读取三份 tokenizer asset，生成四个 module range，先写 catalog/tokenizer region，再写
  weight payload；输出 marker 改为 `mixed_v4_1`；
- `verify.py`：实现 §6 全量校验，按 module 统计 blocks/segments/payload，验证 tokenizer hashes，
  验证 draft module coupling；
- `README.md` 和命令示例：明确 converter 的 tokenizer path 是**构建输入**，不是运行时依赖。

converter 默认从 `--model` 目录读取 tokenizer 与 `tokenizer_config.json`，也允许通过构建期
`--tokenizer DIR` 指定另一份输入；三份 runtime asset 必须全部存在并通过 converter 语义校验。

### 7.2 确定性写入顺序

```text
load HF config/index/tokenizer source
  -> build TEXT manifest (without draft blocks)
  -> optionally build LM_HEAD_DRAFT two-block manifest
  -> build MTP and VISION manifests
  -> freeze module/tensor/segment/fusion/string/tokenizer catalog
  -> compute every absolute offset and file size
  -> write header + catalog + exact zero padding
  -> copy tokenizer bytes and hashes
  -> materialize weight payloads in canonical module order
  -> fsync/close temporary output
  -> atomic rename to final _v4_1.qus
  -> run verifier against final file
```

不得在原 v4.0 文件上原位插入 tokenizer：插入会改变所有 payload offsets，且无法证明原文件完整性。
v4.1 必须从 HF 源重新生成到临时文件，验证成功后再发布。

### 7.3 Manifest 合同

manifest 至少新增/修改：

```json
{
  "format": "q5090_w4g64_mixed_v4_1",
  "format_version": 4,
  "format_minor": 1,
  "binary_spec": "docs/q5090_packed_file_format_v4.md",
  "weights_file": "qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus",
  "modules": ["TEXT_CORE", "LM_HEAD_DRAFT", "MTP_DRAFT", "VISION_ENCODER"],
  "tokenizer": {
    "record_count": 3,
    "assets": [
      {"kind": "TOKENIZER_JSON", "bytes": 0, "crc32": "00000000", "sha256": "..."},
      {"kind": "MERGES_TXT", "bytes": 0, "crc32": "00000000", "sha256": "..."},
      {"kind": "GENERATION_CONFIG_JSON", "bytes": 0, "crc32": "00000000", "sha256": "..."}
    ]
  },
  "lm_head_draft": {
    "present": true,
    "module": "LM_HEAD_DRAFT",
    "n": 131072,
    "k": 5120
  }
}
```

manifest 是 informational mirror，binary header/catalog 仍是 runtime 真值；verifier 必须验证二者完全
一致。

### 7.4 新工件验收

v4.1 canonical artifact 发布前必须证明：

- Python L0：header/catalog/tokenizer/module/padding/CRC/SHA 全量通过；
- Python L1：所有 weight block 反量化值通过现有 oracle；
- 除归属和绝对 offset 外，v4.1 每个既有 weight block 的 shape/qtype/layout/plane bytes/CRC 与 v4.0
  对应 block 一致；draft 两个 block 的 payload CRC 也一致；
- tokenizer 三份 asset 的 SHA-256 与转换输入一致；
- manifest counts、module ranges、payload bytes 和 binary 一致；
- 输出文件使用新 marker，旧 verifier/runtime 对它不得误判为 v4.0。

---

## 8. 格式冻结闸门

### Gate F1：规范冻结

只有以下事项全部完成，才允许实现 converter：

- 本文中的 Header、TokenizerRecord、module enum/order、draft coupling、flags 和命名无未决项；
- `docs/q5090_packed_file_format_v4.md` 已按本文直接更新并完成独立审查；
- Python/C++ 使用的每个常量都有唯一规范来源和精确 offset/size；
- 确认 runtime tokenizer 所需 asset 清单，不再依赖目录中未被打包的文件；
- 确认 canonical artifact 是否包含 optional draft/MTP/vision，manifest 能准确表达缺失 module。

### Gate F2：新权重可独立验证

只有以下事项全部完成，才允许修改 C++ 加载路径：

- converter 和 verifier 只支持 v4.1；
- compact valid/malformed binary fixtures 通过；
- canonical 27B v4.1 权重已重新生成，Python L0/L1 verifier 全通过；
- 新工件和 manifest 的 SHA-256 已记录；
- tokenizer golden ids 已从内嵌 asset 生成并固定。

F2 的关键意义是：在任何 C++ loader 优化之前，已有一份不依赖 C++ 实现即可证明正确的输入工件。

### Gate F3：C++ correctness baseline

加载优化开始前还必须满足：

- C++ 只接受 v4.1，能够从内嵌 tokenizer 启动 CLI/serve；
- runtime `--tokenizer` 和相关 options/constructor 已删除；
- `LM_HEAD_DRAFT` 独立 module 能按请求正确绑定；
- default、MTP、MTP+draft、vision 的短 E2E 输出/tokenizer parity 通过；
- 这一阶段允许暂时复用旧的 full-file host buffer，作为加载优化前的正确性基线。

不把 full-file buffer 的删除混进格式接入提交，可以显著缩小定位范围。

---

## 9. C++ v4.1 正确性接入（优化前）

F2 后先完成最小、直接的格式接入：

1. parser structs/enums 同步 Header 和 ModuleKind；minor 0 直接拒绝；
2. WeightStore 把 draft head 从 TEXT lookup 移到 `ModuleKind::LmHeadDraft`；
3. artifact 读取 tokenizer records，构造 `Q5090TokenizerBundle`；
4. `QwenTokenizer` 改为内存 bundle constructor，删除 `tokenizer_dir_` 和文件读取；
5. `qus`、serve、benchmark 和其他所有入口只接收一个 `.qus`；Engine 保留 CPU raw bundle，上传成功
   后由真正需要文本前端的 CLI/serve 构造 tokenizer；`qus_bench` 不为纯权重基准构造派生表；
6. Engine option 中 `use_lm_head_draft` 表示请求独立 module；请求 draft 时验证 MTP 已启用；
7. 保留一次可对照的 correctness baseline，再进入 §10。

目标启动方式变为：

```bash
./build/src/qus \
  out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus \
  --prompt "$(cat src/kernels/kernel/gqa_attention_decode_i8.cuh)" \
  --max-context 131072 \
  --max-new 65535 \
  --mtp-draft-tokens 3 \
  --lm-head-draft \
  --kv-dtype int8
```

`--tokenizer` 不再存在，也不提供环境变量或默认目录 fallback。

---

## 10. 格式稳定后的加载优化

以下优化只在 Gate F3 后开始。本节保留完整设计边界，但不决定 v4.1 文件合同。

### 10.1 Header-first 与 catalog-first

```text
open fd
  -> pread exactly 4096-byte Header
  -> validate identity/caps/region arithmetic
  -> pread bounded catalog + tokenizer region
  -> validate all structural metadata
  -> build immutable Q5090Catalog
  -> derive ResidencyRequest from runtime options
  -> build exact Q5090LoadPlan
  -> allocate selected device arenas
  -> stream selected payload ranges
  -> transactional publish/bind
```

错误 magic/version/minor/header 必须只读取 4096 bytes、无 CUDA 副作用地失败。catalog allocation 前对
count/bytes 设置模型专用 caps，并做 checked arithmetic。

为了避免打开后文件被替换或覆盖，artifact 在同一个 fd 上工作，记录首次 `fstat` 的
`{dev, ino, size, mtim, ctim}`，每个加载 transaction 前后复查。该检查防止常见部署竞态，不声称抵抗
能伪造 timestamp 的对手；需要更强保证时对 selected payload 增量验证 CRC。

### 10.2 Available 和 Resident 分离

catalog 表示工件中 **available** 的资源，WeightStore 只暴露已经提交的 **resident** view：

```cpp
struct ResidencyRequest {
    bool mtp;
    bool lm_head_draft;
    bool vision;
};

struct Q5090LoadPlan {
    std::vector<PlannedModule> modules;
    std::uint64_t file_read_bytes;
    std::uint64_t h2d_bytes;
};
```

plan 以 ModuleRecord range 为选择边界，对每个 block 预先计算最终 device offset、plane pointers、
allocation bytes 和 copy ranges。planned offset 是唯一 placement 真值，后续分配/绑定不得重算第二套
offset。

### 10.3 精确显存分配

每个 selected module 使用独立 `DeviceArena`：

- `TEXT_CORE` 生命周期等于 Engine；
- `MTP_DRAFT`、`LM_HEAD_DRAFT`、`VISION_ENCODER` 只在请求时分配；
- capacity 等于 plan 中按 256-byte alignment 计算的精确值，不按 `file_size` 估算；
- allocation 后断言 CUDA base 满足 256-byte alignment，所有 view 都来自
  `base + planned_device_offset`；
- 本阶段不实现运行时 unload/hot-load，所有 startup residency commit 后才允许 graph capture。

startup transaction 在 publish 前一次性构造完整的 host record vectors，publish 后不再追加或扩容。
再次 `prepare()` 会先清理旧 resident 状态；不提供运行中保留旧模型并并行 hot-swap 的语义。因此所有
已发布 descriptor address 和 device pointer 在一次已完成加载到下一次显式加载之间稳定。

### 10.4 Pinned 双缓冲流式上传

默认 uploader 使用 2 个固定大小 pinned slots（初始建议每个 64 MiB）和专用 load stream：

```text
pread selected file range -> pinned slot A -> cudaMemcpyAsync -> final arena address
pread next range          -> pinned slot B -> cudaMemcpyAsync -> final arena address
wait event A only when A must be reused
```

它避免 17+ GiB full-file pageable host vector，并让磁盘读取与 H2D 重叠。默认基线使用 buffered
`pread`；`O_DIRECT`、GDS、整文件 mmap/host-register 都不是第一阶段要求，必须在 baseline 有报告后再
用真实硬件数据决定。

copy 以 plan 中相邻 file/device ranges 合并，不能为了方便把 skipped module 读进 staging buffer。
tokenizer 永远不经过 pinned uploader。

### 10.5 事务提交和失败清理

一次 startup residency transaction 对全部 selected modules 执行 prepare -> upload -> validate ->
publish：

- upload 完成前 lookup 不返回半成品 pointer；
- short read、CRC、CUDA enqueue/event、callback 异常首先记录根错误并停止新 enqueue；
- failure guard 必须 drain 已提交的 DMA，之后才能释放 pinned slot 和 arena；
- recoverable error 且 stream/context 健康时回滚全部未发布 selected modules；
- CUDA fatal 使整个 Engine/进程失效，不尝试继续使用旧 resident groups；
- publish 后 descriptor address 和 device pointer 在 Engine 生命周期内稳定。

prepare transaction 按值持有 progress callback；公开的 split `prepare()`/`upload()` 不跨调用保存调用者
提供的裸 callback 指针。DMA guard 的析构顺序必须早于 pinned slots/device arenas：任何 enqueue、event、
callback 或 wait 异常都先 drain stream；无法确认 stream quiescent 时按 CUDA-fatal 终止。stats 描述成功
完成的阶段与最终 load plan，不承诺精确记录失败 attempt 的部分 I/O/H2D 工作量。

### 10.6 可观测性

统一输出或 stats API 至少包含：

- header/catalog/tokenizer read bytes 和耗时；
- selected/skipped modules；
- 每个 module file bytes、H2D bytes、arena capacity、upload 秒数；
- pinned slot count/size；
- total file read、H2D、host peak staging、device resident bytes；
- fail stage（header/catalog/tokenizer/plan/allocate/read/upload/publish）。

---

## 11. 分阶段实施与提交边界

### Phase 1：规范冻结

修改现有 v4 规范和相关 active design docs，完成 Gate F1。此阶段不改 converter/C++。

Definition of done：v4.1 每个新增 byte、enum、flag、module coupling 和文件命名均无歧义，独立审查
通过。

### Phase 2：Converter/verifier 与新权重

实现 §7，先跑 compact binary-contract tests，再生成 canonical 27B artifact，完成 Gate F2。

Definition of done：新 `_v4_1.qus` 和 manifest 发布，Python L0/L1 全通过；旧 v4.0 不再是工具链输出。

### Phase 3：C++ correctness adoption

实现 §9，必要时暂时保留 full-file buffer，建立 Gate F3 baseline。

Definition of done：单文件 CLI/serve 启动；tokenizer golden、default/MTP/draft/vision E2E parity 通过；
无 runtime tokenizer path。

### Phase 4：Header/catalog-first

拆分 CPU-only artifact parser，4096-byte fail-fast，建立 immutable catalog 和 request-based plan。

Definition of done：错误版本失败延迟不随文件大小增长且无 CUDA 副作用；合法 artifact 可在无 CUDA
情况下完成完整 catalog/tokenizer 校验和 load plan。

### Phase 5：Selective residency 与精确 arenas

实现 per-module arena、available/resident 分离、stable descriptor slabs 和精确 byte accounting。

Definition of done：四种 request 的 resident module、显存和 lookup 行为符合 §5.4，未请求 module 不分配
显存。

### Phase 6：Pinned staged uploader

删除 full-file host vector，实现 selected `pread` + 双缓冲 H2D + transactional publish。

Definition of done：未请求 payload 不发生 file read/H2D；host RSS 由文件大小量级降到
metadata + tokenizer + pinned ring；真实输出不变。

### Phase 7：可选 I/O 深化

在 Phase 6 有 cold/warm page-cache、RSS、H2D 和 nsys 报告后，再决定是否评估 `O_DIRECT`/GDS。
没有 direct-path 证据则不加入复杂分支。

---

## 12. 验证策略

### 12.1 格式和 converter

这是 binary/file-format contract，允许并需要高价值 tests：

- Header/TokenizerRecord 精确 pack/unpack 和 offset fixture；
- minor 0/未知 minor、巨大 count、u64 overflow、region overlap/truncation；
- tokenizer record 缺失/重复/乱序、非法 encoding/UTF-8、CRC、非零 padding；
- draft module 缺 block、多 block、错误 module/source kind/qtype/shape/id-map；
- ModuleRecord reserved 非零；presence flags 与 module table 不一致；

### 12.2 Tokenizer 和 E2E

- 用内嵌 asset 对固定中英文、代码、added/special token 语料做 encode/decode golden；
- default stop ids 必须来自内嵌 generation config；
- CLI 和 serve 不访问 tokenizer directory；可在不存在原 HF 目录的环境运行；
- greedy、MTP full head、MTP draft head 的短输出与 correctness baseline 一致。

### 12.3 Selective residency

- §5.4 四种 request 的 selected/resident module 与成功加载统计匹配 plan；
- non-resident lookup 为 null，但 catalog available 状态正确；
- planned offset 256-byte alignment 和 arena bounds 全部断言；
- graph capture 只在 startup residency 完成后发生。

### 12.4 生命周期和性能

- compact fixture 执行 staged upload + bind smoke；
- `/usr/bin/time -v` 比较 max RSS，不再随 17+ GiB artifact 线性增长；
- cold-ish/warm cache 分别报告 file GB/s、H2D GB/s、end-to-end 秒数；
- nsys 确认 disk read 与 H2D 有重叠，slot reuse 才等待 event，没有每 chunk 全 stream sync。

不添加扫描源码字符串或锁定私有调用顺序的测试。

---

## 13. 明确的非目标

- 不改变 qtype、量化值、plane 编码或 CUDA kernel 数值路径；
- 不设计 v4.0 -> v4.1 原位迁移器，也不保留旧 parser/CLI fallback；
- 不把 tokenizer 派生 C++ 容器布局序列化；
- 不在格式冻结阶段同时实现异步 uploader；
- 不在首轮 loader 优化引入 runtime hot-load/unload；
- 不默认引入 GDS、`O_DIRECT`、Unified Memory 或 mmap 全文件方案；
- 不为未来模型抽象通用资源系统，module/tag 只覆盖本项目已存在的四类资源。

---

## 14. 主要风险与审查重点

1. **格式与实现漂移。** v4.1 必须先改唯一规范，再同步 packer/verifier/parser；不能只在 design doc
   声明字段。
2. **Tokenizer 资产不完整。** Gate F1 前要用实际模型目录证明三份 asset 足以让 C++ 完成全部现有
   encode/decode/stop-token 行为。
3. **Draft module 边界。** `LM_HEAD_DRAFT` 和 `MTP_DRAFT` 不得合并；前者依赖后者，但二者有不同
   residency 和 payload 统计。
4. **Payload offset 漂移。** 插入 tokenizer 会改变所有 absolute weight offsets；converter 必须统一
   预计算，禁止后写时局部修补 header。
5. **阶段污染。** 新工件的数值/输出问题必须在 Phase 2/3 解决，不能靠 Phase 4+ 的 loader 重构掩盖。
6. **异步资源释放。** Phase 6 的任何错误路径都要在释放 host/device storage 前处理 in-flight DMA。
7. **指针稳定性。** optional module 发布不能移动已绑定 host descriptors，planned device offset 必须是
   唯一 placement 真值。

本设计的首要交付物不是更快的 loader，而是一份无歧义、可独立生成和验证的 v4.1 工件。只有该工件
成为新的唯一事实来源后，加载优化才有稳定边界。

---

## 15. 实施记录

最终实现遵循 F1 → F2 → F3 → loader optimization 的顺序：

- canonical 工件：`out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus`，17,503,667,712 bytes，
  SHA-256 `58adff47274aff102298139fa604932a51f6d94ae3bac8bcae494cbab4db2162`；
- Python verifier：L0/L1 全量通过，既有 block 数值 mismatch 为 0；格式单测 37 项通过；
- C++ Header-first：错误 version/minor 在读取 4096-byte Header 后失败，CPU prepare 不创建 CUDA
  context 或 device arena；合法工件只读取 bounded catalog/tokenizer prefix 后即可生成 load plan；
- residency：TEXT 16,378,329,088 bytes，LM_HEAD_DRAFT 357,040,128 bytes，MTP 451,267,584
  bytes，VISION 293,396,992 bytes；每个 selected module 的 arena capacity/used 均精确等于 plan；
- uploader：两个最多 64 MiB 的 pinned slot，selected module 使用同一 fd 上的 buffered `pread` 与
  `cudaMemcpyAsync` 轮转；callback 异常路径会 drain in-flight DMA，且不发布 descriptor/arena；
- 实测单文件 CLI 的 maximum RSS 为 331,212 KiB，而非旧实现的 17+ GiB full-file host buffer；
- 验证：39 项非 real-file CTest、real WeightStore 四种 residency、Engine default/MTP/draft parity、
  单文件 CLI 均通过；`compute-sanitizer --tool memcheck` 对 staged upload/bind fixture 报告 0 error。

Phase 7 的 `O_DIRECT`/GDS 仍保持非目标：当前 buffered `pread` 基线已满足正确性、RSS 和 selective
I/O 边界，在没有独立硬件收益证据前不增加第二条 I/O 路径。
