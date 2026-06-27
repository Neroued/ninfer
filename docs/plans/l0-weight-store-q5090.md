# L0 WeightStore q5090 Implementation Plan

> **Current status:** historical plan. The q5090 parser/loader, module-selective `WeightStore`,
> q5090 parser tests, real-file smoke path, and unified `Weight` handle now exist in the current
> tree. This plan is retained for ABI and validation history; checklist items below describe how the
> implementation was driven, not work that is still missing.
>
> **Naming note:** the original plan used the pre-unification name `QuantWeight`. Current code uses
> `Weight` for q5090 quantized payload descriptors and dense control weights.

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans to implement this plan task-by-task. Track steps with
> checkbox syntax. User instruction overrides the generic skill default: do not commit
> unless explicitly told.

**Historical goal:** Replace the obsolete pre-q5090 pipeline with a strict q5090
`Q5090MIXEDV1` parser/loader that matches `tools/q5090_convert` byte-for-byte.

**Architecture:** Keep L0 model-agnostic. A device-free parser validates the entire
file and returns typed metadata plus payload spans. `WeightStore::load()` then uses
`LoadOptions` to upload raw selected-module payload bytes into the weights arena and
exposes generic lookup by module/name and by `source_kind + source_layer`.

**Tech Stack:** C++20, CUDA Runtime, existing `DeviceArena`/`Tensor` views, Python
standard library for test fixture generation only. No zlib or third-party C++ libs.

---

## Authoritative Inputs

- `docs/q5090_packed_file_format_v1.md` is the binary ABI. It wins over all other docs.
- `tools/q5090_convert/format.py` defines exact struct packing:
  - header: `<16s8I8Q14I32s`, 200 bytes, padded to 4096 bytes
  - module record: `<IIQQQQII`, 48 bytes, padded to 64 bytes
  - tensor entry: `<IIQHHHH4I4IIHHQQIIII`, 96 bytes, padded to 128 bytes
- `tools/q5090_convert/qtypes.py` defines enum values and source kinds.
- `tools/q5090_convert/layouts.py` defines payload byte formulas.
- `tools/q5090_convert/convert.py` defines module order, module load policies, and
  manifest shape.
- `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus` and `out/manifest.json` are the real
  integration target.

## File Responsibilities

- Modify `include/qus/core/tensor.h`
  - Replace old W4-only `QuantLayout` with q5090 layouts.
  - Add q5090 qtype, module kind, scale dtype, load policy, and source kind enums.
  - Generalize `Weight` to describe one raw q5090 payload with inline scales.
- Modify `include/qus/core/arena.h` and `src/core/arena.cu`
  - Add `std::size_t mark() const noexcept` and `void rewind(std::size_t mark) noexcept`.
- Rewrite `include/qus/core/weight_store.h`
  - Add `Q5090Expectations`, `LoadOptions`, parsed metadata structs, and lookup API.
  - Remove `WeightEntryKind`, role/layer-only lookup, and pre-q5090 assumptions.
- Create `include/qus/core/weight_store_parser.h`
  - Declare device-free parsed structs and `parse_q5090_file()` for host-only tests and
    for `WeightStore::load()`.
- Rewrite `src/core/weight_store.cpp`
  - Implement CRC-32, FNV-1a-64, little-endian parsing, strict validation, and upload.
  - Keep parser logic independent of CUDA.
- Rewrite `tests/test_weight_store.cpp`
  - GPU-gated synthetic round-trip and module-selective upload checks.
  - GPU-gated real-file inventory/CRC/load smoke for `out/*.qus`.
- Create `tests/test_q5090_parser.cpp`
  - Mandatory host-only malformed parser matrix with no CUDA device construction.
- Add `tests/fixtures/make_q5090_fixture.py`
  - Imports `tools.q5090_convert.format`, `qtypes`, and `layouts.encode_tensor()`.
  - Emits a tiny `.qus` with TEXT_CORE, MTP_DRAFT, and VISION_ENCODER modules covering
    qtypes/layouts required by tests.
- Modify `CMakeLists.txt`
  - Add mandatory host-only `qus_q5090_parser_test` target.
  - Keep `qus_weight_store_test` GPU-gated.
- Delete obsolete files
  - pre-q5090 format document
  - two-stage pack stub
  - two-stage quantize stub
- Update docs
  - `README.md`, `docs/design.md`, `docs/l0-infrastructure-design.md`,
    `docs/qwen3.6-27b-architecture.md`, and stale `docs/plans/*` references must point
    to q5090 docs and state transform ownership correctly.

## Public C++ Surface

Use names in `qus` namespace:

```cpp
enum class QType : std::uint16_t {
    Q4G64_F16S = 0,
    Q5G64_F16S = 1,
    Q6G64_F16S = 2,
    W8G128_F16S = 3,
    BF16_CTRL = 4,
    FP32_CTRL = 5,
};

enum class QuantLayout : std::uint16_t {
    TileN64K64 = 0,
    TileN64K128 = 1,
    RowGroupedG64 = 2,
    Contiguous = 3,
};

enum class ModuleKind : std::uint16_t {
    TextCore = 0,
    MtpDraft = 1,
    VisionEncoder = 2,
};

enum class ScaleDType : std::uint16_t {
    None = 0,
    FP16 = 1,
};

enum class LoadPolicy : std::uint32_t {
    Resident = 0,
    LazyGpu = 1,
    CpuPinnedThenGpu = 2,
};

struct Weight {
    const void* payload = nullptr;
    std::uint64_t payload_bytes = 0;
    QType qtype = QType::Q4G64_F16S;
    QuantLayout layout = QuantLayout::TileN64K64;
    ModuleKind module = ModuleKind::TextCore;
    ScaleDType scale_dtype = ScaleDType::FP16;
    std::uint32_t group_size = 0;
    std::uint32_t source_layer = 0xFFFFFFFFu;
    std::uint32_t source_kind = 0;
    std::int32_t shape[4] = {1, 1, 1, 1};
    std::int32_t padded_shape[4] = {1, 1, 1, 1};
    std::uint32_t ndim = 0;
};
```

`CONTIGUOUS` tensors are exposed as `Tensor` views. Quantized tensors are exposed as
`Weight` descriptors. There is no separate scales pointer; tile payloads already
contain FP16 scales inline.

`WeightStore` API:

```cpp
struct Q5090Expectations {
    std::optional<std::uint32_t> layer_count;
    std::optional<std::uint32_t> hidden_size;
    std::optional<std::uint32_t> intermediate_size;
    std::optional<std::uint32_t> vocab_size;
    std::optional<std::uint32_t> num_attention_heads;
    std::optional<std::uint32_t> num_key_value_heads;
    std::optional<std::uint32_t> head_dim;
    std::optional<std::uint32_t> gdn_key_heads;
    std::optional<std::uint32_t> gdn_value_heads;
    std::optional<std::uint32_t> gdn_key_head_dim;
    std::optional<std::uint32_t> gdn_value_head_dim;
    std::optional<std::uint32_t> gdn_conv_width;
    std::optional<std::uint32_t> full_attention_interval;
    std::optional<std::uint32_t> max_position_embeddings;
};

struct LoadOptions {
    bool load_text = true;
    bool load_mtp = false;
    bool load_vision = false;
    std::vector<std::string> required_text_tensors;
};

class WeightStore {
public:
    explicit WeightStore(Q5090Expectations expected = {});
    void load(const char* path, DeviceArena& weights, DeviceContext& ctx,
              const LoadOptions& options = {});

    const Tensor* tensor(std::string_view name) const noexcept;
    const Weight* qweight(std::string_view name) const noexcept;
    const Tensor* tensor(ModuleKind module, std::uint32_t source_kind,
                         std::uint32_t source_layer) const noexcept;
    const Weight* qweight(ModuleKind module, std::uint32_t source_kind,
                          std::uint32_t source_layer) const noexcept;

    std::size_t tensor_count() const noexcept;
    std::size_t quant_count() const noexcept;
    std::size_t loaded_payload_bytes() const noexcept;
    std::size_t module_tensor_count(ModuleKind module) const noexcept;
    bool module_loaded(ModuleKind module) const noexcept;
    void clear() noexcept;
};
```

## Parser Design

Create device-free internal structs in `weight_store.cpp`:

- `ParsedHeader`
- `ParsedModule`
- `ParsedTensor`
- `ParsedWeightFile`

The parser entry point accepts `std::span<const std::byte>` and expectations:

```cpp
ParsedWeightFile parse_q5090_file(std::span<const std::byte> file,
                                  const Q5090Expectations& expected);
```

Expose this parser through a narrow public header,
`include/qus/core/weight_store_parser.h`, so malformed-input tests can run without
constructing CUDA objects. Keep upload-only code in `WeightStore::load()`.

All integer reads are explicit little-endian helpers, not `reinterpret_cast` of packed
structs. This validates endianness and avoids host alignment dependence.

## Validation Checklist

Header:

- `magic == "Q5090MIXEDV1\0\0\0\0"`
- `version == 1`
- `endian == 0x01020304`
- `header_size == 4096`
- `module_count` in `[1,3]`
- `layer_count == 64` unless caller expectation differs intentionally
- `module_index_bytes == module_count * 64`
- `tensor_index_bytes == tensor_count * 128`
- `module_index_offset == 4096`
- `tensor_index_offset == module_index_offset + module_index_bytes`
- `string_table_offset == tensor_index_offset + tensor_index_bytes`
- `payload_offset >= string_table_offset + string_table_bytes`
- bytes between `string_table_offset + string_table_bytes` and `payload_offset` are zero
- `payload_offset` is 4096 aligned
- `payload_offset + payload_bytes == file.size()`
- `reserved0`, `reserved1`, and header padding bytes are zero
- model dimension fields match supplied expectations

Module records:

- `module_kind` in `{0,1,2}`
- `module_version == 1`
- `load_policy` in `{0,1,2}`
- records are TEXT_CORE then optional MTP_DRAFT then optional VISION_ENCODER
- tensor ranges are contiguous, non-overlapping, and exactly cover the tensor index
- module `payload_offset` equals the first/min payload offset among tensors in the module
- `module.payload_offset + module.payload_bytes` equals the max payload end among tensors
  in the module, including inter-tensor padding inside the span
- reserved bytes are zero

Tensor entries:

- `qtype`, `layout`, `module_kind`, and `scale_dtype` enums in range
- tensor `module_kind` matches the containing module record
- `ndim` in `[1,4]`
- shape dims for active dims are nonzero; inactive dims in the raw entry are one
- padded shape dims for inactive dims are one
- `name_offset + name_len + 1 <= string_table_bytes`
- string table byte after name is NUL
- UTF-8 bytes are copied as canonical names; no fixed-size truncation
- `fnv1a_64(name) == name_hash`
- `payload_offset` is 256 aligned and inside the payload region
- `payload_bytes > 0`
- payload ranges do not overlap
- `crc32(payload) == entry.crc32`
- reserved fields and trailing 32 bytes are zero
- `source_layer <= 63` or `source_layer == 0xFFFFFFFF`
- `source_kind` is one of the q5090 source-kind values listed in the spec
- for quant layouts, `scale_dtype == FP16`; for `CONTIGUOUS`, `scale_dtype == None`
- payload byte count equals the qtype/layout/shape formula below

Payload byte formulas:

- `TILE_N64_K64`: qtype Q4/Q5/Q6 only, group 64, ndim 2.
  - `padded_N == align_up(shape[0], 64)`.
  - `padded_K == align_up(shape[1], 64)`.
  - tile bytes: Q4 2176, Q5 2688, Q6 3200.
  - payload bytes: `(padded_N/64) * (padded_K/64) * tile_bytes`.
- `TILE_N64_K128`: qtype W8 only, group 128, ndim 2.
  - `padded_N == align_up(shape[0], 64)`.
  - `padded_K == align_up(shape[1], 128)`.
  - payload bytes: `(padded_N/64) * (padded_K/128) * 8320`.
- `ROW_GROUPED_G64`: qtype Q6 only, group 64, ndim 2.
  - `padded_N == shape[0]`.
  - `padded_K == align_up(shape[1], 64)`.
  - row group bytes: `2 + 48`.
  - payload bytes: `logical_N * (padded_K/64) * 50`.
- `CONTIGUOUS`: qtype BF16 or FP32 only, group 0.
  - padded shape equals logical shape.
  - payload bytes: `numel * 2` for BF16, `numel * 4` for FP32.

Overflow:

- Use checked add/multiply helpers for every offset, range end, product, and payload
  formula.
- Treat any range whose end wraps or exceeds `file.size()` as invalid.

## Upload and Exception Safety

- `WeightStore::load()` opens and reads the file bytes, calls the parser, validates
  required TEXT_CORE names, then starts CUDA upload.
- Record `const std::size_t mark = weights.mark()` before allocation.
- On any exception after the mark, call `weights.rewind(mark)` and leave store vectors
  unchanged.
- For selected modules only, allocate a raw `DType::U8` tensor of `payload_bytes`.
- Copy bytes H2D exactly. No repacking, transforms, dtype conversion, or scale separation.
- Honor `LoadOptions`:
  - default loads TEXT_CORE only
  - MTP_DRAFT uploads only if `load_mtp == true`
  - VISION_ENCODER uploads only if `load_vision == true`
  - unselected module descriptors remain queryable as metadata, but have null device
    payload pointers and `module_loaded() == false`
- Count APIs include parsed metadata from every module. `tensor_count()`, `quant_count()`,
  and `module_tensor_count()` do not depend on upload selection.
- For unselected modules, `Tensor::data == nullptr` or `Weight::payload == nullptr`,
  while shape, padded shape, qtype/layout, payload bytes, source kind, and source layer
  remain populated.
- `loaded_payload_bytes()` is the sum of selected module `payload_bytes`, including
  inter-tensor padding inside each loaded module span.
- Always parse and validate every module and every payload CRC even if unselected.

## Deletion and Documentation Fixes

Delete:

- pre-q5090 format document
- two-stage pack stub
- two-stage quantize stub

Replace references:

- `README.md` and `docs/design.md` should name `docs/q5090_packed_file_format_v1.md`
  and `tools/q5090_convert/`.
- `docs/l0-infrastructure-design.md` section 5.2, section 10, and section 13 should
  describe q5090 loader behavior and module-selective loading.
- `docs/plans/l0-weight-store.md` and `docs/plans/l0-tensor.md` should be updated or
  marked superseded by this q5090 plan.

Transform ownership fix:

- `docs/qwen3.6-27b-architecture.md` section 11.1 and tensor map section 13 must say:
  weights are stored raw; RMSNorm `+1`, `A = -exp(A_log)`, and related folds are runtime
  responsibilities.
- `docs/l0-infrastructure-design.md` must say L0 loader is transform-neutral and does
  not apply numeric folds.
- No C++ loader code may implement RMSNorm `+1`, `A_log` exponentiation, conv reshaping,
  or other numeric transforms.

## TDD Task List

### Task 1: Fixture generator

**Files:**
- Create: `tests/fixtures/make_q5090_fixture.py`

- [ ] Write a helper that imports `tools.q5090_convert.format`, `qtypes`, and
  `layouts.encode_tensor()`.
- [ ] Emit a tiny q5090 file with TEXT_CORE, MTP_DRAFT, and VISION_ENCODER modules.
- [ ] Include at least:
  - Q4 `TILE_N64_K64`
  - Q5 `TILE_N64_K64`
  - Q6 `TILE_N64_K64`
  - W8 `TILE_N64_K128`
  - Q6 `ROW_GROUPED_G64`
  - BF16 `CONTIGUOUS`
  - FP32 `CONTIGUOUS`
- [ ] Use deterministic tiny tensors and `encode_tensor()` so payload layouts come from the
  real converter.
- [ ] Use converter `pack_header`, `pack_module_record`, `pack_tensor_entry`,
  `build_string_table`, `crc32`, and `fnv1a_64` helpers.
- [ ] Command: `python3 tests/fixtures/make_q5090_fixture.py --out /tmp/qus_fixture.qus`
- [ ] Expected: file is written and can be parsed by the C++ host parser test. Do not
  use `tools.q5090_convert.verify` here because it requires a source model path even for
  `--quick`.

### Task 2: Host parser RED tests

**Files:**
- Create: `tests/test_q5090_parser.cpp`
- Modify: `CMakeLists.txt`

- [ ] Add tests that load the synthetic file into a byte vector and call the device-free
  parser declared by `include/qus/core/weight_store_parser.h`.
- [ ] Assert valid header dims, module count, tensor count, names, enum metadata, shapes,
  padded shapes, source kind/layer, payload bytes, CRC, and FNV.
- [ ] Add malformed cases for bad magic, version, endian, header size, nonzero reserved,
  bad section ordering, OOB ranges, overlapping ranges, unaligned payload, bad enum tags,
  CRC mismatch, name hash mismatch, shape/padded payload mismatch, dimension mismatch,
  and integer overflow.
- [ ] Add `qus_q5090_parser_test` as a host-only target that links `qus_core` but does
  not construct `DeviceContext`, `DeviceArena`, or call CUDA device discovery.
- [ ] Run: `cmake --build build -j && ctest --test-dir build -R qus_q5090_parser_test --output-on-failure`
- [ ] Historical RED expectation: this failed before the q5090 parser/API existed; in the current
  implementation this path should build and pass when run in a suitable environment.

### Task 3: Parser GREEN implementation

**Files:**
- Modify: `include/qus/core/tensor.h`
- Modify: `include/qus/core/weight_store.h`
- Create: `include/qus/core/weight_store_parser.h`
- Rewrite: `src/core/weight_store.cpp`

- [ ] Implement q5090 enums and descriptors.
- [ ] Implement little-endian readers, checked math, FNV-1a-64, CRC-32 with polynomial
  `0xEDB88320`.
- [ ] Implement `parse_q5090_file()` and all validation from this plan.
- [ ] Run the host parser test until it passes.

### Task 4: Arena rewind RED/GREEN

**Files:**
- Modify: `include/qus/core/arena.h`
- Modify: `src/core/arena.cu`
- Modify: `tests/test_arena.cpp`

- [ ] Add failing test that allocates, records mark, allocates again, rewinds, and checks
  `used()` returns to the mark.
- [ ] Implement `mark()` and `rewind(mark)`.
- [ ] Reject marks beyond `capacity()` by clamping only if tests require it; preferred
  behavior is no-throw rewind for marks previously returned by `mark()`.
- [ ] Run: `ctest --test-dir build -R qus_arena_test --output-on-failure`.

### Task 5: GPU upload and module selection RED tests

**Files:**
- Modify: `tests/test_weight_store.cpp`

- [ ] Use CUDA-gated path that skips on no device.
- [ ] Load the synthetic file with default `LoadOptions`.
- [ ] Assert TEXT_CORE payloads upload byte-exact via D2H.
- [ ] Assert MTP_DRAFT descriptors exist but are not loaded by default.
- [ ] Assert VISION_ENCODER descriptors exist but are not loaded by default.
- [ ] Load again with `load_mtp = true`; assert MTP payload byte-exact and VISION remains
  metadata-only.
- [ ] Load with `load_vision = true`; assert VISION payload byte-exact and MTP remains
  metadata-only.
- [ ] Load with both flags; assert TEXT, MTP, and VISION payloads are all uploaded.
- [ ] Add failure test for missing required TEXT_CORE tensor and for small arena rewind.
- [ ] Historical RED expectation: this failed before upload/options API existed; in the current
  implementation this path should build and pass when run in a suitable environment.

### Task 6: GPU upload GREEN implementation

**Files:**
- Modify: `include/qus/core/weight_store.h`
- Modify: `src/core/weight_store.cpp`

- [ ] Implement `LoadOptions`.
- [ ] Implement selected-module allocation and H2D copy.
- [ ] Preserve metadata for unselected modules with null payload pointers.
- [ ] Implement lookup by name and `module + source_kind + source_layer`.
- [ ] Ensure `WeightStore` vectors swap only after successful upload.
- [ ] Run: `ctest --test-dir build -R qus_weight_store_test --output-on-failure`.

### Task 7: Real-file integration

**Files:**
- Modify: `tests/test_weight_store.cpp`

- [ ] If `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus` and `out/manifest.json` exist, parse
  both.
- [ ] Validate manifest segment counts: TEXT_CORE 963, MTP_DRAFT 15, VISION_ENCODER 333.
- [ ] Validate all 1311 entries and every CRC.
- [ ] GPU-gated default load uploads TEXT_CORE. If memory is insufficient, print a clear
  skip instead of failing developer machines without a 32 GB GPU.
- [ ] Optional MTP load uploads MTP_DRAFT when there is enough free memory.
- [ ] Keep this test separate or self-skipping so normal fast unit tests remain useful.

### Task 8: Delete obsolete pipeline and docs fix

**Files:**
- Delete: pre-q5090 format document and two-stage pack/quantize stubs
- Modify: `README.md`, `docs/design.md`, `docs/l0-infrastructure-design.md`,
  `docs/qwen3.6-27b-architecture.md`, `docs/plans/l0-weight-store.md`,
  `docs/plans/l0-tensor.md`

- [ ] Replace all stale pre-q5090 file-format and two-stage tooling references.
- [ ] Replace old offline-transform wording with "runtime applies
  transforms; loader stores raw bytes."
- [ ] Run a stale-reference grep for old format/tooling names and old offline-transform phrases.
- [ ] Expected: no stale references.

### Task 9: Full verification

- [ ] Run `clang-format -i` on changed C/C++/CUDA files.
- [ ] Run `cmake --build build -j`.
- [ ] Run `ctest --test-dir build --output-on-failure`.
- [ ] Run converter Python tests: `python3 -m pytest tools/q5090_convert/tests` if pytest is
  installed; otherwise run the fixture generator and document the missing pytest dependency.
- [ ] Run real-file parser/CRC integration test if `out/*.qus` exists.

## Review Gates

Plan review:

- Fresh subagent compares this plan to `docs/q5090_packed_file_format_v1.md`,
  `tools/q5090_convert/*.py`, and pasted requirements.
- Reviewer checks byte-layout fidelity, validation coverage, no external dependencies,
  transform neutrality, module selection, and deletion completeness.
- Historical note: this review gate required revising the plan before implementation; the file is
  now retained as post-implementation history.

Implementation review:

- Fresh subagent reviews code against this plan and q5090 spec.
- Reviewer focuses on bounds/overflow/overlap, CRC/FNV correctness, exception safety,
  no external C++ deps, and tests that exercise the real converter fixture.

Final review:

- Fresh subagent checks whole pipeline: no obsolete pre-q5090 code path remains, docs are
  reconciled, synthetic and real q5090 files parse, module-selective loading works, and
  build/tests are green.

## Completion Criteria

- No obsolete pre-q5090 loader or files remain.
- Loader parses and validates q5090 metadata and payload CRCs for all modules.
- Default load uploads TEXT_CORE only; MTP/VISION upload only when requested.
- `Weight` describes inline-scale q5090 payloads and never assumes separate FP32
  scales.
- Loader applies no numeric transforms.
- Required TEXT_CORE tensor names can be enforced by caller-supplied options.
- Synthetic converter-backed tests, malformed parser tests, module-selective GPU tests,
  and real-file integration checks pass or skip only for documented environment limits.
- `cmake --build build -j` and `ctest --test-dir build --output-on-failure` pass.
