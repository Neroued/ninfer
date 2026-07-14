#include "ninfer/core/arena.h"
#include "ninfer/core/device.h"
#include "ninfer/core/weight_store.h"
#include "ninfer/core/weight_store_parser.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

constexpr std::uint64_t kSegmentRecordSize = 32;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "ninfer_q5090_weight_store_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open fixture"); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

void write_file(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("failed to create fixture"); }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) { throw std::runtime_error("failed to write fixture"); }
}

std::uint64_t read_u64(const std::vector<std::byte>& bytes, std::uint64_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void write_u32(std::vector<std::byte>& bytes, std::uint64_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

const ninfer::ParsedQ5090Tensor& find_tensor(const ninfer::ParsedQ5090File& parsed,
                                          std::string_view name) {
    for (const ninfer::ParsedQ5090Tensor& tensor : parsed.tensors) {
        if (tensor.name == name) { return tensor; }
    }
    throw std::runtime_error("tensor not found");
}

const ninfer::ParsedQ5090Segment& find_segment(const ninfer::ParsedQ5090File& parsed,
                                            std::string_view name) {
    for (const ninfer::ParsedQ5090Segment& segment : parsed.segments) {
        if (segment.name == name) { return segment; }
    }
    throw std::runtime_error("segment not found");
}

std::size_t find_segment_index(const ninfer::ParsedQ5090File& parsed, std::string_view name) {
    for (std::size_t i = 0; i < parsed.segments.size(); ++i) {
        if (parsed.segments[i].name == name) { return i; }
    }
    throw std::runtime_error("segment not found");
}

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t align) {
    return ((value + align - 1) / align) * align;
}

std::uint64_t nibble_bytes_per_group(ninfer::QType qtype) {
    switch (qtype) {
    case ninfer::QType::Q4G64_F16S:
    case ninfer::QType::Q5G64_F16S:
    case ninfer::QType::Q6G64_F16S:
    case ninfer::QType::W8G32_F16S:
        return 32;
    default:
        throw std::runtime_error("unexpected row-split qtype");
    }
}

std::uint64_t high_bytes_per_group(ninfer::QType qtype) {
    switch (qtype) {
    case ninfer::QType::Q4G64_F16S:
    case ninfer::QType::W8G32_F16S:
        return 0;
    case ninfer::QType::Q5G64_F16S:
        return 8;
    case ninfer::QType::Q6G64_F16S:
        return 16;
    default:
        throw std::runtime_error("unexpected row-split qtype");
    }
}

int expect_plane_offsets(const ninfer::Weight& weight, const ninfer::ParsedQ5090Tensor& tensor,
                         const ninfer::ParsedQ5090Segment& segment, std::string_view label) {
    if (weight.payload == nullptr) { return fail(std::string(label) + " payload null"); }
    const std::uint64_t groups     = tensor.padded_shape[1] / tensor.group_size;
    const std::uint64_t nibble_row = groups * nibble_bytes_per_group(tensor.qtype);
    const std::uint64_t high_row   = groups * high_bytes_per_group(tensor.qtype);
    const std::uint64_t scale_row  = groups * 2ULL;
    const std::uint64_t high_rel   = align_up_u64(tensor.nibble_plane_bytes, 256);
    const std::uint64_t scale_rel  = high_rel + align_up_u64(tensor.high_plane_bytes, 256);
    const auto* base               = static_cast<const std::byte*>(weight.payload);
    int failures                   = 0;
    failures += weight.high_plane_bytes == tensor.high_plane_bytes
                    ? 0
                    : fail(std::string(label) + " high_plane_bytes mismatch");
    failures += weight.qdata == base + segment.row_begin * nibble_row
                    ? 0
                    : fail(std::string(label) + " qdata offset mismatch");
    if (high_row == 0) {
        failures += weight.qhigh == nullptr ? 0 : fail(std::string(label) + " qhigh non-null");
    } else {
        failures += weight.qhigh == base + high_rel + segment.row_begin * high_row
                        ? 0
                        : fail(std::string(label) + " qhigh offset mismatch");
    }
    failures += weight.scales == base + scale_rel + segment.row_begin * scale_row
                    ? 0
                    : fail(std::string(label) + " scales offset mismatch");
    return failures;
}

std::vector<std::byte> payload_bytes(const std::vector<std::byte>& file,
                                     const ninfer::ParsedQ5090Tensor& tensor) {
    const auto begin = file.begin() + static_cast<std::ptrdiff_t>(tensor.payload_offset);
    const auto end   = begin + static_cast<std::ptrdiff_t>(tensor.payload_bytes);
    return std::vector<std::byte>(begin, end);
}

int expect_device_bytes(const void* device, const std::vector<std::byte>& expected,
                        std::string_view label) {
    std::vector<std::byte> actual(expected.size());
    const cudaError_t err =
        cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " cudaMemcpy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual != expected) {
        std::cerr << label << " payload mismatch\n";
        return 1;
    }
    return 0;
}

std::uint32_t segment_row_sum(const ninfer::ParsedQ5090File& parsed,
                              const ninfer::ParsedQ5090Tensor& tensor) {
    std::uint32_t rows = 0;
    for (std::uint32_t i = 0; i < tensor.segment_count; ++i) {
        rows += parsed.segments[static_cast<std::size_t>(tensor.segment_begin + i)].row_count;
    }
    return rows;
}

int expect_counts(const ninfer::WeightStore& store, std::size_t quant_count,
                  std::uint64_t loaded_bytes) {
    int failures = 0;
    failures += store.tensor_count() == 20 ? 0 : fail("tensor_count mismatch");
    failures += store.quant_count() == quant_count ? 0 : fail("quant_count mismatch");
    failures +=
        store.module_tensor_count(ninfer::ModuleKind::TextCore) == 6 ? 0 : fail("TEXT count mismatch");
    failures +=
        store.module_tensor_count(ninfer::ModuleKind::MtpDraft) == 12 ? 0 : fail("MTP count mismatch");
    failures += store.module_tensor_count(ninfer::ModuleKind::VisionEncoder) == 2
                    ? 0
                    : fail("VISION count mismatch");
    failures +=
        store.loaded_payload_bytes() == loaded_bytes ? 0 : fail("loaded payload bytes mismatch");
    return failures;
}

int expect_default_text_load(const ninfer::WeightStore& store, const ninfer::ParsedQ5090File& parsed,
                             const std::vector<std::byte>& file) {
    int failures = 0;
    failures += expect_counts(store, 5, parsed.modules[0].payload_bytes);
    failures += store.module_loaded(ninfer::ModuleKind::TextCore) ? 0 : fail("TEXT not loaded");
    failures += !store.module_loaded(ninfer::ModuleKind::MtpDraft) ? 0 : fail("MTP loaded by default");
    failures +=
        !store.module_loaded(ninfer::ModuleKind::VisionEncoder) ? 0 : fail("VISION loaded by default");

    const auto& text_q             = find_tensor(parsed, "layers.0.mlp.down_proj.weight");
    const ninfer::Weight* text_weight = store.qweight("layers.0.mlp.down_proj.weight");
    failures += text_weight != nullptr ? 0 : fail("missing text quant weight");
    if (text_weight != nullptr) {
        failures += text_weight->payload != nullptr ? 0 : fail("text quant payload is null");
        failures += text_weight->payload_bytes == text_q.payload_bytes
                        ? 0
                        : fail("text payload bytes mismatch");
        failures +=
            text_weight->layout == ninfer::QuantLayout::RowSplit ? 0 : fail("text layout mismatch");
        failures += text_weight->qhigh != nullptr ? 0 : fail("text qhigh null");
        failures += text_weight->scales != nullptr ? 0 : fail("text scales null");
        failures += expect_plane_offsets(*text_weight, text_q,
                                         find_segment(parsed, "layers.0.mlp.down_proj.weight"),
                                         "text qweight");
        failures +=
            expect_device_bytes(text_weight->payload, payload_bytes(file, text_q), "text qweight");
    }

    const ninfer::Weight* gate = store.qweight("layers.0.mlp.gate_proj.weight");
    const ninfer::Weight* up   = store.qweight("layers.0.mlp.up_proj.weight");
    failures += gate != nullptr ? 0 : fail("missing fused gate segment");
    failures += up != nullptr ? 0 : fail("missing fused up segment");
    if (gate != nullptr && up != nullptr) {
        failures += gate->n == 5 && gate->k == 7 ? 0 : fail("gate segment shape mismatch");
        failures += up->n == 4 && up->k == 7 ? 0 : fail("up segment shape mismatch");
        failures += gate->payload == up->payload ? 0 : fail("fused segments should share payload");
        failures +=
            gate->qdata != nullptr && gate->scales != nullptr ? 0 : fail("gate planes null");
        failures += up->qdata != nullptr && up->scales != nullptr ? 0 : fail("up planes null");
        failures += gate->qhigh == nullptr ? 0 : fail("gate qhigh should be null");
        failures += up->qhigh == nullptr ? 0 : fail("up qhigh should be null");
        failures += expect_plane_offsets(*gate, find_tensor(parsed, "layers.0.mlp.gateup"),
                                         find_segment(parsed, "layers.0.mlp.gate_proj.weight"),
                                         "gate segment");
        failures +=
            expect_plane_offsets(*up, find_tensor(parsed, "layers.0.mlp.gateup"),
                                 find_segment(parsed, "layers.0.mlp.up_proj.weight"), "up segment");
    }
    const auto& gateup_tensor = find_tensor(parsed, "layers.0.mlp.gateup");
    const ninfer::Weight* gateup = store.qfused(ninfer::ModuleKind::TextCore, /*MLP_GATEUP*/ 3, 0, 0);
    failures += gateup != nullptr ? 0 : fail("missing fused gate/up block");
    if (gateup != nullptr && gate != nullptr) {
        failures += gateup->n == static_cast<std::int32_t>(segment_row_sum(parsed, gateup_tensor))
                        ? 0
                        : fail("fused gate/up row sum mismatch");
        failures += gateup->k == 7 ? 0 : fail("fused gate/up K mismatch");
        failures += gateup->qtype == ninfer::QType::Q4G64_F16S ? 0 : fail("fused gate/up qtype");
        failures += gateup->source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::Other)
                        ? 0
                        : fail("fused gate/up source_kind");
        failures +=
            gateup->qdata == gate->qdata ? 0 : fail("fused gate/up qdata should start at gate");
        failures +=
            gateup->scales == gate->scales ? 0 : fail("fused gate/up scales should start at gate");
    }

    const auto& text_tensor_meta   = find_tensor(parsed, "layers.0.input_layernorm.weight");
    const ninfer::Tensor* text_tensor = store.tensor("layers.0.input_layernorm.weight");
    failures += text_tensor != nullptr ? 0 : fail("missing text tensor");
    if (text_tensor != nullptr) {
        failures += text_tensor->data != nullptr ? 0 : fail("text tensor payload is null");
        failures += expect_device_bytes(text_tensor->data, payload_bytes(file, text_tensor_meta),
                                        "text tensor");
    }

    const ninfer::Weight* by_source = store.qweight(
        ninfer::ModuleKind::TextCore, static_cast<std::uint32_t>(ninfer::SourceKind::MlpDown), 0);
    failures += by_source == text_weight ? 0 : fail("source lookup mismatch");

    failures += store.qweight("mtp.fc.weight") == nullptr
                    ? 0
                    : fail("unselected MTP descriptor was published");
    failures += store.qweight("model.visual.patch_embed.proj.weight") == nullptr
                    ? 0
                    : fail("unselected VISION descriptor was published");
    return failures;
}

int expect_module_payload(const ninfer::WeightStore& store, const ninfer::ParsedQ5090File& parsed,
                          const std::vector<std::byte>& file, std::string_view name,
                          bool should_be_loaded) {
    const ninfer::ParsedQ5090Tensor& meta = find_tensor(parsed, name);
    const ninfer::Weight* weight          = store.qweight(name);
    if (!should_be_loaded) {
        return weight == nullptr ? 0 : fail("unselected quant descriptor was published");
    }
    if (weight == nullptr) { return fail("missing loaded quant weight"); }
    int failures = 0;
    failures += weight->payload != nullptr ? 0 : fail("expected loaded payload");
    failures +=
        weight->payload_bytes == meta.payload_bytes ? 0 : fail("module payload size mismatch");
    if (weight->payload != nullptr) {
        failures += expect_device_bytes(weight->payload, payload_bytes(file, meta), name);
    }
    return failures;
}

int expect_mtp_expectations_reject_swapped_attn_segments(const std::vector<std::byte>& valid,
                                                         const ninfer::ParsedQ5090File& parsed,
                                                         ninfer::DeviceContext& ctx) {
    std::vector<std::byte> bad         = valid;
    const std::uint64_t segment_offset = read_u64(bad, 200);
    const std::size_t k_index = find_segment_index(parsed, "mtp.layers.0.self_attn.k_proj.weight");
    const std::size_t gate_index = find_segment_index(parsed, "mtp.layers.0.self_attn.q_proj.gate");
    write_u32(bad, segment_offset + k_index * kSegmentRecordSize,
              static_cast<std::uint32_t>(ninfer::SourceKind::AttnGate));
    write_u32(bad, segment_offset + gate_index * kSegmentRecordSize,
              static_cast<std::uint32_t>(ninfer::SourceKind::AttnK));

    const auto path =
        std::filesystem::temp_directory_path() / "ninfer_q5090_bad_mtp_attn_segments.qus";
    write_file(path, bad);

    ninfer::LoadOptions options;
    options.load_mtp = true;
    ninfer::WeightStore store;
    store.load(path.c_str(), ctx, options);
    try {
        store.require_mtp_module_expectations();
    } catch (const std::runtime_error&) { return 0; }
    return fail("swapped MTP attn segment source kinds were accepted");
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures                             = 0;
    const std::filesystem::path fixture_path = make_fixture();
    const std::vector<std::byte> file        = read_file(fixture_path);
    const ninfer::ParsedQ5090File parsed        = ninfer::parse_q5090_file(file);
    ninfer::DeviceContext ctx(0);

    ninfer::WeightStore default_store;
    default_store.load(fixture_path.c_str(), ctx);
    failures += expect_default_text_load(default_store, parsed, file);
    const ninfer::DeviceArena* default_arena = default_store.module_arena(ninfer::ModuleKind::TextCore);
    failures += default_arena != nullptr ? 0 : fail("default TEXT arena missing");
    failures +=
        default_arena != nullptr && default_arena->used() == default_store.loaded_payload_bytes()
            ? 0
            : fail("default arena exact used bytes mismatch");
    failures += default_arena != nullptr &&
                        default_arena->capacity() == default_store.loaded_payload_bytes()
                    ? 0
                    : fail("default arena exact capacity mismatch");
    failures += default_store.load_stats().total_file_read_bytes ==
                        default_store.load_plan().file_read_bytes
                    ? 0
                    : fail("default planned/actual file read bytes mismatch");
    default_store.prepare(fixture_path.c_str());
    default_store.upload(ctx);
    default_store.clear();
    failures += default_store.tensor_count() == 0 ? 0 : fail("clear tensor_count mismatch");
    failures += default_store.quant_count() == 0 ? 0 : fail("clear quant_count mismatch");
    failures += default_store.qfused(ninfer::ModuleKind::TextCore, 3, 0, 0) == nullptr
                    ? 0
                    : fail("clear fused lookup mismatch");
    failures += default_store.loaded_payload_bytes() == 0 ? 0 : fail("clear loaded bytes mismatch");
    failures += !default_store.module_loaded(ninfer::ModuleKind::TextCore)
                    ? 0
                    : fail("clear TEXT module still loaded");

    ninfer::LoadOptions mtp_options;
    mtp_options.load_mtp = true;
    ninfer::WeightStore mtp_store;
    mtp_store.load(fixture_path.c_str(), ctx, mtp_options);
    mtp_store.require_mtp_module_expectations();
    failures += mtp_store.module_loaded(ninfer::ModuleKind::MtpDraft) ? 0 : fail("MTP not loaded");
    failures += !mtp_store.module_loaded(ninfer::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("VISION loaded with MTP");
    failures += expect_module_payload(mtp_store, parsed, file, "mtp.fc.weight", true);
    const ninfer::Weight* mtp_attn = mtp_store.qfused(ninfer::ModuleKind::MtpDraft, /*ATTN_IN*/ 1, 0, 0);
    failures += mtp_attn != nullptr ? 0 : fail("missing MTP attn fused block");
    if (mtp_attn != nullptr) {
        failures += mtp_attn->qtype == ninfer::QType::W8G32_F16S ? 0 : fail("MTP attn qtype");
        failures += mtp_attn->group_size == 32 ? 0 : fail("MTP attn group_size");
        failures += mtp_attn->qhigh == nullptr ? 0 : fail("MTP attn qhigh");
        failures += mtp_attn->high_plane_bytes == 0 ? 0 : fail("MTP attn high_plane_bytes");
    }
    failures += expect_module_payload(mtp_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", false);

    ninfer::LoadOptions vision_options;
    vision_options.load_vision = true;
    ninfer::WeightStore vision_store;
    vision_store.load(fixture_path.c_str(), ctx, vision_options);
    failures +=
        vision_store.module_loaded(ninfer::ModuleKind::VisionEncoder) ? 0 : fail("VISION not loaded");
    failures +=
        !vision_store.module_loaded(ninfer::ModuleKind::MtpDraft) ? 0 : fail("MTP loaded with VISION");
    failures += expect_module_payload(vision_store, parsed, file, "mtp.fc.weight", false);
    failures += expect_module_payload(vision_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", true);

    ninfer::LoadOptions all_options;
    all_options.load_mtp    = true;
    all_options.load_vision = true;
    ninfer::WeightStore all_store;
    all_store.load(fixture_path.c_str(), ctx, all_options);
    all_store.require_mtp_module_expectations();
    failures +=
        all_store.module_loaded(ninfer::ModuleKind::TextCore) ? 0 : fail("TEXT not loaded with all");
    failures +=
        all_store.module_loaded(ninfer::ModuleKind::MtpDraft) ? 0 : fail("MTP not loaded with all");
    failures += all_store.module_loaded(ninfer::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("VISION not loaded with all");
    failures += expect_module_payload(all_store, parsed, file, "mtp.fc.weight", true);
    failures += expect_module_payload(all_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", true);
    failures += expect_mtp_expectations_reject_swapped_attn_segments(file, parsed, ctx);

    ninfer::WeightStore callback_failure_store;
    ninfer::Q5090Progress throwing_progress;
    throwing_progress.min_interval_bytes = 1;
    throwing_progress.callback = [](std::string_view phase, std::uint64_t done, std::uint64_t) {
        if (phase == "upload selected payloads" && done > 0) {
            throw std::runtime_error("injected upload progress failure");
        }
    };
    failures += expect_throws<std::runtime_error>(
        [&] {
            ninfer::LoadOptions options;
            options.progress = &throwing_progress;
            callback_failure_store.load(fixture_path.c_str(), ctx, options);
        },
        "upload callback failure");
    failures += callback_failure_store.module_arena(ninfer::ModuleKind::TextCore) == nullptr &&
                        callback_failure_store.qweight("layers.0.mlp.down_proj.weight") == nullptr
                    ? 0
                    : fail("failed upload published resident state");
    failures += callback_failure_store.load_stats().fail_stage == "upload"
                    ? 0
                    : fail("failed upload stage was not recorded");

    ninfer::WeightStore split_progress_store;
    int upload_progress_calls = 0;
    {
        ninfer::Q5090Progress short_lived_progress;
        short_lived_progress.min_interval_bytes = 1;
        short_lived_progress.callback           = [&](std::string_view phase, std::uint64_t done,
                                            std::uint64_t) {
            if (phase == "upload selected payloads" && done > 0) { ++upload_progress_calls; }
        };
        ninfer::LoadOptions options;
        options.progress = &short_lived_progress;
        split_progress_store.prepare(fixture_path.c_str(), options);
    }
    split_progress_store.upload(ctx);
    failures += upload_progress_calls > 0
                    ? 0
                    : fail("split prepare/upload did not retain progress callback safely");

    failures += expect_throws<std::invalid_argument>(
        [&] {
            ninfer::LoadOptions options;
            options.load_lm_head_draft = true;
            ninfer::WeightStore store;
            store.prepare(fixture_path.c_str(), options);
        },
        "draft residency without MTP");

    failures += expect_throws<std::runtime_error>(
        [&] {
            ninfer::LoadOptions options;
            options.required_text_tensors.push_back("missing.text.tensor");
            ninfer::WeightStore store;
            store.prepare(fixture_path.c_str(), options);
        },
        "missing required text tensor");

    return failures == 0 ? 0 : fail("weight store q5090 test failed");
}
