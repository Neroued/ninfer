#include "ninfer/core/arena.h"
#include "ninfer/core/device.h"
#include "ninfer/core/weight_store.h"
#include "ninfer/core/weight_store_parser.h"

#include "../third_party/nlohmann/json.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kGiB                = 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kProgressChunkBytes = 256ULL * 1024ULL * 1024ULL;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

class ProgressPrinter {
public:
    void operator()(std::string_view phase, std::uint64_t done, std::uint64_t total) {
        using Clock    = std::chrono::steady_clock;
        const auto now = Clock::now();
        if (phase_ != phase) {
            phase_       = std::string(phase);
            phase_start_ = now;
        }

        const double elapsed   = std::chrono::duration<double>(now - phase_start_).count();
        const double done_gib  = static_cast<double>(done) / static_cast<double>(kGiB);
        const double total_gib = static_cast<double>(total) / static_cast<double>(kGiB);
        const double percent =
            total == 0 ? 100.0 : (static_cast<double>(done) * 100.0) / static_cast<double>(total);
        const double rate_gib_s = elapsed > 0.0 ? done_gib / elapsed : 0.0;
        const double eta_s =
            rate_gib_s > 0.0 && done < total ? (total_gib - done_gib) / rate_gib_s : 0.0;

        std::cerr << "[q5090] " << phase_ << ' ' << std::fixed << std::setprecision(2) << done_gib
                  << '/' << total_gib << " GiB (" << std::setprecision(1) << percent << "%)";
        if (rate_gib_s > 0.0) {
            std::cerr << " rate " << std::setprecision(2) << rate_gib_s << " GiB/s ETA "
                      << std::setprecision(0) << eta_s << 's';
        } else {
            std::cerr << " rate -- ETA --";
        }
        std::cerr << '\n' << std::flush;
    }

private:
    std::string phase_;
    std::chrono::steady_clock::time_point phase_start_{std::chrono::steady_clock::now()};
};

class ProgressThrottle {
public:
    explicit ProgressThrottle(ninfer::Q5090Progress* progress) : progress_(progress) {}

    void report(std::string_view phase, std::uint64_t done, std::uint64_t total,
                bool force = false) {
        if (progress_ == nullptr || !progress_->callback) { return; }
        if (!force && done < last_done_ + progress_->min_interval_bytes && done < total) { return; }
        last_done_ = done;
        progress_->callback(phase, done, total);
    }

private:
    ninfer::Q5090Progress* progress_ = nullptr;
    std::uint64_t last_done_      = 0;
};

std::vector<std::byte> read_binary(const std::filesystem::path& path,
                                   ninfer::Q5090Progress* progress) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open real q5090 file"); }
    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < 0) { throw std::runtime_error("failed to size real q5090 file"); }
    in.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    ProgressThrottle reporter(progress);
    std::uint64_t done        = 0;
    const std::uint64_t total = bytes.size();
    reporter.report("inventory read", 0, total, true);
    auto* out = reinterpret_cast<char*>(bytes.data());
    while (done < total) {
        const std::uint64_t chunk = std::min(kProgressChunkBytes, total - done);
        in.read(out + done, static_cast<std::streamsize>(chunk));
        if (!in) { throw std::runtime_error("failed to read real q5090 file"); }
        done += chunk;
        reporter.report("inventory read", done, total);
    }
    reporter.report("inventory read", total, total, true);
    return bytes;
}

Json read_manifest_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("failed to open manifest"); }
    Json manifest;
    try {
        in >> manifest;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("failed to parse manifest JSON: ") + e.what());
    }
    return manifest;
}

int expect_string_field(const Json& object, const char* key, const char* expected,
                        const char* message) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string() || it->get<std::string>() != expected) {
        return fail(message);
    }
    return 0;
}

int expect_uint_field(const Json& object, const char* key, std::uint64_t expected,
                      const char* message) {
    const auto it = object.find(key);
    if (it == object.end()) { return fail(message); }
    if (it->is_number_unsigned()) {
        return it->get<std::uint64_t>() == expected ? 0 : fail(message);
    }
    if (it->is_number_integer()) {
        const std::int64_t value = it->get<std::int64_t>();
        return value >= 0 && static_cast<std::uint64_t>(value) == expected ? 0 : fail(message);
    }
    return fail(message);
}

int expect_bool_field(const Json& object, const char* key, bool expected, const char* message) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_boolean() || it->get<bool>() != expected) {
        return fail(message);
    }
    return 0;
}

template <std::size_t N>
int expect_string_array_field(const Json& object, const char* key,
                              const std::array<const char*, N>& expected, const char* message) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_array() || it->size() != expected.size()) {
        return fail(message);
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!(*it)[i].is_string() || (*it)[i].get<std::string>() != expected[i]) {
            return fail(message);
        }
    }
    return 0;
}

int expect_empty_array_field(const Json& object, const char* key, const char* message) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_array() || !it->empty()) { return fail(message); }
    return 0;
}

template <std::size_t N>
int expect_uint_array_field(const Json& object, const char* key,
                            const std::array<std::uint64_t, N>& expected, const char* message) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_array() || it->size() != expected.size()) {
        return fail(message);
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const Json& value = (*it)[i];
        if (value.is_number_unsigned()) {
            if (value.get<std::uint64_t>() != expected[i]) { return fail(message); }
        } else if (value.is_number_integer()) {
            const std::int64_t signed_value = value.get<std::int64_t>();
            if (signed_value < 0 || static_cast<std::uint64_t>(signed_value) != expected[i]) {
                return fail(message);
            }
        } else {
            return fail(message);
        }
    }
    return 0;
}

int expect_manifest_fields(const Json& manifest, std::uint64_t file_size) {
    if (!manifest.is_object()) { return fail("manifest root must be object"); }

    int failures = 0;
    failures += expect_string_field(manifest, "format", "q5090_w4g64_mixed_v4_2",
                                    "manifest format mismatch");
    failures +=
        expect_uint_field(manifest, "format_version", 4, "manifest format_version mismatch");
    failures += expect_uint_field(manifest, "format_minor", 2, "manifest format_minor mismatch");
    failures += expect_string_field(manifest, "binary_spec", "docs/q5090_packed_file_format_v4.md",
                                    "manifest binary_spec mismatch");
    failures += expect_string_field(manifest, "tensor_plan", "docs/q5090_packed_file_format_v4.md",
                                    "manifest tensor_plan mismatch");
    failures +=
        expect_string_field(manifest, "weights_file", "qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus",
                            "manifest weights_file mismatch");
    failures +=
        expect_uint_field(manifest, "file_bytes", file_size, "manifest file_bytes mismatch");
    failures += expect_bool_field(manifest, "calibrated", false, "manifest calibrated mismatch");
    failures += expect_bool_field(manifest, "lm_head_draft_present", true,
                                  "manifest lm_head_draft_present mismatch");
    failures +=
        expect_empty_array_field(manifest, "absent_modules", "manifest absent_modules mismatch");
    failures += expect_string_array_field(
        manifest, "qtypes",
        std::array<const char*, 7>{"Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S",
                                   "W8G32_F16S", "BF16_CTRL", "FP32_CTRL", "I32_CTRL"},
        "manifest qtypes mismatch");
    failures += expect_string_array_field(manifest, "layouts",
                                          std::array<const char*, 2>{"ROW_SPLIT", "CONTIGUOUS"},
                                          "manifest layouts mismatch");
    failures += expect_string_array_field(manifest, "code_planes",
                                          std::array<const char*, 3>{"nibble", "high", "scale"},
                                          "manifest code_planes mismatch");
    failures += expect_string_array_field(
        manifest, "modules",
        std::array<const char*, 4>{"TEXT_CORE", "LM_HEAD_DRAFT", "MTP_DRAFT", "VISION_ENCODER"},
        "manifest modules mismatch");
    failures += expect_uint_field(manifest, "module_count", 4, "manifest module_count mismatch");
    failures += expect_uint_field(manifest, "tensor_count", 1166, "manifest tensor_count mismatch");
    failures +=
        expect_uint_field(manifest, "segment_count", 1314, "manifest segment_count mismatch");
    failures += expect_uint_field(manifest, "fusion_group_count", 130,
                                  "manifest fusion_group_count mismatch");

    const auto draft_it = manifest.find("lm_head_draft");
    if (draft_it == manifest.end() || !draft_it->is_object()) {
        failures += fail("manifest lm_head_draft mismatch");
    } else {
        failures += expect_bool_field(*draft_it, "present", true,
                                      "manifest lm_head_draft.present mismatch");
        failures += expect_uint_field(*draft_it, "n", 131072, "manifest lm_head_draft.n mismatch");
        failures += expect_uint_field(*draft_it, "k", 5120, "manifest lm_head_draft.k mismatch");
        failures += expect_string_field(*draft_it, "weights_qtype", "Q4G64_F16S",
                                        "manifest lm_head_draft.weights_qtype mismatch");
        failures += expect_string_field(*draft_it, "idmap_qtype", "I32_CTRL",
                                        "manifest lm_head_draft.idmap_qtype mismatch");
        failures += expect_uint_field(*draft_it, "idmap_payload_bytes", 131072ULL * 4ULL,
                                      "manifest lm_head_draft.idmap_payload_bytes mismatch");
    }

    const auto alignment_it = manifest.find("alignment");
    if (alignment_it == manifest.end() || !alignment_it->is_object()) {
        failures += fail("manifest alignment mismatch");
    } else {
        failures +=
            expect_uint_field(*alignment_it, "header", 4096, "manifest alignment.header mismatch");
        failures += expect_uint_field(*alignment_it, "tokenizer", 64,
                                      "manifest alignment.tokenizer mismatch");
        failures += expect_uint_field(*alignment_it, "payload", 4096,
                                      "manifest alignment.payload mismatch");
        failures +=
            expect_uint_field(*alignment_it, "block", 256, "manifest alignment.block mismatch");
        failures +=
            expect_uint_field(*alignment_it, "k_pad", 128, "manifest alignment.k_pad mismatch");
        failures += expect_uint_array_field(*alignment_it, "group_sizes",
                                            std::array<std::uint64_t, 2>{32, 64},
                                            "manifest alignment.group_sizes mismatch");
    }
    return failures;
}

int expect_inventory(const ninfer::ParsedQ5090File& parsed, std::uint64_t file_size) {
    int failures = 0;
    failures += parsed.header.tensor_count == 1166 ? 0 : fail("real tensor_count mismatch");
    failures += parsed.header.module_count == 4 ? 0 : fail("real module_count mismatch");
    failures += parsed.header.format_minor == 2 ? 0 : fail("real format_minor mismatch");
    failures += parsed.tokenizer_records.size() == 3 ? 0 : fail("real tokenizer count mismatch");
    failures += parsed.header.segment_count == 1314 ? 0 : fail("real segment_count mismatch");
    failures += parsed.header.fusion_group_count == 130 ? 0 : fail("real fusion count mismatch");
    failures += parsed.tensors.size() == 1166 ? 0 : fail("real parsed tensor size mismatch");
    failures += parsed.segments.size() == 1314 ? 0 : fail("real parsed segment size mismatch");
    failures += parsed.fusion_groups.size() == 130 ? 0 : fail("real parsed fusion size mismatch");
    failures += parsed.modules.size() == 4 ? 0 : fail("real parsed module size mismatch");
    failures += parsed.header.payload_offset + parsed.header.payload_bytes == file_size
                    ? 0
                    : fail("real file size mismatch");
    failures += parsed.modules[0].module_kind == ninfer::ModuleKind::TextCore
                    ? 0
                    : fail("real TEXT module mismatch");
    failures += parsed.modules[0].tensor_index_count == 819 ? 0 : fail("real TEXT count mismatch");
    failures +=
        parsed.modules[0].payload_bytes == 16378329088ULL ? 0 : fail("real TEXT payload mismatch");
    failures += parsed.modules[1].module_kind == ninfer::ModuleKind::LmHeadDraft
                    ? 0
                    : fail("real LM_HEAD_DRAFT module mismatch");
    failures +=
        parsed.modules[1].tensor_index_count == 2 ? 0 : fail("real LM_HEAD_DRAFT count mismatch");
    failures += parsed.modules[1].payload_bytes == 357040128ULL
                    ? 0
                    : fail("real LM_HEAD_DRAFT payload mismatch");
    failures += parsed.modules[2].module_kind == ninfer::ModuleKind::MtpDraft
                    ? 0
                    : fail("real MTP module mismatch");
    failures += parsed.modules[2].tensor_index_count == 12 ? 0 : fail("real MTP count mismatch");
    failures +=
        parsed.modules[2].payload_bytes == 451267584ULL ? 0 : fail("real MTP payload mismatch");
    failures += parsed.modules[3].module_kind == ninfer::ModuleKind::VisionEncoder
                    ? 0
                    : fail("real VISION module mismatch");
    failures +=
        parsed.modules[3].tensor_index_count == 333 ? 0 : fail("real VISION count mismatch");
    failures +=
        parsed.modules[3].payload_bytes == 295719424ULL ? 0 : fail("real VISION payload mismatch");
    return failures;
}

bool enough_free_memory(std::uint64_t bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    const cudaError_t err   = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::cout << "SKIP: cudaMemGetInfo failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    if (free_bytes < bytes) {
        std::cout << "SKIP: not enough free GPU memory; need " << bytes << ", free " << free_bytes
                  << '\n';
        return false;
    }
    return true;
}

int expect_fused_weight(const ninfer::WeightStore& store, std::uint16_t group_id,
                        std::uint16_t fusion_index, std::uint32_t source_layer, ninfer::QType qtype,
                        std::int32_t n, ninfer::SourceKind first_segment, const char* label) {
    const ninfer::Weight* fused =
        store.qfused(ninfer::ModuleKind::TextCore, group_id, fusion_index, source_layer);
    const ninfer::Weight* first = store.qweight(
        ninfer::ModuleKind::TextCore, static_cast<std::uint32_t>(first_segment), source_layer);
    int failures = 0;
    failures += fused != nullptr ? 0 : fail(std::string(label) + " fused missing");
    failures += first != nullptr ? 0 : fail(std::string(label) + " first segment missing");
    if (fused == nullptr || first == nullptr) { return failures; }
    failures += fused->qtype == qtype ? 0 : fail(std::string(label) + " qtype mismatch");
    failures += fused->layout == ninfer::QuantLayout::RowSplit
                    ? 0
                    : fail(std::string(label) + " layout mismatch");
    failures += fused->n == n ? 0 : fail(std::string(label) + " n mismatch");
    failures += fused->k == 5120 ? 0 : fail(std::string(label) + " k mismatch");
    failures += fused->source_kind == static_cast<std::uint32_t>(ninfer::SourceKind::Other)
                    ? 0
                    : fail(std::string(label) + " source_kind mismatch");
    failures += fused->source_layer == source_layer
                    ? 0
                    : fail(std::string(label) + " source_layer mismatch");
    failures += fused->qdata == first->qdata
                    ? 0
                    : fail(std::string(label) + " qdata first-segment mismatch");
    failures += fused->scales == first->scales
                    ? 0
                    : fail(std::string(label) + " scales first-segment mismatch");
    if (qtype == ninfer::QType::Q5G64_F16S || qtype == ninfer::QType::Q6G64_F16S) {
        failures += fused->qhigh == first->qhigh
                        ? 0
                        : fail(std::string(label) + " qhigh first-segment mismatch");
        failures += fused->qhigh != nullptr ? 0 : fail(std::string(label) + " qhigh null");
        failures +=
            fused->high_plane_bytes > 0 ? 0 : fail(std::string(label) + " high_plane_bytes zero");
    } else {
        failures += fused->qhigh == nullptr ? 0 : fail(std::string(label) + " qhigh non-null");
        failures += fused->high_plane_bytes == 0
                        ? 0
                        : fail(std::string(label) + " high_plane_bytes nonzero");
    }
    return failures;
}

int expect_real_fused_contract(const ninfer::WeightStore& store) {
    int failures = 0;
    failures += expect_fused_weight(store, /*MLP_GATEUP*/ 3, 0, 0, ninfer::QType::Q4G64_F16S, 34816,
                                    ninfer::SourceKind::MlpGate, "mlp.gate_up");
    failures += expect_fused_weight(store, /*ATTN_IN*/ 1, 0, 3, ninfer::QType::Q4G64_F16S, 7168,
                                    ninfer::SourceKind::AttnQ, "attn.qkv.q4");
    failures += expect_fused_weight(store, /*ATTN_IN*/ 1, 1, 3, ninfer::QType::Q5G64_F16S, 7168,
                                    ninfer::SourceKind::AttnGate, "attn.gatev.q5");
    failures += expect_fused_weight(store, /*GDN_IN*/ 2, 0, 0, ninfer::QType::Q4G64_F16S, 4096,
                                    ninfer::SourceKind::GdnInProjQ, "gdn.in_qk.q4");
    failures += store.qfused(ninfer::ModuleKind::TextCore, 0, 0, 3) == nullptr
                    ? 0
                    : fail("standalone o_proj should not have fused record");
    failures += store.qfused(ninfer::ModuleKind::TextCore, 0, 0, 0) == nullptr
                    ? 0
                    : fail("standalone mlp.down should not have fused record");
    return failures;
}

int expect_real_draft_head(const ninfer::WeightStore& store) {
    int failures = 0;
    const ninfer::Weight* draft =
        store.qweight(ninfer::ModuleKind::LmHeadDraft,
                      static_cast<std::uint32_t>(ninfer::SourceKind::LmHeadDraft), ninfer::kQ5090NoLayer);
    failures += draft != nullptr ? 0 : fail("real draft head weight missing");
    if (draft != nullptr) {
        failures +=
            draft->qtype == ninfer::QType::Q4G64_F16S ? 0 : fail("real draft head qtype mismatch");
        failures += draft->layout == ninfer::QuantLayout::RowSplit
                        ? 0
                        : fail("real draft head layout mismatch");
        failures += draft->n == 131072 ? 0 : fail("real draft head n mismatch");
        failures += draft->k == 5120 ? 0 : fail("real draft head k mismatch");
        failures += draft->qhigh == nullptr ? 0 : fail("real draft head qhigh non-null");
        failures += draft->high_plane_bytes == 0 ? 0 : fail("real draft head high plane nonzero");
    }
    const ninfer::Tensor* idmap = store.tensor(
        ninfer::ModuleKind::LmHeadDraft, static_cast<std::uint32_t>(ninfer::SourceKind::LmHeadDraftIdmap),
        ninfer::kQ5090NoLayer);
    failures += idmap != nullptr ? 0 : fail("real draft head idmap missing");
    if (idmap != nullptr) {
        failures +=
            idmap->dtype == ninfer::DType::I32 ? 0 : fail("real draft head idmap dtype mismatch");
        failures += idmap->ne[0] == 131072 ? 0 : fail("real draft head idmap length mismatch");
    }
    return failures;
}

int run_default_load(const std::filesystem::path& file_path, std::uint64_t text_payload_bytes,
                     ninfer::Q5090Progress* progress) {
    if (!enough_free_memory(text_payload_bytes + kGiB)) { return 0; }
    ninfer::DeviceContext ctx(0);
    ninfer::WeightStore store;
    ninfer::LoadOptions options;
    options.progress = progress;
    store.load(file_path.c_str(), ctx, options);
    int failures = 0;
    failures += store.module_loaded(ninfer::ModuleKind::TextCore) ? 0 : fail("real TEXT not loaded");
    failures +=
        !store.module_loaded(ninfer::ModuleKind::MtpDraft) ? 0 : fail("real MTP loaded by default");
    failures += !store.module_loaded(ninfer::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("real VISION loaded by default");
    failures += store.tensor_count() == 1166 ? 0 : fail("real store tensor_count mismatch");
    failures +=
        store.loaded_payload_bytes() == text_payload_bytes ? 0 : fail("real loaded bytes mismatch");
    const ninfer::DeviceArena* text_arena = store.module_arena(ninfer::ModuleKind::TextCore);
    failures += text_arena != nullptr && text_arena->capacity() == text_payload_bytes &&
                        text_arena->used() == text_payload_bytes
                    ? 0
                    : fail("real TEXT exact arena mismatch");
    failures += store.module_arena(ninfer::ModuleKind::MtpDraft) == nullptr &&
                        store.module_arena(ninfer::ModuleKind::LmHeadDraft) == nullptr &&
                        store.module_arena(ninfer::ModuleKind::VisionEncoder) == nullptr
                    ? 0
                    : fail("real default load allocated an unselected module arena");
    const ninfer::Q5090LoadStats& load_stats = store.load_stats();
    failures += load_stats.h2d_bytes == text_payload_bytes
                    ? 0
                    : fail("real default H2D byte count mismatch");
    failures += load_stats.total_file_read_bytes < std::filesystem::file_size(file_path)
                    ? 0
                    : fail("real default load read the full artifact");
    failures += load_stats.pinned_slot_count == 2 &&
                        load_stats.host_peak_staging_bytes <= 128ULL * 1024ULL * 1024ULL
                    ? 0
                    : fail("real default pinned staging stats mismatch");
    failures += expect_real_fused_contract(store);
    failures += store.qweight(ninfer::ModuleKind::LmHeadDraft,
                              static_cast<std::uint32_t>(ninfer::SourceKind::LmHeadDraft),
                              ninfer::kQ5090NoLayer) == nullptr
                    ? 0
                    : fail("real LM_HEAD_DRAFT loaded by default");
    return failures;
}

int run_draft_load(const std::filesystem::path& file_path, std::uint64_t text_payload_bytes,
                   std::uint64_t draft_payload_bytes, std::uint64_t mtp_payload_bytes,
                   ninfer::Q5090Progress* progress) {
    if (!enough_free_memory(text_payload_bytes + draft_payload_bytes + mtp_payload_bytes + kGiB)) {
        return 0;
    }
    ninfer::DeviceContext ctx(0);
    ninfer::LoadOptions options;
    options.load_mtp           = true;
    options.load_lm_head_draft = true;
    options.progress           = progress;
    ninfer::WeightStore store;
    store.load(file_path.c_str(), ctx, options);
    int failures = 0;
    failures += store.module_loaded(ninfer::ModuleKind::LmHeadDraft)
                    ? 0
                    : fail("real LM_HEAD_DRAFT not loaded");
    failures +=
        store.loaded_payload_bytes() == text_payload_bytes + draft_payload_bytes + mtp_payload_bytes
            ? 0
            : fail("real LM_HEAD_DRAFT loaded bytes mismatch");
    failures += expect_real_draft_head(store);
    return failures;
}

int run_mtp_load(const std::filesystem::path& file_path, std::uint64_t text_payload_bytes,
                 std::uint64_t mtp_payload_bytes, ninfer::Q5090Progress* progress) {
    const std::uint64_t needed = text_payload_bytes + mtp_payload_bytes + kGiB;
    if (!enough_free_memory(needed)) { return 0; }
    ninfer::DeviceContext ctx(0);
    ninfer::LoadOptions options;
    options.load_mtp = true;
    options.progress = progress;
    ninfer::WeightStore store;
    store.load(file_path.c_str(), ctx, options);
    store.require_mtp_module_expectations();
    int failures = 0;
    failures +=
        store.module_loaded(ninfer::ModuleKind::TextCore) ? 0 : fail("real TEXT not loaded with MTP");
    failures += store.module_loaded(ninfer::ModuleKind::MtpDraft) ? 0 : fail("real MTP not loaded");
    failures += !store.module_loaded(ninfer::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("real VISION loaded with MTP");
    failures += store.loaded_payload_bytes() == text_payload_bytes + mtp_payload_bytes
                    ? 0
                    : fail("real MTP loaded bytes mismatch");
    return failures;
}

int run_vision_load(const std::filesystem::path& file_path, std::uint64_t text_payload_bytes,
                    std::uint64_t vision_payload_bytes, ninfer::Q5090Progress* progress) {
    if (!enough_free_memory(text_payload_bytes + vision_payload_bytes + kGiB)) { return 0; }
    ninfer::DeviceContext ctx(0);
    ninfer::LoadOptions options;
    options.load_vision = true;
    options.progress    = progress;
    ninfer::WeightStore store;
    store.load(file_path.c_str(), ctx, options);
    int failures = 0;
    failures +=
        store.module_loaded(ninfer::ModuleKind::VisionEncoder) ? 0 : fail("real VISION not loaded");
    failures += store.module_arena(ninfer::ModuleKind::VisionEncoder) != nullptr
                    ? 0
                    : fail("real VISION arena missing");
    failures += store.module_arena(ninfer::ModuleKind::MtpDraft) == nullptr &&
                        store.module_arena(ninfer::ModuleKind::LmHeadDraft) == nullptr
                    ? 0
                    : fail("real VISION load allocated unselected optional arenas");
    failures += store.loaded_payload_bytes() == text_payload_bytes + vision_payload_bytes
                    ? 0
                    : fail("real VISION loaded bytes mismatch");
    return failures;
}

} // namespace

int main() {
    const std::filesystem::path root(NINFER_SOURCE_DIR);
    const std::filesystem::path file_path = root / "out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus";
    const std::filesystem::path manifest_path =
        root / "out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus.manifest.json";
    if (!std::filesystem::exists(file_path) || !std::filesystem::exists(manifest_path)) {
        std::cout << "SKIP: real q5090 file or manifest not present\n";
        return 0;
    }

    int failures                           = 0;
    const std::uint64_t manifest_file_size = std::filesystem::file_size(file_path);
    failures += expect_manifest_fields(read_manifest_json(manifest_path), manifest_file_size);

    ProgressPrinter printer;
    ninfer::Q5090Progress progress;
    progress.min_interval_bytes = 512ULL * 1024ULL * 1024ULL;
    progress.callback           = [&printer](std::string_view phase, std::uint64_t done,
                                   std::uint64_t total) { printer(phase, done, total); };

    std::vector<std::byte> bytes      = read_binary(file_path, &progress);
    const std::uint64_t file_size     = bytes.size();
    const ninfer::ParsedQ5090File parsed = ninfer::parse_q5090_file(bytes, &progress);
    failures += expect_inventory(parsed, file_size);
    const std::uint64_t text_payload_bytes   = parsed.modules[0].payload_bytes;
    const std::uint64_t draft_payload_bytes  = parsed.modules[1].payload_bytes;
    const std::uint64_t mtp_payload_bytes    = parsed.modules[2].payload_bytes;
    const std::uint64_t vision_payload_bytes = parsed.modules[3].payload_bytes;
    bytes.clear();
    bytes.shrink_to_fit();

    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device for real q5090 load\n";
        return failures == 0 ? 0 : fail("real q5090 inventory test failed");
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    failures += run_default_load(file_path, text_payload_bytes, &progress);
    cudaDeviceSynchronize();
    failures += run_draft_load(file_path, text_payload_bytes, draft_payload_bytes,
                               mtp_payload_bytes, &progress);
    cudaDeviceSynchronize();
    failures += run_mtp_load(file_path, text_payload_bytes, mtp_payload_bytes, &progress);
    cudaDeviceSynchronize();
    failures += run_vision_load(file_path, text_payload_bytes, vision_payload_bytes, &progress);
    return failures == 0 ? 0 : fail("real q5090 weight store test failed");
}
