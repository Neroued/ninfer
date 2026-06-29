#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"

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
    explicit ProgressThrottle(qus::Q5090Progress* progress) : progress_(progress) {}

    void report(std::string_view phase, std::uint64_t done, std::uint64_t total,
                bool force = false) {
        if (progress_ == nullptr || !progress_->callback) { return; }
        if (!force && done < last_done_ + progress_->min_interval_bytes && done < total) { return; }
        last_done_ = done;
        progress_->callback(phase, done, total);
    }

private:
    qus::Q5090Progress* progress_ = nullptr;
    std::uint64_t last_done_      = 0;
};

std::vector<std::byte> read_binary(const std::filesystem::path& path,
                                   qus::Q5090Progress* progress) {
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

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = 64;
    expected.hidden_size             = 5120;
    expected.intermediate_size       = 17408;
    expected.vocab_size              = 248320;
    expected.num_attention_heads     = 24;
    expected.num_key_value_heads     = 4;
    expected.head_dim                = 256;
    expected.gdn_key_heads           = 16;
    expected.gdn_value_heads         = 48;
    expected.gdn_key_head_dim        = 128;
    expected.gdn_value_head_dim      = 128;
    expected.gdn_conv_width          = 4;
    expected.full_attention_interval = 4;
    expected.max_position_embeddings = 262144;
    return expected;
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

int expect_manifest_module(const Json& module, const char* kind, const char* policy,
                           std::uint64_t block_count, std::uint64_t segment_count,
                           std::uint64_t fusion_group_count, std::uint64_t payload_bytes,
                           std::uint64_t load_policy) {
    if (!module.is_object()) { return fail("manifest module must be object"); }
    int failures = 0;
    failures += expect_string_field(module, "kind", kind, "manifest module kind mismatch");
    failures += expect_string_field(module, "policy", policy, "manifest module policy mismatch");
    failures +=
        expect_uint_field(module, "block_count", block_count, "manifest module block_count mismatch");
    failures += expect_uint_field(module, "segment_count", segment_count,
                                  "manifest module segment_count mismatch");
    failures += expect_uint_field(module, "fusion_group_count", fusion_group_count,
                                  "manifest module fusion_group_count mismatch");
    failures += expect_uint_field(module, "payload_bytes", payload_bytes,
                                  "manifest module payload_bytes mismatch");
    failures +=
        expect_uint_field(module, "load_policy", load_policy, "manifest module load_policy mismatch");
    return failures;
}

int expect_manifest_fields(const Json& manifest, std::uint64_t file_size) {
    if (!manifest.is_object()) { return fail("manifest root must be object"); }

    int failures = 0;
    failures += expect_string_field(manifest, "format", "q5090_w4g64_mixed_v2",
                                    "manifest format mismatch");
    failures += expect_uint_field(manifest, "format_version", 2, "manifest format_version mismatch");
    failures +=
        expect_string_field(manifest, "model", "Qwen/Qwen3.6-27B", "manifest model mismatch");
    failures += expect_string_field(manifest, "binary_spec", "docs/q5090_packed_file_format_v2.md",
                                    "manifest binary_spec mismatch");
    failures += expect_string_field(manifest, "tensor_plan",
                                    "docs/qwen3_6_27b_q5090_v2_tensor_plan.md",
                                    "manifest tensor_plan mismatch");
    failures += expect_uint_field(manifest, "file_bytes", file_size, "manifest file_bytes mismatch");
    failures += expect_uint_field(manifest, "hidden_size", 5120, "manifest hidden_size mismatch");
    failures +=
        expect_uint_field(manifest, "intermediate_size", 17408, "manifest intermediate_size mismatch");
    failures += expect_uint_field(manifest, "vocab_size", 248320, "manifest vocab_size mismatch");
    failures +=
        expect_uint_field(manifest, "num_hidden_layers", 64, "manifest num_hidden_layers mismatch");
    failures += expect_bool_field(manifest, "zero_point", false, "manifest zero_point mismatch");
    failures +=
        expect_empty_array_field(manifest, "disabled_modules", "manifest disabled_modules mismatch");
    failures += expect_string_array_field(
        manifest, "qtypes",
        std::array<const char*, 6>{"Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G128_F16S",
                                  "BF16_CTRL", "FP32_CTRL"},
        "manifest qtypes mismatch");
    failures += expect_string_array_field(
        manifest, "layouts", std::array<const char*, 2>{"ROW_SPLIT", "CONTIGUOUS"},
        "manifest layouts mismatch");
    failures += expect_uint_field(manifest, "block_count", 1167, "manifest block_count mismatch");
    failures += expect_uint_field(manifest, "segment_count", 1311, "manifest segment_count mismatch");
    failures += expect_uint_field(manifest, "fusion_group_count", 128,
                                  "manifest fusion_group_count mismatch");

    const auto alignment_it = manifest.find("alignment");
    if (alignment_it == manifest.end() || !alignment_it->is_object()) {
        failures += fail("manifest alignment mismatch");
    } else {
        failures += expect_uint_field(*alignment_it, "file_header", 4096,
                                      "manifest alignment.file_header mismatch");
        failures += expect_uint_field(*alignment_it, "tensor_payload", 256,
                                      "manifest alignment.tensor_payload mismatch");
        failures += expect_uint_field(*alignment_it, "payload_region", 4096,
                                      "manifest alignment.payload_region mismatch");
    }

    const auto modules_it = manifest.find("modules");
    if (modules_it == manifest.end() || !modules_it->is_array() || modules_it->size() != 3) {
        failures += fail("manifest modules mismatch");
    } else {
        failures += expect_manifest_module((*modules_it)[0], "TEXT_CORE", "q5090_w4g64_mixed_v2",
                                           819, 963, 128, 16378329088ULL, 0);
        failures += expect_manifest_module((*modules_it)[1], "MTP_DRAFT", "mtp_w8g128_v2", 15,
                                           15, 0, 431361024ULL, 0);
        failures += expect_manifest_module((*modules_it)[2], "VISION_ENCODER",
                                           "vision_q4mix_merger_w8g128_v2", 333, 333, 0,
                                           293396992ULL, 1);
    }
    return failures;
}

int expect_inventory(const qus::ParsedQ5090File& parsed, std::uint64_t file_size) {
    int failures = 0;
    failures += parsed.header.tensor_count == 1167 ? 0 : fail("real tensor_count mismatch");
    failures += parsed.header.module_count == 3 ? 0 : fail("real module_count mismatch");
    failures += parsed.header.segment_count == 1311 ? 0 : fail("real segment_count mismatch");
    failures += parsed.header.fusion_group_count == 128 ? 0 : fail("real fusion count mismatch");
    failures += parsed.tensors.size() == 1167 ? 0 : fail("real parsed tensor size mismatch");
    failures += parsed.segments.size() == 1311 ? 0 : fail("real parsed segment size mismatch");
    failures += parsed.fusion_groups.size() == 128 ? 0 : fail("real parsed fusion size mismatch");
    failures += parsed.modules.size() == 3 ? 0 : fail("real parsed module size mismatch");
    failures += parsed.header.payload_offset + parsed.header.payload_bytes == file_size
                    ? 0
                    : fail("real file size mismatch");
    failures += parsed.modules[0].module_kind == qus::ModuleKind::TextCore
                    ? 0
                    : fail("real TEXT module mismatch");
    failures += parsed.modules[0].tensor_index_count == 819 ? 0 : fail("real TEXT count mismatch");
    failures +=
        parsed.modules[0].payload_bytes == 16378329088ULL ? 0 : fail("real TEXT payload mismatch");
    failures += parsed.modules[1].module_kind == qus::ModuleKind::MtpDraft
                    ? 0
                    : fail("real MTP module mismatch");
    failures += parsed.modules[1].tensor_index_count == 15 ? 0 : fail("real MTP count mismatch");
    failures +=
        parsed.modules[1].payload_bytes == 431361024ULL ? 0 : fail("real MTP payload mismatch");
    failures += parsed.modules[2].module_kind == qus::ModuleKind::VisionEncoder
                    ? 0
                    : fail("real VISION module mismatch");
    failures +=
        parsed.modules[2].tensor_index_count == 333 ? 0 : fail("real VISION count mismatch");
    failures +=
        parsed.modules[2].payload_bytes == 293396992ULL ? 0 : fail("real VISION payload mismatch");
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

int run_default_load(const std::filesystem::path& file_path, std::uint64_t text_payload_bytes,
                     qus::Q5090Progress* progress) {
    if (!enough_free_memory(text_payload_bytes + kGiB)) { return 0; }
    qus::DeviceContext ctx(0);
    qus::DeviceArena arena(static_cast<std::size_t>(text_payload_bytes + kGiB));
    qus::WeightStore store(expectations());
    qus::LoadOptions options;
    options.progress = progress;
    store.load(file_path.c_str(), arena, ctx, options);
    int failures = 0;
    failures += store.module_loaded(qus::ModuleKind::TextCore) ? 0 : fail("real TEXT not loaded");
    failures +=
        !store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("real MTP loaded by default");
    failures += !store.module_loaded(qus::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("real VISION loaded by default");
    failures += store.tensor_count() == 1167 ? 0 : fail("real store tensor_count mismatch");
    failures +=
        store.loaded_payload_bytes() == text_payload_bytes ? 0 : fail("real loaded bytes mismatch");
    return failures;
}

int run_mtp_load(const std::filesystem::path& file_path, std::uint64_t text_payload_bytes,
                 std::uint64_t mtp_payload_bytes, qus::Q5090Progress* progress) {
    const std::uint64_t needed = text_payload_bytes + mtp_payload_bytes + kGiB;
    if (!enough_free_memory(needed)) { return 0; }
    qus::DeviceContext ctx(0);
    qus::DeviceArena arena(static_cast<std::size_t>(needed));
    qus::LoadOptions options;
    options.load_mtp = true;
    options.progress = progress;
    qus::WeightStore store(expectations());
    store.load(file_path.c_str(), arena, ctx, options);
    int failures = 0;
    failures +=
        store.module_loaded(qus::ModuleKind::TextCore) ? 0 : fail("real TEXT not loaded with MTP");
    failures += store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("real MTP not loaded");
    failures += !store.module_loaded(qus::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("real VISION loaded with MTP");
    failures += store.loaded_payload_bytes() == text_payload_bytes + mtp_payload_bytes
                    ? 0
                    : fail("real MTP loaded bytes mismatch");
    return failures;
}

} // namespace

int main() {
    const std::filesystem::path root(QUS_SOURCE_DIR);
    const std::filesystem::path file_path     = root / "out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus";
    const std::filesystem::path manifest_path = root / "out/manifest.v2.json";
    if (!std::filesystem::exists(file_path) || !std::filesystem::exists(manifest_path)) {
        std::cout << "SKIP: real q5090 file or manifest not present\n";
        return 0;
    }

    int failures = 0;
    const std::uint64_t manifest_file_size = std::filesystem::file_size(file_path);
    failures += expect_manifest_fields(read_manifest_json(manifest_path), manifest_file_size);

    ProgressPrinter printer;
    qus::Q5090Progress progress;
    progress.min_interval_bytes = 512ULL * 1024ULL * 1024ULL;
    progress.callback           = [&printer](std::string_view phase, std::uint64_t done,
                                   std::uint64_t total) { printer(phase, done, total); };

    std::vector<std::byte> bytes      = read_binary(file_path, &progress);
    const std::uint64_t file_size     = bytes.size();
    const qus::ParsedQ5090File parsed = qus::parse_q5090_file(bytes, expectations(), &progress);
    failures += expect_inventory(parsed, file_size);
    const std::uint64_t text_payload_bytes = parsed.modules[0].payload_bytes;
    const std::uint64_t mtp_payload_bytes  = parsed.modules[1].payload_bytes;
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
    failures += run_mtp_load(file_path, text_payload_bytes, mtp_payload_bytes, &progress);
    return failures == 0 ? 0 : fail("real q5090 weight store test failed");
}
