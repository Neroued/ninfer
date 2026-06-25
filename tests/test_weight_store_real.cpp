#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("failed to open manifest"); }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
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

int expect_manifest_tokens(const std::string& manifest) {
    int failures = 0;
    failures += manifest.find("\"kind\": \"TEXT_CORE\"") != std::string::npos
                    ? 0
                    : fail("manifest missing TEXT");
    failures += manifest.find("\"kind\": \"MTP_DRAFT\"") != std::string::npos
                    ? 0
                    : fail("manifest missing MTP");
    failures += manifest.find("\"kind\": \"VISION_ENCODER\"") != std::string::npos
                    ? 0
                    : fail("manifest missing VISION");
    failures += manifest.find("\"tensor_count\": 963") != std::string::npos
                    ? 0
                    : fail("manifest TEXT count mismatch");
    failures += manifest.find("\"tensor_count\": 15") != std::string::npos
                    ? 0
                    : fail("manifest MTP count mismatch");
    failures += manifest.find("\"tensor_count\": 333") != std::string::npos
                    ? 0
                    : fail("manifest VISION count mismatch");
    return failures;
}

int expect_inventory(const qus::ParsedQ5090File& parsed, std::uint64_t file_size) {
    int failures = 0;
    failures += parsed.header.tensor_count == 1311 ? 0 : fail("real tensor_count mismatch");
    failures += parsed.header.module_count == 3 ? 0 : fail("real module_count mismatch");
    failures += parsed.tensors.size() == 1311 ? 0 : fail("real parsed tensor size mismatch");
    failures += parsed.modules.size() == 3 ? 0 : fail("real parsed module size mismatch");
    failures += parsed.header.payload_offset + parsed.header.payload_bytes == file_size
                    ? 0
                    : fail("real file size mismatch");
    failures += parsed.modules[0].module_kind == qus::ModuleKind::TextCore
                    ? 0
                    : fail("real TEXT module mismatch");
    failures += parsed.modules[0].tensor_index_count == 963 ? 0 : fail("real TEXT count mismatch");
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
        parsed.modules[2].payload_bytes == 294184960ULL ? 0 : fail("real VISION payload mismatch");
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
    failures += store.tensor_count() == 1311 ? 0 : fail("real store tensor_count mismatch");
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
    const std::filesystem::path file_path     = root / "out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus";
    const std::filesystem::path manifest_path = root / "out/manifest.json";
    if (!std::filesystem::exists(file_path) || !std::filesystem::exists(manifest_path)) {
        std::cout << "SKIP: real q5090 file or manifest not present\n";
        return 0;
    }

    int failures = 0;
    failures += expect_manifest_tokens(read_text(manifest_path));

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
