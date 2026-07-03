#include "qus/model/model.h"
#include "qus/core/arena.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/runtime/round_dump.h"
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct MemoryTap {
    static constexpr bool enabled = true;

    void operator()(qus::model::TapId, int, qus::model::Phase, const qus::Tensor&, cudaStream_t) {}
};

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() {
        if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
    }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() {
        if (ptr_ != nullptr) { (void)cudaFree(ptr_); }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* get() const noexcept { return ptr_; }

private:
    void* ptr_ = nullptr;
};

class TempDir {
public:
    explicit TempDir(std::string_view prefix) {
        const auto base = std::filesystem::temp_directory_path();
        const auto stamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = base / (std::string(prefix) + "_" + std::to_string(stamp) + "_" +
                            std::to_string(attempt));
            if (std::filesystem::create_directory(path_)) { return; }
        }
        throw std::runtime_error("failed to create unique temp directory");
    }

    ~TempDir() {
        if (!path_.empty()) { std::filesystem::remove_all(path_); }
    }

    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

std::vector<float> read_f32_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("failed to open " + path.string()); }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("bad f32 file size: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) { throw std::runtime_error("failed to read " + path.string()); }
    return values;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("failed to open " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int expect_f32_file(const std::filesystem::path& path, std::span<const float> expected) {
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "missing tap output file: " << path << '\n';
        return 1;
    }

    const std::vector<float> actual = read_f32_file(path);
    if (actual.size() != expected.size()) {
        std::cerr << "tap output size mismatch for " << path << ": expected " << expected.size()
                  << ", got " << actual.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "tap output value mismatch for " << path << " at " << i << ": expected "
                      << expected[i] << ", got " << actual[i] << '\n';
            return 1;
        }
    }
    return 0;
}

int expect_contains(std::string_view text, std::string_view needle, std::string_view label) {
    if (text.find(needle) != std::string_view::npos) { return 0; }
    std::cerr << "missing " << label << " in manifest: " << needle << '\n';
    return 1;
}

int expect_exact_file_set(const std::filesystem::path& out_dir,
                          const std::set<std::string>& expected) {
    std::set<std::string> actual;
    for (const auto& entry : std::filesystem::directory_iterator(out_dir)) {
        if (entry.is_regular_file()) { actual.insert(entry.path().filename().string()); }
    }
    if (actual == expected) { return 0; }

    std::cerr << "tap output file set mismatch\nexpected:";
    for (const std::string& name : expected) { std::cerr << ' ' << name; }
    std::cerr << "\nactual:";
    for (const std::string& name : actual) { std::cerr << ' ' << name; }
    std::cerr << '\n';
    return 1;
}

int test_file_tap_output_files() {
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

    CUDA_CHECK(cudaSetDevice(0));

    TempDir out_dir("qus_runtime_tap_behavior");

    const std::array<float, 3> expected{1.0f, -2.5f, 3.25f};
    CudaStream stream;
    DeviceBuffer device(expected.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(device.get(), expected.data(), expected.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.get()));

    qus::Tensor tensor(device.get(), qus::DType::FP32,
                       {static_cast<std::int32_t>(expected.size())});
    qus::model::FileTap tap(out_dir.path());
    tap(qus::model::TapId::AfterEmbed, -1, qus::model::Phase::Prefill, tensor, stream.get());
    tap(qus::model::TapId::AfterMixer, 7, qus::model::Phase::Prefill, tensor, stream.get());
    tap(qus::model::TapId::AfterMlp, 12, qus::model::Phase::Prefill, tensor, stream.get());
    tap(qus::model::TapId::AfterFinalNorm, -1, qus::model::Phase::Decode, tensor, stream.get());
    tap(qus::model::TapId::AfterLogits, -1, qus::model::Phase::Decode, tensor, stream.get());

    int failures = 0;
    failures +=
        std::filesystem::is_directory(out_dir.path()) ? 0 : fail("FileTap did not create out_dir");
    const std::set<std::string> expected_names{
        "embed.f32",
        "final_norm.f32",
        "layer_07_mixer.f32",
        "layer_12_mlp.f32",
        "logits.f32",
    };
    failures += expect_exact_file_set(out_dir.path(), expected_names);
    for (const std::string& name : expected_names) {
        failures += expect_f32_file(out_dir.path() / name, expected);
    }

    return failures;
}

void copy_f32_to_tensor(std::span<const float> values, qus::Tensor tensor, cudaStream_t stream) {
    if (tensor.dtype != qus::DType::FP32 ||
        tensor.numel() != static_cast<std::int64_t>(values.size())) {
        throw std::runtime_error("test tensor shape mismatch");
    }
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, values.data(), values.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
}

int test_round_state_dump_output_files() {
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

    CUDA_CHECK(cudaSetDevice(0));

    CudaStream stream;
    qus::DeviceArena arena(1U << 20);
    qus::GdnState gdn(arena, 1, 2, 2, 1, 2, 2, 2, qus::DType::FP32);
    qus::KVCache mtp_kv(arena, 1, 2, 1, 2, qus::DType::FP32);
    mtp_kv.pos = 2;

    const std::array<float, 4> conv{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> ssm{5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> key(static_cast<std::size_t>(mtp_kv.k[0].numel()), 0.0f);
    std::vector<float> value(static_cast<std::size_t>(mtp_kv.v[0].numel()), 0.0f);
    key[0]   = 9.0f;
    key[1]   = 10.0f;
    value[0] = 11.0f;
    value[1] = 12.0f;

    copy_f32_to_tensor(conv, gdn.conv_slot(0, 0), stream.get());
    copy_f32_to_tensor(ssm, gdn.ssm_slot(0, 0), stream.get());
    copy_f32_to_tensor(key, mtp_kv.k[0], stream.get());
    copy_f32_to_tensor(value, mtp_kv.v[0], stream.get());

    TempDir out_dir("qus_mtp_round_dump_behavior");
    qus::write_mtp_round_state_dump(out_dir.path(), 7, 2, gdn, mtp_kv, stream.get());

    const std::set<std::string> expected_names{
        "round_000007_gdn_00_conv_slot0.f32",
        "round_000007_gdn_00_ssm_slot0.f32",
        "round_000007_manifest.json",
        "round_000007_mtp_kv_00_k.f32",
        "round_000007_mtp_kv_00_v.f32",
    };

    int failures = 0;
    failures += expect_exact_file_set(out_dir.path(), expected_names);
    failures += expect_f32_file(out_dir.path() / "round_000007_gdn_00_conv_slot0.f32", conv);
    failures += expect_f32_file(out_dir.path() / "round_000007_gdn_00_ssm_slot0.f32", ssm);
    failures += expect_f32_file(out_dir.path() / "round_000007_mtp_kv_00_k.f32", key);
    failures += expect_f32_file(out_dir.path() / "round_000007_mtp_kv_00_v.f32", value);

    const std::string manifest = read_text_file(out_dir.path() / "round_000007_manifest.json");
    failures += expect_contains(manifest, "\"artifact_type\": \"qus_mtp_round_state_dump\"",
                                "artifact type");
    failures += expect_contains(manifest, "\"round\": 7", "round index");
    failures += expect_contains(manifest, "\"committed_length\": 2", "committed length");
    failures += expect_contains(manifest, "\"dtype\": \"f32\"", "dtype marker");
    failures += expect_contains(manifest, "\"committed_mtp_kv_positions\": {\"begin\": 0, \"end\": 2}",
                                "committed MTP KV positions");
    failures += expect_contains(manifest, "\"gdn_state\": \"slot0_committed\"", "GDN state marker");
    failures += expect_contains(manifest, "\"round_000007_mtp_kv_00_k.f32\"", "mtp kv filename");

    return failures;
}

} // namespace

static_assert(requires(qus::model::Qwen3_6_27B& card, std::span<const int> ids, MemoryTap& tap) {
    card.prefill(ids, tap);
    card.decode_step(tap);
});

int main() {
    int failures = 0;
    failures += test_file_tap_output_files();
    failures += test_round_state_dump_output_files();

    if (failures != 0) {
        std::cerr << "FAIL runtime FileTap behavior\n";
        return 1;
    }

    std::cout << "OK runtime FileTap behavior\n";
    return 0;
}
