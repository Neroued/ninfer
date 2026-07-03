#include "qus/runtime/round_dump.h"

#include "qus/core/device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qus {
namespace {

float bf16_bits_to_f32(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float out             = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

std::vector<float> tensor_to_f32(const Tensor& x, cudaStream_t stream) {
    if (x.data == nullptr) { throw std::invalid_argument("round dump: tensor data is null"); }
    if (!x.is_contiguous()) { throw std::invalid_argument("round dump: tensor is not contiguous"); }
    const std::int64_t n_i64 = x.numel();
    if (n_i64 < 0) { throw std::overflow_error("round dump: negative tensor size"); }
    const auto n = static_cast<std::size_t>(n_i64);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> out(n);
    if (x.dtype == DType::FP32) {
        CUDA_CHECK(cudaMemcpy(out.data(), x.data, out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    }
    if (x.dtype == DType::BF16) {
        std::vector<std::uint16_t> bits(n);
        CUDA_CHECK(cudaMemcpy(bits.data(), x.data, bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = bf16_bits_to_f32(bits[i]); }
        return out;
    }
    throw std::invalid_argument("round dump: only BF16 and FP32 tensors are supported");
}

void write_f32_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("round dump: failed to open " + path.string()); }
    const auto bytes = values.size() * sizeof(float);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("round dump: file too large");
    }
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!file) { throw std::runtime_error("round dump: failed to write " + path.string()); }
}

std::string round_name(std::uint64_t round_index, const char* suffix) {
    char name[128];
    std::snprintf(name, sizeof(name), "round_%06llu_%s",
                  static_cast<unsigned long long>(round_index), suffix);
    return name;
}

std::string layer_name(std::uint64_t round_index, const char* group, std::uint32_t layer,
                       const char* suffix) {
    char name[160];
    std::snprintf(name, sizeof(name), "round_%06llu_%s_%02u_%s",
                  static_cast<unsigned long long>(round_index), group, layer, suffix);
    return name;
}

void write_manifest(const std::filesystem::path& path, std::uint64_t round_index,
                    std::uint32_t committed_length, const GdnState& gdn, const KVCache& mtp_kv,
                    const std::vector<std::string>& files) {
    std::ofstream file(path);
    if (!file) { throw std::runtime_error("round dump: failed to open " + path.string()); }
    file << "{\n";
    file << "  \"schema_version\": 1,\n";
    file << "  \"artifact_type\": \"qus_mtp_round_state_dump\",\n";
    file << "  \"round\": " << round_index << ",\n";
    file << "  \"committed_length\": " << committed_length << ",\n";
    file << "  \"mtp_kv_position\": " << mtp_kv.pos << ",\n";
    file << "  \"mtp_kv_padded_context\": " << mtp_kv.padded_context << ",\n";
    file << "  \"mtp_kv_layout\": \"[head_dim,padded_context,num_kv_heads]\",\n";
    file << "  \"committed_mtp_kv_positions\": {\"begin\": 0, \"end\": "
         << committed_length << "},\n";
    file << "  \"active_draft_mtp_kv_positions\": {\"begin\": " << committed_length
         << ", \"end\": " << mtp_kv.pos << "},\n";
    file << "  \"dumped_mtp_kv_positions\": {\"begin\": 0, \"end\": " << mtp_kv.padded_context
         << "},\n";
    file << "  \"gdn_state\": \"slot0_committed\",\n";
    file << "  \"gdn_layers\": " << gdn.layer_count() << ",\n";
    file << "  \"mtp_layers\": " << mtp_kv.layer_count() << ",\n";
    file << "  \"dtype\": \"f32\",\n";
    file << "  \"files\": [\n";
    for (std::size_t i = 0; i < files.size(); ++i) {
        file << "    \"" << files[i] << '"';
        if (i + 1 != files.size()) { file << ','; }
        file << '\n';
    }
    file << "  ]\n";
    file << "}\n";
    if (!file) { throw std::runtime_error("round dump: failed to write " + path.string()); }
}

void dump_tensor(const std::filesystem::path& out_dir, const std::string& name, const Tensor& tensor,
                 cudaStream_t stream, std::vector<std::string>& files) {
    write_f32_file(out_dir / name, tensor_to_f32(tensor, stream));
    files.push_back(name);
}

} // namespace

void write_mtp_round_state_dump(const std::filesystem::path& out_dir, std::uint64_t round_index,
                                std::uint32_t committed_length, const GdnState& gdn,
                                const KVCache& mtp_kv, cudaStream_t stream) {
    if (out_dir.empty()) { throw std::invalid_argument("round dump: output directory is empty"); }
    std::filesystem::create_directories(out_dir);

    std::vector<std::string> files;
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        dump_tensor(out_dir, layer_name(round_index, "gdn", layer, "conv_slot0.f32"),
                    gdn.conv_slot(layer, 0), stream, files);
        dump_tensor(out_dir, layer_name(round_index, "gdn", layer, "ssm_slot0.f32"),
                    gdn.ssm_slot(layer, 0), stream, files);
    }

    for (std::uint32_t layer = 0; layer < mtp_kv.layer_count(); ++layer) {
        dump_tensor(out_dir, layer_name(round_index, "mtp_kv", layer, "k.f32"),
                    mtp_kv.k.at(layer), stream, files);
        dump_tensor(out_dir, layer_name(round_index, "mtp_kv", layer, "v.f32"),
                    mtp_kv.v.at(layer), stream, files);
    }

    write_manifest(out_dir / round_name(round_index, "manifest.json"), round_index,
                   committed_length, gdn, mtp_kv, files);
}

} // namespace qus
