#include "models/qwen4_exp/execution/tf_dump.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen4_exp::execution {
namespace {

// .npy v1.0, little-endian, C order, one dimension.
void write_npy(const std::filesystem::path& path, std::string_view descr, std::size_t count,
               const void* data, std::size_t element_bytes) {
    std::ostringstream header;
    header << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (" << count << ",), }";
    std::string text           = header.str();
    const std::size_t unpadded = 10 + text.size() + 1;
    text.append((64 - unpadded % 64) % 64, ' ');
    text.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("tf_dump: cannot write " + path.string()); }
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    out.write(magic, sizeof(magic));
    const auto length          = static_cast<std::uint16_t>(text.size());
    const char length_bytes[2] = {static_cast<char>(length & 0xFF), static_cast<char>(length >> 8)};
    out.write(length_bytes, 2);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * element_bytes));
    if (!out) { throw std::runtime_error("tf_dump: short write " + path.string()); }
}

} // namespace

std::unique_ptr<TfDump> TfDump::from_environment() {
    const char* directory = std::getenv("NINFER_TFDUMP");
    if (directory == nullptr || *directory == '\0') { return nullptr; }
    return std::make_unique<TfDump>(directory);
}

TfDump::TfDump(std::filesystem::path directory) : directory_(std::move(directory)) {
    std::filesystem::create_directories(directory_);
}

void TfDump::capture_layer(std::int32_t layer, const Tensor& value, std::int32_t first_position,
                           cudaStream_t stream) {
    if (value.dtype != DType::BF16) {
        throw std::invalid_argument("tf_dump: taps take BF16 [rows, columns]");
    }
    // The residual is a strided view (a column-slice of a wider buffer), so copy with the
    // row pitch (nb[1]) rather than value.bytes(). The physical buffer size (value.bytes())
    // gives the actual column count.
    const std::size_t element_bytes = 2;
    const std::size_t rows          = static_cast<std::size_t>(value.ne[0]);
    const std::size_t width_bytes   = rows * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t data_bytes    = value.bytes();
    const std::size_t columns       = data_bytes / width_bytes;
    scratch_.resize(data_bytes / element_bytes);
    CUDA_CHECK(cudaMemcpy2DAsync(scratch_.data(), width_bytes, value.data, source_pitch, width_bytes,
                                 columns, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    char name[16];
    std::snprintf(name, sizeof(name), "L%02d", layer);
    // Truncate on the first write of a file this process, append afterwards — avoids stale
    // bytes from a pre-existing file corrupting the record byte count.
    static std::unordered_set<std::string> written;
    const std::string path = (directory_ / ("hidden_" + std::string(name) + ".bin")).string();
    const std::ios::openmode mode = written.count(path) ? std::ios::app : std::ios::trunc;
    written.insert(path);
    std::ofstream out(path, std::ios::binary | mode);
    if (!out) { throw std::runtime_error("tf_dump: cannot open compact hidden tap"); }
    const std::int32_t head[3] = {first_position, static_cast<std::int32_t>(columns),
                                  static_cast<std::int32_t>(rows)};
    out.write(reinterpret_cast<const char*>(head), sizeof(head));
    out.write(reinterpret_cast<const char*>(scratch_.data()),
              static_cast<std::streamsize>(scratch_.size() * sizeof(std::uint16_t)));
    out.flush();
    if (!out) { throw std::runtime_error("tf_dump: short write of a compact hidden tap"); }
    out.close();
    // Derive positions from what is ACTUALLY on disk: re-read the first tap layer's .bin,
    // count the total columns, and write positions.npy = 0..total-1. This guarantees
    // positions always match the features (the dump is constructed per chunk and capture_layer
    // is also called from non-prefill phases, so a running counter would drift).
    const std::string fpath = (directory_ / ("hidden_" + std::string(name) + ".bin")).string();
    std::size_t total_cols = 0;
    if (std::ifstream in(fpath, std::ios::binary); in) {
        std::uint32_t fp, cols, rows32;
        while (in.read(reinterpret_cast<char*>(&fp), 4) &&
               in.read(reinterpret_cast<char*>(&cols), 4) &&
               in.read(reinterpret_cast<char*>(&rows32), 4)) {
            total_cols += cols;
            in.seekg(static_cast<std::streamoff>(static_cast<std::uint64_t>(cols) * rows32 * 2),
                     std::ios::cur);
        }
    }
    std::vector<std::int32_t> pos(total_cols);
    for (std::size_t i = 0; i < total_cols; ++i) { pos[i] = static_cast<std::int32_t>(i); }
    write_npy(directory_ / "positions.npy", "<i4", pos.size(), pos.data(), sizeof(std::int32_t));
}

void TfDump::write_tokens(std::span<const std::int32_t> tokens) {
    std::vector<std::int64_t> wide(tokens.begin(), tokens.end());
    write_npy(directory_ / "tokens.npy", "<i8", wide.size(), wide.data(), sizeof(std::int64_t));
}

void TfDump::write_meta(const std::string& producer, const std::string& artifact) const {
    std::ofstream out(directory_ / "meta.json");
    out << "{\"format\":\"tfdump/1\",\"producer\":\"" << producer << "\",\"artifact\":\""
        << artifact << "\",\"model\":\"qwen4-exp-flashnext\"}\n";
}

} // namespace ninfer::models::qwen4_exp::execution
