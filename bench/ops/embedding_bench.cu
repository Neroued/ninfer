// Performance bench for the registered Qwen3.6 embedding gathers.
//   ./ninfer_embedding_bench --w8 [--decode|--prefill]
//   ./ninfer_embedding_bench --q6 [--decode|--prefill]
#include "ninfer/ops/embedding.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kVocab     = 248320;
constexpr std::int32_t kQ6D       = 5120;
constexpr std::int32_t kGroup     = 64;
constexpr std::int32_t kDPad      = ((kQ6D + 127) / 128) * 128;
constexpr std::int32_t kNibbleBpr = 32;
constexpr std::int32_t kHighBpr   = 16;
constexpr std::int32_t kScaleBpr  = 2;
constexpr std::int32_t kKg        = kDPad / kGroup;
constexpr std::uint64_t kNibblePlaneBytes =
    static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kKg) * kNibbleBpr;
constexpr std::uint64_t kHighPlaneOffset = ((kNibblePlaneBytes + 255ULL) / 256ULL) * 256ULL;
constexpr std::uint64_t kHighPlaneBytes =
    static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kKg) * kHighBpr;
constexpr std::uint64_t kScalePlaneOffset =
    kHighPlaneOffset + ((kHighPlaneBytes + 255ULL) / 256ULL) * 256ULL;
constexpr std::uint64_t kScalePlaneBytes =
    static_cast<std::uint64_t>(kVocab) * static_cast<std::uint64_t>(kKg) * 2ULL;
constexpr std::uint64_t kPayloadBytes = kScalePlaneOffset + kScalePlaneBytes;

DBuf make_ids(std::int32_t t) {
    std::vector<std::int32_t> h(static_cast<std::size_t>(t));
    for (std::int32_t i = 0; i < t; ++i) {
        h[static_cast<std::size_t>(i)] = (i * 9973 + 12345) % kVocab;
    }
    DBuf d(h.size() * sizeof(std::int32_t));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

Weight q6_weight(void* payload) {
    Weight w{};
    w.payload          = payload;
    w.payload_bytes    = kPayloadBytes;
    w.high_plane_bytes = kHighPlaneBytes;
    w.qtype            = QType::Q6G64_F16S;
    w.layout           = QuantLayout::RowSplit;
    w.scale_dtype      = DType::FP16;
    w.group_size       = kGroup;
    w.shape[0]         = kVocab;
    w.shape[1]         = kQ6D;
    w.padded_shape[0]  = kVocab;
    w.padded_shape[1]  = kDPad;
    w.ndim             = 2;
    w.qdata            = payload;
    w.qhigh            = static_cast<std::uint8_t*>(payload) + kHighPlaneOffset;
    w.scales           = static_cast<std::uint8_t*>(payload) + kScalePlaneOffset;
    w.n                = kVocab;
    w.k                = kQ6D;
    w.group            = kGroup;
    return w;
}

void run(std::int32_t t, const char* tag) {
    const std::size_t out_elems = static_cast<std::size_t>(kQ6D) * static_cast<std::size_t>(t);

    DBuf payload(kPayloadBytes);
    DBuf ids = make_ids(t);
    DBuf out = make_zeros(out_elems * sizeof(std::uint16_t));

    // Deterministic nonzero ROW_SPLIT bytes. Correctness values do not
    // matter for this bandwidth bench; tests cover signed LSB-first semantics.
    CUDA_CHECK(cudaMemset(payload.p, 0x3c, payload.bytes));

    Weight table = q6_weight(payload.p);
    Tensor tids(ids.p, DType::I32, {t});
    Tensor tout(out.p, DType::BF16, {kQ6D, t});

    // Task-defined traffic: Q6 row read plus BF16 output write.
    const double bytes = static_cast<double>(t) * static_cast<double>(kKg) *
                             static_cast<double>(kNibbleBpr + kHighBpr + kScaleBpr) +
                         static_cast<double>(t) * static_cast<double>(kQ6D) * 2.0;
    const Result r =
        bench_loop([&](cudaStream_t s) { ops::embedding(tids, table, tout, s); }, bytes);
    print_result(tag, r);
}

constexpr std::int32_t kW8D                 = 2048;
constexpr std::int32_t kW8Group             = 32;
constexpr std::int32_t kW8Kg                = kW8D / kW8Group;
constexpr std::uint64_t kW8CodePlaneBytes   = static_cast<std::uint64_t>(kVocab) * kW8Kg * kW8Group;
constexpr std::uint64_t kW8ScalePlaneOffset = ((kW8CodePlaneBytes + 255ULL) / 256ULL) * 256ULL;
constexpr std::uint64_t kW8ScalePlaneBytes  = static_cast<std::uint64_t>(kVocab) * kW8Kg * 2ULL;
constexpr std::uint64_t kW8PayloadBytes     = kW8ScalePlaneOffset + kW8ScalePlaneBytes;

__global__ void w8_grouped_payload_control_kernel(const std::int32_t* ids,
                                                  const std::uint8_t* codes,
                                                  const std::uint8_t* scales, __nv_bfloat16* out) {
    const int t                    = static_cast<int>(blockIdx.x) / kW8Kg;
    const int g                    = static_cast<int>(blockIdx.x) - t * kW8Kg;
    const int row                  = ids[t];
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kW8Kg + g;
    int scale_bits                 = 0;
    if (threadIdx.x == 0) {
        scale_bits = static_cast<int>(scales[group_index * 2]) |
                     (static_cast<int>(scales[group_index * 2 + 1]) << 8);
    }
    scale_bits      = __shfl_sync(0xffffffffu, scale_bits, 0);
    const auto code = codes[group_index * kW8Group + static_cast<int>(threadIdx.x)];
    const std::uint16_t bits =
        static_cast<std::uint16_t>(code) ^ static_cast<std::uint16_t>(scale_bits);
    out[static_cast<std::int64_t>(t) * kW8D + static_cast<std::int64_t>(g) * kW8Group +
        threadIdx.x] = __ushort_as_bfloat16(bits);
}

__device__ __forceinline__ std::uint32_t expand_code_pair(std::uint32_t word, int shift) {
    return ((word >> shift) & 0xffu) | (((word >> (shift + 8)) & 0xffu) << 16);
}

__global__ void w8_row_payload_control_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                              const std::uint8_t* scales, __nv_bfloat16* out) {
    const int tid                  = static_cast<int>(threadIdx.x);
    const int t                    = static_cast<int>(blockIdx.x);
    const int row                  = ids[t];
    const int k                    = tid * 8;
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kW8Kg + k / kW8Group;
    const std::uint32_t scale_bits = static_cast<std::uint32_t>(scales[group_index * 2]) |
                                     (static_cast<std::uint32_t>(scales[group_index * 2 + 1]) << 8);
    const auto* code_row = codes + static_cast<std::int64_t>(row) * kW8D;
    const uint2 packed   = *reinterpret_cast<const uint2*>(code_row + k);
    uint4 expanded;
    expanded.x = expand_code_pair(packed.x, 0) ^ scale_bits;
    expanded.y = expand_code_pair(packed.x, 16);
    expanded.z = expand_code_pair(packed.y, 0);
    expanded.w = expand_code_pair(packed.y, 16);
    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(t) * kW8D + k) = expanded;
}

Weight w8_weight(void* payload) {
    Weight w{};
    w.payload          = payload;
    w.payload_bytes    = kW8PayloadBytes;
    w.high_plane_bytes = 0;
    w.qtype            = QType::W8G32_F16S;
    w.layout           = QuantLayout::RowSplit;
    w.scale_dtype      = DType::FP16;
    w.group_size       = kW8Group;
    w.shape[0]         = kVocab;
    w.shape[1]         = kW8D;
    w.padded_shape[0]  = kVocab;
    w.padded_shape[1]  = kW8D;
    w.ndim             = 2;
    w.qdata            = payload;
    w.qhigh            = nullptr;
    w.scales           = static_cast<std::uint8_t*>(payload) + kW8ScalePlaneOffset;
    w.n                = kVocab;
    w.k                = kW8D;
    w.group            = kW8Group;
    return w;
}

void run_w8(std::int32_t t, bool control) {
    const std::size_t out_elems = static_cast<std::size_t>(kW8D) * static_cast<std::size_t>(t);
    DBuf payload(kW8PayloadBytes);
    DBuf ids = make_ids(t);
    DBuf out = make_zeros(out_elems * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(payload.p, 0x3c, payload.bytes));

    Weight table = w8_weight(payload.p);
    Tensor tids(ids.p, DType::I32, {t});
    Tensor tout(out.p, DType::BF16, {kW8D, t});
    const double bytes = static_cast<double>(t) *
                         static_cast<double>(kW8D + kW8Kg * 2 + kW8D * 2 + sizeof(std::int32_t));
    const auto launch = [&](cudaStream_t s) {
        if (!control) {
            ops::embedding(tids, table, tout, s);
        } else if (t <= 6) {
            w8_grouped_payload_control_kernel<<<t * kW8Kg, 32, 0, s>>>(
                static_cast<const std::int32_t*>(ids.p),
                static_cast<const std::uint8_t*>(table.qdata),
                static_cast<const std::uint8_t*>(table.scales), static_cast<__nv_bfloat16*>(out.p));
        } else {
            w8_row_payload_control_kernel<<<t, 256, 0, s>>>(
                static_cast<const std::int32_t*>(ids.p),
                static_cast<const std::uint8_t*>(table.qdata),
                static_cast<const std::uint8_t*>(table.scales), static_cast<__nv_bfloat16*>(out.p));
        }
    };
    const Result r = bench_loop(launch, bytes);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "embedding w8%s [248320,2048] T=%d", control ? " control" : "",
                  t);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool q6 = false, w8 = false, decode = false, prefill = false, control = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--q6"))
            q6 = true;
        else if (!std::strcmp(argv[i], "--w8"))
            w8 = true;
        else if (!std::strcmp(argv[i], "--decode"))
            decode = true;
        else if (!std::strcmp(argv[i], "--t64") || !std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--control"))
            control = true;
    }
    if (!q6 && !w8) { w8 = true; }
    if (!decode && !prefill) { decode = prefill = true; }

    if (q6 && decode) run(1, "embedding q6 [248320,5120] T=1");
    if (q6 && prefill) run(64, "embedding q6 [248320,5120] T=64");
    if (w8 && decode) {
        for (std::int32_t t = 1; t <= 6; ++t) { run_w8(t, control); }
    }
    if (w8 && prefill) { run_w8(1024, control); }
    return 0;
}
