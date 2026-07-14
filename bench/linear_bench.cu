// Performance bench for linear at real Qwen3.6 shapes. Dense in_a/in_b use
// [48,5120]; low-bit decode uses real Qwen3.6 GEMV shapes. The printed GB/s is
// informational only; ncu is the performance evidence.
//   ./ninfer_linear_bench [--decode] [--prefill] [--bf16] [--fp32] [--q4] [--q5] [--q6] [--stress]
#include "ninfer/kernels/linear.h"
#include "ninfer_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

// Cold-cache timing: per-layer weight matrices fit the RTX 5090 ~96 MB L2, so the
// shared bench_loop (which reuses one buffer) reports L2 bandwidth, not DRAM. In
// real decode the working set is >>L2, so every weight read is a cold DRAM read.
// --cold flushes a >L2 buffer before each timed launch to model that. ncu remains
// the acceptance gate; this is a fast iteration signal.
bool g_cold              = false;
void* g_flush_buf        = nullptr;
constexpr std::size_t kFlushBytes = std::size_t(256) << 20; // 256 MiB > L2

void flush_l2() {
    if (g_flush_buf == nullptr) { cudaMalloc(&g_flush_buf, kFlushBytes); }
    cudaMemsetAsync(g_flush_buf, 0, kFlushBytes, 0);
}

Result bench_loop_cold(const launch_fn& launch, double bytes_moved, int warmup = 10,
                       int repeat = 80) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < warmup; ++i) launch(nullptr);
    cudaStreamSynchronize(nullptr);

    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        flush_l2();                 // serialized before launch on the default stream
        cudaEventRecord(a, nullptr);
        launch(nullptr);
        cudaEventRecord(b, nullptr);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(double(ms) * 1000.0);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    std::sort(samples.begin(), samples.end());
    Result r;
    r.n_runs      = int(samples.size());
    r.inner_iters = 1;
    r.median_us   = samples[samples.size() / 2];
    r.min_us      = samples.front();
    r.p95_us      = samples[std::min(samples.size() - 1, std::size_t(0.95 * samples.size()))];
    r.mean_us     = r.median_us;
    const double sec = r.median_us * 1e-6;
    r.gbs            = (sec > 0.0) ? bytes_moved / sec / 1e9 : 0.0;
    return r;
}

constexpr std::int32_t kDenseN       = 48;
constexpr std::int32_t kDenseK       = 5120;
constexpr std::int32_t kDenseStressN = 5120;
constexpr std::int32_t kDenseStressK = 6144;
constexpr std::int32_t kDenseStressT = 64;

struct LowbitShape {
    const char* role;
    std::int32_t n;
    std::int32_t k;
};

DBuf make_f32(std::size_t n) {
    std::vector<float> h(n);
    for (std::size_t i = 0; i < n; ++i) { h[i] = 0.25f - static_cast<float>(i % 251) / 1000.0f; }
    DBuf d(n * sizeof(float));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

DBuf make_weight(QType qtype, std::size_t n) {
    if (qtype == QType::FP32_CTRL) { return make_f32(n); }
    return make_bf16(n);
}

Weight dense_weight(void* data, QType qtype, std::int32_t n, std::int32_t k) {
    Weight w{};
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(n) * k * ((qtype == QType::FP32_CTRL) ? 4u : 2u);
    w.qtype         = qtype;
    w.layout        = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.group_size        = 0;
    w.shape[0]          = n;
    w.shape[1]          = k;
    w.padded_shape[0]   = n;
    w.padded_shape[1]   = k;
    w.ndim              = 2;
    w.qdata             = data;
    w.scales            = nullptr;
    w.n                 = n;
    w.k                 = k;
    w.group             = 0;
    return w;
}

const char* qtype_name(QType qtype) { return (qtype == QType::FP32_CTRL) ? "fp32" : "bf16"; }

void run_dense_shape(std::int32_t n, std::int32_t k, std::int32_t t, QType qtype,
                     const char* phase) {
    const std::size_t x_elems   = static_cast<std::size_t>(k) * t;
    const std::size_t w_elems   = static_cast<std::size_t>(n) * k;
    const std::size_t out_elems = static_cast<std::size_t>(n) * t;

    DBuf x    = make_bf16(x_elems);
    DBuf wbuf = make_weight(qtype, w_elems);
    DBuf out  = make_zeros(out_elems * sizeof(std::uint16_t));

    Tensor tx(x.p, DType::BF16, {k, t});
    Tensor tout(out.p, DType::BF16, {n, t});
    Weight w = dense_weight(wbuf.p, qtype, n, k);
    WorkspaceArena ws(64ULL << 20);

    const double weight_bytes = static_cast<double>(w_elems) *
                                ((qtype == QType::FP32_CTRL) ? 4.0 : 2.0) * static_cast<double>(t);
    const double bytes =
        static_cast<double>(x_elems) * 2.0 + weight_bytes + static_cast<double>(out_elems) * 2.0;
    const Result r = bench_loop([&](cudaStream_t s) { kernels::linear(tx, w, tout, ws, s); }, bytes);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "linear %s %s [%d,%d] T=%d", qtype_name(qtype), phase, n, k, t);
    print_result(tag, r);
}

void run_dense(std::int32_t t, QType qtype, const char* phase) {
    run_dense_shape(kDenseN, kDenseK, t, qtype, phase);
}

void run_dense_stress(QType qtype) {
    run_dense_shape(kDenseStressN, kDenseStressK, kDenseStressT, qtype, "stress-prefill");
}

std::int32_t nibble_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return 32;
    case QType::Q5G64_F16S: return 32;
    case QType::Q6G64_F16S: return 32;
    default:                return 0;
    }
}

std::int32_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return 0;
    case QType::Q5G64_F16S: return 8;
    case QType::Q6G64_F16S: return 16;
    default:                return 0;
    }
}

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t align) {
    return ((value + align - 1) / align) * align;
}

DBuf make_row_split_payload(QType qtype, std::int32_t n, std::int32_t k) {
    constexpr std::uint16_t kScaleOne = 0x3c00u;
    const std::int32_t k_pad = static_cast<std::int32_t>(align_up_u64(k, 128));
    const std::int32_t kg    = k_pad / 64;
    const std::uint64_t groups =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg);
    const std::uint64_t nibble_bytes =
        groups * static_cast<std::uint64_t>(nibble_bytes_per_group(qtype));
    const std::uint64_t high_bytes =
        groups * static_cast<std::uint64_t>(high_bytes_per_group(qtype));
    const std::uint64_t high_offset  = align_up_u64(nibble_bytes, 256);
    const std::uint64_t scale_offset = high_offset + align_up_u64(high_bytes, 256);
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) * 2ULL;

    std::vector<std::uint8_t> payload(static_cast<std::size_t>(scale_offset + scale_bytes));
    for (std::uint64_t i = 0; i < nibble_bytes; ++i) {
        payload[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((i * 17u + 31u) & 0xffu);
    }
    for (std::uint64_t i = 0; i < high_bytes; ++i) {
        payload[static_cast<std::size_t>(high_offset + i)] =
            static_cast<std::uint8_t>((i * 13u + 7u) & 0xffu);
    }
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::uint64_t off =
                scale_offset + (static_cast<std::uint64_t>(row) * kg + g) * 2ULL;
            payload[static_cast<std::size_t>(off)] =
                static_cast<std::uint8_t>(kScaleOne & 0xffu);
            payload[static_cast<std::size_t>(off + 1)] =
                static_cast<std::uint8_t>(kScaleOne >> 8);
        }
    }
    DBuf d(payload.size());
    cudaMemcpy(d.p, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    return d;
}

Weight lowbit_weight(void* payload, QType qtype, std::int32_t n, std::int32_t k) {
    const std::int32_t k_pad = static_cast<std::int32_t>(align_up_u64(k, 128));
    const std::int32_t kg    = k_pad / 64;
    const std::uint64_t groups =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg);
    const std::uint64_t nibble_bytes =
        groups * static_cast<std::uint64_t>(nibble_bytes_per_group(qtype));
    const std::uint64_t high_bytes =
        groups * static_cast<std::uint64_t>(high_bytes_per_group(qtype));
    const std::uint64_t high_offset  = align_up_u64(nibble_bytes, 256);
    const std::uint64_t scale_offset = high_offset + align_up_u64(high_bytes, 256);
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) * 2ULL;
    Weight w{};
    w.payload           = payload;
    w.payload_bytes     = scale_offset + scale_bytes;
    w.high_plane_bytes  = high_bytes;
    w.qtype             = qtype;
    w.layout            = QuantLayout::RowSplit;
    w.q5090_scale_dtype = ScaleDType::FP16;
    w.group_size        = 64;
    w.shape[0]          = n;
    w.shape[1]          = k;
    w.padded_shape[0]   = n;
    w.padded_shape[1]   = k_pad;
    w.ndim              = 2;
    w.qdata             = payload;
    w.qhigh             = high_bytes == 0 ? nullptr
                                          : static_cast<std::uint8_t*>(payload) + high_offset;
    w.scales            = static_cast<std::uint8_t*>(payload) + scale_offset;
    w.n                 = n;
    w.k                 = k;
    w.group             = 64;
    return w;
}

Weight q4_weight(void* payload, std::int32_t n, std::int32_t k) {
    return lowbit_weight(payload, QType::Q4G64_F16S, n, k);
}

Weight q5_weight(void* payload, std::int32_t n, std::int32_t k) {
    return lowbit_weight(payload, QType::Q5G64_F16S, n, k);
}

Weight q6_weight(void* payload, std::int32_t n, std::int32_t k) {
    return lowbit_weight(payload, QType::Q6G64_F16S, n, k);
}

const char* lowbit_name(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S: return "q4";
    case QType::Q5G64_F16S: return "q5";
    case QType::Q6G64_F16S: return "q6";
    default:                return "lowbit";
    }
}

void run_lowbit_decode(const LowbitShape& shape, QType qtype) {
    constexpr std::int32_t t    = 1;
    const std::size_t x_elems   = static_cast<std::size_t>(shape.k) * t;
    const std::size_t out_elems = static_cast<std::size_t>(shape.n) * t;

    DBuf x    = make_bf16(x_elems);
    DBuf wbuf = make_row_split_payload(qtype, shape.n, shape.k);
    DBuf out  = make_zeros(out_elems * sizeof(std::uint16_t));

    Tensor tx(x.p, DType::BF16, {shape.k, t});
    Tensor tout(out.p, DType::BF16, {shape.n, t});
    Weight w{};
    switch (qtype) {
    case QType::Q4G64_F16S: w = q4_weight(wbuf.p, shape.n, shape.k); break;
    case QType::Q5G64_F16S: w = q5_weight(wbuf.p, shape.n, shape.k); break;
    case QType::Q6G64_F16S: w = q6_weight(wbuf.p, shape.n, shape.k); break;
    default:                return;
    }

    const double bytes = static_cast<double>(x_elems) * 2.0 + static_cast<double>(w.payload_bytes) +
                         static_cast<double>(out_elems) * 2.0;
    WorkspaceArena ws(64ULL << 20);
    const launch_fn launch = [&](cudaStream_t s) { kernels::linear(tx, w, tout, ws, s); };
    const Result r = g_cold ? bench_loop_cold(launch, bytes) : bench_loop(launch, bytes);

    char tag[112];
    std::snprintf(tag, sizeof(tag), "linear %s decode %-10s [%d,%d] T=1%s", lowbit_name(qtype),
                  shape.role, shape.n, shape.k, g_cold ? " COLD" : "");
    print_result(tag, r);
}

void run_q4_decode() {
    constexpr LowbitShape shapes[] = {
        {"mlp.gate/up", 17408, 5120},
        {"gdn q/k", 2048, 5120},
        {"attn q", 6144, 5120},
    };
    for (const LowbitShape& shape : shapes) {
        run_lowbit_decode(shape, QType::Q4G64_F16S);
    }
}

void run_q5_decode() {
    constexpr LowbitShape shapes[] = {
        {"mlp.down", 5120, 17408},
        {"v/z", 6144, 5120},
        {"out", 5120, 6144},
        {"attn gate", 6144, 5120},
    };
    for (const LowbitShape& shape : shapes) {
        run_lowbit_decode(shape, QType::Q5G64_F16S);
    }
}

void run_q6_decode() {
    constexpr LowbitShape shape{"lm_head", 248320, 5120};
    run_lowbit_decode(shape, QType::Q6G64_F16S);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool decode = false, prefill = false, bf16 = false, fp32 = false, q4 = false, q5 = false;
    bool q6 = false;
    bool stress = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode"))
            decode = true;
        else if (!std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--bf16"))
            bf16 = true;
        else if (!std::strcmp(argv[i], "--fp32"))
            fp32 = true;
        else if (!std::strcmp(argv[i], "--q4"))
            q4 = true;
        else if (!std::strcmp(argv[i], "--q5"))
            q5 = true;
        else if (!std::strcmp(argv[i], "--q6"))
            q6 = true;
        else if (!std::strcmp(argv[i], "--stress"))
            stress = true;
        else if (!std::strcmp(argv[i], "--cold"))
            g_cold = true;
    }
    if (!decode && !prefill) { decode = prefill = true; }
    if (!bf16 && !fp32 && !q4 && !q5 && !q6) { bf16 = fp32 = q4 = q5 = q6 = true; }

    if (stress) {
        if (bf16) run_dense_stress(QType::BF16_CTRL);
        if (fp32) run_dense_stress(QType::FP32_CTRL);
        return 0;
    }

    if (decode && bf16) run_dense(1, QType::BF16_CTRL, "decode");
    if (decode && fp32) run_dense(1, QType::FP32_CTRL, "decode");
    if (prefill && bf16) run_dense(7, QType::BF16_CTRL, "prefill");
    if (prefill && fp32) run_dense(7, QType::FP32_CTRL, "prefill");
    if (decode && q4) run_q4_decode();
    if (decode && q5) run_q5_decode();
    if (decode && q6) run_q6_decode();
    return 0;
}
