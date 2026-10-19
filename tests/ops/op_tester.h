#pragma once
//
// op_tester.h — small reusable building blocks for Op tests: checked CUDA
// calls, device buffers and transfers, deterministic inputs, output guards,
// and exact/numerical verdicts. See docs/maintainer/op-development.md.
//
// Typical per-op test flow (one shape):
//   std::vector<float> x(n); fill_uniform(x, seed, -8.f, 8.f);
//   round_to_bf16(x);                       // what the kernel will read
//   std::vector<double> ref(n);             // compute in double from x
//   ... cpu_ref(x, ref) ...
//   DeviceBuffer dx = to_device_bf16(x);
//   DeviceBuffer dout(cuda_environment().context.device(), n*2);
//   ... launch kernel on dx -> dout ...
//   failures += verify_pointwise("op n=...", from_device_bf16(dout, n), ref,
//                                op_bf16_criterion());

#include "core/buffer.h"
#include "cuda_fixture.h"
#include "core/arena.h"
#include "core/tensor.h" // ninfer::DType, ninfer::Tensor (for op call sites)
#include "ops/op_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ninfer::test {

// --- environment ------------------------------------------------------------
inline void cuda_check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

inline void cuda_check_last_launch(const char* operation) {
    cuda_check(cudaGetLastError(), operation);
}

inline void cuda_synchronize() { cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

inline void cuda_synchronize(cudaStream_t stream) {
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

inline bool cuda_unavailable() {
    int n               = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e == cudaSuccess) { return n == 0; }
    if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) { return true; }
    throw std::runtime_error(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(e));
}

// --- bf16 <-> f32 (round-to-nearest-even) -----------------------------------
inline float bf16_to_f32(std::uint16_t h) {
    std::uint32_t u = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((u >> 16) | 0x0040u); // keep NaN
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return std::uint16_t(u >> 16);
}

// Round an fp32 buffer to the bf16 grid in place, so a CPU reference computes
// from EXACTLY the values the kernel will read.
inline void round_to_bf16(std::vector<float>& v) {
    for (auto& x : v) x = bf16_to_f32(f32_to_bf16(x));
}

// --- seeded honest inputs ---------------------------------------------------
inline void fill_uniform(std::vector<float>& v, std::uint32_t seed, float lo, float hi) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v) x = d(g);
}

inline void fill_iota_i32(std::vector<int>& v, int start = 0) {
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = start + static_cast<int>(i);
}

// --- host <-> device --------------------------------------------------------
template <typename T>
inline DeviceBuffer to_device(const std::vector<T>& h) {
    static_assert(std::is_trivially_copyable_v<T>);
    DeviceBuffer d(cuda_environment().context.device(), h.size() * sizeof(T));
    d.copy_from_host(h.data(), d.size());
    return d;
}

template <typename T>
inline std::vector<T> from_device(const DeviceBuffer& d, std::size_t n) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<T> out(n);
    d.copy_to_host(out.data(), n * sizeof(T));
    return out;
}

template <typename T>
inline std::vector<T> from_device(const void* device, std::size_t n) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<T> out(n);
    if (n != 0) {
        cuda_check(cudaMemcpy(out.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy device-to-host");
    }
    return out;
}

inline DeviceBuffer to_device_bf16(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
    return to_device(b);
}

inline DeviceBuffer to_device_f32(const std::vector<float>& h) { return to_device(h); }

inline DeviceBuffer to_device_i32(const std::vector<int>& h) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    return to_device(h);
}

// --- device -> host (upcast to double for the comparison) -------------------
inline std::vector<double> from_device_bf16(const DeviceBuffer& d, std::size_t n) {
    const std::vector<std::uint16_t> b = from_device<std::uint16_t>(d, n);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(bf16_to_f32(b[i]));
    return o;
}

inline std::vector<double> from_device_bf16(const void* device, std::size_t n) {
    const std::vector<std::uint16_t> b = from_device<std::uint16_t>(device, n);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(bf16_to_f32(b[i]));
    return o;
}

inline std::vector<double> from_device_f32(const DeviceBuffer& d, std::size_t n) {
    const std::vector<float> f = from_device<float>(d, n);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(f[i]);
    return o;
}

inline std::vector<int> from_device_i32(const DeviceBuffer& d, std::size_t n) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    return from_device<int>(d, n);
}

// --- verdict ----------------------------------------------------------------
inline bool error_stats_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_OP_REPORT_STATS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

inline double error_limit_ratio(double error, double limit) {
    if (limit != 0.0) return error / limit;
    return error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

inline void report_pointwise_stats(std::string_view label, std::int64_t count,
                                   const PointwiseStats& stats,
                                   const PointwiseCriterion& criterion) {
    if (!error_stats_enabled()) return;
    std::printf("OP_ERROR_STATS kind=pointwise count=%lld max_abs=%.17g max_rel=%.17g "
                "max_limit_ratio=%.17g abs_limit=%.17g rel_limit=%.17g non_finite=%lld case=%.*s\n",
                static_cast<long long>(count), stats.maximum_absolute_error,
                stats.maximum_relative_error, stats.maximum_criterion_ratio, criterion.absolute,
                criterion.relative, static_cast<long long>(stats.non_finite_count),
                static_cast<int>(label.size()), label.data());
}

inline void report_reduction_stats(std::string_view label, std::int64_t count,
                                   const ReductionStats& stats,
                                   const ReductionCriterion& criterion) {
    if (!error_stats_enabled()) return;
    const double gross_limit = gross_error_limit(stats, criterion);
    const double rms_limit   = reduction_rms_limit(stats, criterion);
    std::printf("OP_ERROR_STATS kind=reduction count=%lld rel_l2=%.17g rel_l2_limit=%.17g "
                "rel_l2_ratio=%.17g rmse=%.17g reference_rms=%.17g max_abs=%.17g "
                "max_reference=%.17g gross_limit=%.17g gross_abs_limit=%.17g "
                "gross_rel_limit=%.17g gross_ratio=%.17g rms_abs_limit=%.17g "
                "rms_limit=%.17g rms_ratio=%.17g non_finite=%lld case=%.*s\n",
                static_cast<long long>(count), stats.relative_l2, criterion.relative_l2,
                error_limit_ratio(stats.relative_l2, criterion.relative_l2),
                stats.root_mean_squared_error, stats.reference_root_mean_square,
                stats.maximum_absolute_error, stats.maximum_absolute_reference, gross_limit,
                criterion.gross_absolute, criterion.gross_relative_to_max_reference,
                error_limit_ratio(stats.maximum_absolute_error, gross_limit),
                criterion.absolute_rms, rms_limit,
                error_limit_ratio(stats.root_mean_squared_error, rms_limit),
                static_cast<long long>(stats.first_non_finite >= 0), static_cast<int>(label.size()),
                label.data());
}

inline void report_scaled_pointwise_stats(std::string_view label, std::int64_t count,
                                          double maximum_absolute_error,
                                          double required_scaled_relative,
                                          double scaled_relative_limit) {
    if (!error_stats_enabled()) return;
    std::printf("OP_ERROR_STATS kind=scaled_pointwise count=%lld max_abs=%.17g "
                "required_scaled_relative=%.17g scaled_relative_limit=%.17g "
                "scaled_relative_ratio=%.17g case=%.*s\n",
                static_cast<long long>(count), maximum_absolute_error, required_scaled_relative,
                scaled_relative_limit,
                error_limit_ratio(required_scaled_relative, scaled_relative_limit),
                static_cast<int>(label.size()), label.data());
}

// Exact transforms, codecs, integer outputs, and state metadata use this path.
template <typename T>
inline int verify_exact(const char* label, const std::vector<T>& got,
                        const std::vector<T>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": size mismatch got=" << got.size()
                  << " expected=" << expected.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!(got[i] == expected[i])) {
            std::cerr << label << ": exact mismatch at index " << i << '\n';
            return 1;
        }
    }
    return 0;
}

// A payload surrounded by byte canaries. Use data() as the Op output and
// verify_guards() after synchronization to detect prefix/suffix overwrites.
class GuardedDeviceBuffer {
public:
    explicit GuardedDeviceBuffer(std::size_t payload_bytes, std::size_t guard_bytes = 256,
                                 std::uint8_t guard_byte = 0xa5)
        : storage_(cuda_environment().context.device(),
                   allocation_bytes(payload_bytes, guard_bytes)),
          payload_bytes_(payload_bytes), guard_bytes_(guard_bytes), guard_byte_(guard_byte) {
        storage_.fill(guard_byte_);
    }

    void* data() noexcept { return static_cast<std::uint8_t*>(storage_.data()) + guard_bytes_; }

    const void* data() const noexcept {
        return static_cast<const std::uint8_t*>(storage_.data()) + guard_bytes_;
    }

    std::size_t bytes() const noexcept { return payload_bytes_; }

    Device device() const noexcept { return storage_.device(); }

    MemorySpan span() noexcept { return storage_.span().subspan(guard_bytes_, payload_bytes_); }

    ConstMemorySpan span() const noexcept {
        return storage_.span().subspan(guard_bytes_, payload_bytes_);
    }

    void fill(int byte_value = 0) {
        if (payload_bytes_ != 0) {
            CudaDeviceScope scope(storage_.device());
            cuda_check(cudaMemset(data(), byte_value, payload_bytes_),
                       "cudaMemset guarded payload");
            cuda_check(cudaStreamSynchronize(nullptr), "guarded payload initialization");
        }
    }

    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0) {
        require_payload_range(byte_offset, count);
        storage_.copy_from_host(source, count, guard_bytes_ + byte_offset);
    }

    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const {
        require_payload_range(byte_offset, count);
        storage_.copy_to_host(destination, count, guard_bytes_ + byte_offset);
    }

    int verify_guards(std::string_view label) const {
        const auto prefix         = from_device<std::uint8_t>(storage_, guard_bytes_);
        const auto* suffix_device = static_cast<const std::uint8_t*>(data()) + payload_bytes_;
        const auto suffix         = from_device<std::uint8_t>(suffix_device, guard_bytes_);
        const auto intact         = [this](const std::vector<std::uint8_t>& guard) {
            return std::all_of(guard.begin(), guard.end(),
                                       [this](std::uint8_t value) { return value == guard_byte_; });
        };
        if (intact(prefix) && intact(suffix)) return 0;
        std::cerr << label << ": device buffer guard was overwritten\n";
        return 1;
    }

private:
    static std::size_t allocation_bytes(std::size_t payload_bytes, std::size_t guard_bytes) {
        if (guard_bytes == 0) {
            throw std::invalid_argument("GuardedDeviceBuffer requires a non-empty guard");
        }
        return payload_bytes + 2 * guard_bytes;
    }

    void require_payload_range(std::size_t byte_offset, std::size_t count) const {
        if (byte_offset > payload_bytes_ || count > payload_bytes_ - byte_offset) {
            throw std::out_of_range("copy exceeds guarded device payload");
        }
    }

    DeviceBuffer storage_;
    std::size_t payload_bytes_;
    std::size_t guard_bytes_;
    std::uint8_t guard_byte_;
};

// Returns 0 (pass) / 1 (fail). Concrete criteria are supplied by the semantic Op suite.
inline int verify_pointwise(std::string_view label, std::span<const double> got,
                            std::span<const double> ref, const PointwiseCriterion& criterion) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size mismatch got=" << got.size() << " ref=" << ref.size() << '\n';
        return 1;
    }
    const PointwiseStats stats = compute_pointwise_stats(
        got.data(), ref.data(), static_cast<std::int64_t>(got.size()), criterion);
    report_pointwise_stats(label, static_cast<std::int64_t>(got.size()), stats, criterion);
    if (pointwise_passes(stats, static_cast<std::int64_t>(got.size()))) return 0;

    std::cerr << label << ": pointwise mismatch" << " max_abs=" << stats.maximum_absolute_error
              << " max_rel=" << stats.maximum_relative_error
              << " max_index=" << stats.maximum_error_index << " actual=" << stats.actual_at_maximum
              << " reference=" << stats.reference_at_maximum
              << " first_violation=" << stats.first_violation
              << " non_finite=" << stats.non_finite_count << '\n';
    return 1;
}

inline int verify_reduction(std::string_view label, std::span<const double> got,
                            std::span<const double> ref, const ReductionCriterion& criterion) {
    if (got.empty() || got.size() != ref.size()) {
        std::cerr << label << ": invalid comparison sizes got=" << got.size()
                  << " ref=" << ref.size() << '\n';
        return 1;
    }
    const ReductionStats stats =
        compute_reduction_stats(got.data(), ref.data(), static_cast<std::int64_t>(got.size()));
    report_reduction_stats(label, static_cast<std::int64_t>(got.size()), stats, criterion);
    if (reduction_passes(stats, static_cast<std::int64_t>(got.size()), criterion)) return 0;

    if (stats.first_non_finite >= 0) {
        std::cerr << label << ": non-finite value at index " << stats.first_non_finite << '\n';
        return 1;
    }
    std::cerr << label << ": reduction criterion failed at index " << stats.maximum_error_index
              << " actual=" << stats.actual_at_maximum
              << " reference=" << stats.reference_at_maximum << '\n';
    return 1;
}

} // namespace ninfer::test
