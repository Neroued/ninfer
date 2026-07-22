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
//   DBuf dx = to_device_bf16(x), dout(n*2);
//   ... launch kernel on dx -> dout ...
//   failures += verify("op n=...", from_device_bf16(dout, n), ref,
//                      Tolerance::bf16_elementwise());

#include "core/tensor.h" // ninfer::DType, ninfer::Tensor (for op call sites)
#include "ops/op_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
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
    return e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || e != cudaSuccess || n == 0;
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

// --- device buffers (RAII) --------------------------------------------------
struct DBuf {
    void* p           = nullptr;
    std::size_t bytes = 0;

    explicit DBuf(std::size_t b) : bytes(b) {
        if (bytes != 0) cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
    }

    ~DBuf() {
        if (p) cudaFree(p);
    }

    DBuf(DBuf&& o) noexcept : p(o.p), bytes(o.bytes) {
        o.p     = nullptr;
        o.bytes = 0;
    }

    DBuf& operator=(DBuf&& o) noexcept {
        if (this != &o) {
            if (p) cudaFree(p);
            p       = o.p;
            bytes   = o.bytes;
            o.p     = nullptr;
            o.bytes = 0;
        }
        return *this;
    }

    DBuf(const DBuf&)            = delete;
    DBuf& operator=(const DBuf&) = delete;

    template <typename T>
    T* data() noexcept {
        return static_cast<T*>(p);
    }

    template <typename T>
    const T* data() const noexcept {
        return static_cast<const T*>(p);
    }

    void fill(int byte_value = 0) {
        if (bytes != 0) cuda_check(cudaMemset(p, byte_value, bytes), "cudaMemset");
    }

    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0) {
        require_range(byte_offset, count, "host-to-device copy");
        if (count == 0) return;
        auto* destination = static_cast<std::uint8_t*>(p) + byte_offset;
        cuda_check(cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice),
                   "cudaMemcpy host-to-device");
    }

    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const {
        require_range(byte_offset, count, "device-to-host copy");
        if (count == 0) return;
        const auto* source = static_cast<const std::uint8_t*>(p) + byte_offset;
        cuda_check(cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device-to-host");
    }

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const {
        if (byte_offset > bytes || count > bytes - byte_offset) {
            throw std::out_of_range(std::string(operation) + " exceeds device buffer");
        }
    }
};

// --- seeded honest inputs ---------------------------------------------------
inline void fill_uniform(std::vector<float>& v, std::uint32_t seed, float lo, float hi) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v) x = d(g);
}

inline void fill_iota_i32(std::vector<int>& v, int start = 0) {
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = start + static_cast<int>(i);
}

// L2-normalize each contiguous `d`-element row (matches RMSNorm/F.normalize
// upstream of GDN q/k; keeps recurrence test inputs from overflowing).
inline void l2_normalize_rows(std::vector<float>& v, int d, long long rows) {
    constexpr double eps = 1e-12;
    for (long long r = 0; r < rows; ++r) {
        float* row   = v.data() + r * d;
        double sumsq = 0.0;
        for (int i = 0; i < d; ++i) sumsq += double(row[i]) * double(row[i]);
        const float inv = float(1.0 / std::sqrt(sumsq + eps));
        for (int i = 0; i < d; ++i) row[i] *= inv;
    }
}

// --- host <-> device --------------------------------------------------------
template <typename T>
inline DBuf to_device(const std::vector<T>& h) {
    static_assert(std::is_trivially_copyable_v<T>);
    DBuf d(h.size() * sizeof(T));
    d.copy_from_host(h.data(), d.bytes);
    return d;
}

template <typename T>
inline std::vector<T> from_device(const DBuf& d, std::size_t n) {
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

inline DBuf to_device_bf16(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
    return to_device(b);
}

inline DBuf to_device_f32(const std::vector<float>& h) { return to_device(h); }

inline DBuf to_device_i32(const std::vector<int>& h) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    return to_device(h);
}

// --- device -> host (upcast to double for the comparison) -------------------
inline std::vector<double> from_device_bf16(const DBuf& d, std::size_t n) {
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

inline std::vector<double> from_device_f32(const DBuf& d, std::size_t n) {
    const std::vector<float> f = from_device<float>(d, n);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(f[i]);
    return o;
}

inline std::vector<int> from_device_i32(const DBuf& d, std::size_t n) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    return from_device<int>(d, n);
}

// --- verdict ----------------------------------------------------------------
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
class GuardedDBuf {
public:
    explicit GuardedDBuf(std::size_t payload_bytes, std::size_t guard_bytes = 256,
                         std::uint8_t guard_byte = 0xa5)
        : storage_(allocation_bytes(payload_bytes, guard_bytes)), payload_bytes_(payload_bytes),
          guard_bytes_(guard_bytes), guard_byte_(guard_byte) {
        storage_.fill(guard_byte_);
    }

    void* data() noexcept { return static_cast<std::uint8_t*>(storage_.p) + guard_bytes_; }

    const void* data() const noexcept {
        return static_cast<const std::uint8_t*>(storage_.p) + guard_bytes_;
    }

    std::size_t bytes() const noexcept { return payload_bytes_; }

    void fill(int byte_value = 0) {
        if (payload_bytes_ != 0) {
            cuda_check(cudaMemset(data(), byte_value, payload_bytes_),
                       "cudaMemset guarded payload");
        }
    }

    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0) {
        require_payload_range(byte_offset, count);
        if (count == 0) return;
        auto* destination = static_cast<std::uint8_t*>(data()) + byte_offset;
        cuda_check(cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice),
                   "cudaMemcpy host-to-guarded-device");
    }

    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const {
        require_payload_range(byte_offset, count);
        if (count == 0) return;
        const auto* source = static_cast<const std::uint8_t*>(data()) + byte_offset;
        cuda_check(cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost),
                   "cudaMemcpy guarded-device-to-host");
    }

    int verify_guards(const char* label) const {
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
        if (guard_bytes == 0) throw std::invalid_argument("GuardedDBuf requires a non-empty guard");
        return payload_bytes + 2 * guard_bytes;
    }

    void require_payload_range(std::size_t byte_offset, std::size_t count) const {
        if (byte_offset > payload_bytes_ || count > payload_bytes_ - byte_offset) {
            throw std::out_of_range("copy exceeds guarded device payload");
        }
    }

    DBuf storage_;
    std::size_t payload_bytes_;
    std::size_t guard_bytes_;
    std::uint8_t guard_byte_;
};

// Returns 0 (pass) / 1 (fail). Prints the diff line with the chosen preset.
inline int verify(const char* label, const std::vector<double>& got, const std::vector<double>& ref,
                  const Tolerance& tol) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size mismatch got=" << got.size() << " ref=" << ref.size() << '\n';
        return 1;
    }
    const DiffStats s = compute_diff(got.data(), ref.data(), (long long)got.size(), tol);
    print_diff(label, s, tol);
    const bool ok = diff_passes(s, tol);
    if (!ok) std::cerr << label << ": FAIL (see diff above)\n";
    return ok ? 0 : 1;
}

} // namespace ninfer::test
