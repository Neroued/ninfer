#include "qus/kernels/gated_delta_rule.h"

#include "kernels/launcher/gated_delta_rule.h"
#include "qus/core/device.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace qus::kernels {
namespace {

constexpr std::int32_t kS   = 128;
constexpr std::int32_t kHqk = 16;
constexpr std::int32_t kHv  = 48;

void require_dtype(const Tensor& t, DType dtype, const char* name) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string("gated_delta_rule: ") + name); }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gated_delta_rule: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_delta_rule: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gated_delta_rule: ") + name +
                                    " data must be non-null");
    }
}

void validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                        const Tensor& beta, float scale, const Tensor& ssm_state,
                        const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_dtype(ssm_state, DType::FP32, "ssm_state must be FP32");

    const std::int32_t T = q.ne[2];
    if (T <= 0) { throw std::invalid_argument("gated_delta_rule: T must be positive"); }
    require_shape(q, kS, kHqk, T, 1, "q");
    require_shape(k, kS, kHqk, T, 1, "k");
    require_shape(v, kS, kHv, T, 1, "v");
    require_shape(out, kS, kHv, T, 1, "out");
    require_shape(g, kHv, T, 1, 1, "g");
    require_shape(beta, kHv, T, 1, 1, "beta");
    require_shape(ssm_state, kS, kS, kHv, 1, "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    constexpr float expected_scale = 0.08838834764831845f;
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_rule: scale must be 1/sqrt(128)");
    }
}

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > static_cast<std::size_t>(-1) - a) {
        throw std::overflow_error("gated_delta_rule: scratch size overflow");
    }
    return a + b;
}

std::size_t transient_scratch_bytes(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& out) {
    std::size_t bytes = 0;
    bytes             = checked_add(bytes, static_cast<std::size_t>(q.numel()) * sizeof(float));
    bytes             = checked_add(bytes, static_cast<std::size_t>(k.numel()) * sizeof(float));
    bytes             = checked_add(bytes, static_cast<std::size_t>(v.numel()) * sizeof(float));
    bytes             = checked_add(bytes, static_cast<std::size_t>(out.numel()) * sizeof(float));
    return checked_add(bytes, 4 * 256);
}

std::size_t align_up(std::size_t offset, std::size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

Tensor scratch_tensor(unsigned char* base, std::size_t& offset, DType dtype,
                      std::initializer_list<std::int32_t> shape) {
    offset = align_up(offset, 256);
    Tensor t(base + offset, dtype, shape);
    offset = checked_add(offset, t.bytes());
    return t;
}

struct ScratchBuffer {
    void* ptr            = nullptr;
    std::size_t capacity = 0;

    ScratchBuffer() = default;

    ~ScratchBuffer() {
        if (ptr != nullptr) { cudaFree(ptr); }
    }

    ScratchBuffer(const ScratchBuffer&)            = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;

    ScratchBuffer(ScratchBuffer&& other) noexcept : ptr(other.ptr), capacity(other.capacity) {
        other.ptr      = nullptr;
        other.capacity = 0;
    }

    ScratchBuffer& operator=(ScratchBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) { cudaFree(ptr); }
            ptr            = other.ptr;
            capacity       = other.capacity;
            other.ptr      = nullptr;
            other.capacity = 0;
        }
        return *this;
    }

    void* get(std::size_t bytes) {
        if (capacity >= bytes) { return ptr; }
        if (ptr != nullptr) { CUDA_CHECK(cudaFree(ptr)); }
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        capacity = bytes;
        return ptr;
    }
};

ScratchBuffer& recurrent_scratch_for(cudaStream_t stream) {
    thread_local std::unordered_map<cudaStream_t, ScratchBuffer> scratch_by_stream;
    return scratch_by_stream[stream];
}

} // namespace

void gated_delta_rule_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                const Tensor& beta, float scale, Tensor& ssm_state, Tensor& out,
                                cudaStream_t stream) {
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    auto* scratch_base = static_cast<unsigned char*>(
        recurrent_scratch_for(stream).get(transient_scratch_bytes(q, k, v, out)));
    std::size_t off = 0;
    Tensor q_f32    = scratch_tensor(scratch_base, off, DType::FP32, {q.ne[0], q.ne[1], q.ne[2]});
    Tensor k_f32    = scratch_tensor(scratch_base, off, DType::FP32, {k.ne[0], k.ne[1], k.ne[2]});
    Tensor v_f32    = scratch_tensor(scratch_base, off, DType::FP32, {v.ne[0], v.ne[1], v.ne[2]});
    Tensor out_f32 =
        scratch_tensor(scratch_base, off, DType::FP32, {out.ne[0], out.ne[1], out.ne[2]});

    detail::gdn_cast_qkv_bf16_to_f32_launch(q, k, v, q_f32, k_f32, v_f32, stream);
    detail::gated_delta_rule_recurrent_launch(q_f32, k_f32, v_f32, g, beta, scale, ssm_state,
                                              out_f32, stream);
    detail::gdn_cast_f32_to_bf16_launch(out_f32, out, stream);
}

void gated_delta_rule_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, int chunk_size, WorkspaceArena& ws,
                              Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    (void)q;
    (void)k;
    (void)v;
    (void)g;
    (void)beta;
    (void)scale;
    (void)chunk_size;
    (void)ws;
    (void)ssm_state;
    (void)out;
    (void)stream;
    throw std::logic_error("gated_delta_rule_chunked: not implemented in gdn-2");
}

} // namespace qus::kernels
