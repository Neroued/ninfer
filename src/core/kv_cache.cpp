#include "qus/core/kv_cache.h"

#include <limits>
#include <new>
#include <stdexcept>

namespace qus {
namespace {

void validate_layer(const KVCache& cache, std::uint32_t layer) {
    if (layer >= cache.layer_count()) { throw std::out_of_range("KVCache layer out of range"); }
}

void validate_kv_head(const KVCache& cache, std::int32_t kv_head) {
    if (kv_head < 0 || kv_head >= cache.num_kv_heads) {
        throw std::out_of_range("KVCache kv_head out of range");
    }
}

std::uint32_t align_up_u32(std::uint32_t value, std::uint32_t alignment) {
    const std::uint64_t mask    = static_cast<std::uint64_t>(alignment) - 1U;
    const std::uint64_t aligned = (static_cast<std::uint64_t>(value) + mask) & ~mask;
    if (aligned > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("KVCache padded_context out of range");
    }
    return static_cast<std::uint32_t>(aligned);
}

KVHeadSlot slot_at(const KVCache& cache, std::uint32_t layer, std::uint32_t position,
                   std::int32_t kv_head) {
    validate_layer(cache, layer);
    validate_kv_head(cache, kv_head);
    if (position >= cache.max_context ||
        position > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::out_of_range("KVCache position out of range");
    }
    const std::int64_t element_offset =
        static_cast<std::int64_t>(cache.head_dim) *
        (static_cast<std::int64_t>(position) +
         static_cast<std::int64_t>(cache.padded_context) * kv_head);
    const std::int64_t byte_offset =
        element_offset * static_cast<std::int64_t>(dtype_size(cache.dtype));

    auto* k_ptr = static_cast<unsigned char*>(cache.k[layer].data) + byte_offset;
    auto* v_ptr = static_cast<unsigned char*>(cache.v[layer].data) + byte_offset;
    return KVHeadSlot{Tensor(k_ptr, cache.dtype, {cache.head_dim}),
                      Tensor(v_ptr, cache.dtype, {cache.head_dim})};
}

std::size_t checked_mul_size(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("KVCache size overflow");
    }
    return a * b;
}

std::size_t checked_add_size(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("KVCache size overflow");
    }
    return a + b;
}

void preflight_cache_arena(DeviceArena& arena, std::uint32_t full_layers, std::size_t layer_bytes) {
    if (arena.used() > arena.capacity()) {
        throw std::runtime_error("cache arena used exceeds capacity");
    }
    const std::size_t tensors    = checked_mul_size(static_cast<std::size_t>(full_layers), 2);
    const std::size_t per_tensor = checked_add_size(layer_bytes, 255);
    const std::size_t required   = checked_mul_size(tensors, per_tensor);
    const std::size_t remaining  = arena.capacity() - arena.used();
    if (required > remaining) { throw std::bad_alloc(); }
}

} // namespace

KVCache::KVCache(DeviceArena& cache_arena, std::uint32_t full_layers, std::uint32_t max_context_in,
                 std::int32_t num_kv_heads_in, std::int32_t head_dim_in, DType dtype_in)
    : pos(0), max_context(max_context_in), padded_context(align_up_u32(max_context_in, 128)),
      num_kv_heads(num_kv_heads_in), head_dim(head_dim_in), dtype(dtype_in) {
    if (full_layers == 0) { throw std::invalid_argument("KVCache full_layers must be nonzero"); }
    if (max_context_in == 0 ||
        max_context_in > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("KVCache max_context out of range");
    }
    if (num_kv_heads_in <= 0) {
        throw std::invalid_argument("KVCache num_kv_heads must be positive");
    }
    if (head_dim_in <= 0) { throw std::invalid_argument("KVCache head_dim must be positive"); }

    const auto padded_context_i32 = static_cast<std::int32_t>(padded_context);
    const Tensor layer_shape(nullptr, dtype_in,
                             {head_dim_in, padded_context_i32, num_kv_heads_in});
    preflight_cache_arena(cache_arena, full_layers, layer_shape.bytes());

    k.reserve(full_layers);
    v.reserve(full_layers);
    for (std::uint32_t layer = 0; layer < full_layers; ++layer) {
        k.push_back(cache_arena.alloc(dtype_in, {head_dim_in, padded_context_i32, num_kv_heads_in}));
        v.push_back(cache_arena.alloc(dtype_in, {head_dim_in, padded_context_i32, num_kv_heads_in}));
    }
}

std::uint32_t KVCache::layer_count() const noexcept { return static_cast<std::uint32_t>(k.size()); }

KVHeadSlot KVCache::slot(std::uint32_t layer, std::uint32_t position,
                         std::int32_t kv_head) const {
    if (position >= pos) { throw std::out_of_range("KVCache read position has not been written"); }
    return slot_at(*this, layer, position, kv_head);
}

KVHeadSlot KVCache::append_slot(std::uint32_t layer, std::int32_t kv_head) const {
    if (pos >= max_context) { throw std::out_of_range("KVCache append position is full"); }
    return slot_at(*this, layer, pos, kv_head);
}

void KVCache::advance() {
    if (pos >= max_context) { throw std::out_of_range("KVCache position is full"); }
    ++pos;
}

void KVCache::reset() noexcept { pos = 0; }

} // namespace qus
