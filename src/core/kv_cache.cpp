#include "qus/core/kv_cache.h"

#include <limits>
#include <new>
#include <stdexcept>

namespace qus {
namespace {

void validate_layer(const KVCache& cache, std::uint32_t layer) {
    if (layer >= cache.layer_count()) { throw std::out_of_range("KVCache layer out of range"); }
}

KVSlot slot_at(const KVCache& cache, std::uint32_t layer, std::uint32_t position) {
    validate_layer(cache, layer);
    if (position >= cache.max_context ||
        position > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::out_of_range("KVCache position out of range");
    }
    const auto pos_i32 = static_cast<std::int32_t>(position);
    return KVSlot{cache.k[layer].slice(2, pos_i32, 1), cache.v[layer].slice(2, pos_i32, 1)};
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
    : pos(0), max_context(max_context_in), num_kv_heads(num_kv_heads_in), head_dim(head_dim_in),
      dtype(dtype_in) {
    if (full_layers == 0) { throw std::invalid_argument("KVCache full_layers must be nonzero"); }
    if (max_context_in == 0 ||
        max_context_in > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("KVCache max_context out of range");
    }
    if (num_kv_heads_in <= 0) {
        throw std::invalid_argument("KVCache num_kv_heads must be positive");
    }
    if (head_dim_in <= 0) { throw std::invalid_argument("KVCache head_dim must be positive"); }

    const auto max_context_i32 = static_cast<std::int32_t>(max_context_in);
    const Tensor layer_shape(nullptr, dtype_in, {num_kv_heads_in, head_dim_in, max_context_i32});
    preflight_cache_arena(cache_arena, full_layers, layer_shape.bytes());

    k.reserve(full_layers);
    v.reserve(full_layers);
    for (std::uint32_t layer = 0; layer < full_layers; ++layer) {
        k.push_back(cache_arena.alloc(dtype_in, {num_kv_heads_in, head_dim_in, max_context_i32}));
        v.push_back(cache_arena.alloc(dtype_in, {num_kv_heads_in, head_dim_in, max_context_i32}));
    }
}

std::uint32_t KVCache::layer_count() const noexcept { return static_cast<std::uint32_t>(k.size()); }

KVSlot KVCache::slot(std::uint32_t layer, std::uint32_t position) const {
    if (position >= pos) { throw std::out_of_range("KVCache read position has not been written"); }
    return slot_at(*this, layer, position);
}

KVSlot KVCache::append_slot(std::uint32_t layer) const {
    if (pos >= max_context) { throw std::out_of_range("KVCache append position is full"); }
    return slot_at(*this, layer, pos);
}

void KVCache::advance() {
    if (pos >= max_context) { throw std::out_of_range("KVCache position is full"); }
    ++pos;
}

void KVCache::reset() noexcept { pos = 0; }

} // namespace qus
