#include "ninfer/core/kv_cache.h"

#include <limits>
#include <new>
#include <stdexcept>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

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

std::int32_t normalize_quant_group(DType dtype, std::int32_t quant_group, std::int32_t head_dim) {
    if (dtype == DType::BF16) {
        if (quant_group != 0) {
            throw std::invalid_argument("KVCache BF16 mode must not set quant_group");
        }
        return 0;
    }
    if (dtype != DType::I8) { throw std::invalid_argument("KVCache dtype must be BF16 or I8"); }

    const std::int32_t group = quant_group == 0 ? kKvQuantGroup : quant_group;
    if (group != kKvQuantGroup) {
        throw std::invalid_argument("KVCache I8 mode requires quant_group 64");
    }
    if (head_dim % group != 0) {
        throw std::invalid_argument("KVCache head_dim must be divisible by quant_group");
    }
    return group;
}

std::int32_t scale_groups(const KVCache& cache) {
    if (cache.quant_group <= 0) { return 0; }
    return cache.head_dim / cache.quant_group;
}

KVHeadSlot slot_at(const KVCache& cache, std::uint32_t layer, std::uint32_t position,
                   std::int32_t kv_head) {
    validate_layer(cache, layer);
    validate_kv_head(cache, kv_head);
    if (position >= cache.max_context ||
        position > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::out_of_range("KVCache position out of range");
    }
    const std::int64_t element_offset = static_cast<std::int64_t>(cache.head_dim) *
                                        (static_cast<std::int64_t>(position) +
                                         static_cast<std::int64_t>(cache.padded_context) * kv_head);
    const std::int64_t byte_offset =
        element_offset * static_cast<std::int64_t>(dtype_size(cache.dtype));

    auto* k_ptr = static_cast<unsigned char*>(cache.k[layer].data) + byte_offset;
    auto* v_ptr = static_cast<unsigned char*>(cache.v[layer].data) + byte_offset;
    KVHeadSlot slot{Tensor(k_ptr, cache.dtype, {cache.head_dim}),
                    Tensor(v_ptr, cache.dtype, {cache.head_dim}),
                    {},
                    {}};
    if (cache.dtype == DType::I8) {
        const std::int32_t groups = scale_groups(cache);
        const std::int64_t scale_element_offset =
            static_cast<std::int64_t>(groups) *
            (static_cast<std::int64_t>(position) +
             static_cast<std::int64_t>(cache.padded_context) * kv_head);
        const std::int64_t scale_byte_offset =
            scale_element_offset * static_cast<std::int64_t>(dtype_size(DType::FP16));
        auto* k_scale_ptr =
            static_cast<unsigned char*>(cache.k_scale[layer].data) + scale_byte_offset;
        auto* v_scale_ptr =
            static_cast<unsigned char*>(cache.v_scale[layer].data) + scale_byte_offset;
        slot.k_scale = Tensor(k_scale_ptr, DType::FP16, {groups});
        slot.v_scale = Tensor(v_scale_ptr, DType::FP16, {groups});
    }
    return slot;
}

std::size_t checked_add_size(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("KVCache size overflow");
    }
    return a + b;
}

std::size_t align_up_size(std::size_t value, std::size_t alignment) {
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error("KVCache size overflow");
    }
    return (value + mask) & ~mask;
}

std::size_t arena_cursor_after(std::size_t cursor, std::size_t bytes) {
    return checked_add_size(align_up_size(cursor, kArenaAlign), bytes);
}

void preflight_cache_arena(DeviceArena& arena, std::uint32_t full_layers, std::size_t code_bytes,
                           std::size_t scale_bytes, bool quantized) {
    if (arena.used() > arena.capacity()) {
        throw std::runtime_error("cache arena used exceeds capacity");
    }
    std::size_t cursor = arena.used();
    for (std::uint32_t layer = 0; layer < full_layers; ++layer) {
        cursor = arena_cursor_after(cursor, code_bytes);
        cursor = arena_cursor_after(cursor, code_bytes);
        if (quantized) {
            cursor = arena_cursor_after(cursor, scale_bytes);
            cursor = arena_cursor_after(cursor, scale_bytes);
        }
    }
    if (cursor > arena.capacity()) { throw std::bad_alloc(); }
}

} // namespace

KVCache::KVCache(DeviceArena& cache_arena, std::uint32_t full_layers, std::uint32_t max_context_in,
                 std::int32_t num_kv_heads_in, std::int32_t head_dim_in, DType dtype_in,
                 std::int32_t quant_group_in)
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
    quant_group = normalize_quant_group(dtype_in, quant_group_in, head_dim_in);

    const auto padded_context_i32 = static_cast<std::int32_t>(padded_context);
    const Tensor code_shape(nullptr, dtype_in, {head_dim_in, padded_context_i32, num_kv_heads_in});
    const bool quantized      = dtype_in == DType::I8;
    const std::int32_t groups = quantized ? head_dim_in / quant_group : 0;
    const std::size_t scale_bytes =
        quantized
            ? Tensor(nullptr, DType::FP16, {groups, padded_context_i32, num_kv_heads_in}).bytes()
            : 0;
    preflight_cache_arena(cache_arena, full_layers, code_shape.bytes(), scale_bytes, quantized);

    k.reserve(full_layers);
    v.reserve(full_layers);
    if (quantized) {
        k_scale.reserve(full_layers);
        v_scale.reserve(full_layers);
    }
    for (std::uint32_t layer = 0; layer < full_layers; ++layer) {
        k.push_back(
            cache_arena.alloc(dtype_in, {head_dim_in, padded_context_i32, num_kv_heads_in}));
        v.push_back(
            cache_arena.alloc(dtype_in, {head_dim_in, padded_context_i32, num_kv_heads_in}));
        if (quantized) {
            k_scale.push_back(
                cache_arena.alloc(DType::FP16, {groups, padded_context_i32, num_kv_heads_in}));
            v_scale.push_back(
                cache_arena.alloc(DType::FP16, {groups, padded_context_i32, num_kv_heads_in}));
        }
    }
}

std::uint32_t KVCache::layer_count() const noexcept { return static_cast<std::uint32_t>(k.size()); }

KVHeadSlot KVCache::slot(std::uint32_t layer, std::uint32_t position, std::int32_t kv_head) const {
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

void KVCache::rewind(std::uint32_t position) {
    if (position > pos) { throw std::out_of_range("KVCache rewind cannot move forward"); }
    pos = position;
}

void KVCache::reset() noexcept { pos = 0; }

} // namespace ninfer
