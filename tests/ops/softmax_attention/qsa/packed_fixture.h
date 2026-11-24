#pragma once

#include "core/layout.h"
#include "ops/ragged_fixture.h"

namespace ninfer::test::qsa {

inline Tensor packed_tensor(MemoryArena& arena, DType dtype, std::initializer_list<int> shape,
                            const void* host = nullptr) {
    auto t = arena.alloc(dtype, shape, 256);
    if (host) CUDA_CHECK(cudaMemcpy(t.data, host, t.bytes(), cudaMemcpyHostToDevice));
    return t;
}

inline ops::RaggedSequenceView packed_metadata(MemoryArena& arena, const RaggedFixture& ragged) {
    return {packed_tensor(arena, DType::I32, {ragged.segments() + 1}, ragged.prefix.data()),
            packed_tensor(arena, DType::I32, {ragged.segments()}, ragged.lengths.data()),
            packed_tensor(arena, DType::I32, {ragged.tokens()}, ragged.mapping.data()),
            ragged.prefix};
}

// The prefix ends on a fresh scratch alignment boundary.
inline std::size_t prefix_bytes(std::initializer_list<Tensor> tensors) {
    LayoutBuilder layout;
    for (const auto& t : tensors) (void)layout.add(t.bytes(), 256, "packed test operand");
    return tensors.size() ? layout.finish(256) : 0;
}

inline void finish_prefix(MemoryArena& arena, std::size_t bytes) {
    if (bytes > arena.used()) (void)arena.alloc_bytes(bytes - arena.used(), 1);
}

} // namespace ninfer::test::qsa
