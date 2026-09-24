#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// RTX 5090 codec frontiers balance trace-like and independent expert distributions. The public
// workspace query starts at the earliest codec-specific prefill route.
inline constexpr std::int32_t kSparseMoePrefillWorkspaceMin = 20;
inline constexpr std::int32_t kSparseMoePrefillQ4Q5Min      = 47;
inline constexpr std::int32_t kSparseMoePrefillQ4Q6Min      = 47;
inline constexpr std::int32_t kSparseMoePrefillQ8Q8Min      = 20;
// Swept cold against this profile's own small-T route over [2,46], each route measured through the
// operator benchmark at its own schedule. Small-T is ahead through T=8 (79.6 us against 81.9), and
// from T=9 to T=12 it is at most 4.5 % behind - the price of keeping the activation represented,
// which is worth paying for the accuracy: rel_l2 at T=1 is 1.60e-3 against 1.70e-1 for a four-bit
// activation. From T=13 the prefill route's lead becomes real (+9 % at T=16, +29 % at T=32) and
// outgrows that.
inline constexpr std::int32_t kSparseMoePrefillNvfp4Min = 13;
inline constexpr std::int32_t kSparseMoePrefillWideMin  = 768;
inline constexpr std::int32_t kSparseMoePrefillSliceMax = 4096;
inline constexpr std::int32_t kSparseMoeRouteTileTokens = 8;
// 257 logits padded to a 16-byte-aligned per-token stride.
inline constexpr std::int32_t kSparseMoeRouterScoreRows = 260;

struct SparseMoePrefillPlan {
    std::int32_t tokens         = 0;
    std::int32_t slice_tokens   = 0;
    std::size_t workspace_bytes = 0;
    // Whether the route stages its operands in NVFP4. The staging buffers are the profile's own,
    // so the flag has to reach the allocation and not only the launch.
    bool nvfp4 = false;
};

struct SparseMoePrefillWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    // Selection writes one rank local to a routing tile. The gather must keep this source separate
    // from the final inverse map because all threads in an assignment block consume the rank while
    // thread 0 publishes the packed column. The index kernel does both from one thread per
    // assignment, so the separation is there for the gather the Q8 routed codec still takes.
    Tensor local_rank;
    Tensor shared_scale;
    // Scan is the final consumer of tile_counts. Both maps below alias its dead prefix for the
    // remaining lifetime; 256 * ceil(T / 8) is at least 32 * T elements and they take 8 * T each.
    Tensor tile_counts;
    Tensor packed_index;
    // The token each packed column was routed from, so the routed gate/up GEMM can stage its
    // activation tile from `x` instead of from a materialised copy. Sits in the same dead prefix
    // behind packed_index and needs no allocation of its own.
    Tensor packed_token;
    Tensor tile_bases;
    Tensor expert_offsets;
    Tensor route_job_experts;
    Tensor route_job_columns;
    // A negative count selects the token-oriented adaptive route; its magnitude is the unused
    // grouped-route job count. Nonnegative values select the normal grouped route.
    Tensor route_job_count;

    // The three large allocations are lifetime unions:
    //   router scores FP32 <-> shared SwiGLU BF16
    //   gathered X BF16    <-> routed down output BF16 <-> adaptive FP32 activations
    //     (a Q4 routed gate/up stages from x and gathers nothing, so it only ever writes here)
    //   routed SwiGLU BF16 <-> routed FP32 token reduction <-> every NVFP4 staging plane
    Tensor score_storage;
    Tensor shared_activation;
    Tensor grouped_io;
    Tensor routed_storage;
    Tensor routed_sum;

    // The NVFP4 route quantises the chunk once and keeps both SwiGLU intermediates encoded, so
    // `down` needs no separate quantiser: 288 bytes a row against the bf16 plane's 1024. All six
    // are carved out of `routed_storage`, which this profile never writes -- the encoded
    // intermediate replaces the routed SwiGLU plane and the shared-down epilogue replaces the
    // token reduction, so the branch returns before either view has a reader. 3744 bytes a token
    // inside 8192, so the profile costs no workspace at all.
    Tensor nvfp4_input_codes;
    Tensor nvfp4_input_scales;
    Tensor nvfp4_routed_codes;
    Tensor nvfp4_routed_scales;
    Tensor nvfp4_shared_codes;
    Tensor nvfp4_shared_scales;
};

template <class Arena>
SparseMoePrefillWorkspace
allocate_sparse_moe_prefill_workspace(Arena& arena, std::int32_t capacity_tokens, bool nvfp4) {
    SparseMoePrefillWorkspace out;
    const std::int32_t assignments = 8 * capacity_tokens;
    const std::int32_t route_tiles =
        (capacity_tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;

    out.token_ids    = arena.alloc(DType::I32, {assignments}, 256);
    out.token_alpha  = arena.alloc(DType::FP32, {assignments}, 256);
    out.local_rank   = arena.alloc(DType::I32, {assignments}, 256);
    out.shared_scale = arena.alloc(DType::FP32, {capacity_tokens}, 256);
    out.tile_counts  = arena.alloc(DType::I32, {256, route_tiles}, 256);
    // Both maps alias the dead prefix of tile_counts, whose last reader is the scan. The
    // layout builder allocates nothing, so the offset is only formed when there is a pointer.
    auto* const counts = static_cast<std::int32_t*>(out.tile_counts.data);
    out.packed_index   = Tensor(counts, DType::I32, {assignments});
    out.packed_token =
        Tensor(counts != nullptr ? counts + assignments : nullptr, DType::I32, {assignments});
    out.tile_bases     = arena.alloc(DType::I32, {256, route_tiles}, 256);
    out.expert_offsets = arena.alloc(DType::I32, {257}, 256);
    // A route job is one nonempty expert column tile. The bound is for the
    // narrowest prefill tile (32 assignments) and includes every expert tail.
    const std::int32_t max_route_jobs = assignments / 32 + 256;
    out.route_job_experts             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    out.route_job_columns             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    // [0] is the job count, negative when the scan hands the extent to the decode path;
    // [1] is how many experts the route touched, which picks the gate/up pipeline depth.
    out.route_job_count = arena.alloc(DType::I32, {2}, 256);

    out.score_storage = arena.alloc(DType::FP32, {kSparseMoeRouterScoreRows, capacity_tokens}, 256);
    out.shared_activation = Tensor(out.score_storage.data, DType::BF16, {512, capacity_tokens});

    out.grouped_io = arena.alloc(DType::BF16, {2048, assignments}, 256);

    out.routed_storage = arena.alloc(DType::BF16, {512, assignments}, 256);
    out.routed_sum     = Tensor(out.routed_storage.data, DType::FP32, {2048, capacity_tokens});

    if (nvfp4) {
        // The layout builder allocates nothing, so an offset is only formed when there is a
        // pointer -- the same guard the index maps above use.
        auto* const base    = static_cast<std::uint8_t*>(out.routed_storage.data);
        std::int64_t cursor = 0;
        const auto carve    = [&](std::int32_t rows, std::int32_t columns) {
            std::uint8_t* const at = base != nullptr ? base + cursor : nullptr;
            cursor += (static_cast<std::int64_t>(rows) * columns + 255) / 256 * 256;
            return Tensor(at, DType::U8, {rows, columns});
        };
        out.nvfp4_input_codes   = carve(1024, capacity_tokens);
        out.nvfp4_input_scales  = carve(128, capacity_tokens);
        out.nvfp4_routed_codes  = carve(256, assignments);
        out.nvfp4_routed_scales = carve(32, assignments);
        out.nvfp4_shared_codes  = carve(256, capacity_tokens);
        out.nvfp4_shared_scales = carve(32, capacity_tokens);
        // 3744 bytes a token, and five alignment gaps of under 256 each, inside the 8192 a token
        // that `routed_storage` holds for a profile that has a routed SwiGLU plane. Asserted
        // rather than trusted: a wider staging plane has to move the union, not overrun it. The
        // eight is `assignments / capacity_tokens`, which is kTopK.
        static_assert(1024 + 128 + 8 * 256 + 8 * 32 + 256 + 32 + 5 * 255 <= 512 * 8 * 2,
                      "the NVFP4 staging planes must fit the routed SwiGLU plane they alias");
        static_cast<void>(cursor);
    }
    return out;
}

[[nodiscard]] bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                                           QType routed_down) noexcept;
[[nodiscard]] std::size_t sparse_moe_prefill_workspace_bytes(std::int32_t max_tokens, bool nvfp4);
[[nodiscard]] SparseMoePrefillPlan
resolve_sparse_moe_prefill_plan(std::int32_t tokens, QType routed_gate_up, QType routed_down);

void sparse_moe_prefill_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
