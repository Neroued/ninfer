#include "ninfer/ops/loop_message.h"
#include "core/platform.h"
#include "ops/common/host/operands.h"
#include "ops/loop_message/implementation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
namespace operand = common::operand;
using namespace detail::loop_message;
constexpr const char* kOp = "loop_message_blend_rmsnorm";

#if NINFER_TARGET_RTX5090
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string(kOp) + ": " + message);
}

void check_extent(int x, int y, int z, int element_bytes) {
    // The common Tensor metadata uses signed 64-bit byte strides.
    if (std::int64_t(x) * y > std::numeric_limits<std::int64_t>::max() / z / element_bytes)
        throw std::overflow_error(std::string(kOp) + ": tensor byte extent overflows");
}

void validate_extents(int d, int r, int s, int first, int last) {
    require(d > 0 && d % 8 == 0 && d <= std::numeric_limits<int>::max() / 2 && r > 0 &&
                r % 16 == 0 && s > 0 && first > 0 && last >= first,
            "invalid hidden, rank, streams or token interval");
    check_extent(d, last, s, 4);
    check_extent(2 * d, r, s, 2);
}

operand::Range checked_tensor(const Tensor& value, DType dtype, std::initializer_list<int> shape,
                              const char* name, Device device) {
    operand::tensor(value, dtype, shape, 16, kOp, name, device);
    return operand::range(value, kOp);
}

// Decode uses a short PDL chain. Dense RTX 5090 sweeps favor a separate
// finish for the first few rows, a last-CTA reduction for the next wave,
// and token-tiled MMA once per-row weight replication starts to dominate.
constexpr int kSeparateRows = 4;
constexpr int kDecodeRows   = 6;
constexpr int kMediumRows   = 144;

constexpr std::size_t kChunkBudget      = 42ULL << 20;
constexpr std::size_t kWeightsPerStream = std::size_t(8) * kHidden * kRank;
// Public FP32 activations, BF16 output, blend, split-K partials, latent,
// inverse RMS and tile sums/counters. Destination broadcasting can only
// reduce this conservative estimate.
constexpr std::size_t kBytesPerRow = 14ULL * kHidden + 4 * kRank * kShortSplits + 2 * kRank +
                                     sizeof(float) * kUpTiles + sizeof(float2) +
                                     (kRank / kShortRankTile) * sizeof(unsigned);
constexpr int kChunkGranularity = 64;

int chunk_streams(int streams) {
    constexpr auto bytes = kWeightsPerStream + kBytesPerRow * kChunkGranularity;
    return std::min(streams, int(kChunkBudget / bytes));
}

int chunk_limit(int streams) {
    const auto room = kChunkBudget / chunk_streams(streams) - kWeightsPerStream;
    return int(room / (kBytesPerRow * kChunkGranularity)) * kChunkGranularity;
}

// A wave is the measured resident cluster count times the 16-token tile.
// Cluster8 serves the first wave without workspace (including the rows=145..256 band
// where multi-kernel Medium/PrefillShort incur extra launch and scratch overhead);
// PrefillShort and Cluster4 cover wave-two transitions and large-M residency.
constexpr int kTileTokens        = 16;
constexpr int kCluster8Wave      = 40;
constexpr int kShortSecondWave   = 56;
constexpr int kCluster4Wave      = 84;
constexpr std::int64_t kL2Bytes  = 96LL << 20;

Schedule select(int hidden, int rank, int tokens, int streams) {
    if (hidden != kHidden || rank != kRank) return {Route::Vector, tokens, streams};
    const auto rows = std::int64_t(tokens) * streams;
    if (rows <= kSeparateRows) return {Route::DecodeSeparate, tokens, streams};
    if (rows <= kDecodeRows) return {Route::DecodeLast, tokens, streams};
    if (rows <= kMediumRows) return {Route::Medium, tokens, chunk_streams(streams)};

    const auto tiles     = std::int64_t((tokens - 1) / kTileTokens + 1) * streams;
    const bool next_wave = tiles > kCluster8Wave;
    if (next_wave && tiles <= kShortSecondWave)
        return {Route::PrefillShort, std::min(tokens, chunk_limit(streams)),
                chunk_streams(streams)};
    const bool outside_l2 = rows > kL2Bytes / (2 * std::int64_t(sizeof(float)) * kHidden);
    const bool narrow     = outside_l2 || (next_wave && tiles <= kCluster4Wave);
    return {narrow ? Route::Cluster4 : Route::Cluster8, tokens, streams};
}

ScratchLayout scratch_layout(Schedule plan, int rank) {
    const auto rows = std::int64_t(plan.tokens) * plan.streams;
    ScratchLayout layout;
    if (plan.route == Route::Cluster4 || plan.route == Route::Cluster8) return layout;
    if (plan.route == Route::Vector) {
        layout.bytes = operand::bytes(rows, std::size_t(rank) * 2, kOp);
        return layout;
    }
    const bool decode  = plan.route == Route::DecodeSeparate || plan.route == Route::DecodeLast;
    const bool prefill = plan.route == Route::PrefillShort;
    const int splits   = decode                        ? kDecodeSplits
                         : plan.route == Route::Medium ? kMediumSplits
                                                       : kShortSplits;

    auto append = [&](std::size_t width) {
        const auto offset = layout.bytes;
        const auto size   = operand::bytes(rows, width, kOp);
        if (size > std::numeric_limits<std::size_t>::max() - offset)
            throw std::overflow_error("loop message scratch size overflows");
        layout.bytes += size;
        return offset;
    };
    layout.sum_tiles  = decode ? kDecodeTiles : kUpTiles;
    layout.projection = append(kRank * splits * sizeof(float));
    layout.blended    = append(kHidden * sizeof(float));
    layout.sums       = append(layout.sum_tiles * sizeof(float));
    if (prefill) {
        // Vector loads from latent are 16-byte aligned. Scalar counters go last.
        layout.latent        = append(kRank * sizeof(__nv_bfloat16));
        layout.inv           = append(sizeof(float2));
        constexpr int planes = kRank / kShortRankTile;
        layout.done          = append(planes * sizeof(unsigned));
    } else if (plan.route == Route::DecodeLast) {
        layout.done = append(sizeof(unsigned));
    }
    return layout;
}
#else
[[noreturn]] void unsupported_target() {
    throw std::runtime_error(std::string(kOp) + ": requires the RTX 5090 build target");
}
#endif
} // namespace

std::size_t loop_message_blend_rmsnorm_workspace_capacity_bytes(std::int32_t hidden,
                                                                std::int32_t rank,
                                                                std::int32_t streams,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens) {
#if NINFER_TARGET_RTX5090
    validate_extents(hidden, rank, streams, min_tokens, max_tokens);
    const auto size_at = [=](int m) {
        return scratch_layout(select(hidden, rank, m, streams), rank).bytes;
    };
    auto peak = size_at(max_tokens);
    if (hidden == kHidden && rank == kRank) {
        // Each schedule segment has a monotone or capped layout. Include the
        // last extent of each segment whose layout can shrink at the boundary.
        for (const int last : {kSeparateRows / streams, kDecodeRows / streams,
                               kMediumRows / streams, kShortSecondWave / streams * kTileTokens}) {
            if (last >= min_tokens && last < max_tokens) peak = std::max(peak, size_at(last));
        }
    }
    return peak;
#else
    (void)hidden;
    (void)rank;
    (void)streams;
    (void)min_tokens;
    (void)max_tokens;
    unsupported_target();
#endif
}

void loop_message_blend_rmsnorm(const Tensor& dest, const Tensor& message,
                                const LoopMessageWeights& weights, const Tensor& norm_weight,
                                float pair_eps, float norm_eps, MemoryArena& workspace, Tensor& out,
                                CudaExecutionView execution) {
#if NINFER_TARGET_RTX5090
    const int d = message.ne[0], m = message.ne[1], s = message.ne[2], r = weights.down.ne[1];
    const int sd = dest.ne[2], sn = norm_weight.ne[1];
    validate_extents(d, r, s, m, m);
    require((sd == 1 || sd == s) && (sn == 1 || sn == s), "invalid broadcast extent");
    require(pair_eps > 0 && std::isfinite(pair_eps) && norm_eps > 0 && std::isfinite(norm_eps),
            "epsilons must be finite and positive");

    const auto device = execution.device();
    operand::device(workspace.device(), device, kOp, "workspace");
    const std::array inputs{
        checked_tensor(dest, DType::FP32, {d, m, sd}, "dest", device),
        checked_tensor(message, DType::FP32, {d, m, s}, "message", device),
        checked_tensor(weights.down, DType::BF16, {2 * d, r, s}, "down", device),
        checked_tensor(weights.up, DType::BF16, {r, 2 * d, s}, "up", device),
        checked_tensor(norm_weight, DType::BF16, {d, sn}, "norm weight", device),
    };
    const auto output = checked_tensor(out, DType::BF16, {d, m, s}, "out", device);
    for (auto input : inputs) operand::disjoint(output, input, kOp);

    const auto plan   = select(d, r, m, s);
    const auto layout = scratch_layout(plan, r);
    auto scope        = workspace.scope();
    std::byte* base   = nullptr;
    if (layout.bytes) {
        const auto scratch = workspace.alloc_bytes(layout.bytes, 16);
        const auto range   = operand::range(scratch.data, layout.bytes, kOp);
        for (auto input : inputs) operand::disjoint(range, input, kOp);
        operand::disjoint(range, output, kOp);
        base = static_cast<std::byte*>(scratch.data);
    }
    const Arguments a{static_cast<const float*>(dest.data),
                      static_cast<const float*>(message.data),
                      static_cast<const __nv_bfloat16*>(weights.down.data),
                      static_cast<const __nv_bfloat16*>(weights.up.data),
                      static_cast<const __nv_bfloat16*>(norm_weight.data),
                      static_cast<__nv_bfloat16*>(out.data),
                      base ? reinterpret_cast<__nv_bfloat16*>(base + layout.latent) : nullptr,
                      d,
                      r,
                      m,
                      s,
                      sd == 1,
                      sn == 1,
                      pair_eps,
                      norm_eps};
    if (plan.route == Route::Cluster4 || plan.route == Route::Cluster8) {
        launch_cluster_prefill(a, plan.route, execution.stream());
        return;
    }
    if (plan.route == Route::Vector) {
        launch_generic(a, execution.stream());
        return;
    }
    for (int first_s = 0; first_s < s;) {
        const int ss = std::min(plan.streams, s - first_s);
        for (int first_m = 0; first_m < m;) {
            const int mm = std::min(plan.tokens, m - first_m);
            const Chunk chunk{a,
                              first_m,
                              first_s,
                              mm,
                              ss,
                              layout.sum_tiles,
                              reinterpret_cast<float*>(base + layout.projection),
                              reinterpret_cast<float2*>(base + layout.inv),
                              reinterpret_cast<float*>(base + layout.blended),
                              reinterpret_cast<float*>(base + layout.sums),
                              reinterpret_cast<unsigned*>(base + layout.done)};
            if (plan.route == Route::DecodeSeparate || plan.route == Route::DecodeLast)
                launch_decode(chunk, plan.route, execution.stream());
            else
                launch_prefill(chunk, plan.route, execution.stream());
            first_m += mm;
        }
        first_s += ss;
    }
#else
    (void)dest;
    (void)message;
    (void)weights;
    (void)norm_weight;
    (void)pair_eps;
    (void)norm_eps;
    (void)workspace;
    (void)out;
    (void)execution;
    unsupported_target();
#endif
}
} // namespace ninfer::ops
