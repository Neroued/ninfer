#pragma once

namespace ninfer::ops::detail::moe_cuda::block_scaled {

enum class Nvfp4Pipeline { K128, K256Deep, K512, K256 };

// Low bits bypass L1 for weight/input codes; high bits request L2 eviction
// priorities for streaming weights and reused activations. Scales stay cached.
enum class Nvfp4CachePolicy {
    Cached               = 0,
    BypassWeightL1       = 1,
    BypassInputL1        = 2,
    BypassL1             = 3,
    StreamingWeights     = 5,
    StreamingAll         = 7,
    RetainInputs         = 13,
    RetainInputsBypassL1 = 14,
};

struct Nvfp4Tuning {
    Nvfp4Pipeline pipeline = Nvfp4Pipeline::K128;
    Nvfp4CachePolicy cache = Nvfp4CachePolicy::Cached;
    int down_row_window    = 0;
};

struct Nvfp4LoadSchedule {
    int gate_columns = 0, down_columns = 0;
    int max_load = 0x7fffffff;
    Nvfp4Tuning tuning;
    int queued_ctas_per_sm = 6;
};

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
