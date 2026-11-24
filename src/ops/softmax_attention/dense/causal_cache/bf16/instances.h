#pragma once

#include "ops/softmax_attention/kernels/causal/bf16/schedule.cuh"

namespace ninfer::ops::detail {

enum class Bf16KvInstance { Grouped32, Grouped64, Tiled64, Tiled128, GroupedDecode };

struct Bf16KvInstanceDescription {
    int query_rows;
    int key_rows;
    bool grouped;
};

template <Bf16KvInstance>
struct Bf16KvInstanceTraits;
#define NINFER_BF16_KV_INSTANCE(ID, GROUPED, ...)                                                  \
    template <>                                                                                    \
    struct Bf16KvInstanceTraits<Bf16KvInstance::ID> {                                              \
        using Schedule = __VA_ARGS__;                                                              \
        static constexpr Bf16KvInstanceDescription description{Schedule::kQueryRows,               \
                                                               Schedule::kKeyRows, GROUPED};       \
    }
NINFER_BF16_KV_INSTANCE(Grouped32, true, Bf16KvGroupedMmaSchedule<32, 32>);
NINFER_BF16_KV_INSTANCE(Grouped64, true, Bf16KvGroupedMmaSchedule<64, 32>);
NINFER_BF16_KV_INSTANCE(Tiled64, false, Bf16KvTiledMmaSchedule<64, 64>);
NINFER_BF16_KV_INSTANCE(Tiled128, false, Bf16KvTiledMmaSchedule<128, 32>);
NINFER_BF16_KV_INSTANCE(GroupedDecode, true, Bf16KvGroupedMmaSchedule<16, 32, 2, 2, 1>);
#undef NINFER_BF16_KV_INSTANCE

constexpr Bf16KvInstanceDescription bf16_kv_instance_description(Bf16KvInstance id) {
    switch (id) {
#define NINFER_BF16_KV_DESCRIBE(ID)                                                                \
    case Bf16KvInstance::ID:                                                                       \
        return Bf16KvInstanceTraits<Bf16KvInstance::ID>::description
        NINFER_BF16_KV_DESCRIBE(GroupedDecode);
        NINFER_BF16_KV_DESCRIBE(Grouped32);
        NINFER_BF16_KV_DESCRIBE(Grouped64);
        NINFER_BF16_KV_DESCRIBE(Tiled64);
        NINFER_BF16_KV_DESCRIBE(Tiled128);
#undef NINFER_BF16_KV_DESCRIBE
    }
    return {};
}

} // namespace ninfer::ops::detail
