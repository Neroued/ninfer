#include "ops/linear/projection/q8/q8_dispatch.h"
#include "ops/linear/projection/q8/shapes.h"
#include "core/platform.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using ShapeSelector = Q8Launch (*)(std::int32_t);

struct ShapeEntry {
    std::int32_t n;
    std::int32_t k;
    ShapeSelector select;
};

template <class Geometry>
constexpr ShapeEntry shape(ShapeSelector select) {
    return {Geometry::kOutputRows, Geometry::kInputRows, select};
}

constexpr std::array kShapes{
#if NINFER_TARGET_SPARK
    shape<Q8N4096K1024>(select_q8_n4096_k1024),   shape<Q8N4096K4096>(select_q8_n4096_k4096),
    shape<Q8N12288K4096>(select_q8_n12288_k4096), shape<Q8N2560K3072>(select_q8_n2560_k3072),
    shape<Q8N8192K2560>(select_q8_n8192_k2560),
#endif
    shape<Q8N1024K2048>(select_q8_n1024_k2048),   shape<Q8N1024K5120>(select_q8_n1024_k5120),
    shape<Q8N2048K4096>(select_q8_n2048_k4096),   shape<Q8N2048K4608>(select_q8_n2048_k4608),
    shape<Q8N2048K16384>(select_q8_n2048_k16384), shape<Q8N2560K6144>(select_q8_n2560_k6144),
    shape<Q8N4608K4608>(select_q8_n4608_k4608),   shape<Q8N5120K4608>(select_q8_n5120_k4608),
    shape<Q8N5120K6144>(select_q8_n5120_k6144),   shape<Q8N5120K10240>(select_q8_n5120_k10240),
    shape<Q8N5120K17408>(select_q8_n5120_k17408), shape<Q8N5120K25600>(select_q8_n5120_k25600),
    shape<Q8N6144K2560>(select_q8_n6144_k2560),   shape<Q8N6144K5120>(select_q8_n6144_k5120),
    shape<Q8N9216K2048>(select_q8_n9216_k2048),   shape<Q8N10240K2560>(select_q8_n10240_k2560),
    shape<Q8N12288K2048>(select_q8_n12288_k2048), shape<Q8N12288K2560>(select_q8_n12288_k2560),
    shape<Q8N14336K5120>(select_q8_n14336_k5120), shape<Q8N16384K2560>(select_q8_n16384_k2560),
    shape<Q8N34816K5120>(select_q8_n34816_k5120), shape<Q8N248320K5120>(select_q8_n248320_k5120),
};
} // namespace

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q8 linear: unsupported shape");
}

} // namespace ninfer::ops::detail
