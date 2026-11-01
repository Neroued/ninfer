#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_shapes.h"

#include <algorithm>
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
    shape<Q8N1024K2048>(select_q8_n1024_k2048),
    shape<Q8N1024K5120>(select_q8_n1024_k5120),
    shape<Q8N2048K4096>(select_q8_n2048_k4096),
    shape<Q8N2048K4608>(select_q8_n2048_k4608),
    shape<Q8N2048K16384>(select_q8_n2048_k16384),
    shape<Q8N4608K4608>(select_q8_n4608_k4608),
    shape<Q8N5120K4608>(select_q8_n5120_k4608),
    shape<Q8N5120K6144>(select_q8_n5120_k6144),
    shape<Q8N5120K10240>(select_q8_n5120_k10240),
    shape<Q8N5120K17408>(select_q8_n5120_k17408),
    shape<Q8N5120K25600>(select_q8_n5120_k25600),
    shape<Q8N6144K5120>(select_q8_n6144_k5120),
    shape<Q8N9216K2048>(select_q8_n9216_k2048),
    shape<Q8N12288K2048>(select_q8_n12288_k2048),
    shape<Q8N14336K5120>(select_q8_n14336_k5120),
    shape<Q8N34816K5120>(select_q8_n34816_k5120),
    shape<Q8N248320K5120>(select_q8_n248320_k5120),
    shape<Q8N5120K3072>(select_q8_n5120_k3072),
    shape<Q8N5120K5120>(select_q8_n5120_k5120),
    shape<Q8N5120K8704>(select_q8_n5120_k8704),
    shape<Q8N7168K5120>(select_q8_n7168_k5120),
    shape<Q8N17408K5120>(select_q8_n17408_k5120),
    shape<Q8N5120K2048>(select_q8_n5120_k2048),
    shape<Q8N124160K5120>(select_q8_n124160_k5120),
    // Qwen4Exp hyper-connection Down(+inject) and Up, and the PLE key/value parent.
    ShapeEntry{324, 10240, select_q8_few_rows_wide_k},
    ShapeEntry{320, 10240, select_q8_few_rows_wide_k},
    ShapeEntry{10240, 320, select_q8_generic},
    ShapeEntry{12800, 2560, select_q8_qwen4_rows_k2560},
    // Qwen4Exp dense projections of the `qwen3_8_flash_next_nvfp4` recipe (Q8): QSA q|k|gate|v,
    // QSA indexer q|k, attention and GDN output, GDN q|k|v|z and a|b, shared-expert gate|up and
    // down (ninfer-ext's registration, untuned on the RTX 5070 Ti).
    ShapeEntry{13312, 2560, select_q8_qwen4_n13312_k2560},
    ShapeEntry{640, 2560, select_q8_qwen4_n640_k2560},
    ShapeEntry{2560, 6144, select_q8_qwen4_n2560_k6144},
    ShapeEntry{16384, 2560, select_q8_qwen4_rows_k2560},
    ShapeEntry{96, 2560, select_q8_qwen4_n96_k2560},
    ShapeEntry{1280, 2560, select_q8_qwen4_rows_k2560},
    ShapeEntry{2560, 640, select_q8_qwen4_n2560_k640},
    // Qwen4Exp at tp 2 (models/qwen3_5/load/sharding.h), one rank's shard: the Q8 recipe's
    // attention q|k|gate|v (12 query + 1 KV heads), GDN q|k|v|z (8 key + 24 value heads) and a|b
    // (24 + 24 rows), the row-parallel attention and GDN outputs, and the stream-split
    // hyper-connection Down(+inject) and Up.
    ShapeEntry{6656, 2560, select_q8_qwen4_n6656_k2560},
    ShapeEntry{8192, 2560, select_q8_qwen4_rows_k2560},
    ShapeEntry{48, 2560, select_q8_qwen4_n48_k2560},
    ShapeEntry{2560, 3072, select_q8_qwen4_n2560_k3072},
    ShapeEntry{324, 5120, select_q8_few_rows_wide_k},
    ShapeEntry{320, 5120, select_q8_few_rows_wide_k},
    ShapeEntry{5120, 320, select_q8_generic},
    // Hyper-connection Down(+inject) [R + I, 4*5120] and Up [4*5120, R] at H = 5120, on the same
    // generic routes as H = 2560.
    ShapeEntry{324, 20480, select_q8_few_rows_wide_k},
    ShapeEntry{320, 20480, select_q8_few_rows_wide_k},
    ShapeEntry{20480, 320, select_q8_generic},
    // Low rank 640: Down(+inject) and Up at H = 2560 and 5120, same generic routes.
    ShapeEntry{644, 10240, select_q8_few_rows_wide_k},
    ShapeEntry{640, 10240, select_q8_few_rows_wide_k},
    ShapeEntry{10240, 640, select_q8_generic},
    ShapeEntry{644, 20480, select_q8_few_rows_wide_k},
    ShapeEntry{640, 20480, select_q8_few_rows_wide_k},
    ShapeEntry{20480, 640, select_q8_generic},
    // Qwen4Exp MTP stem: fc_embedding and fc_hidden (the latter over S columns per token).
    ShapeEntry{2560, 2560, select_q8_qwen4_rows_k2560},
    // Qwen4Exp Vision merger output (the Qwen3.5 tower's [4608 -> H] projection at H = 2560): the
    // 27B merger's K and T selection (its MMA and SIMT routes are generic over 64-row multiples).
    ShapeEntry{2560, 4608, select_q8_n5120_k4608},
    // Qwen3.5 Vision tower stored Q8 (a Flash-Next tower recipe): q|k|v, attention output, MLP
    // fc1/fc2 and the patch embedding, on the generic routes (K 4304 is padded to whole groups).
    ShapeEntry{3456, 1152, select_q8_generic},
    ShapeEntry{1152, 1152, select_q8_generic},
    ShapeEntry{4304, 1152, select_q8_generic},
    ShapeEntry{1152, 4304, select_q8_generic},
    ShapeEntry{1152, 1536, select_q8_generic},
};
} // namespace

bool q8_shape_registered(std::int32_t n, std::int32_t k) noexcept {
    return std::any_of(kShapes.begin(), kShapes.end(),
                       [&](const ShapeEntry& entry) { return entry.n == n && entry.k == k; });
}

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q8 linear: unsupported shape");
}

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy)(x, w, out, stream);
}

} // namespace ninfer::ops::detail
