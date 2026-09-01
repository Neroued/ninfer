#include "ops/linear/q4/q4_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q4 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 4096:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 34816:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 131072:
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 8) { return launch_q4_draft_head_small_t; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    case 2048:
        if (n == 131072) {
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 20) { return launch_q4_draft_head_small_t; }
            if (t <= 32) { return launch_q4_mma_r64_c32; }
            if (t <= 48) { return launch_q4_mma_r64_c48; }
            if (t <= 56) { return launch_q4_mma_r64_c56; }
            if (t <= 63) { return launch_q4_mma_r64_c72; }
            if (t == 64) { return launch_q4_mma_r64_c64_endpoint; }
            if (t <= 72) { return launch_q4_mma_r64_c72; }
            if (t <= 80) { return launch_q4_mma_r64_c80; }
            if (t <= 96) { return launch_q4_mma_r64_c96; }
            if (t <= 104) { return launch_q4_mma_r64_c104_bounded; }
            if (t <= 111) { return launch_q4_mma_r64_c112_partial; }
            if (t == 112) { return launch_q4_mma_r64_c112; }
            if (t <= 119) { return launch_q4_mma_r64_c120_partial; }
            if (t == 120) { return launch_q4_mma_r64_c120; }
            return launch_q4_mma_r64_c128;
        }
        break;
    case 1152:
        if (t < 4 || t > 131072 || (t % 4) != 0) { break; }
        switch (n) {
        case 3456:
            if (t <= 36) { return launch_q4_simt_r8_c4; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        case 4304:
            if (t == 4) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t == 12) { return launch_q4_simt_r8_c4; }
            if (t <= 24) { return launch_q4_simt_r8_c8; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q4 linear: unsupported shape or T");
}

#ifdef NINFER_VOLTA_BUILD
bool q4_launch_needs_volta_fallback(Q4Launch launch) noexcept {
    return launch != launch_q4_gemv_r1_w8_direct && launch != launch_q4_gemv_r4_w1_direct &&
           launch != launch_q4_draft_head_small_t && launch != launch_q4_simt_r8_c4 &&
           launch != launch_q4_simt_r8_c8;
}
#endif

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8: {
        const Q4Launch launch = select_q4_a16_launch(n, k, t);
#ifdef NINFER_VOLTA_BUILD
        if (q4_launch_needs_volta_fallback(launch)) { return launch_q4_simt_r8_c8; }
#endif
        return launch;
    }
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q4 linear: unsupported policy");
}

#ifdef NINFER_VOLTA_BUILD
// Band in which the fused tensor-core route is routed. The lower bound is the measured crossover
// at N=4096 K=5120, both routes through the same bench back to back: T=8 SIMT 72.7us vs fused
// 93.2us, then T=9 129.0 vs ~93, T=12 146.4 vs 95.2, T=32 259.1 vs 110.6. T=8 is the only point
// SIMT wins, because it is exactly one eight-column tile; from T=9 it pays a second, mostly empty
// tile while the fused route is nearly flat in T.
// Upper bound keeps the fp32 split-K buffer bounded (n*T*4 bytes; 8.9 MB at the widest Q4 shape)
// and covers the whole concurrency range, since MTP3 puts C8 at T=32 and C16 at T=64.
constexpr std::int32_t kVoltaMmaMinT = 9;
constexpr std::int32_t kVoltaMmaMaxT = 64;

#endif

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
#ifdef NINFER_VOLTA_BUILD
    // Quadpair-split-N maps T to the 8-row axis instead of the 32-row one, so it is the right
    // geometry exactly where the companion kernel pads its A rows away. Needs no workspace at
    // all, so unlike that route there is nothing to fall back from.
    if ((policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        q4_uses_volta_qpn(out.ne[0], x.ne[0], t)) {
        launch_q4_volta_qpn(x, w, out, stream);
        return;
    }
    if (workspace != nullptr && t >= kVoltaMmaMinT && t <= kVoltaMmaMaxT &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) &&
        q4_volta_mma_supported(out.ne[0], x.ne[0], t)) {
        // Fall back rather than trust the caller to have sized the arena: linear's workspace
        // contract predates this route, so a caller that sized for the old zero-byte Q4
        // requirement must degrade to SIMT, not overrun.
        const std::size_t need = q4_volta_mma_workspace_bytes(out.ne[0], x.ne[0], t);
        if (workspace->capacity() - workspace->used() >= need) {
            launch_q4_volta_mma(x, w, out, *workspace, stream);
            return;
        }
    }
#else
    (void)workspace;
#endif
    const Q4Launch launch = select_q4_launch(w.n, w.k, t, policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
