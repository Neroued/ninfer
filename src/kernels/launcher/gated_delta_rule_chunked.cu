#include "kernels/launcher/gated_delta_rule.h"

#include "kernels/kernel/gdn_chunked_common.cuh"
#include "qus/core/device.h"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <new>

namespace qus::kernels::detail {
namespace {

constexpr std::int64_t kS   = 128;
constexpr std::int64_t kHqk = 16;
constexpr std::int64_t kHv  = 48;
constexpr std::int64_t kB   = 1;

} // namespace

std::size_t gdn_chunked_workspace_bytes(std::int64_t T) {
    if (T <= 0) { return 0; }
    return static_cast<std::size_t>(gdn_chunked::workspace_bytes(kS, kHqk, kHv, T, kB));
}

void gated_delta_rule_chunked_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                                     void* workspace, std::size_t workspace_bytes,
                                     cudaStream_t stream) {
    const auto layout =
        gdn_chunked::compute_workspace_layout(q.ne[0], q.ne[1], v.ne[1], q.ne[2], kB);
    if (workspace == nullptr || workspace_bytes < static_cast<std::size_t>(layout.total_bytes)) {
        throw std::bad_alloc();
    }

    auto* base     = static_cast<unsigned char*>(workspace);
    auto* g_cumsum = reinterpret_cast<float*>(base + layout.g_cumsum_off);
    auto* W        = reinterpret_cast<__nv_bfloat16*>(base + layout.W_off);
    auto* U        = reinterpret_cast<__nv_bfloat16*>(base + layout.U_off);
    auto* v_new    = reinterpret_cast<__nv_bfloat16*>(base + layout.v_new_off);
    auto* h_chunk  = reinterpret_cast<__nv_bfloat16*>(base + layout.h_chunk_off);

    gdn_chunked::prepare_wy_wu_config prepare{};
    prepare.S            = q.ne[0];
    prepare.H_qk         = q.ne[1];
    prepare.H_v          = v.ne[1];
    prepare.L            = q.ne[2];
    prepare.B            = kB;
    prepare.k            = static_cast<const __nv_bfloat16*>(k.data);
    prepare.v            = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in         = static_cast<const float*>(g.data);
    prepare.beta         = static_cast<const float*>(beta.data);
    prepare.W            = W;
    prepare.U            = U;
    prepare.g_cumsum_out = g_cumsum;
    prepare.stream       = stream;
    CUDA_CHECK(gdn_prepare_wy_wu::launch_prepare_wy_wu(prepare));

    gdn_chunked::state_passing_config state{};
    state.S         = q.ne[0];
    state.H_qk      = q.ne[1];
    state.H_v       = v.ne[1];
    state.L         = q.ne[2];
    state.B         = kB;
    state.W         = W;
    state.U         = U;
    state.k         = static_cast<const __nv_bfloat16*>(k.data);
    state.g_cumsum  = g_cumsum;
    state.state_in  = static_cast<const float*>(ssm_state_in.data);
    state.v_new     = v_new;
    state.h_chunk   = h_chunk;
    state.state_out = static_cast<float*>(ssm_state_out.data);
    state.stream    = stream;
    CUDA_CHECK(gdn_state_passing::launch_state_passing(state));

    gdn_chunked::chunk_output_config output{};
    output.S        = q.ne[0];
    output.H_qk     = q.ne[1];
    output.H_v      = v.ne[1];
    output.L        = q.ne[2];
    output.B        = kB;
    output.q        = static_cast<const __nv_bfloat16*>(q.data);
    output.k        = static_cast<const __nv_bfloat16*>(k.data);
    output.v_new    = v_new;
    output.g_cumsum = g_cumsum;
    output.h_chunk  = h_chunk;
    output.attn_out = static_cast<__nv_bfloat16*>(out.data);
    output.scale    = scale;
    output.stream   = stream;
    CUDA_CHECK(gdn_chunk_output::launch_chunk_output(output));
}

} // namespace qus::kernels::detail
