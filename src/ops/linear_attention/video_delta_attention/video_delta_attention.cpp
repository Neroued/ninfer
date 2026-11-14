#include "ninfer/ops/video_delta_attention.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/video_delta_attention/launch.h"
#include "ops/common/host/launch.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <array>

namespace ninfer::ops {
namespace {
namespace vd               = detail::vda;
namespace val              = common::operand;
constexpr const char* Name = "video_delta_attention";
using Range                = val::Range;

std::size_t mul(std::size_t a, std::size_t b) { return val::bytes(a, b, Name); }

std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("video_delta_attention: workspace size overflow");
    return a + b;
}

void geometry(int h, int u, int first, int last, int batch, VideoDeltaAttentionWindow w) {
    if (h < 1 || h > 56 || u < 1 || batch < 1 || first < 0 || last < first ||
        std::int64_t(last) * u > std::numeric_limits<std::int32_t>::max() || w.chunk_frames < 1 ||
        w.radius < 0 || w.frame_offset < 0 || w.frame_offset >= w.chunk_frames)
        throw std::invalid_argument("video_delta_attention: invalid geometry or extent");
}

struct Layout {
    std::size_t matrix_words{}, boundary_words{}, bytes{};
};

Layout layout(int groups, int f, VideoDeltaAttentionWindow w, bool init = false) {
    if (!f) return {};
    const auto g      = std::size_t(groups);
    const auto chunks = (std::int64_t(f) + w.frame_offset + w.chunk_frames - 1) / w.chunk_frames;
    Layout l;
    l.matrix_words   = mul(mul(f, g), 128 * 128);
    l.boundary_words = init ? 0 : mul(mul(2 * std::size_t(chunks), g), 128 * 128);
    l.bytes          = mul(add(mul(2, l.matrix_words), l.boundary_words), sizeof(float));
    return l;
}

int working_groups(int heads, int batch, int limit) {
    return static_cast<int>(std::min<std::int64_t>(std::int64_t(heads) * batch, limit));
}

Layout attention_layout(int heads, int frames, int batch, VideoDeltaAttentionWindow window) {
    return layout(working_groups(heads, batch, vd::HeadWave), frames, window);
}

Layout state_layout(int heads, int tokens, int batch) {
    if (!tokens) return {};
    const int groups = working_groups(heads, batch, vd::StateHeadWave);
    auto result      = layout(groups, vd::state_statistics_partitions(groups, tokens), {}, true);
    // The last wave can choose more partitions than a full wave. Account for
    // both plans rather than assuming that the largest head count costs most.
    const int tail = std::int64_t(heads) * batch % vd::StateHeadWave;
    if (tail && tail != groups) {
        const auto last = layout(tail, vd::state_statistics_partitions(tail, tokens), {}, true);
        if (last.bytes > result.bytes) result = last;
    }
    return result;
}

Range dense(const Tensor& t, DType dt, std::array<int, 4> shape, Device device, const char* name) {
    val::device(t.device, device, Name, name);
    val::dtype(t, dt, Name, name);
    val::shape(t, shape, Name, name);
    const auto count = val::numel(t, Name);
    if (!count) return {0, 0};
    if (!t.is_contiguous() || !val::aligned(t.data, dtype_size(dt)))
        throw std::invalid_argument(std::string(Name) + ": invalid contiguous operand " + name);
    return val::range(t.data, mul(count, dtype_size(dt)), Name);
}

Range vector(const Tensor& t, int h, int tokens, int b, Device device, const char* name) {
    val::device(t.device, device, Name, name);
    val::dtype(t, DType::BF16, Name, name);
    val::shape(t, {128, h, tokens, b}, Name, name);
    if (!val::numel(t, Name)) return {0, 0};
    if (!val::aligned(t.data, 2) || t.nb[0] != 2)
        throw std::invalid_argument(std::string(Name) + ": invalid vector " + name);
    std::size_t span = 256;
    for (int axis = 1; axis < 4; ++axis) {
        if (t.nb[axis] <= 0 || t.nb[axis] % 2 || std::uint64_t(t.nb[axis]) < span)
            throw std::invalid_argument(std::string(Name) + ": unordered/overlapping strides " +
                                        name);
        span = add(span, mul(std::size_t(t.ne[axis] - 1), std::size_t(t.nb[axis])));
    }
    return val::range(t.data, span, Name);
}

vd::Vector view(const Tensor& t) {
    return {static_cast<const __nv_bfloat16*>(t.data), t.nb[1] / 2, t.nb[2] / 2, t.nb[3] / 2};
}

void execute(vd::Arguments a, std::span<const Range> inputs, Range output, MemoryArena& ws,
             CudaExecutionView execution, bool init) {
    val::device(ws.device(), execution.device(), Name, "workspace");
    for (auto r : inputs) val::disjoint(r, output, Name);
    if (!a.frames || (init && (a.tokens == 0 || a.state_scale == 0))) {
        if (init)
            CUDA_CHECK(cudaMemsetAsync(a.state_out, 0, mul(mul(a.heads, a.batch), 128 * 128 * 4),
                                       execution.stream()));
        return;
    }
    const auto l = init ? state_layout(a.heads, a.tokens, a.batch)
                        : attention_layout(a.heads, a.frames, a.batch, a.window);
    auto scope   = ws.scope();
    auto memory  = ws.alloc_bytes(l.bytes);
    auto wr      = val::range(memory.data, l.bytes, Name);
    val::disjoint(wr, output, Name);
    for (auto r : inputs) val::disjoint(wr, r, Name);
    auto* p              = static_cast<float*>(memory.data);
    auto* j              = p + l.matrix_words;
    auto* boundaries     = j + l.matrix_words;
    const int wave_limit = init ? vd::StateHeadWave : vd::HeadWave;
    const auto groups    = std::int64_t(a.heads) * a.batch;
    for (std::int64_t begin = 0; begin < groups; begin += wave_limit) {
        const int count = static_cast<int>(std::min<std::int64_t>(wave_limit, groups - begin));
        vd::launch(a, p, j, boundaries, begin % a.heads, count, begin / a.heads, init,
                   execution.stream());
    }
}

void attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& alpha,
               const Tensor& beta, const Tensor& initial, const Tensor* weight, const Tensor* gate,
               float eps, int u, VideoDeltaAttentionWindow w, MemoryArena& ws, Tensor& out,
               CudaExecutionView ex) {
    const int h = q.ne[1], f = alpha.ne[2], b = q.ne[3];
    geometry(h, u, f, f, b, w);
    const int t = f * u;
    std::array<Range, 8> reads{
        vector(q, h, t, b, ex.device(), "q"),
        vector(k, h, t, b, ex.device(), "k"),
        vector(v, h, t, b, ex.device(), "v"),
        dense(alpha, DType::FP32, {128, h, f, b}, ex.device(), "alpha"),
        dense(beta, DType::FP32, {h, t, b, 1}, ex.device(), "beta"),
        dense(initial, DType::FP32, {128, 128, h, b}, ex.device(), "initial state")};
    if (weight) {
        if (!(eps > 0) || !std::isfinite(eps))
            throw std::invalid_argument("video_delta_attention: invalid norm epsilon");
        reads[6] = dense(*weight, DType::BF16, {128, 1, 1, 1}, ex.device(), "norm weight");
        reads[7] = vector(*gate, h, t, b, ex.device(), "gate logits");
    }
    auto dest = dense(out, DType::BF16, {128, h, t, b}, ex.device(), "out");
    vd::Arguments a;
    a.q = view(q);
    a.k = view(k);
    a.v = view(v);
    if (gate) a.gate = view(*gate);
    a.alpha     = static_cast<const float*>(alpha.data);
    a.beta      = static_cast<const float*>(beta.data);
    a.initial   = static_cast<const float*>(initial.data);
    a.weight    = weight ? static_cast<const __nv_bfloat16*>(weight->data) : nullptr;
    a.out       = static_cast<__nv_bfloat16*>(out.data);
    a.heads     = h;
    a.frames    = f;
    a.per_frame = u;
    a.tokens    = t;
    a.batch     = b;
    a.window    = w;
    a.eps       = eps;
    execute(a, std::span<const Range>(reads.data(), weight ? 8 : 6), dest, ws, ex, false);
}
} // namespace

std::size_t video_delta_attention_workspace_capacity_bytes(int h, int u, int first, int last, int b,
                                                           VideoDeltaAttentionWindow w) {
    geometry(h, u, first, last, b, w);
    const auto bytes = attention_layout(h, last, b, w).bytes;
    return bytes ? add(bytes, 255) : 0;
}

std::size_t video_delta_state_init_workspace_capacity_bytes(int h, int first, int last, int b) {
    geometry(h, 1, first, last, b, {});
    return last ? add(state_layout(h, last, b).bytes, 255) : 0;
}

void video_delta_state_init(const Tensor& k, const Tensor& v, const Tensor& beta, float scale,
                            MemoryArena& ws, Tensor& state, CudaExecutionView ex) {
    const int h = k.ne[1], t = k.ne[2], b = k.ne[3];
    geometry(h, 1, t, t, b, {});
    if (!std::isfinite(scale))
        throw std::invalid_argument("video_delta_state_init: nonfinite scale");
    const Range reads[]{vector(k, h, t, b, ex.device(), "k"), vector(v, h, t, b, ex.device(), "v"),
                        dense(beta, DType::FP32, {h, t, b, 1}, ex.device(), "beta")};
    auto dest = dense(state, DType::FP32, {128, 128, h, b}, ex.device(), "state");
    vd::Arguments a;
    a.k           = view(k);
    a.v           = view(v);
    a.beta        = static_cast<const float*>(beta.data);
    a.state_out   = static_cast<float*>(state.data);
    a.heads       = h;
    a.frames      = t ? 1 : 0;
    a.per_frame   = t;
    a.tokens      = t;
    a.batch       = b;
    a.state_scale = scale;
    execute(a, reads, dest, ws, ex, true);
}

void video_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& alpha,
                           const Tensor& beta, const Tensor& initial, int u,
                           VideoDeltaAttentionWindow w, MemoryArena& ws, Tensor& out,
                           CudaExecutionView ex) {
    attention(q, k, v, alpha, beta, initial, nullptr, nullptr, 0, u, w, ws, out, ex);
}

void video_delta_attention_sigmoid_gated(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& alpha, const Tensor& beta,
                                         const Tensor& initial, const Tensor& weight,
                                         const Tensor& gate, float eps, int u,
                                         VideoDeltaAttentionWindow w, MemoryArena& ws, Tensor& out,
                                         CudaExecutionView ex) {
    attention(q, k, v, alpha, beta, initial, &weight, &gate, eps, u, w, ws, out, ex);
}
} // namespace ninfer::ops
