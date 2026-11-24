#include "models/glm5next/vision/execution.h"

#include "core/layout.h"
#include "core/transfer.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/softmax_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ninfer::models::glm5next::vision {
namespace {
struct Layout {
    LayoutRegion x, y, z, scratch;
    TensorRegion positions, segments;
    std::size_t bytes;
};

template <class Profile>
Layout layout(const VisionConfig& c, const Profile& p, std::uint32_t features) {
    const int n      = dimension(std::uint64_t(features) * c.merge_size * c.merge_size);
    const auto bytes = [](std::uint64_t rows, std::uint64_t cols) {
        return Tensor::metadata(DType::BF16, {dimension(rows), dimension(cols)}).bytes();
    };
    const auto linear = [&](const auto& weight) {
        const auto profile = ops::linear_profile(weight);
        return ops::linear_workspace_capacity_bytes(profile.qtype, profile.n, profile.k,
                                                    profile.policy, 1, features);
    };
    auto scratch  = std::max(linear(p.merger_projection), linear(p.merger_down));
    const auto gu = ops::linear_profile(p.merger_gate_up);
    scratch       = std::max(scratch, ops::linear_swiglu_workspace_capacity_bytes(
                                    gu.qtype, gu.n, gu.k, gu.policy, 1, features));
    scratch =
        std::max(scratch, ops::packed_softmax_attention_workspace_capacity_bytes(
                              {dimension(c.head_dim()), dimension(c.heads), dimension(c.heads)}, 1,
                              n, 1, features));
    LayoutBuilder b;
    Layout out;
    const auto root = std::max(bytes(c.hidden, n), bytes(c.output_hidden, features));
    out.x           = b.add(root, 256, "Vision residual/downsample");
    out.y           = b.add(root, 256, "Vision branch/merger projection");
    out.z           = b.add(std::max({bytes(4ULL * c.hidden, n), bytes(c.intermediate, n),
                                      bytes(c.patch_width(), n), bytes(c.merger_intermediate, features)}),
                            256, "Vision norm/QKV/FFN");
    out.positions   = b.add_tensor(DType::I32, {n, 2}, 256, "Vision 2D positions");
    out.segments = b.add_tensor(DType::I32, {dimension(features + 1ULL)}, 256, "Vision segments");
    out.scratch  = b.add(std::max<std::size_t>(1, scratch), 256, "Vision Op scratch");
    out.bytes    = b.finish(256);
    return out;
}
} // namespace

models::vision::Capacity plan_resources(const VisionConfig& c, const execution::VisionProfile& p,
                                        std::uint32_t maximum, std::size_t budget) {
    return models::vision::plan_capacity(
        budget, maximum, dimension(c.output_hidden), [&](std::uint32_t f) {
            const auto n = dimension(std::uint64_t(f) * c.merge_size * c.merge_size);
            LayoutBuilder host;
            (void)host.add_tensor(DType::BF16, {dimension(c.patch_width()), n}, 256, "patches");
            (void)host.add_tensor(DType::I32, {n, 2}, 256, "positions");
            (void)host.add_tensor(DType::I32, {dimension(f + 1ULL)}, 256, "segments");
            return models::vision::EncoderMemory{layout(c, p, f).bytes, host.finish(256)};
        });
}

std::size_t encode(const VisionConfig& c, const execution::VisionParameters& p,
                   const EncoderBatch& input, Tensor& output, MemorySpan backing,
                   MemorySpan staging, CudaExecutionView cuda) {
    if (!input.features || input.cu_seqlens.size() < 2 || input.cu_seqlens.front() != 0)
        throw std::invalid_argument("GLM Vision batch has no independent segments");
    const auto n = dimension(std::uint64_t(input.features) * c.merge_size * c.merge_size);
    const auto f = dimension(input.features), h = dimension(c.hidden);
    if (input.cu_seqlens.back() != n || input.positions.size() != 2ULL * n)
        throw std::invalid_argument("GLM Vision metadata does not cover its patches");
    for (std::size_t i = 1; i < input.cu_seqlens.size(); ++i)
        if (input.cu_seqlens[i] <= input.cu_seqlens[i - 1])
            throw std::invalid_argument("GLM Vision segments are not strictly ordered");
    if (output.dtype != DType::BF16 || output.ne[0] != dimension(c.output_hidden) ||
        output.ne[1] != f || output.ne[2] != 1 || output.ne[3] != 1 || !output.is_contiguous() ||
        output.device != cuda.device() || backing.device != cuda.device() ||
        staging.device != Device::cpu() || !staging.host_pinned || p.layers.size() != c.depth ||
        !c.depth)
        throw std::invalid_argument("GLM Vision operands differ from the prepared geometry");
    const auto segment_length = input.cu_seqlens[1];
    bool equal_segments       = true;
    for (std::size_t i = 2; i < input.cu_seqlens.size(); ++i)
        equal_segments &= input.cu_seqlens[i] - input.cu_seqlens[i - 1] == segment_length;
    const auto lo  = reinterpret_cast<std::uintptr_t>(backing.data);
    const auto dst = reinterpret_cast<std::uintptr_t>(output.data);
    if (dst < lo + backing.bytes && lo < dst + output.bytes())
        throw std::invalid_argument("GLM Vision output overlaps encoder workspace");
    const auto l = layout(c, p, input.features);
    if (backing.bytes < l.bytes) throw std::invalid_argument("GLM Vision workspace is too small");
    const auto tensor = [&](LayoutRegion region, std::int32_t rows, std::int32_t columns) {
        return Tensor(region.bind(backing), DType::BF16, {rows, columns});
    };
    auto x = tensor(l.x, h, n), y = tensor(l.y, h, n);
    const auto z_matrix = [&](std::uint64_t begin, int rows, int columns) {
        const auto bytes = Tensor::metadata(DType::BF16, {rows, columns}).bytes();
        return Tensor(l.z.bind(backing).subspan(begin * sizeof(std::uint16_t), bytes), DType::BF16,
                      {rows, columns});
    };
    MemoryArena host(staging);
    auto patch         = z_matrix(0, dimension(c.patch_width()), n);
    auto patch_host    = host.alloc_bytes(patch.bytes());
    std::size_t cursor = 0;
    for (auto part : input.patches) {
        if (part.size_bytes() > patch_host.bytes - cursor)
            throw std::invalid_argument("GLM Vision patch parts exceed the batch");
        std::memmove(static_cast<std::byte*>(patch_host.data) + cursor, part.data(),
                     part.size_bytes());
        cursor += part.size_bytes();
    }
    if (cursor != patch_host.bytes)
        throw std::invalid_argument("GLM Vision patch batch is incomplete");
    copy_bytes_async({patch.data, patch.bytes(), patch.device}, patch_host, cuda);
    auto positions    = l.positions.bind(backing);
    auto segments     = l.segments.bind(backing).slice(0, 0, input.cu_seqlens.size());
    const auto upload = [&](const void* source, Tensor& destination) {
        auto memory = host.alloc_bytes(destination.bytes());
        std::memcpy(memory.data, source, destination.bytes());
        copy_bytes_async({destination.data, destination.bytes(), destination.device}, memory, cuda);
    };
    upload(input.positions.data(), positions);
    upload(input.cu_seqlens.data(), segments);
    ops::linear_bias(patch, p.patch.weight, p.patch_bias, x, cuda);
    auto norm = z_matrix(0, h, n);
    ops::rmsnorm(x, p.layers.front().input_norm, c.norm_epsilon, false, norm, cuda);
    for (std::size_t i = 0; i < p.layers.size(); ++i) {
        const auto& layer = p.layers[i];
        std::array<Tensor, 3> qkv;
        for (int j = 0; j < 3; ++j) qkv[j] = z_matrix(std::uint64_t(j + 1) * h * n, h, n);
        ops::linear_split(norm, layer.qkv.weight, layer.qkv_bias, qkv, cuda);
        auto q = qkv[0].view({dimension(c.head_dim()), dimension(c.heads), n});
        auto k = qkv[1].view({dimension(c.head_dim()), dimension(c.heads), n});
        auto v = qkv[2].view({dimension(c.head_dim()), dimension(c.heads), n});
        ops::rmsnorm_rope_2d(positions, layer.query_norm, layer.key_norm, c.norm_epsilon, 10000.0F,
                             q, k, cuda);
        auto attended = y.view({dimension(c.head_dim()), dimension(c.heads), n});
        const ops::AttentionHeadGeometry geometry{dimension(c.head_dim()), dimension(c.heads),
                                                  dimension(c.heads)};
        const auto scale = 1.0F / std::sqrt(float(c.head_dim()));
        if (equal_segments)
            ops::packed_softmax_attention(q, k, v, geometry, scale, segment_length, attended, cuda);
        else {
            MemoryArena attention(l.scratch.bind(backing));
            ops::packed_softmax_attention(q, k, v, geometry, scale, segments, attention, attended,
                                          cuda);
        }
        ops::linear_bias(y, layer.output.weight, layer.output_bias, norm, cuda);
        ops::residual_add_rmsnorm(x, norm, layer.post_attention_norm, c.norm_epsilon, x, y, cuda);
        auto intermediate = z_matrix(0, dimension(c.intermediate), n);
        ops::linear_swiglu(y, layer.gate_up.weight, layer.gate_up_bias, intermediate,
                           c.swiglu_limit, cuda);
        ops::linear_bias(intermediate, layer.down.weight, layer.down_bias, y, cuda);
        const auto& gamma = i + 1 < p.layers.size() ? p.layers[i + 1].input_norm : p.post_norm;
        ops::residual_add_rmsnorm(x, y, gamma, c.norm_epsilon, x, norm, cuda);
    }
    auto merged = norm.view({dimension(c.merge_width()), f});
    auto first  = tensor(l.x, dimension(c.output_hidden), f);
    auto second = tensor(l.y, dimension(c.output_hidden), f);
    ops::linear_bias(merged, p.downsample.weight, p.downsample_bias, first, cuda);
    {
        MemoryArena scratch(l.scratch.bind(backing));
        ops::linear(first, p.merger_projection.weight, second, p.merger_projection.policy, scratch,
                    cuda);
    }
    ops::layer_norm(second, p.merger_norm_weight, p.merger_norm_bias, c.norm_epsilon, first, cuda);
    ops::gelu(first, ops::GeluMode::Exact, cuda);
    auto intermediate = z_matrix(0, dimension(c.merger_intermediate), f);
    {
        MemoryArena scratch(l.scratch.bind(backing));
        ops::linear_swiglu(first, p.merger_gate_up.weight, intermediate, c.swiglu_limit,
                           p.merger_gate_up.policy, scratch, cuda);
    }
    {
        MemoryArena scratch(l.scratch.bind(backing));
        ops::linear(intermediate, p.merger_down.weight, output, p.merger_down.policy, scratch,
                    cuda);
    }
    return l.bytes;
}

} // namespace ninfer::models::glm5next::vision
