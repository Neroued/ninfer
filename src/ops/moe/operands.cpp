#include "ops/moe/operands.h"
#include "core/platform.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
namespace operand = common::operand;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

struct MatrixBytes {
    std::size_t codes, high, scales;
};

MatrixBytes matrix_bytes(QType type, int rows, int columns) {
    const auto elements = std::size_t(rows) * columns;
    switch (type) {
    case QType::BF16:
        return {elements * 2, 0, 0};
    case QType::NVFP4:
        return {elements / 2, 0, elements / 16};
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
        return {elements, 0, elements / (128 * 128) * 2};
    case QType::Q4_G64_FP16:
        return {elements / 2, 0, elements / 32};
    case QType::Q5_G64_FP16:
        return {elements / 2, elements / 8, elements / 32};
    case QType::Q6_G64_FP16:
        return {elements / 2, elements / 4, elements / 32};
    case QType::Q8_G32_FP16:
        return {elements, 0, elements / 16};
    default:
        throw std::invalid_argument("MoE matrix format is unsupported");
    }
}
} // namespace

void validate_route_profile(const RouteProfile& p) {
    require((p.hidden == 2048 && p.experts == 256 && p.top_k == 8) ||
                (p.hidden == 2560 && p.experts == 512 && p.top_k == 10),
            "topk softmax route: unsupported geometry");
}

void validate_expert_profile(const RoutedSwiGLUProfile& p) {
    if (p.ffn.hidden == 4096) {
#if NINFER_TARGET_SPARK
        const auto& f    = p.ffn;
        const bool nvfp4 = f.gate_up_type == QType::NVFP4 &&
                           f.gate_up_layout == QuantLayout::BlockScaleK16M128x4 &&
                           f.gate_up_policy == LinearPolicy::AllowA4;
        const bool fp8 = f.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_BF16 &&
                         f.gate_up_layout == QuantLayout::BlockScaleM128K128 &&
                         f.gate_up_policy == LinearPolicy::AllowA8;
        require(f.intermediate == 2048 && p.experts == 288 && p.top_k == 8 && f.clamp_limit &&
                    std::isfinite(*f.clamp_limit) && *f.clamp_limit > 0 && (nvfp4 || fp8) &&
                    f.down_type == f.gate_up_type && f.down_layout == f.gate_up_layout &&
                    f.down_policy == f.gate_up_policy,
                "routed SwiGLU: unsupported clamped block-scaled profile");
        return;
#else
        throw std::invalid_argument("routed SwiGLU: unregistered geometry on this target");
#endif
    }

    require(!p.ffn.clamp_limit, "routed SwiGLU: clamped profile is not registered");
    validate_route_profile({p.ffn.hidden, p.experts, p.top_k});
    const auto& f = p.ffn;
    if (f.hidden == 2560) {
        const bool nvfp4 = f.gate_up_type == QType::NVFP4 &&
                           f.gate_up_layout == QuantLayout::BlockScaleK16M128x4 &&
                           f.gate_up_policy == LinearPolicy::AllowA4;
        const bool fp8 = f.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_BF16 &&
                         f.gate_up_layout == QuantLayout::BlockScaleM128K128 &&
                         f.gate_up_policy == LinearPolicy::AllowA8;
        require(f.intermediate == 640 && (nvfp4 || fp8) && f.down_type == f.gate_up_type &&
                    f.down_layout == f.gate_up_layout && f.down_policy == f.gate_up_policy,
                "routed SwiGLU: unsupported block-scaled profile");
        return;
    }
    require(f.intermediate == 512 && f.gate_up_layout == QuantLayout::RowSplit &&
                f.down_layout == QuantLayout::RowSplit && valid_linear_policy(f.gate_up_policy) &&
                valid_linear_policy(f.down_policy),
            "routed SwiGLU: unsupported matrix geometry, layout or policy");
    require((f.gate_up_type == QType::Q4_G64_FP16 &&
             (f.down_type == QType::Q5_G64_FP16 || f.down_type == QType::Q6_G64_FP16)) ||
                (f.gate_up_type == QType::Q8_G32_FP16 && f.down_type == QType::Q8_G32_FP16),
            "routed SwiGLU: unsupported weight profile");
}

void validate_shared_profile(const SwiGLUProfile& p, int hidden) {
    require(!p.clamp_limit, "shared SwiGLU: clamped profile is not registered");
    if (hidden == 2560) {
        require(p.hidden == hidden && (p.intermediate == 320 || p.intermediate == 640) &&
                    p.gate_up_type == QType::BF16 && p.down_type == QType::BF16 &&
                    p.gate_up_layout == QuantLayout::Contiguous &&
                    p.down_layout == QuantLayout::Contiguous &&
                    p.gate_up_policy == LinearPolicy::A16Only &&
                    p.down_policy == LinearPolicy::A16Only,
                "shared SwiGLU: unsupported BF16 profile");
        return;
    }
    require(p.hidden == hidden && p.intermediate == 512 && p.gate_up_type == QType::Q8_G32_FP16 &&
                p.down_type == QType::Q8_G32_FP16 && p.gate_up_layout == QuantLayout::RowSplit &&
                p.down_layout == QuantLayout::RowSplit && valid_linear_policy(p.gate_up_policy) &&
                valid_linear_policy(p.down_policy),
            "shared SwiGLU: unsupported matrix geometry, format or policy");
}

void validate_moe_profile(const MoeProfile& p) {
    validate_expert_profile(p.routed);
    validate_shared_profile(p.shared, p.routed.ffn.hidden);
}

void validate_interval(int first, int last) {
    require(first > 0 && last >= first, "MoE workspace: invalid token interval");
}

void validate_subset_kind(ExpertSubsetKind kind) {
    require(kind == ExpertSubsetKind::All || kind == ExpertSubsetKind::None ||
                kind == ExpertSubsetKind::MaskOnes || kind == ExpertSubsetKind::MaskZeros,
            "routed SwiGLU: invalid expert subset");
}

void validate_output(DType type, MoeEpilogue mode) {
    require((type == DType::BF16 || type == DType::FP32) &&
                (mode == MoeEpilogue::Write || mode == MoeEpilogue::AddResidual) &&
                (type == DType::BF16 || mode == MoeEpilogue::Write),
            "MoE: output requires FP32 Write, BF16 Write or BF16 AddResidual");
}

MoeOperands::MoeOperands(const char* operation, CudaExecutionView execution)
    : op_(operation), device_(execution.device()) {
    require(device_.type() == DeviceType::CUDA && execution.multiprocessor_count() > 0,
            "MoE: invalid CUDA execution view");
}

void MoeOperands::read(const void* pointer, std::size_t bytes) {
    if (bytes) { reads_.at(read_count_++) = operand::range(pointer, bytes, op_); }
}

void MoeOperands::write(const Tensor& tensor) {
    writes_.at(write_count_++) = operand::range(tensor, op_);
}

int MoeOperands::input(const Tensor& x, int hidden) {
    require(x.ne[1] > 0, "MoE: T must be positive");
    operand::tensor(x, DType::BF16, {hidden, x.ne[1]}, 16, op_, "x", device_);
    read(x.data, x.bytes());
    return x.ne[1];
}

void MoeOperands::output(const Tensor& y, int hidden, int tokens, MoeEpilogue mode) {
    validate_output(y.dtype, mode);
    operand::tensor(y, y.dtype, {hidden, tokens}, 16, op_, "destination", device_);
    write(y);
}

void MoeOperands::control(const Weight& w, const RouteProfile& p) {
    operand::device(w.device, device_, op_, "control");
    require(w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.n == p.experts + 1 &&
                w.k == p.hidden && w.ndim == 2 && w.shape[0] == w.n && w.shape[1] == w.k &&
                w.shape[2] == 1 && w.shape[3] == 1 && w.padded_shape[0] == w.n &&
                w.padded_shape[1] == w.k && operand::aligned(w.qdata, 16) && !w.qhigh && !w.scales,
            "MoE: invalid control weight");
    const auto bytes = std::size_t(w.n) * w.k * 2;
    require(w.payload_bytes >= bytes, "MoE: truncated control weight");
    read(w.qdata, bytes);
}

void MoeOperands::sigmoid_control(const Weight& w, const RouteProfile& p) {
    require(w.group == 0 && w.group_size == 0 && w.high_plane_bytes == 0,
            "MoE: invalid sigmoid control metadata");
    matrix(w, p.experts, p.hidden, QType::BF16);
}

void MoeOperands::routes(const ExpertRoutes& r, int top_k, int tokens, bool writing) {
    operand::tensor(r.ids, DType::I32, {top_k, tokens}, 16, op_, "route IDs", device_);
    operand::tensor(r.coefficients, DType::FP32, {top_k, tokens}, 16, op_, "route coefficients",
                    device_);
    if (writing) {
        write(r.ids);
        write(r.coefficients);
    } else {
        read(r.ids.data, r.ids.bytes());
        read(r.coefficients.data, r.coefficients.bytes());
    }
}

void MoeOperands::coefficient(const Tensor& value, int tokens, bool writing) {
    operand::tensor(value, DType::FP32, {tokens}, 4, op_, "shared coefficients", device_);
    if (writing) {
        write(value);
    } else {
        read(value.data, value.bytes());
    }
}

void MoeOperands::subset(ExpertSubset subset, int experts) {
    validate_subset_kind(subset.kind);
    if (subset.kind == ExpertSubsetKind::MaskOnes || subset.kind == ExpertSubsetKind::MaskZeros) {
        operand::tensor(subset.mask, DType::I32, {experts}, 4, op_, "expert mask", device_);
        read(subset.mask.data, subset.mask.bytes());
    }
}

void MoeOperands::experts(const CudaSwiGLUExperts& w, const RoutedSwiGLUProfile& p) {
    operand::device(w.bank.device, device_, op_, "expert bank");
    require(w.bank.count == std::uint32_t(p.experts) &&
                bool(w.bank.gate_up.offsets) == bool(w.bank.down.offsets),
            "MoE: expert bank count or addressing differs");
    for (const auto divisor : {w.activation_divisors.gate_up, w.activation_divisors.down}) {
        require(!divisor || (std::isfinite(*divisor) && *divisor > 0),
                "MoE: activation divisor must be finite and positive");
    }
    if (p.ffn.gate_up_type == QType::NVFP4) {
        require(w.activation_divisors.gate_up && w.activation_divisors.down,
                "MoE: NVFP4 A4 requires both activation divisors");
    } else if (p.ffn.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_BF16) {
        require(!w.activation_divisors.gate_up && !w.activation_divisors.down,
                "MoE: block FP8 has no activation divisors");
    }
    const auto matrix = [&](const ExpertMatrixDeviceView& view, QType type, int n, int k) {
        const auto bytes = matrix_bytes(type, n, k);
        if (type == QType::NVFP4) {
            require(operand::aligned(view.divisors, alignof(float)) &&
                        view.divisor_stride <= (SIZE_MAX / sizeof(float) - 1) / (p.experts - 1),
                    "MoE: invalid NVFP4 matrix divisors");
            read(view.divisors, ((p.experts - 1) * view.divisor_stride + 1) * sizeof(float));
        } else {
            require(!view.divisors && view.divisor_stride == 0,
                    "MoE: this weight format has no matrix divisors");
        }
        if (view.offsets) {
            require(operand::aligned(view.offsets, 8), "MoE: unaligned expert offsets");
            read(view.offsets, std::size_t(p.experts) * 8);
            operand::device(view.indexed_storage.device, device_, op_, "indexed expert storage");
            require(view.indexed_storage.data && view.indexed_storage.bytes,
                    "MoE: missing indexed expert storage");
            read(view.indexed_storage.data, view.indexed_storage.bytes);
        }
        const auto plane = [&](const std::byte* base, std::size_t size, std::uint64_t stride,
                               std::size_t alignment = 16) {
            if (!size) {
                require(!base && !stride, "MoE: unexpected encoded plane");
                return;
            }
            require(operand::aligned(base, alignment), "MoE: missing or unaligned encoded plane");
            if (view.offsets) {
                const auto begin   = reinterpret_cast<std::uintptr_t>(view.indexed_storage.data);
                const auto address = reinterpret_cast<std::uintptr_t>(base);
                require(address >= begin && address - begin <= view.indexed_storage.bytes &&
                            size <= view.indexed_storage.bytes - (address - begin),
                        "MoE: indexed plane lies outside its storage");
            } else {
                require(stride >= size && stride % alignment == 0 &&
                            stride <= (SIZE_MAX - size) / std::size_t(p.experts - 1),
                        "MoE: invalid expert stride");
                read(base, (p.experts - 1) * stride + size);
            }
        };
        plane(view.codes, bytes.codes, view.code_stride);
        plane(view.high, bytes.high, view.high_stride);
        plane(view.scales, bytes.scales, view.scale_stride,
              type == QType::FP8_E4M3FN_BLOCK128X128_BF16 ? 2 : 16);
    };
    matrix(w.bank.gate_up, p.ffn.gate_up_type, 2 * p.ffn.intermediate, p.ffn.hidden);
    matrix(w.bank.down, p.ffn.down_type, p.ffn.hidden, p.ffn.intermediate);
}

void MoeOperands::matrix(const Weight& w, int rows, int columns, QType type) {
    operand::device(w.device, device_, op_, "shared weight");
    require((type == QType::BF16 || type == QType::Q8_G32_FP16) && w.qtype == type && w.n == rows &&
                w.k == columns && w.ndim == 2 && w.shape[0] == rows && w.shape[1] == columns &&
                w.shape[2] == 1 && w.shape[3] == 1 && w.padded_shape[0] == rows &&
                w.padded_shape[1] == columns && !w.qhigh && !w.high_plane_bytes &&
                operand::aligned(w.qdata, 16),
            "MoE: invalid shared matrix");
    require(type == QType::BF16
                ? w.layout == QuantLayout::Contiguous && !w.scales
                : w.layout == QuantLayout::RowSplit && w.group_size == 32 && w.group == 32 &&
                      w.scale_dtype == DType::FP16 && operand::aligned(w.scales, 16),
            "MoE: invalid shared matrix encoding");
    const auto bytes = matrix_bytes(type, rows, columns);
    require(w.payload_bytes >= bytes.codes + bytes.scales, "MoE: truncated shared matrix");
    read(w.qdata, bytes.codes);
    read(w.scales, bytes.scales);
}

void MoeOperands::shared(const SwiGLUWeights& weights, const SwiGLUProfile& profile) {
    validate_shared_profile(profile, profile.hidden);
    const auto actual = swiglu_profile(weights);
    require(actual.gate_up_policy == profile.gate_up_policy &&
                actual.down_policy == profile.down_policy,
            "MoE: shared weight policy differs from profile");
    matrix(weights.gate_up.weight, 2 * profile.intermediate, profile.hidden, profile.gate_up_type);
    matrix(weights.down.weight, profile.hidden, profile.intermediate, profile.down_type);
}

void MoeOperands::unscaled_shared(const SwiGLUWeights& weights, const SwiGLUProfile& p) {
    auto prepared        = swiglu_profile(weights);
    prepared.clamp_limit = p.clamp_limit;
    require(prepared == p, "MoE: unscaled shared weights differ from profile");
    for (const Weight* w : {&weights.gate_up.weight, &weights.down.weight})
        if (w->qtype == QType::BF16)
            require(w->group == 0 && w->group_size == 0 && w->high_plane_bytes == 0,
                    "MoE: invalid unscaled BF16 shared metadata");
    matrix(weights.gate_up.weight, 2 * p.intermediate, p.hidden, p.gate_up_type);
    matrix(weights.down.weight, p.hidden, p.intermediate, p.down_type);
}

void MoeOperands::workspace(const MemoryArena& workspace, std::size_t bytes) {
    operand::device(workspace.device(), device_, op_, "workspace");
    require(workspace.used() <= workspace.capacity() &&
                bytes <= workspace.capacity() - workspace.used(),
            "MoE: insufficient workspace");
    if (bytes) {
        const auto span = workspace.storage().subspan(workspace.used(), bytes);
        require(operand::aligned(span.data, 256), "MoE: workspace is not aligned");
        writes_.at(write_count_++) = operand::range(span.data, span.bytes, op_);
    }
}

void MoeOperands::finish() const {
    const auto writes = std::span(writes_).first(write_count_);
    operand::disjoint(writes, op_);
    for (const auto w : writes) {
        for (const auto r : std::span(reads_).first(read_count_)) { operand::disjoint(w, r, op_); }
    }
}
} // namespace ninfer::ops::detail
