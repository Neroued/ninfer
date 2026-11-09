#include "ninfer/ops/mhc.h"
#include "ops/mhc/implementation.h"
#include "ops/common/host/operands.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail::mhc {
namespace {
namespace operand         = common::operand;
constexpr const char* kOp = "mhc";

void require(bool value, const char* reason) {
    if (!value) throw std::invalid_argument(std::string(kOp) + ": " + reason);
}

// Only actual live ranges are checked: unrelated arena storage may share backing.
struct Validation {
    Device device;
    std::array<operand::Range, 12> inputs{}, outputs{};
    int input_count = 0, output_count = 0;

    operand::Range tensor(const Tensor& t, DType type, std::initializer_list<int> dims,
                          const char* name) {
        operand::tensor(t, type, dims, 16, kOp, name, device);
        return operand::range(t, kOp);
    }

    void input(const Tensor& t, DType type, std::initializer_list<int> dims, const char* name) {
        inputs[input_count++] = tensor(t, type, dims, name);
    }

    void parameter(const Tensor& t, std::initializer_list<int> dims, const char* name) {
        if (t.dtype != DType::BF16 && t.dtype != DType::FP32)
            throw std::invalid_argument(std::string(kOp) + ": " + name +
                                        " requires BF16 or FP32 storage");
        input(t, t.dtype, dims, name);
    }

    void output(const Tensor& t, DType type, std::initializer_list<int> dims, const char* name,
                bool residual_alias = false) {
        output_range(tensor(t, type, dims, name), residual_alias);
    }

    void output_range(operand::Range range, bool residual_alias = false) {
        for (int i = residual_alias ? 1 : 0; i < input_count; ++i)
            operand::disjoint(range, inputs[i], kOp);
        for (int i = 0; i < output_count; ++i) operand::disjoint(range, outputs[i], kOp);
        outputs[output_count++] = range;
    }

    void scratch(MemorySpan span) {
        operand::device(span.device, device, kOp, "workspace");
        const auto range = operand::range(span.data, span.bytes, kOp);
        for (int i = 0; i < input_count; ++i) operand::disjoint(range, inputs[i], kOp);
        for (int i = 0; i < output_count; ++i) operand::disjoint(range, outputs[i], kOp);
    }

    void weights(const MhcPreWeights& w) {
        const auto& f = w.projection;
        operand::device(f.device, device, kOp, "projection");
        require(f.qtype == QType::BF16 && f.layout == QuantLayout::Contiguous && f.ndim == 2 &&
                    f.n == kRows && f.k == kWidth && f.shape[0] == kRows && f.shape[1] == kWidth &&
                    f.shape[2] == 1 && f.shape[3] == 1 && f.padded_shape[0] == kRows &&
                    f.padded_shape[1] == kWidth && f.padded_shape[2] == 1 &&
                    f.padded_shape[3] == 1 && !f.qhigh && !f.scales && f.high_plane_bytes == 0 &&
                    f.group_size == 0 && f.group == 0 &&
                    f.payload_bytes >= std::size_t(kRows) * kWidth * 2 &&
                    operand::aligned(f.qdata, 16),
                "projection requires compact BF16 [24,16384]");
        inputs[input_count++] = operand::range(f.qdata, std::size_t(kRows) * kWidth * 2, kOp);
        parameter(w.base, {kRows}, "base");
        parameter(w.scale, {3}, "scale");
        parameter(w.gamma, {kHidden}, "gamma");
    }
};

void parameters(const MhcPreParameters& p) {
    for (float value : {p.coefficient_epsilon, p.branch_epsilon, p.pre_epsilon, p.sinkhorn_epsilon,
                        p.post_multiplier})
        require(value > 0 && std::isnormal(value),
                "parameters must be positive normal FP32 values");
    require(p.sinkhorn_iterations > 0, "Sinkhorn iterations must be positive");
}

void execute(Form form, const Tensor& x, const Tensor* block, const MhcCoefficients* c,
             const MhcPreWeights* weights, const MhcPreParameters* p, Tensor* updated,
             const MhcPreOutputs* outputs, MemoryArena* arena, CudaExecutionView execution,
             const Tensor* final_gamma = nullptr, float final_epsilon = 0, Tensor* hidden = nullptr,
             const Tensor* mean_tap = nullptr) {
    const bool expand    = form == Form::ExpandPre;
    const auto& geometry = expand ? *updated : x;
    validate_geometry({geometry.ne[0], geometry.ne[1]});
    const int m = geometry.ne[2];
    require(m > 0, "columns must be positive");
    const auto plan = select(form, m, execution.multiprocessor_count(), mean_tap && mean_tap->data);
    Validation v{execution.device()};
    if (expand)
        v.input(x, DType::BF16, {kHidden, m}, "embedding");
    else
        v.input(x, DType::BF16, {kHidden, kStreams, m}, "residual");
    Operands o;
    o.residual = x.data;
    o.columns  = m;
    if (block) {
        v.input(*block, DType::BF16, {kHidden, m}, "block output");
        v.input(c->post, DType::FP32, {kStreams, m}, "post coefficients");
        v.input(c->mix, DType::FP32, {kStreams, kStreams, m}, "mix coefficients");
        o.block = block->data;
        o.post  = static_cast<const float*>(c->post.data);
        o.mix   = static_cast<const float*>(c->mix.data);
    }
    if (weights) {
        parameters(*p);
        v.weights(*weights);
        o.projection = weights->projection.qdata;
        o.base       = weights->base.data;
        o.scale      = weights->scale.data;
        o.gamma      = weights->gamma.data;
        o.base_type  = weights->base.dtype;
        o.scale_type = weights->scale.dtype;
        o.gamma_type = weights->gamma.dtype;
        o.parameters = *p;
    }
    if (updated) {
        // Shapes/strides are checked independently below, so pointer equality implies exact view.
        v.output(*updated, DType::BF16, {kHidden, kStreams, m}, "updated residual",
                 !expand && updated->data == x.data);
        o.updated = updated->data;
    }
    if (outputs) {
        v.output(outputs->block_input, DType::BF16, {kHidden, m}, "block input");
        v.output(outputs->coefficients.post, DType::FP32, {kStreams, m}, "next post coefficients");
        v.output(outputs->coefficients.mix, DType::FP32, {kStreams, kStreams, m},
                 "next mix coefficients");
        o.output    = outputs->block_input.data;
        o.next_post = static_cast<float*>(outputs->coefficients.post.data);
        o.next_mix  = static_cast<float*>(outputs->coefficients.mix.data);
    }
    if (final_gamma) {
        require(final_epsilon > 0 && std::isfinite(final_epsilon),
                "final epsilon must be positive and finite");
        v.parameter(*final_gamma, {kHidden}, "final gamma");
        v.output(*hidden, DType::BF16, {kHidden, m}, "hidden");
        o.gamma                     = final_gamma->data;
        o.gamma_type                = final_gamma->dtype;
        o.output                    = hidden->data;
        o.parameters.branch_epsilon = final_epsilon;
    }
    if (mean_tap && mean_tap->data) {
        const auto& tap = *mean_tap;
        require(form == Form::PostPre && std::int64_t(tap.ne[1]) * tap.ne[2] == m && tap.nb[0] == 2,
                "mean tap requires feature-contiguous [4096,C,B], C*B=M");
        operand::strided_tensor(tap, DType::BF16, {kHidden, tap.ne[1], tap.ne[2]}, 16, kOp,
                                "mean tap", execution.device());
        v.output_range(operand::strided_range(tap, kOp));
        o.mean_tap          = tap.data;
        o.tap_columns       = tap.ne[1];
        o.tap_column_stride = tap.nb[1] / 2;
        o.tap_batch_gap     = tap.nb[2] / 2 - std::int64_t(tap.ne[1]) * o.tap_column_stride;
    }
    if (arena) {
        operand::device(arena->device(), execution.device(), kOp, "workspace");
        auto scope       = arena->scope();
        const auto bytes = scratch_bytes(plan, m);
        const auto scratch =
            bytes ? arena->alloc_bytes(bytes) : MemorySpan{nullptr, 0, execution.device()};
        v.scratch(scratch);
        launch(form, o, plan, scratch.data, execution);
    } else {
        launch(form, o, plan, nullptr, execution);
    }
}
} // namespace

void validate_geometry(const MhcGeometry& g) {
    require(g.hidden == kHidden && g.streams == kStreams, "unsupported hidden/stream geometry");
}

std::size_t scratch_bytes(Plan p, int m) {
    // The last projection record ends at its square sum; no trailing stride padding.
    const auto floats = p.method == Method::Projection || p.method == Method::VectorProjection
                            ? (std::uint64_t(p.partials) * m - 1) * 32 + 25
                        : p.method == Method::SplitMean ? std::uint64_t(kHidden + p.partials) * m
                                                        : 0;
    return common::operand::bytes(floats, 4, "mhc");
}

std::size_t capacity(Form form, const MhcGeometry& g, int sm, int first, int last) {
    validate_geometry(g);
    require(sm > 0 && first > 0 && last >= first, "invalid SM count or column interval");
    std::size_t maximum = 0;
    for (std::int64_t cursor = first; cursor <= last;) {
        const auto plan   = select(form, static_cast<int>(cursor), sm, false);
        const auto tapped = select(form, static_cast<int>(cursor), sm, form == Form::PostPre);
        const int end     = std::min({last, plan.interval_end, tapped.interval_end});
        // Within a selected region the partial reservation grows monotonically with M.
        maximum = std::max(maximum, scratch_bytes(plan, end));
        maximum = std::max(maximum, scratch_bytes(tapped, end));
        cursor  = std::int64_t(end) + 1;
    }
    return maximum;
}
} // namespace ninfer::ops::detail::mhc

namespace ninfer::ops {
namespace mhc = detail::mhc;

void mhc_pre(const Tensor& x, const MhcPreWeights& w, const MhcPreParameters& p, MhcPreOutputs out,
             MemoryArena& workspace, CudaExecutionView e) {
    mhc::execute(mhc::Form::Pre, x, nullptr, nullptr, &w, &p, nullptr, &out, &workspace, e);
}

void mhc_post(const Tensor& x, const Tensor& y, const MhcCoefficients& c, Tensor& out,
              CudaExecutionView e) {
    mhc::execute(mhc::Form::Post, x, &y, &c, nullptr, nullptr, &out, nullptr, nullptr, e);
}

void mhc_expand_pre(const Tensor& x, const MhcPreWeights& w, const MhcPreParameters& p,
                    Tensor& residual, MhcPreOutputs out, MemoryArena& workspace,
                    CudaExecutionView e) {
    mhc::execute(mhc::Form::ExpandPre, x, nullptr, nullptr, &w, &p, &residual, &out, &workspace, e);
}

void mhc_post_pre(const Tensor& x, const Tensor& y, const MhcCoefficients& c,
                  const MhcPreWeights& w, const MhcPreParameters& p, Tensor& residual,
                  MhcPreOutputs out, MemoryArena& workspace, CudaExecutionView e,
                  const Tensor& mean_tap) {
    mhc::execute(mhc::Form::PostPre, x, &y, &c, &w, &p, &residual, &out, &workspace, e, nullptr, 0,
                 nullptr, &mean_tap);
}

void mhc_post_mean_norm(const Tensor& x, const Tensor& y, const MhcCoefficients& c,
                        const Tensor& gamma, float epsilon, Tensor& hidden, MemoryArena& workspace,
                        CudaExecutionView e) {
    mhc::execute(mhc::Form::PostMeanNorm, x, &y, &c, nullptr, nullptr, nullptr, nullptr, &workspace,
                 e, &gamma, epsilon, &hidden);
}

std::size_t mhc_pre_workspace_capacity_bytes(const MhcGeometry& g, int sm, int first, int last) {
    return mhc::capacity(mhc::Form::Pre, g, sm, first, last);
}

std::size_t mhc_expand_pre_workspace_capacity_bytes(const MhcGeometry& g, int sm, int first,
                                                    int last) {
    return mhc::capacity(mhc::Form::ExpandPre, g, sm, first, last);
}

std::size_t mhc_post_pre_workspace_capacity_bytes(const MhcGeometry& g, int sm, int first,
                                                  int last) {
    return mhc::capacity(mhc::Form::PostPre, g, sm, first, last);
}

std::size_t mhc_post_mean_norm_workspace_capacity_bytes(const MhcGeometry& g, int sm, int first,
                                                        int last) {
    return mhc::capacity(mhc::Form::PostMeanNorm, g, sm, first, last);
}
} // namespace ninfer::ops
