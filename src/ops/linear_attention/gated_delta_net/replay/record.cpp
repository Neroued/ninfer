#include "ninfer/ops/gated_delta_net.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <array>
#include <span>

namespace ninfer::ops {
namespace {
using detail::gated_delta_net::kStateDim;
using detail::gated_delta_net::kMaximumReplayRows;

void validate_replay_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, ConstStateFieldView initial_state,
                            const Tensor& valid_columns, const Tensor& key_record,
                            const Tensor& value_record, const Tensor& gate_record,
                            const Tensor& out, Device device) {
    constexpr const char* kOp      = "gated_delta_net_replay_record";
    const std::int32_t qk_heads    = q.ne[1];
    const std::int32_t value_heads = v.ne[1];
    const std::int32_t width       = q.ne[2];
    const std::int32_t rows        = q.ne[3];
    const bool registered_heads    = (qk_heads == 16 && (value_heads == 48 || value_heads == 32)) ||
                                  (qk_heads == 8 && value_heads == 24);
    if (!registered_heads || width < 2 || width > 16 || rows <= 0 || rows > kMaximumReplayRows) {
        throw std::invalid_argument(std::string(kOp) + ": unsupported geometry");
    }


    common::operand::tensor(q, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "q",
                            device);
    common::operand::tensor(k, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "k",
                            device);
    common::operand::tensor(v, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "v",
                            device);
    common::operand::tensor(g, DType::FP32, {value_heads, width, rows}, 4, kOp, "g", device);
    common::operand::tensor(beta, DType::FP32, {value_heads, width, rows}, 4, kOp, "beta", device);
    if (initial_state.field.layers != 1)
        throw std::invalid_argument("gated_delta_net_replay_record: select one state layer");
    common::operand::state_field(initial_state, DType::FP32, {kStateDim, kStateDim, value_heads},
                                 16, kOp, "initial state", device);
    common::operand::shape(initial_state.records.byte_offsets, {rows, 1, 1, 1}, kOp,
                           "initial state byte offsets");
    if (valid_columns.data != nullptr) {
        common::operand::tensor(valid_columns, DType::I32, {rows}, 4, kOp, "valid columns", device);
    }
    common::operand::tensor(key_record, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp,
                            "key record", device);
    common::operand::tensor(value_record, DType::BF16, {kStateDim, value_heads, width, rows}, 2,
                            kOp, "value record", device);
    common::operand::tensor(gate_record, DType::FP32, {2, value_heads, width, rows}, 8, kOp,
                            "gate record", device);
    common::operand::tensor(out, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "out",
                            device);
    detail::gated_delta_net::require_scale(scale, kOp);

    const std::array<const Tensor*, 11> tensors{&q,
                                                &k,
                                                &v,
                                                &g,
                                                &beta,
                                                &valid_columns,
                                                &initial_state.records.byte_offsets,
                                                &key_record,
                                                &value_record,
                                                &gate_record,
                                                &out};
    std::array<common::operand::Blocks, 11> ranges{};
    std::size_t count = 0;
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr) ranges[count++] = common::operand::blocks(*tensor, kOp);
    }
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (common::operand::overlaps(ranges[i], ranges[j]))
                throw std::invalid_argument("gated_delta_net_replay_record: tensors overlap");
}

} // namespace

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   ConstStateFieldView initial_state, const Tensor& valid_columns,
                                   GdnRecurrenceRecords records, Tensor& out,
                                   CudaExecutionView execution) {
    const auto stream = execution.stream();
    validate_replay_record(q, k, v, g, beta, scale, initial_state, valid_columns, records.key,
                           records.value, records.gate, out, execution.device());
    detail::gated_delta_net::launch_recurrent_record(q, k, v, g, beta, scale, initial_state,
                                                     valid_columns, records.key, records.value,
                                                     records.gate, out, stream);
}

} // namespace ninfer::ops
