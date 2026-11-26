#include "launch.h"

#include "ops/common/host/operands.h"

#include <cmath>
#include <limits>

namespace ninfer::ops {
namespace {

namespace operand = common::operand;
using operand::Range;

constexpr char kContext[] = "markov_rnn_context";
constexpr char kStep[]    = "markov_rnn_step";
constexpr char kLogits[]  = "markov_logits_add_softcap";

void validate_rank(int rank, const char* op) {
    if (rank <= 0 || rank % 16 || rank > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument(std::string(op) + ": invalid rank");
}

std::size_t context_partial_bytes(int h, int r, int steps, int batch) {
    validate_rank(r, kContext);
    if (h <= 0 || h % 8 || steps <= 0 || batch <= 0)
        throw std::invalid_argument(std::string(kContext) + ": invalid context extents");
    const auto k_tiles = 1 + (h - 1) / markov::kContextKTile;
    auto bytes         = operand::bytes(3ULL * r * batch, sizeof(float), kContext);
    bytes              = operand::bytes(std::uint64_t(steps) + 1, bytes, kContext);
    return operand::bytes(k_tiles, bytes, kContext);
}

Range tensor_range(const Tensor& t, DType dtype, std::initializer_list<int> dims, const char* op,
                   const char* name, Device device) {
    operand::tensor(t, dtype, dims, 16, op, name, device);
    return operand::range(t, op);
}

Range bf16_weight(const Weight& w, int n, int k, const char* op, const char* name, Device device) {
    operand::device(w.device, device, op, name);
    const auto bytes = operand::bytes(std::uint64_t(n) * k, 2, op);
    const std::array shape{n, k, 1, 1};
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous || w.ndim != 2 || w.n != n ||
        w.k != k || !std::equal(shape.begin(), shape.end(), w.shape) ||
        !std::equal(shape.begin(), shape.end(), w.padded_shape) || !operand::aligned(w.qdata, 16) ||
        !w.payload || w.group != 0 || w.group_size != 0 || w.scales || w.qhigh ||
        w.high_plane_bytes != 0)
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    const auto data    = operand::range(w.qdata, bytes, op);
    const auto backing = operand::range(w.payload, w.payload_bytes, op);
    if (data.begin < backing.begin || data.end > backing.end)
        throw std::invalid_argument(std::string(op) + ": truncated " + name);
    return data;
}

Range mask_range(const Tensor& mask, int batch, const char* op, Device device) {
    if (!mask.data) return {};
    return tensor_range(mask, DType::I32, {batch}, op, "live_rows", device);
}

void disjoint_unless_exact(Range output, Range input, const char* op) {
    if (output.begin != input.begin || output.end != input.end)
        operand::disjoint(output, input, op);
}

} // namespace

std::size_t markov_rnn_context_workspace_capacity_bytes(int hidden, int rank, int max_steps,
                                                        int max_batch) {
    const auto bytes = context_partial_bytes(hidden, rank, max_steps, max_batch);
    if (bytes > std::numeric_limits<std::size_t>::max() - 15)
        throw std::overflow_error(std::string(kContext) + ": workspace capacity overflows");
    return bytes + 15; // Cover alignment at any arena cursor.
}

void markov_rnn_context(const Tensor& hidden, const Tensor& f_rnn, const Weight& w_hidden,
                        const Weight& w_frnn, const Tensor& bias, MemoryArena& workspace,
                        Tensor& context, CudaExecutionView execution) {
    const int h = hidden.ne[0], steps = hidden.ne[1], batch = hidden.ne[2];
    const int rank    = w_hidden.n / 3;
    const auto bytes  = context_partial_bytes(h, rank, steps, batch);
    const int rows    = 3 * rank;
    const auto device = execution.device();
    const std::array inputs{
        tensor_range(hidden, DType::BF16, {h, steps, batch}, kContext, "hidden", device),
        tensor_range(f_rnn, DType::BF16, {h, batch}, kContext, "f_rnn", device),
        bf16_weight(w_hidden, rows, h, kContext, "w_hidden", device),
        bf16_weight(w_frnn, rows, h, kContext, "w_frnn", device),
        tensor_range(bias, DType::FP32, {rows}, kContext, "bias", device)};
    const auto output =
        tensor_range(context, DType::FP32, {rows, batch, steps}, kContext, "context", device);
    operand::device(workspace.device(), device, kContext, "workspace");
    auto scope          = workspace.scope();
    const auto scratch  = workspace.alloc_bytes(bytes, 16);
    const auto writable = operand::range(scratch.data, scratch.bytes, kContext);
    operand::disjoint(output, writable, kContext);
    for (const auto input : inputs) {
        operand::disjoint(output, input, kContext);
        operand::disjoint(writable, input, kContext);
    }
    markov::launch_context(hidden, f_rnn, w_hidden, w_frnn, bias, static_cast<float*>(scratch.data),
                           context, execution);
}

void markov_rnn_step(const Tensor& state_in, const Tensor& prev_tokens, const Tensor& context_step,
                     const Tensor& w1_embed, const Weight& w_recurrent, const Tensor& live_rows,
                     Tensor& state_out, Tensor& latent, CudaExecutionView execution) {
    const int rank = state_in.ne[0], batch = state_in.ne[1];
    validate_rank(rank, kStep);
    const auto device = execution.device();
    const auto state =
        tensor_range(state_in, DType::BF16, {rank, batch}, kStep, "state_in", device);
    const std::array inputs{
        tensor_range(prev_tokens, DType::I32, {batch}, kStep, "prev_tokens", device),
        tensor_range(context_step, DType::FP32, {3 * rank, batch}, kStep, "context_step", device),
        tensor_range(w1_embed, DType::BF16, {rank, w1_embed.ne[1]}, kStep, "w1_embed", device),
        bf16_weight(w_recurrent, 3 * rank, 2 * rank, kStep, "w_recurrent", device),
        mask_range(live_rows, batch, kStep, device)};
    const auto output =
        tensor_range(state_out, DType::BF16, {rank, batch}, kStep, "state_out", device);
    const auto projected =
        tensor_range(latent, DType::BF16, {rank, batch}, kStep, "latent", device);
    disjoint_unless_exact(output, state, kStep);
    operand::disjoint(projected, state, kStep);
    operand::disjoint(output, projected, kStep);
    for (const auto input : inputs) {
        operand::disjoint(output, input, kStep);
        operand::disjoint(projected, input, kStep);
    }
    if (output.begin == state.begin && rank > kMarkovRnnStepMaxInPlaceRank)
        throw std::invalid_argument(std::string(kStep) +
                                    ": exact state aliasing requires rank <= " +
                                    std::to_string(kMarkovRnnStepMaxInPlaceRank));
    markov::launch_step(state_in, prev_tokens, context_step, w1_embed, w_recurrent, live_rows,
                        state_out, latent, execution);
}

void markov_logits_add_softcap(const Tensor& latent, const Weight& folded_weight,
                               const Tensor& folded_bias, const Tensor& base_logits,
                               int token_domain, float softcap, const Tensor& live_rows,
                               Tensor& logits_out, CudaExecutionView execution) {
    const int rank = latent.ne[0], batch = latent.ne[1], vocab = folded_weight.n;
    validate_rank(rank, kLogits);
    if (token_domain < 1 || token_domain > vocab || !(softcap > 0) || !std::isfinite(softcap))
        throw std::invalid_argument(std::string(kLogits) + ": invalid token domain or softcap");
    const auto device = execution.device();
    const std::array inputs{
        tensor_range(latent, DType::BF16, {rank, batch}, kLogits, "latent", device),
        bf16_weight(folded_weight, vocab, rank, kLogits, "folded_weight", device),
        tensor_range(folded_bias, DType::BF16, {vocab}, kLogits, "folded_bias", device),
        mask_range(live_rows, batch, kLogits, device)};
    const auto base =
        tensor_range(base_logits, DType::BF16, {vocab, batch}, kLogits, "base_logits", device);
    const auto output =
        tensor_range(logits_out, DType::BF16, {vocab, batch}, kLogits, "logits_out", device);
    disjoint_unless_exact(output, base, kLogits);
    for (const auto input : inputs) operand::disjoint(output, input, kLogits);
    markov::launch_logits(latent, folded_weight, folded_bias, base_logits, token_domain, softcap,
                          live_rows, logits_out, execution);
}

} // namespace ninfer::ops
