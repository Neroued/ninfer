#include "launch.h"

#include "ops/common/host/operands.h"

#include <cmath>
#include <limits>

namespace ninfer::ops {
namespace {

namespace operand = common::operand;
using operand::Range;

constexpr char kContext[]      = "markov_rnn_context";
constexpr char kStep[]         = "markov_rnn_step";
constexpr char kLogits[]       = "markov_logits_add_softcap";
constexpr char kLogitsArgmax[] = "markov_logits_add_softcap_argmax";

void validate_rank(int rank, const char* op) {
    if (rank <= 0 || rank % 16 || rank > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument(std::string(op) + ": invalid rank");
}

std::size_t context_partial_bytes(int h, int r, int steps, int batch, bool capacity = false) {
    validate_rank(r, kContext);
    if (h <= 0 || h % 8 || steps <= 0 || batch <= 0)
        throw std::invalid_argument(std::string(kContext) + ": invalid context extents");
    const std::uint64_t columns = std::uint64_t(steps) * batch;
    const auto base             = operand::bytes(3ULL * r * batch, sizeof(float), kContext);
    if (!capacity && columns >= markov::kContextLargeColumns) return base;
    const auto k_tiles                = 1 + (h - 1) / markov::kContextKTile;
    constexpr std::uint64_t max_small = markov::kContextLargeColumns - 1;
    const auto small_columns =
        std::min(columns, max_small) + std::min<std::uint64_t>(batch, max_small);
    const auto partial = operand::bytes(small_columns, 3ULL * r * sizeof(float), kContext);
    return std::max(base, operand::bytes(k_tiles, partial, kContext));
}

Range tensor_range(const Tensor& t, DType dtype, std::initializer_list<int> dims, const char* op,
                   const char* name, Device device, std::size_t alignment = 16) {
    operand::tensor(t, dtype, dims, alignment, op, name, device);
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

auto logits_inputs(const Tensor& latent, const Weight& weight, const Tensor& bias,
                   const Tensor& base, int domain, float cap, const Tensor& live, const char* op,
                   Device device) {
    const int rank = latent.ne[0], batch = latent.ne[1], vocab = weight.n;
    validate_rank(rank, op);
    if (domain < 1 || domain > vocab || !(cap > 0) || !std::isfinite(cap))
        throw std::invalid_argument(std::string(op) + ": invalid token domain or softcap");
    return std::array{tensor_range(latent, DType::BF16, {rank, batch}, op, "latent", device),
                      bf16_weight(weight, vocab, rank, op, "folded_weight", device),
                      tensor_range(bias, DType::BF16, {vocab}, op, "folded_bias", device),
                      mask_range(live, batch, op, device),
                      tensor_range(base, DType::BF16, {vocab, batch}, op, "base_logits", device)};
}

std::size_t logit_partial_bytes(int domain, int batch) {
    if (domain <= 0 || batch <= 0)
        throw std::invalid_argument(std::string(kLogitsArgmax) + ": invalid extents");
    const auto tiles = 1 + (domain - 1) / markov::kLogitTile;
    return operand::bytes(std::uint64_t(tiles) * batch, sizeof(markov::LogitTileResult),
                          kLogitsArgmax);
}

} // namespace

std::size_t markov_rnn_context_workspace_capacity_bytes(int hidden, int rank, int max_steps,
                                                        int max_batch) {
    const auto bytes = context_partial_bytes(hidden, rank, max_steps, max_batch, true);
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
    const auto device = execution.device();
    const auto inputs = logits_inputs(latent, folded_weight, folded_bias, base_logits, token_domain,
                                      softcap, live_rows, kLogits, device);
    const auto output = tensor_range(logits_out, DType::BF16, {folded_weight.n, latent.ne[1]},
                                     kLogits, "logits_out", device);
    disjoint_unless_exact(output, inputs.back(), kLogits);
    for (std::size_t i = 0; i + 1 < inputs.size(); ++i)
        operand::disjoint(output, inputs[i], kLogits);
    markov::launch_logits(latent, folded_weight, folded_bias, base_logits, token_domain, softcap,
                          live_rows, logits_out, execution);
}

std::size_t markov_logits_add_softcap_argmax_workspace_capacity_bytes(int max_domain,
                                                                      int max_batch) {
    const auto bytes       = logit_partial_bytes(max_domain, max_batch);
    constexpr auto padding = alignof(markov::LogitTileResult) - 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - padding)
        throw std::overflow_error(std::string(kLogitsArgmax) + ": workspace capacity overflows");
    return bytes + padding;
}

void markov_logits_add_softcap_argmax(const Tensor& latent, const Weight& folded_weight,
                                      const Tensor& folded_bias, const Tensor& base_logits,
                                      int token_domain, float softcap, const Tensor& live_rows,
                                      Tensor& ids, Tensor& probabilities, MemoryArena& workspace,
                                      CudaExecutionView execution) {
    const auto device = execution.device();
    const auto inputs = logits_inputs(latent, folded_weight, folded_bias, base_logits, token_domain,
                                      softcap, live_rows, kLogitsArgmax, device);
    const int batch   = latent.ne[1];
    const auto tokens = tensor_range(ids, DType::I32, {batch}, kLogitsArgmax, "ids", device, 4);
    const auto probs  = tensor_range(probabilities, DType::FP32, {batch}, kLogitsArgmax,
                                     "probabilities", device, 4);
    operand::disjoint(tokens, probs, kLogitsArgmax);
    operand::device(workspace.device(), device, kLogitsArgmax, "workspace");
    auto scope          = workspace.scope();
    const auto scratch  = workspace.alloc_bytes(logit_partial_bytes(token_domain, batch),
                                                alignof(markov::LogitTileResult));
    const auto writable = operand::range(scratch.data, scratch.bytes, kLogitsArgmax);
    operand::disjoint(tokens, writable, kLogitsArgmax);
    operand::disjoint(probs, writable, kLogitsArgmax);
    for (const auto input : inputs) {
        operand::disjoint(tokens, input, kLogitsArgmax);
        operand::disjoint(probs, input, kLogitsArgmax);
        operand::disjoint(writable, input, kLogitsArgmax);
    }
    markov::launch_logits_argmax(latent, folded_weight, folded_bias, base_logits, token_domain,
                                 softcap, live_rows, ids, probabilities,
                                 static_cast<markov::LogitTileResult*>(scratch.data), execution);
}

} // namespace ninfer::ops
