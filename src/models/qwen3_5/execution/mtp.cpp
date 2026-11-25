#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "core/nvtx.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/workspace.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"

#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_tensor(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
                    Device device, const char* name) {
    if (tensor.dtype != dtype || tensor.device != device || !tensor.data ||
        !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("MTP: invalid ") + name);
    }
    std::size_t axis = 0;
    for (const auto extent : shape) {
        if (tensor.ne[axis++] != extent) {
            throw std::invalid_argument(std::string("MTP: invalid ") + name + " shape");
        }
    }
    for (; axis < 4; ++axis) {
        if (tensor.ne[axis] != 1) {
            throw std::invalid_argument(std::string("MTP: invalid ") + name + " shape");
        }
    }
}

const MtpParameters& require_mtp(const Parameters& parameters) {
    if (!parameters.mtp) { throw std::logic_error("MTP parameters were not selected"); }
    return *parameters.mtp;
}

std::int32_t validate_inputs(MtpExecutionView state, const MtpInputs& inputs,
                             const ops::RaggedSequenceView* ragged = nullptr) {
    const auto& config = state.parameters.profile->config.text;
    const auto width   = inputs.ids.ne[0];
    const auto batch   = inputs.ids.ne[1];
    if (width <= 0 || batch <= 0) {
        throw std::invalid_argument("MTP: width and batch must be positive");
    }
    const auto columns = dimension(static_cast<std::uint64_t>(width) * batch);
    const auto hidden  = dimension(config.hidden_size);
    const auto device  = state.execution.device();
    require_tensor(inputs.ids, DType::I32, {width, batch}, device, "ids");
    require_tensor(inputs.conditioning, DType::BF16, {hidden, width, batch}, device,
                   "conditioning");
    require_tensor(inputs.cache_positions, DType::I32, {width, batch}, device, "cache positions");
    if (ragged && (batch != 1 || width != ragged->total_tokens() || !ragged->num_seqs()))
        throw std::invalid_argument("MTP: invalid packed history geometry");
    const auto rows = ragged ? ragged->num_seqs() : batch;
    require_tensor(inputs.valid_columns, DType::I32, {rows}, device, "valid columns");
    require_tensor(inputs.kv_table_rows, DType::I32, {rows}, device, "KV table rows");
    const auto axes = inputs.rope_positions.ne[1];
    if (axes != 1 && axes != 3) {
        throw std::invalid_argument("MTP: RoPE positions require one or three axes");
    }
    require_tensor(inputs.rope_positions, DType::I32, {columns, axes}, device, "RoPE positions");
    if (inputs.input_embeddings) {
        require_tensor(*inputs.input_embeddings, DType::BF16, {hidden, width, batch}, device,
                       "input embeddings");
    }
    return columns;
}

workspace::MtpStemRoots mtp_stem(MtpExecutionView state, const MtpInputs& inputs,
                                 std::int32_t columns) {
    const auto& parameters = require_mtp(state.parameters);
    const auto& config     = state.parameters.profile->config.text;
    auto roots =
        workspace::mtp_stem(state.workspace, config, columns, inputs.input_embeddings == nullptr);
    Tensor embedding = roots.embedding;
    if (inputs.input_embeddings) {
        embedding = inputs.input_embeddings->view({dimension(config.hidden_size), columns});
    } else {
        ops::embedding(inputs.ids.view({columns}), state.parameters.text.token_embedding, embedding,
                       state.execution);
    }
    ops::rmsnorm(embedding, parameters.embedding_norm, config.rms_norm_eps, true,
                 roots.normalized_embedding, state.execution);
    ops::rmsnorm(inputs.conditioning.view({dimension(config.hidden_size), columns}),
                 parameters.hidden_norm, config.rms_norm_eps, true, roots.normalized_hidden,
                 state.execution);
    ops::concat_bf16_features(roots.normalized_embedding, roots.normalized_hidden,
                              roots.packed_input, state.execution);
    ops::linear(roots.packed_input, parameters.input_projection.weight, roots.residual,
                parameters.input_projection.policy, state.workspace, state.execution);
    ops::rmsnorm(roots.residual, parameters.input_norm, config.rms_norm_eps, true,
                 roots.attention_hidden, state.execution);
    return roots;
}

CausalAttentionReadout
mtp_attention_readout(MtpExecutionView state, const MtpInputs& inputs,
                      const workspace::MtpAttentionProjectionRoots& projection,
                      const workspace::MtpAttentionResultRoots& result) {
    const auto& attention = *state.parameters.profile->config.text.attention;
    const auto width = inputs.ids.ne[0], batch = inputs.ids.ne[1];
    const auto head_dim = dimension(attention.head_dim);
    const auto heads    = dimension(attention.num_attention_heads);
    const auto kv_heads = dimension(attention.num_key_value_heads);
    return {result.normalized_query.view({head_dim, heads, width, batch}),
            result.normalized_key.view({head_dim, kv_heads, width, batch}),
            projection.value.view({head_dim, kv_heads, width, batch}),
            inputs.cache_positions,
            inputs.valid_columns,
            inputs.kv_table_rows,
            result.attention.view({head_dim, heads, width, batch}),
            state.cache,
            {head_dim, heads, kv_heads},
            static_cast<float>(1.0 / std::sqrt(double(attention.head_dim))),
            {}};
}

} // namespace

std::size_t mtp_projection_workspace_bytes(const MtpProjectionProfile& parameters,
                                           std::int32_t first, std::int32_t last) {
    const auto& p = parameters.packed;
    if (!parameters.rows) {
        return ops::linear_split_workspace_capacity_bytes(ops::ProjectionProfile{p, {}},
                                                          parameters.output_rows, first, last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {p.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionProfile& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (parameters.rows) {
        return ops::linear_pair_workspace_capacity_bytes((*parameters.rows)[1],
                                                         (*parameters.rows)[3], first, last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, MemoryArena& workspace, CudaExecutionView execution) {
    const auto& p = parameters.packed;
    if (!parameters.rows) {
        std::array outputs{query, key, gate, value};
        ops::linear_split(hidden, p, outputs, workspace, execution);
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, execution);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::split_bf16_features(packed, q, k, g, v, execution);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       MemoryArena& workspace, CudaExecutionView execution) {
    if (parameters.rows) {
        ops::linear_pair(hidden, hidden, (*parameters.rows)[1].weight, (*parameters.rows)[3].weight,
                         key, value, execution);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, execution);
}

void mtp_forward(MtpExecutionView state, const MtpInputs& inputs, ops::ContextBucket bucket,
                 Tensor& hidden, const CausalAttentionSubmit& submit) {
    const auto& parameters = require_mtp(state.parameters);
    const auto& config     = state.parameters.profile->config.text;
    const auto columns     = validate_inputs(state, inputs);
    const auto width       = inputs.ids.ne[0];
    const auto batch       = inputs.ids.ne[1];
    require_tensor(hidden, DType::BF16, {dimension(config.hidden_size), width, batch},
                   state.execution.device(), "output hidden");
    nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(columns));
    auto scope            = state.workspace.scope();
    auto stem             = mtp_stem(state, inputs, columns);
    const auto& attention = *config.attention;
    auto projection       = workspace::mtp_attention_projection(state.workspace, config, columns);
    mtp_projection(stem.attention_hidden, parameters.projection, attention, projection.query,
                   projection.gate, projection.key, projection.value, state.workspace,
                   state.execution);

    const auto head_dim     = dimension(attention.head_dim);
    const auto heads        = dimension(attention.num_attention_heads);
    const auto kv_heads     = dimension(attention.num_key_value_heads);
    Tensor query            = projection.query.view({head_dim, heads, columns});
    Tensor key              = projection.key.view({head_dim, kv_heads, columns});
    auto result             = workspace::mtp_attention_results(state.workspace, config, columns);
    Tensor normalized_query = result.normalized_query.view({head_dim, heads, columns});
    Tensor normalized_key   = result.normalized_key.view({head_dim, kv_heads, columns});
    ops::rmsnorm(query, parameters.query_norm, config.rms_norm_eps, true, normalized_query,
                 state.execution);
    ops::rmsnorm(key, parameters.key_norm, config.rms_norm_eps, true, normalized_key,
                 state.execution);
    text_rope(inputs.rope_positions, *config.rope_parameters, normalized_query, normalized_key,
              state.execution);

    auto readout = mtp_attention_readout(state, inputs, projection, result);
    if (submit)
        readout.submit(state.workspace, state.execution, submit);
    else
        readout.execute(bucket, state.workspace, state.execution);
    Tensor gated_attention = result.attention;
    ops::sigmoid_mul(projection.gate, gated_attention, state.execution);

    auto post = workspace::mtp_post_attention(state.workspace, config, columns);
    ops::linear(gated_attention, parameters.output.weight, post.output, parameters.output.policy,
                state.workspace, state.execution);
    ops::residual_add(post.output, stem.residual, state.execution);
    ops::rmsnorm(stem.residual, parameters.post_attention_norm, config.rms_norm_eps, true,
                 post.post_mixer_hidden, state.execution);
    {
        auto mlp_scope = state.workspace.scope();
        if (const auto* dense = std::get_if<DenseParameters>(&parameters.ffn)) {
            dense_mlp(post.post_mixer_hidden, *dense, stem.residual, DenseMlpUse::Mtp,
                      state.workspace, state.execution);
        } else {
            const auto& moe = std::get<moe::Parameters>(parameters.ffn);
            moe::execute(post.post_mixer_hidden, moe, moe.resident, ops::MoeEpilogue::AddResidual,
                         stem.residual, {inputs.valid_columns, inputs.ids.ne[0]}, state.workspace,
                         state.execution);
        }
    }
    Tensor output = hidden.view({dimension(config.hidden_size), columns});
    ops::rmsnorm(stem.residual, parameters.final_norm, config.rms_norm_eps, true, output,
                 state.execution);
}

void bind_mtp_attention_readout(MtpExecutionView state, const MtpInputs& inputs,
                                const CausalAttentionSubmit& submit) {
    const auto& config = state.parameters.profile->config.text;
    const auto columns = validate_inputs(state, inputs);
    auto scope         = state.workspace.scope();
    (void)workspace::mtp_stem(state.workspace, config, columns, inputs.input_embeddings == nullptr);
    const auto projection = workspace::mtp_attention_projection(state.workspace, config, columns);
    const auto result     = workspace::mtp_attention_results(state.workspace, config, columns);
    auto readout          = mtp_attention_readout(state, inputs, projection, result);
    readout.submit(state.workspace, state.execution, submit);
}

namespace {

std::pair<Tensor, Tensor> project_history(MtpExecutionView state, const MtpInputs& inputs,
                                          int columns) {
    const auto& parameters = require_mtp(state.parameters);
    const auto& config     = state.parameters.profile->config.text;
    const auto& attention  = *config.attention;
    const auto head_dim    = dimension(attention.head_dim);
    const auto kv_heads    = dimension(attention.num_key_value_heads);
    auto stem              = mtp_stem(state, inputs, columns);
    Tensor key   = state.workspace.alloc(DType::BF16, {dimension(attention.key_width()), columns});
    Tensor value = state.workspace.alloc(DType::BF16, {dimension(attention.key_width()), columns});
    mtp_kv_projection(stem.attention_hidden, parameters.projection, attention, key, value,
                      state.workspace, state.execution);
    Tensor normalized_key = state.workspace.alloc(DType::BF16, {head_dim, kv_heads, columns});
    ops::rmsnorm(key.view({head_dim, kv_heads, columns}), parameters.key_norm, config.rms_norm_eps,
                 true, normalized_key, state.execution);
    text_rope(inputs.rope_positions, *config.rope_parameters, normalized_key, state.execution);
    return {normalized_key, value.view({head_dim, kv_heads, columns})};
}

} // namespace

void mtp_history(MtpExecutionView state, const MtpInputs& inputs,
                 ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    const auto columns = validate_inputs(state, inputs);
    const auto width = inputs.ids.ne[0], batch = inputs.ids.ne[1];
    nvtx::ScopedRange range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(columns));
    auto scope              = state.workspace.scope();
    const auto [key, value] = project_history(state, inputs, columns);
    ops::kv_cache_append_prefix(key.view({key.ne[0], key.ne[1], width, batch}),
                                value.view({value.ne[0], value.ne[1], width, batch}),
                                inputs.cache_positions, inputs.valid_columns, inputs.kv_table_rows,
                                envelope, state.cache, state.execution);
}

void mtp_history(MtpExecutionView state, const MtpInputs& inputs,
                 const ops::RaggedSequenceView& ragged) {
    const auto columns = validate_inputs(state, inputs, &ragged);
    nvtx::ScopedRange range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(columns));
    auto scope              = state.workspace.scope();
    const auto [key, value] = project_history(state, inputs, columns);
    ops::kv_cache_append(key, value, ragged, inputs.cache_positions.view({columns}),
                         inputs.kv_table_rows, state.cache, state.execution);
}

void mtp_propose(MtpExecutionView state, const Tensor& hidden, Tensor& full_logits, Tensor& ids,
                 Tensor& confidences) {
    const auto& parameters = require_mtp(state.parameters);
    const auto& config     = state.parameters.profile->config.text;
    const auto columns     = hidden.ne[1];
    if (columns <= 0) { throw std::invalid_argument("MTP: proposal columns must be positive"); }
    require_tensor(hidden, DType::BF16, {dimension(config.hidden_size), columns},
                   state.execution.device(), "proposal hidden");
    require_tensor(ids, DType::I32, {columns}, state.execution.device(), "proposal ids");
    nvtx::ScopedRange range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                            static_cast<std::uint64_t>(columns));
    auto scope = state.workspace.scope();
    switch (state.proposal_head) {
    case ProposalHead::Full:
        require_tensor(full_logits, DType::BF16, {dimension(config.vocab_size), columns},
                       state.execution.device(), "proposal logits");
        ops::linear(hidden, parameters.output_head.weight, full_logits,
                    parameters.output_head.policy, state.workspace, state.execution);
        ops::argmax_with_probability(full_logits, ids, confidences,
                                     dimension(state.parameters.profile->public_tokens),
                                     state.workspace, state.execution);
        return;
    case ProposalHead::Optimized: {
        if (!state.parameters.proposal) {
            throw std::logic_error("MTP: optimized proposal head was not selected");
        }
        const auto& proposal = *state.parameters.proposal;
        const auto rows      = dimension(proposal.rows);
        Tensor logits        = state.workspace.alloc(DType::BF16, {rows, columns});
        ops::linear(hidden, proposal.head.weight, logits, proposal.head.policy, state.workspace,
                    state.execution);
        ops::argmax_with_probability(
            logits, ids, confidences,
            proposal.token_ids ? rows : dimension(state.parameters.profile->public_tokens),
            state.workspace, state.execution);
        if (proposal.token_ids) {
            ops::remap_ids_inplace(ids, static_cast<const std::int32_t*>(proposal.token_ids->data),
                                   rows, state.execution);
        }
        return;
    }
    }
    throw std::invalid_argument("MTP: invalid proposal head policy");
}

} // namespace ninfer::models::qwen3_5::execution
