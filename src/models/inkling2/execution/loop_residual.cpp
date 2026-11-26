#include "models/inkling2/execution/loop_residual.h"

#include "core/transfer.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/short_conv.h"

namespace ninfer::models::inkling2::execution {
namespace {

constexpr float pair_eps = 1e-6F;

void copy_tensor_async(const Tensor& dst, const Tensor& src, CudaExecutionView cuda) {
    copy_bytes_async({dst.data, dst.bytes(), cuda.device()}, {src.data, src.bytes(), cuda.device()},
                     cuda);
}

int attention_streams(const LogicalLayerConfig& logical) { return logical.is_swa ? 1 : 3; }

} // namespace

LoopResidualRun::LoopResidualRun(const Parameters& p, workspace::MainRoots& roots,
                                 const BatchView& batch, const StateViews& states,
                                 const ops::RaggedSequenceView* ragged, const RecordWindow* records,
                                 Tensor aux, std::optional<std::uint32_t> final_aux_slot,
                                 MemoryArena& arena, CudaExecutionView cuda)
    : parameters_(p.text), config_(p.profile->config.text), profile_(p.profile->text),
      roots_(roots), batch_(batch), states_(states), ragged_(ragged), records_(records), aux_(aux),
      final_aux_slot_(final_aux_slot), arena_(arena), cuda_(cuda) {}

void LoopResidualRun::capture_aux(std::uint32_t slot, const Tensor& source) {
    const auto width = std::size_t(source.ne[0]) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpy2DAsync(aux_.slice(1, extent(slot), 1).data, aux_.nb[2], source.data,
                                 width, width, roots_.stack.ne[1], cudaMemcpyDeviceToDevice,
                                 cuda_.stream()));
}

bool LoopResidualRun::at_core_start(const LogicalLayerConfig& logical) const {
    return config_.loop && logical.physical_layer == config_.loop->core_start;
}

bool LoopResidualRun::starts_block(const LogicalLayerConfig& logical) const {
    return logical.physical_layer % config_.attn_res_block_size == 0;
}

Tensor LoopResidualRun::stack_slot(std::uint32_t slot) const {
    return roots_.stack.slice(2, extent(slot), 1).view({roots_.stack.ne[0], roots_.stack.ne[1]});
}

Tensor LoopResidualRun::view_tokens(const Tensor& t) const {
    if (ragged_) return t;
    return t.view({t.ne[0], batch_.tokens, batch_.rows});
}

void LoopResidualRun::restart(const PhysicalLayerParameters& layer) {
    // Save the previous pass's bank before restoring the prelude partial in place.
    copy_tensor_async(
        roots_.bank,
        roots_.stack.slice(2, dimension(prelude_blocks_ + 1U), dimension(profile_.bank_blocks)),
        cuda_);
    num_blocks_ = prelude_blocks_;
    copy_tensor_async(stack_slot(num_blocks_), roots_.prelude_partial, cuda_);
    if (!prelude_dests_ready_) {
        ops::block_attn_res(roots_.stack.slice(2, 0, dimension(num_blocks_)),
                            roots_.prelude_partial, Tensor{}, layer.attn_res_query, Tensor{},
                            profile_.attn_res, roots_.prelude_dests, cuda_);
        prelude_dests_ready_ = true;
    }
}

void LoopResidualRun::blend_attention(const LogicalLayerConfig& logical,
                                      const PhysicalLayerParameters& layer) {
    const int blocks      = dimension(profile_.bank_blocks);
    const auto bank_stack = blocks > 1 ? roots_.bank.slice(2, 0, blocks - 1) : Tensor{};
    const auto bank_partial =
        roots_.bank.slice(2, blocks - 1, 1).view({roots_.stack.ne[0], roots_.stack.ne[1]});
    ops::block_attn_res(bank_stack, bank_partial, Tensor{}, layer.loop->message_queries, Tensor{},
                        profile_.attn_res, roots_.messages, cuda_);
    const auto dest = at_core_start(logical)
                          ? roots_.prelude_dests
                          : roots_.attn_in_fp32.slice(2, 0, attention_streams(logical));
    ops::loop_message_blend_rmsnorm(dest, roots_.messages.slice(2, 0, 3), layer.loop->attn_kv,
                                    layer.attn_input_norm, pair_eps, config_.norm_epsilon, arena_,
                                    roots_.attn_in_bf16, cuda_);
}

void LoopResidualRun::initial_attention_norm(const LogicalLayerConfig& logical,
                                             const PhysicalLayerParameters& layer) {
    const auto input = stack_slot(0);
    for (int j = 0; j < attention_streams(logical); ++j) {
        auto out          = roots_.attn_in_bf16.slice(2, j, 1).view({input.ne[0], input.ne[1]});
        const auto weight = layer.attn_input_norm.slice(1, j, 1).view({input.ne[0]});
        ops::rmsnorm(input, weight, config_.norm_epsilon, /*unit_offset=*/false, out, cuda_);
    }
}

void LoopResidualRun::begin_visit(const LogicalLayerConfig& logical, std::size_t visit_index) {
    const auto& layer = parameters_.layers.at(logical.physical_layer);
    const bool looped = logical.pass_index > 0;
    if (at_core_start(logical)) {
        if (looped) {
            restart(layer);
        } else {
            prelude_blocks_ = num_blocks_;
            copy_tensor_async(roots_.prelude_partial, stack_slot(num_blocks_), cuda_);
        }
    }
    if (aux_.data && logical.capture_aux_slot)
        capture_aux(*logical.capture_aux_slot, stack_slot(num_blocks_));
    if (looped)
        blend_attention(logical, layer);
    else if (visit_index == 0)
        initial_attention_norm(logical, layer);
}

void LoopResidualRun::after_attention(const LogicalLayerConfig& logical, const Tensor& attn_proj,
                                      Tensor& mlp_in_bf16) {
    const auto& layer = parameters_.layers.at(logical.physical_layer);
    const bool looped = logical.pass_index > 0;
    const bool first  = starts_block(logical);
    const auto stack  = roots_.stack.slice(2, 0, dimension(num_blocks_ + first));
    const auto addend = first ? Tensor{} : view_tokens(stack_slot(num_blocks_));
    const auto norm   = looped ? Tensor{} : layer.mlp_input_norm;
    const auto src    = states_.attn_sconv(logical.logical_index, batch_.source_offsets);
    const auto dst    = states_.attn_sconv(logical.logical_index, batch_.destination_offsets);
    const auto input  = view_tokens(attn_proj);
    auto output       = view_tokens(roots_.attn_out);
    auto scope        = arena_.scope();
    auto mixed =
        looped ? arena_.alloc(DType::FP32, {mlp_in_bf16.ne[0], mlp_in_bf16.ne[1], 1}) : mlp_in_bf16;

    if (records_) {
        auto snapshot = records_->layers.at(logical.logical_index).attn_conv;
        ops::short_conv_block_attn_res_snapshot(
            input, layer.attn_short_conv, src, snapshot, batch_.valid_columns, records_->base_slots,
            Tensor{}, Tensor{}, stack, addend, layer.mlp_attn_res_query, norm, profile_.attn_res,
            output, mixed, cuda_);
    } else if (ragged_) {
        ops::short_conv_block_attn_res(input, layer.attn_short_conv, *ragged_, src, dst, Tensor{},
                                       Tensor{}, stack, addend, layer.mlp_attn_res_query, norm,
                                       profile_.attn_res, output, mixed, cuda_);
    } else {
        ops::short_conv_block_attn_res(input, layer.attn_short_conv, src, dst, batch_.valid_columns,
                                       Tensor{}, Tensor{}, stack, addend, layer.mlp_attn_res_query,
                                       norm, profile_.attn_res, output, mixed, cuda_);
    }
    if (looped)
        ops::loop_message_blend_rmsnorm(mixed, roots_.messages.slice(2, 3, 1), layer.loop->mlp,
                                        layer.mlp_input_norm, pair_eps, config_.norm_epsilon,
                                        arena_, mlp_in_bf16, cuda_);
}

void LoopResidualRun::after_mlp(const LogicalLayerConfig& logical, std::size_t visit_index,
                                const Tensor& moe_out, Tensor& final_hidden) {
    const auto& layer   = parameters_.layers.at(logical.physical_layer);
    const bool first    = starts_block(logical);
    const auto residual = first ? Tensor{} : view_tokens(stack_slot(num_blocks_));
    if (first) ++num_blocks_;
    auto conv_out     = view_tokens(stack_slot(num_blocks_));
    const auto src    = states_.mlp_sconv(logical.logical_index, batch_.source_offsets);
    const auto dst    = states_.mlp_sconv(logical.logical_index, batch_.destination_offsets);
    const auto input  = view_tokens(moe_out);
    const auto addend = view_tokens(roots_.attn_out);

    const auto& visits = config_.logical_layers;
    const auto next    = visit_index + 1 < visits.size() ? &visits[visit_index + 1] : nullptr;
    if (next && at_core_start(*next) && next->pass_index > 0) {
        // The restart computes its mix from the preserved prelude.
        if (records_) {
            auto snapshot = records_->layers.at(logical.logical_index).mlp_conv;
            ops::short_conv_residual_snapshot(input, layer.mlp_short_conv, src, snapshot,
                                              batch_.valid_columns, records_->base_slots, residual,
                                              addend, conv_out, cuda_);
        } else if (ragged_)
            ops::short_conv_residual(input, layer.mlp_short_conv, *ragged_, src, dst, residual,
                                     addend, conv_out, cuda_);
        else
            ops::short_conv_residual(input, layer.mlp_short_conv, src, dst, batch_.valid_columns,
                                     residual, addend, conv_out, cuda_);
        return;
    }

    Tensor query, norm, mixed;
    if (next) {
        const auto& next_layer = parameters_.layers.at(next->physical_layer);
        const bool looped      = next->pass_index > 0;
        query                  = next_layer.attn_res_query;
        norm                   = looped ? Tensor{} : next_layer.attn_input_norm;
        mixed                  = (looped ? roots_.attn_in_fp32 : roots_.attn_in_bf16)
                    .slice(2, 0, attention_streams(*next));
    } else {
        query = parameters_.output_query;
        norm  = parameters_.final_norm;
        mixed = final_hidden.view({roots_.stack.ne[0], roots_.stack.ne[1], 1});
    }

    const auto stack = roots_.stack.slice(2, 0, dimension(num_blocks_));
    if (records_) {
        auto snapshot = records_->layers.at(logical.logical_index).mlp_conv;
        ops::short_conv_block_attn_res_snapshot(input, layer.mlp_short_conv, src, snapshot,
                                                batch_.valid_columns, records_->base_slots,
                                                residual, addend, stack, Tensor{}, query, norm,
                                                profile_.attn_res, conv_out, mixed, cuda_);
    } else if (ragged_) {
        ops::short_conv_block_attn_res(input, layer.mlp_short_conv, *ragged_, src, dst, residual,
                                       addend, stack, Tensor{}, query, norm, profile_.attn_res,
                                       conv_out, mixed, cuda_);
    } else {
        ops::short_conv_block_attn_res(input, layer.mlp_short_conv, src, dst, batch_.valid_columns,
                                       residual, addend, stack, Tensor{}, query, norm,
                                       profile_.attn_res, conv_out, mixed, cuda_);
    }
    if (!next && aux_.data && final_aux_slot_) capture_aux(*final_aux_slot_, final_hidden);
}

} // namespace ninfer::models::inkling2::execution
