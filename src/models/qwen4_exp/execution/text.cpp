#include "models/qwen4_exp/execution/text.h"
#include "models/qwen4_exp/execution/gdn.h"
#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/ple.h"
#include "core/transfer.h"
#include "core/nvtx.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"

namespace ninfer::models::qwen4_exp::execution {

TextRun::TextRun(const Parameters& p, TextExecution execution, BatchView batch, StateUpdate update,
                 CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), update_(update), cuda_(cuda),
      scope_(execution.workspace.scope()),
      roots_(
          workspace::roots(execution.workspace, p.model.config().text, batch.tokens * batch.batch)),
      expert_activity_(execution.experts ? execution.experts->begin(batch.tokens * batch.batch)
                                         : std::nullopt) {
    if (batch.tokens <= 0 || batch.batch < 1 || batch.batch > kMaximumConcurrency ||
        (std::holds_alternative<SequenceSlots>(update) && batch.batch != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1) ||
        (std::holds_alternative<VerifyRecords>(update) && batch.tokens < 2))
        throw std::invalid_argument("invalid Qwen4Exp sequence/step window");
}

void TextRun::prefix(std::optional<VisualEmbeddingView> visual) {
    stem(visual);
    for (std::uint32_t i = 0; i < parameters_.model.config().text.ple.layer; ++i) layer(i);
}

void TextRun::stem(std::optional<VisualEmbeddingView> visual) {
    const auto& p = parameters_.text;
    ops::embedding(batch_.ids.view({batch_.tokens * batch_.batch}), p.embedding, roots_.input,
                   cuda_);
    if (visual) ops::scatter(visual->features, visual->columns, roots_.input, cuda_);
    ops::hc_expand_norm(roots_.input, p.layers.front().attention_hc.norm,
                        parameters_.model.config().text.norm_epsilon, roots_.residual,
                        roots_.normalized, cuda_);
}

void TextRun::layer(std::uint32_t i) {
    nvtx::ScopedRange layer(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
    const auto& p = parameters_.text.layers[i];
    if (execution_.experts) execution_.experts->before_layer(p.moe.bank);
    mixer(i);
    {
        auto scratch = execution_.workspace.scope();
        if (execution_.experts)
            execution_.experts->execute(p.moe.bank, roots_.input, roots_.output,
                                        execution_.workspace, {});
        else
            resident_moe(i);
    }
    combine(i);
}

void TextRun::mixer(std::uint32_t i) {
    const auto& config = parameters_.model.config().text;
    const auto& p      = parameters_.text.layers[i];
    const auto& state  = config.layers[i];
    auto& arena        = execution_.workspace;
    {
        auto scratch = arena.scope();
        ops::hc_mix(roots_.normalized, p.attention_hc.mix, arena, roots_.input, roots_.injection,
                    cuda_);
    }
    if (const auto* g = std::get_if<GdnParameters>(&p.mixer)) {
        gdn(roots_.input, config.gdn, *g, batch_, execution_.states.gdn(), state.state_index,
            update_, config.norm_epsilon, roots_.output, arena, cuda_);
    } else {
        ops::QsaRawRecordWriteView records;
        if (!std::holds_alternative<SequenceSlots>(update_)) {
            records = {
                execution_.qsa_raw_records.slice(3, state.state_index, 1).slice(2, 0, batch_.batch),
                execution_.qsa_coordinate_records.slice(3, state.state_index, 1)
                    .slice(2, 0, batch_.batch),
                batch_.record_offsets};
        }
        const auto after = std::holds_alternative<VerifyRecords>(update_)
                               ? std::nullopt
                               : std::optional{execution_.states.main_tail(
                                     state.state_index, batch_.destination_slots)};
        qsa(roots_.input, config.qsa, std::get<QsaParameters>(p.mixer), batch_,
            execution_.states.main_tail(state.state_index, batch_.source_slots), after,
            execution_.cache.layer(state.state_index), records, roots_.output, arena, cuda_);
    }
    ops::hc_combine_norm(roots_.residual, roots_.output, roots_.injection, p.moe_hc.norm,
                         config.norm_epsilon, roots_.residual, roots_.normalized, cuda_);
    {
        auto scratch = arena.scope();
        ops::hc_mix(roots_.normalized, p.moe_hc.mix, arena, roots_.input, roots_.injection, cuda_);
    }
}

void TextRun::resident_moe(std::uint32_t i) {
    const auto& p = parameters_.text.layers[i].moe;
    moe::execute(roots_.input, p, p.resident, ops::MoeEpilogue::Write, roots_.output,
                 execution_.workspace, cuda_);
}

void TextRun::gpu_moe(const moe::GpuCall& call) {
    auto scratch = execution_.workspace.scope();
    call(roots_.input, roots_.output, execution_.workspace, {});
}

void TextRun::combine(std::uint32_t i) {
    const auto& config = parameters_.model.config().text;
    if (i + 1 == config.ple.layer) {
        ops::hc_combine(roots_.residual, roots_.output, roots_.injection, roots_.residual, cuda_);
    } else {
        const auto& norm = i + 1 < config.layers.size()
                               ? parameters_.text.layers[i + 1].attention_hc.norm
                               : parameters_.text.final_hc.norm;
        ops::hc_combine_norm(roots_.residual, roots_.output, roots_.injection, norm,
                             config.norm_epsilon, roots_.residual, roots_.normalized, cuda_);
    }
}

void TextRun::consume_ple(const Tensor& embedding) {
    const auto& p    = parameters_.text;
    const auto start = parameters_.model.config().text.ple.layer;
    ops::PleConvRecordWindowView records;
    if (!std::holds_alternative<SequenceSlots>(update_))
        records = {execution_.ple_records.slice(2, 0, batch_.batch), batch_.record_offsets};
    const auto after = std::holds_alternative<VerifyRecords>(update_)
                           ? std::nullopt
                           : std::optional{execution_.states.ple(batch_.destination_slots)};
    ple(embedding, parameters_.model.config().text, p.ple, batch_,
        execution_.states.ple(batch_.source_slots), after, records, p.layers[start].attention_hc,
        roots_.residual, roots_.normalized, execution_.workspace, cuda_);
}

void TextRun::suffix(TextOutputs output) {
    const auto start = parameters_.model.config().text.ple.layer;
    for (std::uint32_t i = start; i < parameters_.text.layers.size(); ++i) layer(i);
    finish(output);
}

void TextRun::finish(TextOutputs output) {
    const auto& p = parameters_.text;
    if (output.conditioning.data)
        copy_bytes_async({output.conditioning.data, output.conditioning.bytes(), cuda_.device()},
                         {roots_.residual.data, roots_.residual.bytes(), cuda_.device()}, cuda_);
    auto sample = output.sample.view(
        {dimension(parameters_.model.config().text.hidden), batch_.tokens * batch_.batch});
    ops::hc_mix(roots_.normalized, p.final_hc.mix, execution_.workspace, sample, cuda_);
}

void project_head(const Tensor& sample, const LinearParameters& head, Tensor& logits,
                  MemoryArena& arena, CudaExecutionView cuda) {
    auto scope = arena.scope();
    ops::linear(sample, head.weight, logits, head.policy, arena, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
