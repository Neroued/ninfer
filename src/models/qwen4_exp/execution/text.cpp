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
      partition_(std::holds_alternative<SequenceState>(update)
                     ? sequence_parallel_policy(p.profile->local.size)
                           .partition(batch.tokens, p.profile->local.rank)
                     : TokenPartition(batch.tokens * batch.batch)),
      scope_(execution.workspace.scope()),
      roots_(workspace::roots(execution.workspace, p.profile->config.text,
                              batch.tokens * batch.batch, partition_.padded)),
      expert_activity_(execution.experts ? execution.experts->begin(batch.tokens * batch.batch,
                                                                    batch.active_tokens)
                                         : std::nullopt) {
    if (batch.tokens <= 0 || batch.batch < 1 ||
        (std::holds_alternative<SequenceState>(update) && batch.batch != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1) ||
        (std::holds_alternative<VerifyRecords>(update) && batch.tokens < 2))
        throw std::invalid_argument("invalid Qwen4Exp sequence/step window");
    if (partition_.split() && !execution_.parallel)
        throw std::invalid_argument("Qwen4Exp sequence partition requires its parallel group");
    partition_.clear_padding(roots_.exchange_input, 1, cuda_);
    partition_.clear_padding(roots_.exchange_output, 1, cuda_);
}

void TextRun::mix(const HcParameters& p) {
    auto input = partitioned_ ? partition_.local(roots_.exchange_input, 1) : roots_.input;
    ops::hc_mix(roots_.normalized, p.mix, execution_.workspace, input, roots_.injection, cuda_);
    if (partitioned_) {
        const auto local = partition_.exchange(roots_.exchange_input, 1);
        execution_.parallel->gather(local, roots_.exchange_input, cuda_);
    }
}

Tensor TextRun::reduce_output() {
    if (partitioned_) {
        auto local = partition_.exchange(roots_.exchange_output, 1);
        execution_.parallel->sum_scatter(roots_.exchange_output, local, cuda_);
        return partition_.local(roots_.exchange_output, 1);
    }
    if (execution_.parallel) execution_.parallel->sum(roots_.output, cuda_);
    return roots_.output;
}

void TextRun::stem(std::optional<models::vision::EmbeddingView> visual) {
    const auto& p = parameters_.text;
    ops::embedding(batch_.ids.view({batch_.tokens * batch_.batch}), p.embedding, roots_.input,
                   cuda_);
    if (visual)
        for (const auto& slice : visual->slices)
            ops::scatter(slice.features, slice.columns, roots_.input, cuda_);
    ops::hc_expand_norm(roots_.input, p.layers.front().attention_hc.norm,
                        parameters_.profile->config.text.norm_epsilon, roots_.residual,
                        roots_.normalized, cuda_);
}

bool TextRun::advance_layer(std::uint32_t i) {
    const nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
    const auto& p = parameters_.text.layers[i];
    if (waiting_moe_) {
        if (!execution_.experts->poll()) return false;
        waiting_moe_ = false;
    } else {
        if (execution_.experts && !execution_.experts->try_before_layer(p.moe.bank)) return false;
        mixer(i);
        if (execution_.experts) {
            if (!execution_.experts->start(p.moe.bank, roots_.input, roots_.output,
                                           {batch_.valid_columns, batch_.tokens},
                                           execution_.workspace, {})) {
                waiting_moe_ = true;
                return false;
            }
        } else
            resident_moe(i);
    }
    combine(i);
    return true;
}

void TextRun::mixer(std::uint32_t i) {
    local_mixer(i);
    finish_mixer(i);
}

void TextRun::local_mixer(std::uint32_t i) {
    const auto& config = parameters_.profile->config.text;
    const auto& p      = parameters_.text.layers[i];
    const auto& state  = config.layers[i];
    auto& arena        = execution_.workspace;
    mix(p.attention_hc);
    if (const auto* g = std::get_if<GdnParameters>(&p.mixer)) {
        gdn(roots_.input, parameters_.profile->local.gdn, *g, batch_, execution_.states.gdn(),
            state.state_index, update_, config.norm_epsilon, roots_.output, arena, cuda_);
    } else {
        ops::QsaRawRecordWriteView records;
        if (!std::holds_alternative<SequenceState>(update_)) {
            records = {
                execution_.qsa_raw_records.slice(3, state.state_index, 1).slice(2, 0, batch_.batch),
                execution_.qsa_coordinate_records.slice(3, state.state_index, 1)
                    .slice(2, 0, batch_.batch),
                batch_.record_offsets};
        }
        const auto after = std::holds_alternative<VerifyRecords>(update_)
                               ? std::nullopt
                               : std::optional{execution_.states.main_tail(
                                     state.state_index, batch_.destination_offsets)};
        qsa(roots_.input, parameters_.profile->local.qsa, std::get<QsaParameters>(p.mixer), batch_,
            execution_.states.main_tail(state.state_index, batch_.source_offsets), after,
            execution_.cache.layer(state.state_index), records, roots_.output, arena, cuda_);
    }
}

void TextRun::finish_mixer(std::uint32_t i) {
    const auto& config = parameters_.profile->config.text;
    const auto& p      = parameters_.text.layers[i];
    if (i == 0 && execution_.first_collective_ready)
        execution_.first_collective_ready->wait(cuda_.stream());
    const auto output = reduce_output();
    ops::hc_combine_norm(roots_.residual, output, roots_.injection, p.moe_hc.norm,
                         config.norm_epsilon, roots_.residual, roots_.normalized, cuda_);
    mix(p.moe_hc);
}

void TextRun::resident_moe(std::uint32_t i) {
    const auto& p = parameters_.text.layers[i].moe;
    moe::execute(roots_.input, p, p.resident, ops::MoeEpilogue::Write, roots_.output,
                 {batch_.valid_columns, batch_.tokens}, execution_.workspace, cuda_);
}

void TextRun::gpu_moe(const moe::GpuCall& call) {
    auto scratch = execution_.workspace.scope();
    call(roots_.input, roots_.output, {batch_.valid_columns, batch_.tokens}, execution_.workspace,
         {});
}

void TextRun::combine(std::uint32_t i) {
    const auto output  = reduce_output();
    const auto& config = parameters_.profile->config.text;
    if (i + 1 == config.ple.layer) {
        ops::hc_combine(roots_.residual, output, roots_.injection, roots_.residual, cuda_);
    } else {
        const auto& norm = i + 1 < config.layers.size()
                               ? parameters_.text.layers[i + 1].attention_hc.norm
                               : parameters_.text.final_hc.norm;
        ops::hc_combine_norm(roots_.residual, output, roots_.injection, norm, config.norm_epsilon,
                             roots_.residual, roots_.normalized, cuda_);
    }
}

void TextRun::consume_ple(const Tensor& embedding) {
    const auto& p    = parameters_.text;
    const auto start = parameters_.profile->config.text.ple.layer;
    ops::PleConvRecordWindowView records;
    if (!std::holds_alternative<SequenceState>(update_))
        records = {execution_.ple_records.slice(2, 0, batch_.batch), batch_.record_offsets};
    const auto after = std::holds_alternative<VerifyRecords>(update_)
                           ? std::nullopt
                           : std::optional{execution_.states.ple(batch_.destination_offsets)};
    ple(embedding, parameters_.profile->config.text, p.ple, batch_,
        execution_.states.ple(batch_.source_offsets), after, records, p.layers[start].attention_hc,
        roots_.residual, roots_.normalized, execution_.workspace, cuda_);
    if (partition_.split()) {
        roots_.residual   = partition_.local(roots_.residual, 2);
        roots_.normalized = partition_.local(roots_.normalized, 2);
        roots_.injection  = partition_.local(roots_.injection, 1);
        partitioned_      = true;
    }
}

void TextRun::finish(TextOutputs output, Tensor logits) {
    const auto& p = parameters_.text;
    if (output.conditioning.data)
        copy_bytes_async({output.conditioning.data, output.conditioning.bytes(), cuda_.device()},
                         {roots_.residual.data, roots_.residual.bytes(), cuda_.device()}, cuda_);
    auto sample =
        output.sample.view({dimension(parameters_.profile->config.text.hidden), partition_.padded});
    auto local = partitioned_ ? partition_.local(sample, 1) : sample;
    ops::hc_mix(roots_.normalized, p.final_hc.mix, execution_.workspace, local, cuda_);
    if (partitioned_) {
        partition_.clear_padding(sample, 1, cuda_);
        execution_.parallel->gather(partition_.exchange(sample, 1), sample, cuda_);
    }
    sample = sample.slice(1, 0, partition_.columns);
    if (logits.data) {
        logits = logits.view({logits.ne[0], batch_.tokens * batch_.batch});
        project_head(sample, p.head, logits, execution_.workspace, cuda_);
    }
}

void project_head(const Tensor& sample, const LinearParameters& head, Tensor& logits,
                  MemoryArena& arena, CudaExecutionView cuda) {
    auto scope = arena.scope();
    ops::linear(sample, head.weight, logits, head.policy, arena, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
