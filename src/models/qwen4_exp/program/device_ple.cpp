#include "models/qwen4_exp/program/internal.h"

namespace ninfer::models::qwen4_exp::detail {

Tensor RankProgramImpl::device_embedding(std::int32_t tokens, std::int32_t batch) {
    return packed_tensor(*resources.plan.persistent.buffers.ple_embedding,
                         {dimension(parameters.profile->config.text.ple.embedding), tokens, batch});
}

void RankProgramImpl::prepare_device_ple(const execution::BatchView& batch) {
    const auto& table    = std::get<DevicePleTable>(parameters.model.ple_table());
    const auto& controls = parameters.model.definition().ple_parameters().hash;
    ops::NgramHashParameters hash;
    for (std::size_t i = 0; i < hash.multipliers.size(); ++i)
        hash.multipliers[i] = static_cast<std::uint64_t>(controls.multipliers[i]);
    for (std::size_t i = 0; i < hash.offsets.size(); ++i) {
        hash.vocab_sizes[i] = controls.vocab_sizes[i];
        hash.offsets[i]     = controls.offsets[i];
    }
    const auto& io      = resources.plan.persistent.buffers.batch;
    const auto tail     = packed_tensor(io.raw_tail, {2, batch.batch});
    const auto slots    = device_tensor(io.row_indices).slice(0, 0, batch.batch);
    auto embedding      = device_embedding(batch.tokens, batch.batch);
    const auto producer = resources.ple_transfer();
    ple_inputs.record(cuda.stream());
    ple_inputs.wait(producer.stream());
    ops::ngram_embedding_lookup(batch.ids, batch.valid_columns, {tail, slots}, hash,
                                parameters.profile->config.text.ple.eos, table.shards, embedding,
                                producer);
    if (parallel) parallel->merge_embedding(embedding, producer);
    ple_ready.record(producer.stream());
}

} // namespace ninfer::models::qwen4_exp::detail
