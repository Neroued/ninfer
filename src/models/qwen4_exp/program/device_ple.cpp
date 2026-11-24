#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"

namespace ninfer::models::qwen4_exp::detail {
namespace {
ops::NgramHashParameters device_hash(const ple::HashControls& controls) {
    ops::NgramHashParameters hash;
    for (std::size_t i = 0; i < hash.multipliers.size(); ++i)
        hash.multipliers[i] = static_cast<std::uint64_t>(controls.multipliers[i]);
    for (std::size_t i = 0; i < hash.offsets.size(); ++i) {
        hash.vocab_sizes[i] = controls.vocab_sizes[i];
        hash.offsets[i]     = controls.offsets[i];
    }
    return hash;
}

StateFieldView token_tails(const Tensor& tail, const Tensor& offsets) {
    return {{{tail.data, tail.bytes(), tail.device}, offsets},
            {{{0, 2 * sizeof(TokenId), alignof(TokenId)}, DType::I32, {2, 1, 1, 1}},
             1,
             2 * sizeof(TokenId)}};
}
} // namespace

Tensor RankProgramImpl::device_embedding(std::int32_t tokens, std::int32_t batch) {
    return packed_tensor(*resources.buffers().ple_embedding,
                         {dimension(parameters.profile->config.text.ple.embedding), tokens, batch});
}

void RankProgramImpl::prepare_device_ple(const execution::BatchView& batch, bool read_drafts,
                                         const StreamSectionRecorder& readback) {
    const auto& table     = std::get<DevicePleTable>(parameters.model.ple_table());
    const auto hash       = device_hash(parameters.model.definition().ple_parameters().hash);
    const auto& io        = resources.buffers().batch;
    const auto tail       = packed_tensor(io.raw_tail, {2, batch.batch});
    const auto offsets    = device_tensor(io.raw_tail_offsets).slice(0, 0, batch.batch);
    const auto tail_field = token_tails(tail, offsets);
    auto embedding        = device_embedding(batch.tokens, batch.batch);
    const auto producer   = resources.ple_transfer();
    ple_inputs.record(cuda.stream());
    ple_inputs.wait(producer.stream());
    if (read_drafts) {
        const auto copy = [&] {
            copy_bytes_async(draft_output.span().subspan(0, batch.ids.bytes()),
                             {batch.ids.data, batch.ids.bytes(), cuda.device()}, producer);
        };
        if (readback) {
            readback(producer.stream(), copy);
            draft_copied.record_external(producer.stream());
        } else {
            copy();
            draft_copied.record(producer.stream());
        }
    }
    ops::ngram_embedding_lookup(batch.ids, batch.valid_columns, {tail_field}, hash,
                                parameters.profile->config.text.ple.eos, table.shards, embedding,
                                producer);
    if (parallel) parallel->merge_unique(embedding, producer);
    ple_ready.record(producer.stream());
}

void RankProgramImpl::prepare_packed_device_ple(const PackedInputLayout& io) {
    const auto& f     = *compute;
    const auto& table = std::get<DevicePleTable>(parameters.model.ple_table());
    const auto hash   = device_hash(parameters.model.definition().ple_parameters().hash);
    const auto tail = token_tails(device_tensor(*io.raw_tail), device_tensor(*io.raw_tail_offsets));
    auto embedding =
        device_embedding(f.tokens, 1)
            .view({dimension(parameters.profile->config.text.ple.embedding), dimension(f.tokens)});
    const auto producer = resources.ple_transfer();
    ple_inputs.record(cuda.stream());
    ple_inputs.wait(producer.stream());
    ops::ngram_embedding_lookup(f.batch.ids, f.packed->ragged, {tail}, hash,
                                parameters.profile->config.text.ple.eos, table.shards, embedding,
                                producer);
    if (parallel) parallel->merge_unique(embedding, producer);
    ple_ready.record(producer.stream());
}

} // namespace ninfer::models::qwen4_exp::detail
