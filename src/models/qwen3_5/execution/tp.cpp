#include "models/qwen3_5/execution/tp.h"

#include "core/device_scope.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/tp2/argmax.h"
#include "ops/launcher/concat_rows.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::uint32_t divide(std::uint32_t extent, int width, const char* label) {
    const auto divisor = static_cast<std::uint32_t>(width);
    if (extent == 0 || extent % divisor != 0) {
        throw std::invalid_argument(std::string("tensor-parallel Text config: ") + label +
                                    " is not divisible by the tensor-parallel width");
    }
    return extent / divisor;
}

} // namespace

TextConfig shard_text_config(const TextConfig& config, int width, bool hc_split_streams) {
    if (width != 1 && width != kTensorParallelWidth) {
        throw std::invalid_argument("tensor-parallel Text config: width must be 1 or 2");
    }
    TextConfig out = config;
    if (width == 1) { return out; }
    if (config.qwen4()) {
        // Routed experts are partitioned by expert ID and the shared expert is held whole by one
        // rank per block (load/sharding.h), so the MoE geometry is unchanged on each rank; a dense
        // FFN splits its intermediate rows like the Qwen3.5 dense block's. The QSA indexer and the
        // PLE are replicated.
        if (const auto* dense = std::get_if<DenseConfig>(&out.ffn)) {
            out.ffn = DenseConfig{divide(dense->intermediate_size, width, "intermediate_size")};
        }
        if (hc_split_streams) {
            auto& hc    = out.hyper_connection.value();
            hc.hc_count = divide(hc.hc_count, width, "hc_count");
        }
    } else {
        const auto& dense = std::get<DenseConfig>(out.ffn);
        out.ffn = DenseConfig{divide(dense.intermediate_size, width, "intermediate_size")};
    }
    out.vocab_size = divide(out.vocab_size, width, "vocab_size");
    if (out.attention) {
        out.attention->num_attention_heads =
            divide(out.attention->num_attention_heads, width, "num_attention_heads");
        out.attention->num_key_value_heads =
            divide(out.attention->num_key_value_heads, width, "num_key_value_heads");
    }
    if (out.gdn) {
        out.gdn->linear_num_key_heads =
            divide(out.gdn->linear_num_key_heads, width, "linear_num_key_heads");
        out.gdn->linear_num_value_heads =
            divide(out.gdn->linear_num_value_heads, width, "linear_num_value_heads");
    }
    return out;
}

std::size_t output_head_split_workspace_bytes(const LinearParameters& shard, std::int32_t first,
                                              std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(shard.weight.qtype, shard.weight.n, shard.weight.k,
                                                shard.policy, first, last);
}

void output_logits_split_rank0(const std::array<Tensor, 2>& hidden,
                               const std::array<const LinearParameters*, 2>& head,
                               const std::array<Tensor, 2>& partial, const Tensor& logits,
                               const Tensor& staging,
                               const std::array<WorkspaceArena*, 2>& workspace,
                               const ExecutionContext& execution, const ops::PeerEvents& events) {
    const std::int32_t columns = hidden[0].ne[1];
    const std::int32_t rows0   = partial[0].ne[0];
    const std::int32_t rows1   = partial[1].ne[0];
    for (std::size_t r = 0; r < 2; ++r) {
        if (partial[r].dtype != DType::BF16 || partial[r].ne[1] != columns ||
            !partial[r].is_contiguous()) {
            throw std::invalid_argument("tensor-parallel logits: partial logits do not match");
        }
    }
    if (logits.dtype != DType::BF16 || logits.ne[0] != rows0 + rows1 || logits.ne[1] != columns ||
        logits.ne[2] != 1 || logits.ne[3] != 1 || !logits.is_contiguous() ||
        staging.dtype != DType::BF16 || staging.ne[0] != rows1 || staging.ne[1] != columns ||
        !staging.is_contiguous()) {
        throw std::invalid_argument(
            "tensor-parallel logits: gathered logits or staging do not match the vocabulary");
    }
    if (!events.live()) { throw std::invalid_argument("tensor-parallel logits: dead events"); }
    project_column_parallel(hidden, head, partial, workspace, execution);

    const std::size_t bytes1   = static_cast<std::size_t>(rows1) * sizeof(std::uint16_t);
    const auto height          = static_cast<std::size_t>(columns);
    const DeviceContext& rank0 = *execution.dev[0];
    const DeviceContext& rank1 = *execution.dev[1];
    int previous               = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(rank1.device));
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(1), rank1.stream));
    CUDA_CHECK(cudaSetDevice(rank0.device));
    CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(1), 0));
    // The one cross-device transfer: a plain D2D cudaMemcpyAsync over UVA, the form the
    // collectives use because it is capturable with and without peer access. Eagerly with a
    // HostExchange attached it goes through pinned host memory instead (T68: rank 1's half may
    // sit in the elastic VMM pool lent to a prompt chunk, which rank 0 cannot address without P2P).
    if (ops::HostExchange* exchange = ops::eager_exchange(execution, events)) {
        exchange->copy(1, partial[1].data, staging.data, bytes1 * height, execution, events);
        CUDA_CHECK(cudaSetDevice(rank0.device));
    } else {
        CUDA_CHECK(::ninfer::memcpy_device_async(staging.data, partial[1].data, bytes1 * height,
                                                 rank0.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));
    // A kernel, not two pitched copies: the column count and the buffers differ between the CUDA
    // Graph profiles of one class, and a captured 2D memcpy node cannot take such a change in
    // place.
    Tensor gathered = logits;
    ops::detail::concat_rows_bf16_launch(partial[0], staging, gathered, rank0.stream);
    // Rank 1 may overwrite partial[1] only after rank 0's pull has read it.
    CUDA_CHECK(cudaSetDevice(rank1.device));
    CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.pull_done(0), 0));
    CUDA_CHECK(cudaSetDevice(previous));
}

void proposal_argmax_split(const std::array<Tensor, 2>& hidden,
                           const std::array<const LinearParameters*, 2>& head,
                           const std::array<Tensor, 2>& partial, const std::array<Tensor, 2>& local,
                           const std::array<Tensor, 2>& candidates,
                           const std::array<Tensor, 2>& staging, const Tensor& tokens,
                           const std::array<WorkspaceArena*, 2>& workspace,
                           const ExecutionContext& execution, const ops::PeerEvents& events) {
    const std::int32_t rows = partial[0].ne[0];
    if (partial[1].ne[0] != rows || head[0] == nullptr || head[1] == nullptr ||
        head[0]->weight.n != rows || head[1]->weight.n != rows) {
        throw std::invalid_argument(
            "tensor-parallel proposal argmax: the ranks' head blocks differ");
    }
    if (!events.live()) {
        throw std::invalid_argument("tensor-parallel proposal argmax: dead events");
    }
    project_column_parallel(hidden, head, partial, workspace, execution);
    {
        const ScopedCurrentDevice restore;
        for (int rank = 0; rank < kTensorParallelWidth; ++rank) {
            const auto r               = static_cast<std::size_t>(rank);
            const DeviceContext& owner = *execution.dev[r];
            ScopedCurrentDevice::select(owner.device);
            Tensor argmax = local[r];
            ops::argmax(partial[r], argmax, rows, owner.stream);
            Tensor packed = candidates[r];
            ops::argmax_split_pack(partial[r], argmax, rank, packed, owner.stream);
        }
    }
    // The ranks' candidates occupy disjoint digits, so the summing exchange is their exact union.
    ops::allreduce_sum(candidates, staging, execution, events);
    const ScopedCurrentDevice rank0(execution.dev[0]->device);
    Tensor selected = tokens;
    ops::argmax_split_select(candidates[0], rows, selected, execution.dev[0]->stream);
}

OrdinaryPeerFrame ordinary_peer_frame(const qwen3_5::OrdinaryDecodeState& frame) {
    return {frame.tokens,
            frame.cache_positions,
            frame.rope_positions,
            frame.text_kv_table_rows,
            frame.state_source_slots,
            frame.state_destination_slots};
}

} // namespace ninfer::models::qwen3_5::execution
