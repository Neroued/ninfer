#pragma once

#include "kpool_indexer_fixture.h"
#include "kpool_prepare_fixture.h"
#include "kpool_fp8_fixture.h"
#include "ninfer/ops/linear_split.h"

namespace ninfer::tool::kpool {

// Actual public Op composition on one native cache, one stream and one scratch arena.
// Persistent outputs/selection/tail never occupy scratch. Replays overwrite the same
// logical interval from an immutable before-tail, so no timed restoration is needed.
struct ChainFixture {
    Weights weights;
    IndexerFixture index;
    Buffer<std::uint16_t> x, rank, query, key_up, value_up, output, w_b;
    Weight query_weight;
    Buffer<int> selection_rows;
    Buffer<std::int64_t> before_offsets, after_offsets;
    StateImageLayout state_layout;
    ops::KpoolTailFields fields;
    DeviceBuffer state;
    std::vector<std::uint16_t> initial_state, latent;
    std::vector<std::uint8_t> latent_codes, latent_scales;
    ops::KpoolTailStateView before, after;
    ops::KpoolMlaCacheView cache;
    std::array<std::size_t, 4> capacities;
    float scale = 1.F / 16;

    ChainFixture(CudaExecutionView cuda, int T, int B, int E, int layers = 11, bool ragged = false,
                 const IndexerCase& shape = {}, bool repeated_inputs = true,
                 bool initialize_history          = true,
                 KvCacheVectorFormat index_format = KvCacheVectorFormat::BFloat16,
                 ops::KpoolMlaStorage storage     = ops::KpoolMlaStorage::BFloat16)
        : weights(cuda.device(), Projection::Full),
          index(cuda, T, B, E, layers, ragged, storage, shape, initialize_history, index_format),
          x(cuda.device(), DType::BF16, {4096, T, B}),
          rank(cuda.device(), DType::BF16, {1536, T, B}),
          query(cuda.device(), DType::BF16, {256, 64, T, B}),
          key_up(cuda.device(), DType::BF16, {512, 256, 64}),
          value_up(cuda.device(), DType::BF16, {512, 256, 64}),
          output(cuda.device(), DType::BF16, {256, 64, T, B}),
          w_b(cuda.device(), DType::BF16, {1536, 20480}),
          selection_rows(cuda.device(), DType::I32, {T, B}),
          before_offsets(cuda.device(), DType::I64, {B}),
          after_offsets(cuda.device(), DType::I64, {B}),
          latent(initialize_history ? std::size_t(index.physical_pages) * 64 * 512 : 0) {
        StateImageBuilder builder;
        fields            = ops::append_kpool_tail_fields(builder, 128, layers);
        state_layout      = builder.finish();
        const auto stride = std::int64_t(state_layout.image_bytes + 256);
        initial_state.resize((256 + 2 * B * stride) / 2, 0xa5a5);
        state = DeviceBuffer(cuda.device(), initial_state.size() * 2);
        StateImageStorage images(state.span(), state_layout);
        before = {images.field(fields.raw_keys, before_offsets.view, index.layer),
                  images.field(fields.raw_gates, before_offsets.view, index.layer)};
        after  = {images.field(fields.raw_keys, after_offsets.view, index.layer),
                  images.field(fields.raw_gates, after_offsets.view, index.layer)};
        cache  = index.layout.layer(PagedRecordView{index.packed.storage(), index.tables.view, 64},
                                    index.layer);
        for (int b = 0; b < B; ++b) {
            before_offsets.at(b) = index.valid.at(b) ? 256 + b * stride : -1;
            after_offsets.at(b)  = index.valid.at(b) ? 256 + (B + b) * stride : -1;
            for (int t = 0; t < T; ++t) {
                selection_rows.at(t, b) = t < index.valid.at(b) ? b * T + t : -1;
                for (int d = 0; d < 4096; ++d)
                    // A bounded set of independent inputs permits complete long-T validation.
                    x.at(d, t, b) = sample(
                        d + 4096 * (repeated_inputs ? t % 7 : t) + 9397 * b + 317, 1.F / 256);
            }
            if (!index.valid.at(b)) continue;
            for (int kind = 0; kind < 2; ++kind)
                for (int r = 0; r < 3; ++r)
                    for (int d = 0; d < 128; ++d)
                        initial_state[state_index(false, kind, b, r, d)] =
                            r < index.positions.at(0, b) % 4
                                ? sample(d + 128 * r + 777 * b + 99 * kind, 1.F / 512)
                                : 0;
        }
        for (std::size_t i = 0; i < latent.size(); ++i) latent[i] = sample(i + 3351, 1.F / 512);
        for (std::size_t i = 0; i < key_up.host.size(); ++i) {
            key_up.host[i]   = sample(i + 1157, 1.F / 8192);
            value_up.host[i] = sample(i + 7759, 1.F / 8192);
        }
        for (std::size_t i = 0; i < w_b.host.size(); ++i)
            w_b.host[i] = sample(i + 997, 1.F / 16384);
        w_b.upload();
        query_weight.device  = cuda.device();
        query_weight.payload = query_weight.qdata = w_b.view.data;
        query_weight.payload_bytes                = w_b.view.bytes();
        query_weight.qtype                        = QType::BF16;
        query_weight.layout                       = QuantLayout::Contiguous;
        query_weight.n = query_weight.shape[0] = query_weight.padded_shape[0] = 20480;
        query_weight.k = query_weight.shape[1] = query_weight.padded_shape[1] = 1536;
        query_weight.ndim                                                     = 2;
        x.upload();
        key_up.upload();
        value_up.upload();
        selection_rows.upload();
        before_offsets.upload();
        after_offsets.upload();
        state.copy_from_host(initial_state.data(), state.size());
        if (initialize_history) {
            if (storage == ops::KpoolMlaStorage::BFloat16) {
                Tensor dense({latent.data(), latent.size() * 2, Device::cpu()}, DType::BF16,
                             {512, 64, 1, index.physical_pages});
                index.packed.upload_plane(index.packed.plane_index(cache.latent.data.plane), dense);
            } else {
                encode_fp8_latent(latent, latent_codes, latent_scales);
                Tensor codes({latent_codes.data(), latent_codes.size(), Device::cpu()},
                             DType::FP8_E4M3FN, {512, 64, 1, index.physical_pages});
                Tensor scales({latent_scales.data(), latent_scales.size(), Device::cpu()},
                              DType::U8, {4, 64, 1, index.physical_pages});
                index.packed.upload_plane(index.packed.plane_index(cache.latent.data.plane), codes);
                index.packed.upload_plane(index.packed.plane_index(cache.latent.scales.plane),
                                          scales);
            }
        }
        const std::array<std::int32_t, 2> split_rows{16384, 4096};
        capacities = {
            ops::kpool_mla_prepare_update_workspace_capacity_bytes(
                weights.profile(), storage, index_format, B, 1, T, cuda.multiprocessor_count()),
            ops::linear_split_workspace_capacity_bytes(
                ops::ProjectionWeights{ops::SingleProjectionWeight{query_weight}}, split_rows, B,
                T * B),
            index.workspace_bytes,
            ops::kpool_sparse_mla_workspace_capacity_bytes({64, 256, 256, 512}, storage,
                                                           index.envelope, B, 1, T,
                                                           cuda.multiprocessor_count())};
        index.workspace.reset();
        index.workspace_bytes = *std::max_element(capacities.begin(), capacities.end());
        index.scratch         = DeviceBuffer(cuda.device(), index.workspace_bytes + 512);
        index.scratch.fill(0xa5);
        index.workspace =
            std::make_unique<MemoryArena>(index.scratch.span().subspan(256, index.workspace_bytes));
    }

    std::size_t state_index(bool dst, int kind, int b, int r, int d) const {
        const auto& f     = kind ? before.raw_gates.field : before.raw_keys.field;
        const auto offset = dst ? after_offsets.at(b) : before_offsets.at(b);
        return (offset + f.tensor.region.offset + r * f.byte_strides[1]) / 2 + d;
    }

    std::size_t physical_token(int b, int token) const {
        const auto offset = index.tables.at(token / 64, index.selectors.at(b));
        return (offset - index.packed.physical_offset(0)) / index.packed.record_stride() * 64 +
               token % 64;
    }

    void prepare(CudaExecutionView cuda) {
        ops::kpool_mla_prepare_update(
            x.view, weights.params, {index.positions.view, index.valid.view, index.selectors.view},
            before, cache, after, {rank.view, index.coeff.view}, *index.workspace, cuda);
    }

    void queries(CudaExecutionView cuda) {
        const int columns = index.T * index.B;
        const std::array<Tensor, 2> outputs{query.view.reshape({16384, columns}),
                                            index.query.view.reshape({4096, columns})};
        ops::linear_split(rank.view.reshape({1536, columns}),
                          ops::ProjectionWeights{ops::SingleProjectionWeight{query_weight}},
                          outputs, *index.workspace, cuda);
    }

    void attend(CudaExecutionView cuda) {
        ops::kpool_sparse_mla(query.view, {key_up.view, value_up.view}, cache.latent,
                              {index.positions.view, index.valid.view, index.selectors.view},
                              index.selection, selection_rows.view, scale, index.envelope,
                              *index.workspace, output.view, cuda);
    }

    void launch(CudaExecutionView cuda) {
        prepare(cuda);
        queries(cuda);
        index.launch(cuda);
        attend(cuda);
    }
};
} // namespace ninfer::tool::kpool
