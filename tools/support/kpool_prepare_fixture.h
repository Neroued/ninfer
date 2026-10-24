#pragma once

#include "kpool_fixture.h"
#include "core/arena.h"
#include "ninfer/ops/kpool_mla_layout.h"
#include "paged_kv_fixture.h"

#include <memory>
#include <numeric>

namespace ninfer::tool::kpool {
using Projection = ops::KpoolMlaPrepareProjection;

inline const char* name(Projection p) {
    return p == Projection::Full ? "full" : p == Projection::History ? "history" : "query_history";
}

struct Weights {
    Projection mode;
    int q, heads, rows;
    Buffer<std::uint16_t> weight, cn, qn, kn, kb, ape;
    ops::KpoolMlaPrepareParameters params;

    Weights(Device device, Projection p)
        : mode(p), q(p == Projection::History ? 0 : 1536), heads(p == Projection::Full ? 32 : 0),
          rows(768 + q + heads), weight(device, DType::BF16, {4096, rows}),
          cn(device, DType::BF16, {512}), qn(device, DType::BF16, {1536}),
          kn(device, DType::BF16, {128}), kb(device, DType::BF16, {128}),
          ape(device, DType::BF16, {128, 4}) {
        for (std::size_t i = 0; i < weight.host.size(); ++i)
            weight.host[i] = sample(i + 171, 1.F / 16384);
        for (auto* v : {&cn, &qn, &kn})
            for (std::size_t i = 0; i < v->host.size(); ++i)
                v->host[i] = encode(1 + decode(sample(i + 42, 1.F / 2048)));
        for (std::size_t i = 0; i < kb.host.size(); ++i) kb.host[i] = sample(i + 111, 1.F / 2048);
        for (std::size_t i = 0; i < ape.host.size(); ++i) ape.host[i] = sample(i + 811, 1.F / 256);
        for (auto* v : {&weight, &cn, &qn, &kn, &kb, &ape}) v->upload();
        Weight w{};
        w.device  = device;
        w.payload = w.qdata = weight.view.data;
        w.payload_bytes     = weight.view.bytes();
        w.qtype             = QType::BF16;
        w.layout            = QuantLayout::Contiguous;
        w.n = w.shape[0] = w.padded_shape[0] = rows;
        w.k = w.shape[1] = w.padded_shape[1] = 4096;
        w.ndim                               = 2;
        params = {p, w, q ? qn.view : Tensor{}, cn.view, kn.view, kb.view, ape.view, 1e-5F, 1e-6F};
    }

    ops::KpoolMlaPrepareProfile profile() const { return {mode, 4096, 512, 128, q, heads, 4}; }
};

// Storage and input generation shared by tests/benchmarks. No implementation/oracle dependency.
// Every cache record contains the complete 11-layer page group, placed in reverse table order.
struct Fixture {
    Weights& w;
    int T, B, L = 11, layer = 5, W, P;
    bool update, inplace;
    Buffer<std::uint16_t> x, q, keys, gates;
    Buffer<float> coeff;
    Buffer<int> positions, valid, selectors, offsets;
    Buffer<std::int64_t> tables, before_offsets, after_offsets;
    std::vector<int> bases;
    ops::KpoolMlaCacheLayout cache_layout;
    PackedKVFixture packed;
    StateImageLayout state_layout;
    ops::KpoolTailFields fields;
    std::vector<std::uint16_t> state_host;
    DeviceBuffer state, workspace_buffer;
    std::unique_ptr<MemoryArena> workspace;
    ops::KpoolMlaCacheView cache;
    ops::KpoolTailStateView before, after;
    ops::KpoolRawRecordWriteView records;
    std::size_t workspace_bytes;

    Fixture(Weights& weights, CudaExecutionView cuda, int tokens, int batch, bool upd,
            bool ragged = false, bool strided = false, bool alias = false, int phase = 0)
        : w(weights), T(tokens), B(batch), W(T + 8), P((T + 128 + 63) / 64), update(upd),
          inplace(alias), x(cuda.device(), DType::BF16, {4096, T, B}),
          q(cuda.device(), DType::BF16, {1536, T, B}),
          keys(cuda.device(), DType::BF16, {128, W, B}, strided),
          gates(cuda.device(), DType::BF16, {128, W, B}, strided),
          coeff(cuda.device(), DType::FP32, {32, T, B}),
          positions(cuda.device(), DType::I32, {T, B}), valid(cuda.device(), DType::I32, {B}),
          selectors(cuda.device(), DType::I32, {B}), offsets(cuda.device(), DType::I32, {B}),
          tables(cuda.device(), DType::I64, {P, B}, strided),
          before_offsets(cuda.device(), DType::I64, {B}),
          after_offsets(cuda.device(), DType::I64, {B}), bases(B),
          cache_layout(
              ops::plan_kpool_mla_cache_layout({ops::KpoolMlaStorage::BFloat16, 512, 128, L})),
          packed(cuda.device(), cache_layout.page.geometry, P * B) {
        StateImageBuilder sb;
        (void)sb.add(DType::I32, {64}, 1, "guard");
        fields = ops::append_kpool_tail_fields(sb, 128, L);
        (void)sb.add(DType::I32, {64}, 1, "guard");
        state_layout      = sb.finish();
        const auto stride = std::int64_t(state_layout.image_bytes + 256);
        state_host.resize((256 + 2 * B * stride) / 2, 0xa5a5);
        state = DeviceBuffer(cuda.device(), state_host.size() * 2);
        StateImageStorage images(state.span(), state_layout);
        before  = {images.field(fields.raw_keys, before_offsets.view, layer),
                   images.field(fields.raw_gates, before_offsets.view, layer)};
        after   = {images.field(fields.raw_keys, after_offsets.view, layer),
                   images.field(fields.raw_gates, after_offsets.view, layer)};
        cache   = cache_layout.layer(PagedRecordView{packed.storage(), tables.view, 64}, layer);
        records = {keys.view, gates.view, offsets.view};
        for (std::size_t i = 0; i < x.host.size(); ++i) x.host[i] = sample(i + 331, 1.F / 256);
        for (int b = 0; b < B; ++b) {
            bases[b]             = 60 + (phase + b) % 4;
            valid.at(b)          = ragged ? (b == B - 1 && B > 1 ? 0 : std::max(1, T - b)) : T;
            selectors.at(b)      = valid.at(b) ? B - 1 - b : -1;
            offsets.at(b)        = valid.at(b) ? 3 + b % 3 : -1;
            before_offsets.at(b) = valid.at(b) ? 256 + b * stride : -1;
            after_offsets.at(b)  = valid.at(b) ? 256 + (alias ? b : b + B) * stride : -1;
            for (int t = 0; t < T; ++t) positions.at(t, b) = t < valid.at(b) ? bases[b] + t : -1;
            for (int p = 0; p < P; ++p)
                tables.at(p, B - 1 - b) = packed.physical_offset(P * B - 1 - (b * P + p));
            if (valid.at(b))
                for (int kind = 0; kind < 2; ++kind)
                    for (int r = 0; r < 3; ++r)
                        for (int d = 0; d < 128; ++d)
                            state_host[state_index(false, kind, b, r, d)] =
                                r < bases[b] % 4
                                    ? sample(d + 128 * r + 777 * b + 99 * kind, 1.F / 512)
                                    : 0;
        }
        for (auto* v : {&x, &q, &keys, &gates}) v->upload();
        coeff.upload();
        for (auto* v : {&positions, &valid, &selectors, &offsets}) v->upload();
        for (auto* v : {&tables, &before_offsets, &after_offsets}) v->upload();
        state.copy_from_host(state_host.data(), state_host.size() * 2);
        auto capacity    = update ? ops::kpool_mla_prepare_update_workspace_capacity_bytes
                                  : ops::kpool_mla_prepare_record_workspace_capacity_bytes;
        workspace_bytes  = capacity(w.profile(), ops::KpoolMlaStorage::BFloat16, B, 1, T,
                                    cuda.multiprocessor_count());
        workspace_buffer = DeviceBuffer(cuda.device(), workspace_bytes);
        workspace        = std::make_unique<MemoryArena>(workspace_buffer.span());
    }

    std::size_t state_index(bool dst, int kind, int b, int r, int d) const {
        const auto& f     = kind ? before.raw_gates.field : before.raw_keys.field;
        const auto offset = dst ? after_offsets.at(b) : before_offsets.at(b);
        return (offset + f.tensor.region.offset + r * f.byte_strides[1]) / 2 + d;
    }

    std::size_t cache_index(int kind, int b, int p, int d) const {
        const auto& plane = kind ? cache.index.keys : cache.latent.data;
        const int span = kind ? 4 : 1, dim = kind ? 128 : 512;
        return (tables.at((p * span) / 64, selectors.at(b)) + plane.plane.tensor.region.offset) /
                   2 +
               (p % (64 / span)) * dim + d;
    }

    ops::KpoolMlaPrepareOutputs outputs() const {
        return {w.q ? q.view : Tensor{}, w.heads ? coeff.view : Tensor{}};
    }

    void launch(CudaExecutionView cuda) {
        const ops::KpoolMlaBatchView batch{positions.view, valid.view, selectors.view};
        if (update)
            ops::kpool_mla_prepare_update(x.view, w.params, batch, before, cache, after, outputs(),
                                          *workspace, cuda);
        else
            ops::kpool_mla_prepare_record(x.view, w.params, batch, before, cache, records,
                                          outputs(), *workspace, cuda);
    }

    // Caller performs restoration outside timing; original host data outlives the async copy.
    void restore(CudaExecutionView cuda) {
        if (update && inplace)
            CUDA_CHECK(cudaMemcpyAsync(state.data(), state_host.data(), state.size(),
                                       cudaMemcpyHostToDevice, cuda.stream()));
    }
};
} // namespace ninfer::tool::kpool
