#pragma once
#include "options.h"
#include "quantized_weight.cuh"
#include "direct_bf16_weight.cuh"
#include "core/transfer.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ninfer/ops/kda_replay.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/rmsnorm.h"

#include <optional>

namespace ninfer::bench::kda {

inline ops::LinearPolicy input_policy(const Options& o) {
    return o.policy == "a8" || (o.policy == "default" && o.format == "mxfp8")
               ? ops::LinearPolicy::AllowA8
               : ops::LinearPolicy::A16Only;
}

inline PackedQuantizedWeight make_input_weight(const Options& o) {
    return o.format == "mxfp8" ? make_mxfp8_weight(12288, 4096, 101)
                               : make_row_split_weight(QType::Q8_G32_FP16, 12288, 4096, 4096, 101);
}

inline DeviceBuffer make_conv_weight(const Options& o) {
    return o.format == "mxfp8" ? make_bf16(12288 * 4, 105, -0.1F, 0.1F)
                               : make_f32(12288 * 4, 105, -0.1F, 0.1F);
}

struct Fixture {
    const Options& o;
    int t, b, h, qh, layers, channels;
    std::vector<int> valid_counts, commit_counts;
    DeviceBuffer q, k, v, g, beta, a_log, bias, out;
    DeviceBuffer state, record_buffer, sources, destinations, valid, commits;
    DeviceBuffer input, conv_weight, scratch;
    PackedQuantizedWeight weight;
    DirectBf16Weight control_weight, forget_weight, gate_weight;
    PackedQuantizedWeight output_weight;
    DeviceBuffer forget_lowrank, gate_lowrank, gate, gamma, normalized, layer_output;
    StateImageLayout state_layout;
    LinearAttentionStateLayout fields;
    std::unique_ptr<StateImageStorage> storage;
    std::unique_ptr<LinearAttentionStateView> states;
    KdaReplayRecords records;
    std::unique_ptr<ops::KdaReplayFoldPlan> fold;
    std::unique_ptr<MemoryArena> workspace;
    std::size_t capacity = 0;
    std::vector<std::int64_t> so, ds;

    bool full_layer() const { return o.op.starts_with("layer-"); }

    bool recorded_layer() const { return o.op == "layer-record" || o.op == "layer-chain"; }

    bool projects_fold() const {
        return o.op == "fold" && o.format == "mxfp8" && h == 32 && qh == 32;
    }

    Fixture(const Options& options, int tokens)
        : o(options), t(tokens), b(o.batch), h(o.heads), qh(o.qk_heads),
          layers(o.op == "fold" ? o.layers : 1), channels((2 * qh + h) * 128),
          valid_counts(counts(o.valid, t, b, false)), commit_counts(counts(o.commits, t, b, true)),
          q(make_bf16(std::size_t(128) * qh * t * b, 11)),
          k(make_bf16(std::size_t(128) * qh * t * b, 12)),
          v(make_bf16(std::size_t(128) * h * t * b, 13)),
          g(make_bf16(std::size_t(128) * h * t * b, 14, -4, 4)),
          beta(make_bf16(std::size_t(h) * t * b, 15)), a_log(make_f32(h, 16, -0.5, 0.5)),
          bias(make_f32(128 * h, 17, -1, 1)), out(make_zeros(std::size_t(128) * h * t * b * 2)),
          valid(measurement::make_i32(valid_counts)),
          commits(measurement::make_i32(commit_counts)) {
        if ((o.op == "update" || o.op == "layer-update") && t != 1)
            throw std::invalid_argument("update requires T=1");
        if ((o.op == "core" || o.op == "layer-prefill") && o.valid != "dense")
            throw std::invalid_argument("core consumes every physical token");
        if (o.commits == "dense") {
            commit_counts = valid_counts;
            commits.copy_from_host(commit_counts.data(), commit_counts.size() * sizeof(int));
        }
        for (int i = 0; i < b; ++i)
            if (commit_counts[i] > valid_counts[i])
                throw std::invalid_argument("commit exceeds valid input");
        StateImageBuilder builder;
        (void)builder.add(DType::I32, {64}, 1, "unrelated field");
        fields = plan_linear_attention_state(
            builder,
            {static_cast<unsigned>(layers), channels, 3, h, 128, 128, DType::BF16, o.state_dtype});
        state_layout       = builder.finish();
        std::size_t cursor = 256;
        for (int i = 0; i < b; ++i) {
            so.push_back(cursor);
            cursor += state_layout.image_bytes + (i + 1) * 256;
            ds.push_back(cursor);
            cursor += state_layout.image_bytes + (i + 2) * 256;
        }
        state   = make_zeros(cursor);
        sources = make_buffer(b * sizeof(std::int64_t));
        sources.copy_from_host(so.data(), sources.size());
        destinations = make_buffer(b * sizeof(std::int64_t));
        destinations.copy_from_host(ds.data(), destinations.size());
        storage        = std::make_unique<StateImageStorage>(state.span(), state_layout);
        states         = std::make_unique<LinearAttentionStateView>(*storage, fields);
        const auto n_r = std::size_t(128) * 128 * h * layers * b;
        auto recurrent = o.state_dtype == DType::BF16 ? make_bf16(n_r, 19, -0.1F, 0.1F)
                                                      : make_f32(n_r, 19, -0.1F, 0.1F);
        auto conv         = make_bf16(std::size_t(channels) * 3 * layers * b, 23, -0.1F, 0.1F);
        const auto stream = device_context().execution_view().stream();
        for (int i = 0; i < b; ++i)
            for (int l = 0; l < layers; ++l) {
                const auto r     = states->recurrent_tensor(l, so[i]);
                const auto c     = states->conv_tensor(l, so[i]);
                const auto index = std::size_t(i) * layers + l;
                CUDA_CHECK(cudaMemcpyAsync(
                    r.data, static_cast<std::byte*>(recurrent.data()) + index * r.bytes(),
                    r.bytes(), cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(c.data,
                                           static_cast<std::byte*>(conv.data()) + index * c.bytes(),
                                           c.bytes(), cudaMemcpyDeviceToDevice, stream));
            }
        if (o.op == "record" || o.op == "fold" || o.op == "input-record" || o.op == "chain" ||
            recorded_layer()) {
            LayoutBuilder rb;
            auto layout   = plan_kda_replay_records(rb, {layers, b, t, channels, qh, h, 128, 128});
            record_buffer = make_zeros(rb.finish(256));
            records       = KdaReplayRecords(record_buffer.span(), layout);
            fold = std::make_unique<ops::KdaReplayFoldPlan>(records, states->all_layers_view());
            if (o.op == "fold" && !projects_fold()) {
                auto conv_records = make_bf16(records.conv.numel(), 29, -0.1F, 0.1F);
                CUDA_CHECK(cudaMemcpyAsync(records.conv.data, conv_records.data(),
                                           records.conv.bytes(), cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (o.op == "core" || o.op == "layer-prefill")
            capacity = ops::kimi_delta_attention_workspace_capacity_bytes(qh, h, t, t);
        if (o.op.starts_with("input-") || o.op == "chain" || full_layer() || projects_fold()) {
            if (qh != 32 || h != 32) throw std::invalid_argument("input geometry is Hq=Hv=32");
            weight      = make_input_weight(o);
            input       = make_bf16(std::size_t(4096) * t * b, 103, -0.01F, 0.01F);
            conv_weight = make_conv_weight(o);
            const ops::LinearProfile profile{weight.weight.qtype, 12288, 4096, input_policy(),
                                             weight.weight.layout};
            const bool update =
                o.op == "input-update" || o.op == "layer-update" || o.op == "layer-prefill";
            const auto input_capacity =
                update ? ops::kda_input_proj_conv_update_workspace_capacity_bytes(profile, b, t, t)
                       : ops::kda_input_proj_conv_record_workspace_capacity_bytes(profile, b, t, t);
            capacity = std::max(capacity, input_capacity);
        }
        if (full_layer()) {
            const int m    = t * b;
            control_weight = make_direct_bf16_weight(288, 4096, 111);
            forget_weight  = make_direct_bf16_weight(4096, 128, 113);
            gate_weight    = make_direct_bf16_weight(4096, 128, 115);
            output_weight  = make_mxfp8_weight(4096, 4096, 117);
            forget_lowrank = make_buffer(std::size_t(128) * m * 2);
            gate_lowrank   = make_buffer(std::size_t(128) * m * 2);
            gate           = make_buffer(std::size_t(4096) * m * 2);
            gamma          = make_bf16(128, 119, 0.9F, 1.1F);
            normalized     = make_buffer(std::size_t(4096) * m * 2);
            layer_output   = make_buffer(std::size_t(4096) * m * 2);
            const std::array<std::int32_t, 3> segments{32, 128, 128};
            capacity = std::max(
                capacity,
                ops::linear_split_workspace_capacity_bytes(
                    ops::ProjectionWeights{ops::SingleProjectionWeight{control_weight.weight}},
                    segments, m, m));
            capacity =
                std::max(capacity, ops::linear_workspace_capacity_bytes(
                                       QType::MXFP8, 4096, 4096, ops::LinearPolicy::AllowA8, m, m));
        }
        scratch   = make_buffer(std::max<std::size_t>(256, capacity));
        workspace = std::make_unique<MemoryArena>(scratch.span());
        if (o.op == "fold")
            for (int l = 0; l < layers; ++l) {
                if (projects_fold()) input_call(true, device_context().execution_view(), l);
                record(l, device_context().execution_view());
            }
        device_context().synchronize();
    }

    Tensor st() { return Tensor(sources.span(), DType::I64, {b}); }

    Tensor dt() { return Tensor(destinations.span(), DType::I64, {b}); }

    Tensor vt() { return Tensor(valid.span(), DType::I32, {b}); }

    Tensor ct() { return Tensor(commits.span(), DType::I32, {b}); }

    ops::LinearPolicy input_policy() const { return kda::input_policy(o); }

    Tensor vector(DeviceBuffer& x, int heads) {
        return Tensor(x.span(), DType::BF16, {128, heads, t, b});
    }

    void record(int layer, CudaExecutionView cuda) {
        const auto rr = records.layer(layer, b);
        auto y        = vector(out, h);
        ops::kimi_delta_attention_replay_record(
            vector(q, qh), vector(k, qh), vector(v, h),
            vector(g, h), Tensor(beta.span(), DType::BF16, {h, t, b}),
            Tensor(a_log.span(), DType::FP32, {h}), Tensor(bias.span(), DType::FP32, {128, h}), -5,
            0.08838834764831845F, states->layer_view(layer, st()).recurrent, vt(),
            rr.recurrence, y, cuda);
    }

    void input_call(bool record_mode, CudaExecutionView cuda, int layer = 0) {
        auto x = Tensor(input.span(), DType::BF16, {4096, t, b}),
             w = Tensor(conv_weight.span(), o.format == "mxfp8" ? DType::BF16 : DType::FP32,
                        {12288, 4});
        const ops::KdaConvOutputs outputs{vector(q, qh).view({4096, t, b}),
                                          vector(k, qh).view({4096, t, b}),
                                          vector(v, h).view({4096, t, b})};
        const ops::SingleProjectionWeight qw{weight.weight, input_policy()};
        if (record_mode) {
            auto rr = records.layer(layer, b);
            ops::kda_input_proj_conv_record(x, qw, w, {states->layer_view(layer, st()).conv}, vt(),
                                            outputs, rr.conv, *workspace, cuda);
        } else
            ops::kda_input_proj_conv_update(
                x, qw, w,
                {states->layer_view(layer, st()).conv, states->layer_view(layer, dt()).conv}, vt(),
                outputs, *workspace, cuda);
    }

    void launch(CudaExecutionView cuda) {
        if (full_layer()) {
            launch_layer(cuda);
            return;
        }
        if (o.op == "fold") {
            fold->execute(st(), dt(), ct(), cuda);
            return;
        }
        if (o.op == "record") {
            record(0, cuda);
            return;
        }
        if (o.op == "chain") {
            input_call(true, cuda);
            record(0, cuda);
            fold->execute(st(), dt(), ct(), cuda);
            return;
        }
        if (o.op.starts_with("input-")) {
            input_call(o.op == "input-record", cuda);
            return;
        }
        auto y = vector(out, h), bb = Tensor(beta.span(), DType::BF16, {h, t, b});
        auto al = Tensor(a_log.span(), DType::FP32, {h}),
             db = Tensor(bias.span(), DType::FP32, {128, h});
        if (o.op == "update")
            ops::kimi_delta_attention_batch_update(
                vector(q, qh), vector(k, qh), vector(v, h), vector(g, h), bb, al, db, -5,
                0.08838834764831845F, states->layer_view(0, st()).recurrent,
                states->layer_view(0, dt()).recurrent, vt(), y, cuda);
        else {
            auto si = states->recurrent_tensor(0, so[0]), sd = states->recurrent_tensor(0, ds[0]);
            ops::kimi_delta_attention(vector(q, qh), vector(k, qh), vector(v, h), vector(g, h), bb,
                                      al, db, -5, 0.08838834764831845F, *workspace, si, sd, y,
                                      cuda);
        }
    }

    void launch_layer(CudaExecutionView cuda) {
        const int m = t * b;
        input_call(recorded_layer(), cuda);
        const auto x  = Tensor(input.span(), DType::BF16, {4096, m});
        const auto fl = Tensor(forget_lowrank.span(), DType::BF16, {128, m}),
                   gl = Tensor(gate_lowrank.span(), DType::BF16, {128, m});
        const std::array controls{Tensor(beta.span(), DType::BF16, {32, m}), fl, gl};
        ops::linear_split(
            x, ops::ProjectionWeights{ops::SingleProjectionWeight{control_weight.weight}}, controls,
            *workspace, cuda);
        auto forget = Tensor(g.span(), DType::BF16, {4096, m}),
             gates  = Tensor(gate.span(), DType::BF16, {4096, m});
        ops::linear_pair(fl, gl, forget_weight.weight, gate_weight.weight, forget, gates, cuda);
        auto y = vector(out, h), bb = Tensor(beta.span(), DType::BF16, {h, t, b});
        const auto al = Tensor(a_log.span(), DType::FP32, {h}),
                   db = Tensor(bias.span(), DType::FP32, {128, h});
        if (recorded_layer()) {
            record(0, cuda);
        } else if (o.op == "layer-update") {
            ops::kimi_delta_attention_batch_update(
                vector(q, qh), vector(k, qh), vector(v, h), vector(g, h), bb, al, db, -5,
                0.08838834764831845F, states->layer_view(0, st()).recurrent,
                states->layer_view(0, dt()).recurrent, vt(), y, cuda);
        } else {
            auto si = states->recurrent_tensor(0, so[0]), sd = states->recurrent_tensor(0, ds[0]);
            ops::kimi_delta_attention(vector(q, qh), vector(k, qh), vector(v, h), vector(g, h), bb,
                                      al, db, -5, 0.08838834764831845F, *workspace, si, sd, y,
                                      cuda);
        }
        auto norm = Tensor(normalized.span(), DType::BF16, {128, h, m});
        ops::sigmoid_gated_rmsnorm(y.view({128, h, m}), Tensor(gamma.span(), DType::BF16, {128}),
                                   gates.view({128, h, m}), 1e-6F, norm, cuda);
        auto result = Tensor(layer_output.span(), DType::BF16, {4096, m});
        ops::linear(norm.view({4096, m}), output_weight.weight, result, ops::LinearPolicy::AllowA8,
                    *workspace, cuda);
        if (o.op == "layer-chain") fold->execute(st(), dt(), ct(), cuda);
    }
};

struct RaggedWeights {
    PackedQuantizedWeight projection;
    DeviceBuffer conv, a_log, bias;
    ops::LinearPolicy policy;
    DType conv_dtype;

    explicit RaggedWeights(const Options& o)
        : policy(input_policy(o)), conv_dtype(o.format == "mxfp8" ? DType::BF16 : DType::FP32) {
        if (o.op == "input-update") {
            if (o.qk_heads != 32 || o.heads != 32)
                throw std::invalid_argument("input geometry is Hq=Hv=32");
            projection = make_input_weight(o);
            conv       = make_conv_weight(o);
        } else {
            a_log = make_f32(o.heads, 16, -0.5, 0.5);
            bias  = make_f32(128 * o.heads, 17, -1, 1);
        }
    }

    ops::SingleProjectionWeight projection_view() const { return {projection.weight, policy}; }
};

struct RaggedFixture {
    RaggedWeights& weights;
    const bool input_update;
    const int h, qh, channels;
    const ragged::RaggedDeviceFixture workload;
    const int n, s;
    const std::size_t state_bytes;
    const ragged::StateOffsets offsets;
    DeviceBuffer sources, destinations, state;
    std::optional<SavedBuffer> initial;
    DeviceBuffer x, q, k, v, g, beta, out, scratch;
    StateFieldLayout layout;
    std::size_t capacity;
    std::unique_ptr<MemoryArena> workspace;

    RaggedFixture(const Options& o, const ragged::SegmentSpec& lengths, RaggedWeights& w)
        : weights(w), input_update(o.op == "input-update"), h(o.heads), qh(o.qk_heads),
          channels((2 * qh + h) * 128), workload(lengths, device_context().device()),
          n(workload.view().total_tokens()), s(workload.view().num_seqs()),
          state_bytes(input_update ? 6ULL * channels
                                   : dtype_size(o.state_dtype) * 128ULL * 128 * h),
          offsets(ragged::make_state_offsets(s, o.state_form, state_bytes, 16, 16)),
          sources(ragged::upload_offsets(device_context().device(), offsets.source_offsets_host)),
          destinations(
              ragged::upload_offsets(device_context().device(), offsets.destination_offsets_host)) {
        const auto qk_elements    = std::size_t(128) * qh * n;
        const auto value_elements = std::size_t(128) * h * n;
        if (input_update) {
            state    = make_bf16(offsets.backing_bytes() / 2, 23, -0.1F, 0.1F);
            layout   = {{{0, state_bytes, 16}, DType::BF16, {channels, 3, 1, 1}}, 1, 0};
            x        = make_bf16(std::size_t(4096) * n, 103, -0.01F, 0.01F);
            q        = make_zeros(qk_elements * 2);
            k        = make_zeros(qk_elements * 2);
            v        = make_zeros(value_elements * 2);
            capacity = ops::kda_input_proj_conv_update_ragged_workspace_capacity_bytes(
                ops::linear_profile(weights.projection_view()), n, n, s, s);
        } else {
            state = o.state_dtype == DType::BF16
                        ? make_bf16(offsets.backing_bytes() / 2, 19, -0.1F, 0.1F)
                        : make_f32(offsets.backing_bytes() / 4, 19, -0.1F, 0.1F);
            layout = {{{0, state_bytes, 16}, o.state_dtype, {128, 128, h, 1}}, 1, 0};
            q        = make_bf16(qk_elements, 11);
            k        = make_bf16(qk_elements, 12);
            v        = make_bf16(value_elements, 13);
            g        = make_bf16(value_elements, 14, -4, 4);
            beta     = make_bf16(std::size_t(h) * n, 15);
            out      = make_zeros(value_elements * 2);
            capacity = ops::kimi_delta_attention_ragged_workspace_capacity_bytes(qh, h, n, n, s, s);
        }
        scratch   = make_buffer(std::max<std::size_t>(256, capacity));
        workspace = std::make_unique<MemoryArena>(scratch.span());
        if (offsets.mutates_source) initial.emplace(state);
    }

    void restore(CudaExecutionView cuda) const {
        if (initial) initial->restore(cuda);
    }

    void launch(CudaExecutionView cuda) {
        const auto seq = workload.view();
        const ConstStateFieldView source{{state.span(), ragged::offset_tensor(sources)}, layout};
        const StateFieldView destination{{state.span(), ragged::offset_tensor(destinations)},
                                         layout};
        if (input_update) {
            const ops::KdaConvOutputs outputs{Tensor(q.span(), DType::BF16, {4096, n}),
                                              Tensor(k.span(), DType::BF16, {4096, n}),
                                              Tensor(v.span(), DType::BF16, {4096, n})};
            ops::kda_input_proj_conv_update(
                Tensor(x.span(), DType::BF16, {4096, n}), weights.projection_view(),
                Tensor(weights.conv.span(), weights.conv_dtype, {channels, 4}),
                {source, destination}, seq, outputs, *workspace, cuda);
        } else {
            Tensor y(out.span(), DType::BF16, {128, h, n});
            ops::kimi_delta_attention(Tensor(q.span(), DType::BF16, {128, qh, n}),
                                      Tensor(k.span(), DType::BF16, {128, qh, n}),
                                      Tensor(v.span(), DType::BF16, {128, h, n}),
                                      Tensor(g.span(), DType::BF16, {128, h, n}),
                                      Tensor(beta.span(), DType::BF16, {h, n}),
                                      Tensor(weights.a_log.span(), DType::FP32, {h}),
                                      Tensor(weights.bias.span(), DType::FP32, {128, h}), -5,
                                      0.08838834764831845F, seq, *workspace, source, destination, y,
                                      cuda);
        }
    }
};
} // namespace ninfer::bench::kda
