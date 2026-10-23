#pragma once
#include "options.h"
#include "quantized_weight.cuh"
#include "core/transfer.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ninfer/ops/kda_replay.h"
#include "ninfer/ops/kimi_delta_attention.h"

namespace ninfer::bench::kda {
struct Fixture {
    const Options& o;
    int t, b, h, qh, layers, channels;
    std::vector<int> valid_counts, commit_counts;
    DeviceBuffer q, k, v, g, beta, a_log, bias, out;
    DeviceBuffer state, record_buffer, sources, destinations, valid, commits;
    DeviceBuffer input, conv_weight, scratch;
    PackedQuantizedWeight weight;
    StateImageLayout state_layout;
    LinearAttentionStateLayout fields;
    std::unique_ptr<StateImageStorage> storage;
    std::unique_ptr<LinearAttentionStateView> states;
    KdaReplayRecords records;
    std::unique_ptr<ops::KdaReplayFoldPlan> fold;
    std::unique_ptr<MemoryArena> workspace;
    std::size_t capacity = 0;
    std::vector<std::int64_t> so, ds;

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
        if (o.op == "update" && t != 1) throw std::invalid_argument("update requires T=1");
        if (o.op == "core" && o.valid != "dense")
            throw std::invalid_argument("core consumes every physical token");
        if (o.commits == "dense") commit_counts = valid_counts;
        for (int i = 0; i < b; ++i)
            if (commit_counts[i] > valid_counts[i])
                throw std::invalid_argument("commit exceeds valid input");
        StateImageBuilder builder;
        (void)builder.add(DType::I32, {64}, 1, "unrelated field");
        fields = plan_linear_attention_state(
            builder, {static_cast<unsigned>(layers), channels, 3, h, 128, 128, DType::BF16});
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
        storage = std::make_unique<StateImageStorage>(state.span(), state_layout);
        states  = std::make_unique<LinearAttentionStateView>(*storage, fields);
        if (o.op == "record" || o.op == "fold" || o.op == "input-record" || o.op == "chain") {
            LayoutBuilder rb;
            auto layout   = plan_kda_replay_records(rb, {layers, b, t, channels, qh, h, 128, 128});
            record_buffer = make_zeros(rb.finish(256));
            records       = KdaReplayRecords(record_buffer.span(), layout);
            fold = std::make_unique<ops::KdaReplayFoldPlan>(records, states->all_layers_view());
        }
        if (o.op == "core")
            capacity = ops::kimi_delta_attention_workspace_capacity_bytes(qh, h, t, t);
        if (o.op.starts_with("input-") || o.op == "chain") {
            if (qh != 32 || h != 32) throw std::invalid_argument("input geometry is Hq=Hv=32");
            weight      = make_row_split_weight(QType::Q8_G32_FP16, 12288, 4096, 4096, 101);
            input       = make_bf16(std::size_t(4096) * t * b, 103, -0.01F, 0.01F);
            conv_weight = make_f32(12288 * 4, 105, -0.1F, 0.1F);
            capacity    = ops::kda_input_proj_conv_update_workspace_capacity_bytes(
                {QType::Q8_G32_FP16, 12288, 4096, ops::LinearPolicy::A16Only,
                    QuantLayout::RowSplit},
                b, t, t);
        }
        scratch   = make_buffer(std::max<std::size_t>(256, capacity));
        workspace = std::make_unique<MemoryArena>(scratch.span());
        if (o.op == "fold")
            for (int l = 0; l < layers; ++l) record(l, device_context().execution_view());
        device_context().synchronize();
    }

    Tensor st() { return Tensor(sources.span(), DType::I64, {b}); }

    Tensor dt() { return Tensor(destinations.span(), DType::I64, {b}); }

    Tensor vt() { return Tensor(valid.span(), DType::I32, {b}); }

    Tensor ct() { return Tensor(commits.span(), DType::I32, {b}); }

    Tensor vector(DeviceBuffer& x, int heads) {
        return Tensor(x.span(), DType::BF16, {128, heads, t, b});
    }

    void record(int layer, CudaExecutionView cuda) {
        auto rr = records.layer(layer, b);
        auto y  = vector(out, h);
        ops::kimi_delta_attention_replay_record(
            vector(q, qh), o.op == "chain" ? rr.key : vector(k, qh),
            o.op == "chain" ? rr.value : vector(v, h), vector(g, h),
            Tensor(beta.span(), DType::BF16, {h, t, b}), Tensor(a_log.span(), DType::FP32, {h}),
            Tensor(bias.span(), DType::FP32, {128, h}), -5, 0.08838834764831845F,
            states->layer_view(layer, st()).recurrent, vt(),
            {rr.key, rr.value, rr.log_decay, rr.beta}, y, cuda);
    }

    void input_call(bool record_mode, CudaExecutionView cuda) {
        auto x           = Tensor(input.span(), DType::BF16, {4096, t, b}),
             w           = Tensor(conv_weight.span(), DType::FP32, {12288, 4});
        const auto key   = o.op == "chain" ? records.layer(0, b).key : vector(k, qh);
        const auto value = o.op == "chain" ? records.layer(0, b).value : vector(v, h);
        const ops::KdaConvOutputs outputs{vector(q, qh).view({4096, t, b}), key.view({4096, t, b}),
                                          value.view({4096, t, b})};
        const ops::SingleProjectionWeight qw{weight.weight, ops::LinearPolicy::A16Only};
        if (record_mode) {
            auto rr = records.layer(0, b);
            ops::kda_input_proj_conv_record(x, qw, w, {states->layer_view(0, st()).conv}, vt(),
                                            outputs, rr.conv, *workspace, cuda);
        } else
            ops::kda_input_proj_conv_update(
                x, qw, w, {states->layer_view(0, st()).conv, states->layer_view(0, dt()).conv},
                vt(), outputs, *workspace, cuda);
    }

    void launch(CudaExecutionView cuda) {
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
                states->layer_view(0, dt()).recurrent, y, cuda);
        else {
            auto si = states->recurrent_tensor(0, so[0]), sd = states->recurrent_tensor(0, ds[0]);
            ops::kimi_delta_attention(vector(q, qh), vector(k, qh), vector(v, h), vector(g, h), bb,
                                      al, db, -5, 0.08838834764831845F, *workspace, si, sd, y,
                                      cuda);
        }
    }

    // The chunk consumer's captured grid gives the actual value-stripe replication count.
    // This accounting inspects launches, without invoking a private entry or duplicating dispatch.
    std::uint64_t workspace_reads() {
        if (!capacity) return 0;
        if (o.op != "core") {
            std::uint64_t columns = 0;
            for (int n : valid_counts)
                columns +=
                    n + 3 * ((n + 3) / 4 - 1) + (o.op == "input-update" ? std::min(n, 3) : 0);
            return 4ULL * channels * columns;
        }
        auto cuda = device_context().execution_view();
        launch(cuda);
        device_context().synchronize();
        cudaGraph_t gr;
        CUDA_CHECK(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
        launch(cuda);
        CUDA_CHECK(cudaStreamEndCapture(cuda.stream(), &gr));
        std::size_t n = 0;
        CUDA_CHECK(cudaGraphGetNodes(gr, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        CUDA_CHECK(cudaGraphGetNodes(gr, nodes.data(), &n));
        std::uint64_t reads = 0;
        for (auto node : nodes) {
            cudaGraphNodeType type;
            CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            std::size_t dependents = 0;
            CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &dependents));
            if (type == cudaGraphNodeTypeKernel && !dependents) {
                cudaKernelNodeParams params;
                CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
                reads = std::uint64_t(capacity) * (params.gridDim.x / h);
            }
        }
        CUDA_CHECK(cudaGraphDestroy(gr));
        return reads;
    }
};
} // namespace ninfer::bench::kda
