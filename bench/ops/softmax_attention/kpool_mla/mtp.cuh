#pragma once

static Json mtp_round(CudaExecutionView cuda, const Json& c, int repeats, int warmup) {
    const int B = c.at("B"), S = c.at("draft_steps"), W = S + 1, E = c.at("context"), F = E - 1;
    const int U = bucket(F + W);
    kp::IndexerCase ms, ds;
    ms.ends.assign(B, F + W);
    ms.valid.assign(B, W);
    ms.envelope = U;
    ds.ends.assign(B, F + 1);
    ds.valid.assign(B, 1);
    ds.reserve_ends.assign(B, F + W);
    ds.envelope = U;
    app::Fixture main(cuda, W, B, ms, true), draft(cuda, 1, B, ds, false, 1);
    main.after           = main.before;
    const auto canonical = draft.before, first = draft.after;
    DeviceBuffer provisional(cuda.device(), draft.state.size());
    StateImageStorage images(provisional.span(), draft.state_layout);
    ops::KpoolTailStateView temporary{
        images.field(draft.fields.raw_keys, draft.after_offsets.view, 0),
        images.field(draft.fields.raw_gates, draft.after_offsets.view, 0)};
    std::vector<int> accepted(B);
    const std::string condition = c.at("acceptance");
    for (int b = 0; b < B; ++b)
        accepted[b] = condition == "zero"   ? 0
                      : condition == "half" ? W / 2
                      : condition == "all"  ? W
                                            : (b % (W + 1));
    const int history_columns =
        std::max(0, *std::max_element(accepted.begin(), accepted.end()) - 1);
    const int HW = std::max(1, history_columns);
    kp::Buffer<std::uint16_t> inputs(cuda.device(), DType::BF16, {4096, 1, B, S}),
        history(cuda.device(), DType::BF16, {4096, HW, B});
    kp::Buffer<int> positions(cuda.device(), DType::I32, {1, B, S}),
        history_positions(cuda.device(), DType::I32, {HW, B}),
        valid(cuda.device(), DType::I32, {B});
    for (int b = 0; b < B; ++b) {
        const int a         = accepted[b];
        main.accepted.at(b) = a;
        valid.at(b)         = std::max(0, a - 1);
        for (int s = 0; s < S; ++s) {
            positions.at(0, b, s) = F + s;
            if (s < HW) history_positions.at(s, b) = s < valid.at(b) ? F + 1 + s : -1;
            for (int d = 0; d < 4096; ++d) {
                inputs.at(d, 0, b, s) =
                    kp::sample(d + 4096 * (s + 1) + kp::hash(1357 + b), 1.F / 256);
                if (s < HW)
                    history.at(d, s, b) =
                        kp::sample(d + 4096 * (s + 1) + kp::hash(9871 + b), 1.F / 256);
            }
        }
    }
    for (auto* v : {&inputs, &history}) v->upload();
    positions.upload();
    history_positions.upload();
    valid.upload();
    main.accepted.upload();
    const auto history_capacity = ops::kpool_mla_prepare_update_workspace_capacity_bytes(
        {kp::Projection::History, 4096, 512, 128, 0, 0, 4}, ops::KpoolMlaStorage::BFloat16,
        KvCacheVectorFormat::BFloat16, B, 1, HW, cuda.multiprocessor_count());
    DeviceBuffer history_scratch(cuda.device(), history_capacity);
    MemoryArena history_arena(history_scratch.span());
    const auto history_params = draft.params(app::Phase::History);
    std::vector<cudaEvent_t> events(S + 5);
    for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));
    auto body = [&](CudaExecutionView v, bool diagnostic) {
        int event       = 0;
        const auto mark = [&] {
            if (diagnostic)
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[event], v.stream(), cudaEventRecordExternal));
            ++event;
        };
        mark();
        for (int s = 0; s < S; ++s) {
            auto x  = inputs.view;
            x.data  = static_cast<std::byte*>(inputs.view.data) + std::size_t(s) * B * 4096 * 2;
            x.ne[3] = 1;
            draft.x.view               = x.reshape({4096, 1, B});
            auto pos                   = positions.view;
            pos.data                   = static_cast<std::byte*>(positions.view.data) + s * B * 4;
            pos.ne[2]                  = 1;
            draft.index.positions.view = pos.reshape({1, B});
            draft.before               = s ? temporary : canonical;
            draft.after                = s ? temporary : first;
            draft.launch(v, s ? app::Phase::Reuse : app::Phase::Full);
            // The frozen first tail remains in draft.state; later steps borrow a provisional copy.
            if (!s)
                CUDA_CHECK(cudaMemcpyAsync(provisional.data(), draft.state.data(),
                                           draft.state.size(), cudaMemcpyDeviceToDevice,
                                           v.stream()));
            mark();
        }
        main.launch(v);
        mark();
        main.extract(v);
        mark();
        // Adopt the first true-conditioning tail only for accepted rows. Acceptance zero
        // retains the canonical image; History fills later accepted conditioning columns.
        for (int b = 0; b < B; ++b)
            if (accepted[b])
                for (int kind = 0; kind < 2; ++kind) {
                    auto* bytes = static_cast<std::byte*>(draft.state.data());
                    CUDA_CHECK(cudaMemcpyAsync(bytes + draft.state_index(false, kind, b, 0, 0) * 2,
                                               bytes + draft.state_index(true, kind, b, 0, 0) * 2,
                                               128 * 3 * 2, cudaMemcpyDeviceToDevice, v.stream()));
                }
        mark();
        if (history_columns)
            ops::kpool_mla_prepare_update(
                history.view, history_params,
                {history_positions.view, valid.view, draft.index.selectors.view}, canonical,
                draft.cache, canonical, {}, history_arena, v);
        mark();
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    Json out = {
        {"case", c},
        {"accepted_columns", accepted},
        {"scope", "controlled_attention_round"},
        {"history_columns", history_columns},
        {"selection_end", E},
        {"envelope", U},
        {"native_kv_bytes", main.index.packed.storage().bytes + draft.index.packed.storage().bytes},
        {"workspace_bytes",
         main.index.workspace_bytes + draft.index.workspace_bytes + history_capacity}};
    {
        measurement::BodyTimedGraph graph;
        graph.capture(cuda, [&](auto v) { body(v, false); });
        std::vector<double> values;
        for (int r = -warmup; r < repeats; ++r) {
            main.restore(cuda);
            draft.restore(cuda);
            graph.launch(cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            if (r >= 0) values.push_back(graph.elapsed_us());
        }
        out["body"]          = stats(values);
        out["graph_kernels"] = graph.kernels();
        measurement::BodyTimedGraph staged;
        staged.capture(cuda, [&](auto v) { body(v, true); });
        std::vector<std::vector<double>> timings(S + 4);
        for (int r = -3; r < 20; ++r) {
            main.restore(cuda);
            draft.restore(cuda);
            staged.launch(cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            if (r < 0) continue;
            for (int j = 0; j < S + 4; ++j) {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, events[j], events[j + 1]));
                timings[j].push_back(ms * 1000.);
            }
        }
        for (int j = 0; j < S + 4; ++j) out["phase_intervals"].push_back(stats(timings[j]));
    }
    for (auto e : events) cudaEventDestroy(e);
    const auto ends = draft.index.ends.download();
    for (int end : ends)
        if (end != E) throw std::runtime_error("MTP selection end changed");
    out["status"] = "measured";
    return out;
}
