#include "fixture.h"

namespace ninfer::test::causal_attention {
namespace {
template <class Launch>
void launch_attention_case(Launch&& launch, bool graph_replay) {
    if (!graph_replay) {
        launch(cuda_environment().default_execution);
        cuda_synchronize();
        return;
    }

    // Prime route-local CUDA function attributes before stream capture. The append route writes
    // the same represented values to the same positions, so this does not change its oracle.
    launch(cuda_environment().default_execution);
    cuda_synchronize();

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create causal-attention graph stream");
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin causal-attention capture");
    launch(stream_execution);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end causal-attention capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate causal-attention graph");
    for (int replay = 0; replay < 2; ++replay) {
        cuda_check(cudaGraphLaunch(executable, stream), "launch causal-attention graph");
    }
    cuda_check(cudaStreamSynchronize(stream), "synchronize causal-attention graph");
    cuda_check(cudaGraphExecDestroy(executable), "destroy causal-attention graph executable");
    cuda_check(cudaGraphDestroy(graph), "destroy causal-attention graph");
    cuda_check(cudaStreamDestroy(stream), "destroy causal-attention graph stream");
}

// Qualify large production widths against selected independent oracle rows;
// execution, cache-state checks and guards still cover the entire request.
template <class T>
std::vector<T> select_query_columns(const std::vector<T>& values, std::size_t stride,
                                    std::span<const int> queries) {
    if (queries.empty()) return values;
    std::vector<T> result(stride * queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i)
        std::copy_n(values.begin() + queries[i] * stride, stride, result.begin() + i * stride);
    return result;
}


} // namespace

int run_append_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const AttentionCase& test_case, MappingPattern mapping,
                    std::span<const int> oracle_queries,
                    std::span<const std::uint32_t> graph_limits) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    const float amplitude = test_case.qk_amplitude;
    std::vector<float> q  = make_bf16_values(q_elements, test_case.seed, -amplitude, amplitude);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<float> k =
        make_bf16_values(kv_elements, test_case.seed + 1u, -amplitude, amplitude);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                   test_case.envelope_max};

    const HostCache initial =
        make_cache(geometry, storage, total + 3, test_case.seed + 10u, amplitude);
    HostCache expected = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference =
        ideal_attention(select_query_columns(q, kHeadDim * geometry.q_heads, oracle_queries),
                        expected, select_query_columns(positions, 1, oracle_queries));
    DeviceCache cache(initial, mapping, max_context);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.span(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.span(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.span(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.span(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.span(), DType::I32, {1});
    Tensor tout(dout.span(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, test_case.tokens, test_case.tokens,
        execution.multiprocessor_count());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    MemoryArena workspace(workspace_buffer.span());

    int graph_failures = 0;
    const auto launch  = [&](CudaExecutionView cuda) {
        ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, ttable_row, op_geometry(geometry),
                                       kAttentionScale, cache.batch_view(), envelope, workspace,
                                       tout, cuda);
    };
    if (graph_limits.empty()) {
        launch_attention_case(launch, test_case.graph_replay);
    } else {
        DecodeGraphExecutable executable;
        for (const auto limit : graph_limits) {
            envelope.max_visible_keys = limit;
            launch(execution);
            cuda_synchronize();
            DecodeGraphDefinition definition;
            definition.capture(execution.stream(), [&] { launch(execution); });
            if (executable.ready())
                executable.update(definition);
            else
                executable.instantiate(definition);
            dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));
            cuda_synchronize();
            executable.launch(execution.stream());
            cuda_synchronize();
            const auto actual = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
            graph_failures +=
                verify_attention(std::string(cache_name(storage)) + " attention envelope update",
                                 bf16_bits_to_double(select_query_columns(
                                     actual, kHeadDim * geometry.q_heads, oracle_queries)),
                                 reference, attention_criterion(storage));
        }
    }

    const std::string label =
        case_label("causal_softmax_attention", geometry, storage, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = graph_failures +
                   verify_attention(label,
                                    bf16_bits_to_double(select_query_columns(
                                        output_bits, kHeadDim * geometry.q_heads, oracle_queries)),
                                    reference, attention_criterion(storage));
    failures += verify_cache(label, cache.snapshot(), expected);
    if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        DeviceCache standalone(initial, mapping, max_context);
        ops::kv_cache_append(tk, tv, tp, standalone.view(),
                             ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        failures += verify_cache(label + " fused/standalone byte parity", cache.snapshot(),
                                 standalone.snapshot());
        failures += standalone.verify_guards(label + " standalone append");
    }
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_cached_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const AttentionCase& test_case, MappingPattern mapping,
                    std::span<const int> oracle_queries) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const float amplitude = test_case.qk_amplitude;
    std::vector<float> q  = make_bf16_values(q_elements, test_case.seed, -amplitude, amplitude);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};

    const HostCache cache_host =
        make_cache(geometry, storage, total + 3, test_case.seed + 10u, amplitude);
    const std::vector<double> reference =
        ideal_attention(select_query_columns(q, kHeadDim * geometry.q_heads, oracle_queries),
                        cache_host, select_query_columns(positions, 1, oracle_queries));
    DeviceCache cache(cache_host, mapping, max_context);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.span(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.span(), DType::I32, {test_case.tokens});
    Tensor tout(dout.span(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, test_case.tokens, test_case.tokens,
        execution.multiprocessor_count());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    MemoryArena workspace(workspace_buffer.span());

    launch_attention_case(
        [&](CudaExecutionView cuda) {
            ops::causal_softmax_attention_cached(tq, tp, op_geometry(geometry), kAttentionScale,
                                                 cache.view(), envelope, workspace, tout, cuda);
        },
        test_case.graph_replay);

    const std::string label =
        case_label("causal_softmax_attention_cached", geometry, storage, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label,
                                    bf16_bits_to_double(select_query_columns(
                                        output_bits, kHeadDim * geometry.q_heads, oracle_queries)),
                                    reference, attention_criterion(storage));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                   const BatchAttentionCase& test_case, int envelope_max,
                   std::span<const std::uint32_t> graph_limits) {
    const int batch = test_case.contexts.size(), width = test_case.width;
    const int pool_rows = std::max(
        batch, *std::max_element(test_case.table_rows.begin(), test_case.table_rows.end()) + 1);
    int maximum_visible = 1;
    for (int b = 0; b < batch; ++b)
        maximum_visible = std::max(
            maximum_visible,
            test_case.contexts[b] + (test_case.graph_replay ? width : test_case.valid_columns[b]));
    for (const auto& contexts : test_case.replay_contexts)
        for (const int context : contexts)
            maximum_visible = std::max(maximum_visible, context + width);
    for (const auto limit : graph_limits)
        envelope_max = std::max(envelope_max, static_cast<int>(limit));
    ops::CausalAttentionExecutionEnvelope envelope{
        test_case.graph_replay ? 1U : static_cast<unsigned>(maximum_visible),
        static_cast<unsigned>(std::max(maximum_visible, envelope_max))};
    const std::size_t q_column_elements  = std::size_t(kHeadDim) * geometry.q_heads,
                      kv_column_elements = std::size_t(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = std::size_t(width) * batch;
    auto q = make_bf16_values(q_column_elements * columns, test_case.seed, -.25f, .25f);
    auto k = make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -.25f, .25f);
    auto v = make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.f, 1.f);
    inject_codec_edges(geometry, columns, k, v);
    std::vector<HostCache> initial;
    // Cover every replay's live prefix plus a writable-tail guard; future table capacity does
    // not require generating, encoding or copying nonexistent KV rows.
    for (int row = 0; row < pool_rows; ++row)
        initial.push_back(
            make_cache(geometry, storage, maximum_visible + 3, test_case.seed + 20u + 3u * row));
    auto expected = initial;
    BatchDeviceCache cache(initial, test_case.mapping, envelope.max_visible_keys);
    std::optional<BatchDeviceCache> control;
    if (test_case.graph_replay)
        control.emplace(initial, test_case.mapping, envelope.max_visible_keys);
    GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), dp(columns * 4),
        dvalid(batch * 4), dlanes(batch * 4), dout(q.size() * 2);
    Tensor tq(dq.span(), DType::BF16, {kHeadDim, geometry.q_heads, width, batch});
    Tensor tk(dk.span(), DType::BF16, {kHeadDim, geometry.kv_heads, width, batch}),
        tv(dv.span(), DType::BF16, {kHeadDim, geometry.kv_heads, width, batch});
    Tensor tp(dp.span(), DType::I32, {width, batch}), tvalid(dvalid.span(), DType::I32, {batch}),
        tlanes(dlanes.span(), DType::I32, {batch});
    Tensor tout(dout.span(), DType::BF16, {kHeadDim, geometry.q_heads, width, batch});
    const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, batch, width, width,
        execution.multiprocessor_count());
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(scratch.span());
    const bool masked = test_case.graph_replay ||
                        std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](int count) { return count != width; });
    const auto launch = [&] {
        ops::causal_softmax_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, tlanes,
                                      op_geometry(geometry), kAttentionScale, cache.view(),
                                      envelope, workspace, tout, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    auto valid = test_case.valid_columns, lanes = test_case.table_rows;
    std::vector<int> positions(columns);
    int failures = 0;
    for (int phase = 0; phase < (test_case.graph_replay ? 3 : 1); ++phase) {
        if (!graph_limits.empty()) envelope.max_visible_keys = graph_limits[phase];
        const auto& contexts = test_case.replay_contexts.empty() ? test_case.contexts
                                                                 : test_case.replay_contexts[phase];
        if (phase) {
            for (auto& x : q) x *= -0.5f;
            for (auto& x : k) x *= -0.5f;
            for (auto& x : v) x *= -0.5f;
            round_to_bf16(q);
            round_to_bf16(k);
            round_to_bf16(v);
            std::rotate(lanes.begin(), lanes.begin() + 1, lanes.end());
            for (auto& count : valid) count = (count + 1) % (width + 1);
        }
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < width; ++j)
                positions[b * width + j] =
                    valid[b] == 0 ? 0 : contexts[b] + std::min(j, valid[b] - 1);
        const auto q_bits = to_bf16_bits(q), k_bits = to_bf16_bits(k), v_bits = to_bf16_bits(v);
        dq.copy_from_host(q_bits.data(), q_bits.size() * 2);
        dk.copy_from_host(k_bits.data(), k_bits.size() * 2);
        dv.copy_from_host(v_bits.data(), v_bits.size() * 2);
        dp.copy_from_host(positions.data(), positions.size() * 4);
        dvalid.copy_from_host(valid.data(), batch * 4);
        dlanes.copy_from_host(lanes.data(), batch * 4);
        dout.fill(0xff);
        scratch.fill(phase ? 0xa5 : 0x5a);
        cuda_synchronize();
        const auto before = cache.snapshot_bytes();
        if (control) control->copy_from(cache);
        std::vector<double> reference(q.size(), 0.0);
        for (int b = 0; b < batch; ++b)
            if (valid[b]) {
                const std::vector<int> row_positions(positions.begin() + b * width,
                                                     positions.begin() + b * width + valid[b]);
                auto row_q = extract_request_columns(q, q_column_elements, width, b, valid[b]);
                auto row_k = extract_request_columns(k, kv_column_elements, width, b, valid[b]);
                auto row_v = extract_request_columns(v, kv_column_elements, width, b, valid[b]);
                append_cache(expected[lanes[b]], row_k, row_v, row_positions);
                insert_request_columns(ideal_attention(row_q, expected[lanes[b]], row_positions),
                                       q_column_elements, width, b, reference);
            }
        if (test_case.graph_replay && (phase == 0 || !graph_limits.empty())) {
            launch();
            cuda_synchronize(
                execution.stream()); // Warm lazy CUDA attributes; repeated append is idempotent.
            definition.capture(execution.stream(), launch);
            if (graph.ready())
                graph.update(definition);
            else
                graph.instantiate(definition);
            dout.fill(0xff);
            scratch.fill(0x5a);
            cuda_synchronize();
        }
        if (test_case.graph_replay)
            graph.launch(execution.stream());
        else
            launch();
        cuda_synchronize(execution.stream());
        const std::string label = std::string("causal batch ") + geometry.name + " " +
                                  cache_name(storage) + " W=" + std::to_string(width) +
                                  " B=" + std::to_string(batch) + " phase=" + std::to_string(phase);
        const auto output = copy_from_guarded<std::uint16_t>(dout, q.size());
        failures += verify_attention(label, bf16_bits_to_double(output), reference,
                                     attention_criterion(storage));
        failures += verify_invalid_columns_zero(label, output, geometry, width, valid);
        failures += cache.verify(label, expected);
        failures += cache.verify_untouched(label, before, positions, lanes, valid, width);
        if (control) {
            for (int b = 0; b < batch; ++b)
                if (valid[b]) {
                    Tensor ck = tk.slice(3, b, 1).slice(2, 0, valid[b]);
                    Tensor cv = tv.slice(3, b, 1).slice(2, 0, valid[b]);
                    Tensor cp = tp.slice(1, b, 1).slice(0, 0, valid[b]);
                    ops::kv_cache_append(ck, cv, cp, control->single_view(lanes[b]), execution);
                }
            cuda_synchronize(execution.stream());
            const auto actual = cache.snapshot_bytes(), standalone = control->snapshot_bytes();
            for (int plane = 0; plane < 4; ++plane)
                failures += verify_exact(
                    (label + " standalone parity plane=" + std::to_string(plane)).c_str(),
                    actual[plane], standalone[plane]);
        }
        failures += verify_input(label + " q unchanged", dq, q_bits) +
                    verify_input(label + " k unchanged", dk, k_bits) +
                    verify_input(label + " v unchanged", dv, v_bits);
        failures += verify_positions(label + " positions unchanged", dp, positions) +
                    verify_positions(label + " counts unchanged", dvalid, valid) +
                    verify_positions(label + " lanes unchanged", dlanes, lanes);
        failures += dout.verify_guards(label) + scratch.verify_guards(label);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) {
            std::cerr << label << ": workspace mismatch\n";
            ++failures;
        }
    }
    return failures;
}

int verify_workspace_capacity_contract(CudaExecutionView execution, KvCacheStorage storage) {
    int failures = 0;
    constexpr ops::CausalAttentionExecutionEnvelope envelope{1, 1025};
    constexpr ops::AttentionHeadGeometry geometry{kHeadDim, 16, 2};
    const std::size_t interval = ops::causal_softmax_attention_workspace_capacity_bytes(
        geometry, storage, envelope, 1, 1, 17, execution.multiprocessor_count());
    std::size_t witness = 0;
    for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
        witness = std::max(witness, ops::causal_softmax_attention_workspace_capacity_bytes(
                                        geometry, storage, envelope, 1, tokens, tokens,
                                        execution.multiprocessor_count()));
    }
    if (interval != witness) {
        std::cerr << "causal_softmax_attention interval capacity has no exact route witness\n";
        ++failures;
    }

    {
        // Prefill split counts can decrease as query width grows. The interval
        // query must still cover every supported point, including before a drop.
        for (const auto& item : kGeometries) {
            constexpr ops::CausalAttentionExecutionEnvelope prefill_envelope{1, 131072};
            std::size_t largest = 0;
            for (int width = 17; width <= 1025; ++width)
                largest = std::max(largest, ops::causal_softmax_attention_workspace_capacity_bytes(
                                                op_geometry(item), storage, prefill_envelope, 1,
                                                width, width, execution.multiprocessor_count()));
            const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
                op_geometry(item), storage, prefill_envelope, 1, 17, 1025,
                execution.multiprocessor_count());
            if (capacity != largest) {
                std::cerr << "causal_softmax_attention prefill interval capacity mismatch\n";
                ++failures;
            }
        }
    }

    // Device capacity is a planning input. Host-only queries cover batched decode
    // and single-request prefill on several SM counts without launching synthetic devices.
    for (const int sm_count : {96, 160, 170, 192}) {
        constexpr ops::AttentionHeadGeometry capacity_geometry{kHeadDim, 24, 4};
        constexpr ops::CausalAttentionExecutionEnvelope capacity_envelope{1, 131072};
        for (const int batch : {1, 3}) {
            const int max_width = batch == 1 ? 257 : 16;
            std::size_t largest = 0;
            for (int width = 1; width <= max_width; ++width)
                largest = std::max(largest, ops::causal_softmax_attention_workspace_capacity_bytes(
                                                capacity_geometry, storage, capacity_envelope,
                                                batch, width, width, sm_count));
            const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
                capacity_geometry, storage, capacity_envelope, batch, 1, max_width, sm_count);
            if (capacity != largest) {
                std::cerr << "causal_softmax_attention capacity interval mismatch for SM="
                          << sm_count << " B=" << batch << '\n';
                ++failures;
            }
        }
    }
    for (const int sm_count : {0, -1}) {
        try {
            (void)ops::causal_softmax_attention_workspace_capacity_bytes(
                geometry, storage, envelope, 1, 1, 17, sm_count);
            std::cerr << "causal_softmax_attention accepted a nonpositive SM count\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }

    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            {kHeadDim, 16, 2}, storage, {1, ops::kCausalAttentionMaximumVisibleKeys}, 1, 1, 1,
            execution.multiprocessor_count());
    } catch (const std::invalid_argument&) {
        std::cerr << "causal_softmax_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            {kHeadDim, 16, 2}, storage, {1, ops::kCausalAttentionMaximumVisibleKeys + 1}, 1, 1, 1,
            execution.multiprocessor_count());
        std::cerr << "causal_softmax_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}
} // namespace ninfer::test::causal_attention
