#include "fixture.h"
#include "core/platform.h"
#include "ops/ragged_fixture.h"
#include "ops/graph_patch_check.h"

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

ContextBucketRange replay_range(ContextBucket bucket, std::span<const ContextBucket> sequence) {
    if (sequence.empty()) return bucket;
    const auto [first, last] = std::minmax_element(sequence.begin(), sequence.end());
    return {*first, *last};
}

template <class Launch>
std::vector<CapturedKernelSequence>
capture_bucket_sequence(CudaExecutionView execution, const Geometry& geometry,
                        KvCacheStorage storage, int width, int batch, bool masked,
                        std::span<const ContextBucket> buckets, std::size_t expected_nodes,
                        Launch&& launch, int& failures) {
    std::vector<CapturedKernelSequence> captures;
    std::vector<ops::OpGraphProfile> profiles;
    for (const auto bucket : buckets) {
        const auto profile = ops::causal_softmax_attention_graph_profile(
            op_geometry(geometry), storage, bucket, batch, width, execution.multiprocessor_count(),
            true, masked);
        captures.emplace_back(execution, [&](auto cuda) { launch(bucket, cuda); });
        if (captures.back().params().size() != expected_nodes) {
            std::cerr << "causal attention unexpected kernel node count for "
                      << ops::context_bucket_name(bucket) << '\n';
            ++failures;
        }
        if (batch == 8 && width == 1 && bucket == ContextBucket::K8 &&
            buckets.front() == ContextBucket::K2 &&
            (profile == profiles.front()) != (storage == KvCacheStorage::BFloat16)) {
            std::cerr << "causal attention unexpected K2/K8 decode profile equality\n";
            ++failures;
        }
        for (std::size_t i = 0; i < profiles.size(); ++i)
            if (profile == profiles[i] && !same_kernel_launches(captures[i], captures.back())) {
                std::cerr << "causal attention equal graph profiles have different launches\n";
                ++failures;
            }
        profiles.push_back(profile);
    }
    return captures;
}

std::size_t causal_verify_nodes(KvCacheStorage storage, int width) {
    return storage != KvCacheStorage::BFloat16 && width > 8 ? 3 : 2;
}

} // namespace

int run_append_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const AttentionCase& test_case, MappingPattern mapping,
                    std::span<const int> oracle_queries,
                    std::span<const ContextBucket> graph_buckets) {
    const std::int32_t total          = test_case.base + test_case.tokens;
    const std::int32_t table_capacity = std::max(total + 3, test_case.table_capacity);
    const std::size_t q_elements      = static_cast<std::size_t>(kHeadDim) *
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

    const HostCache initial =
        make_cache(geometry, storage, total + 3, test_case.seed + 10u, amplitude);
    HostCache expected = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference =
        ideal_attention(select_query_columns(q, kHeadDim * geometry.q_heads, oracle_queries),
                        expected, select_query_columns(positions, 1, oracle_queries));
    DeviceCache cache(initial, mapping, table_capacity);

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
        op_geometry(geometry), storage, replay_range(test_case.bucket, graph_buckets), 1,
        test_case.tokens, test_case.tokens, execution.multiprocessor_count());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    MemoryArena workspace(workspace_buffer.span());

    int graph_failures = 0;
    const auto launch  = [&](ContextBucket bucket, CudaExecutionView cuda) {
        ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, ttable_row, op_geometry(geometry),
                                       kAttentionScale, cache.batch_view(), bucket, workspace, tout,
                                       cuda);
    };
    if (graph_buckets.empty()) {
        launch_attention_case([&](auto cuda) { launch(test_case.bucket, cuda); },
                              test_case.graph_replay);
    } else {
        const auto captures = capture_bucket_sequence(
            execution, geometry, storage, test_case.tokens, 1, false, graph_buckets,
            causal_verify_nodes(storage, test_case.tokens), launch, graph_failures);
        FlatGraphPatchHarness patched(captures.front()), updated(captures.front());
        const auto verify = [&](std::string_view mode) {
            cuda_synchronize(execution.stream());
            const auto actual = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
            graph_failures += verify_attention(
                std::string(cache_name(storage)) + " attention " + std::string(mode),
                bf16_bits_to_double(
                    select_query_columns(actual, kHeadDim * geometry.q_heads, oracle_queries)),
                reference, attention_criterion(storage));
        };
        for (const auto& capture : captures) {
            dout.fill(0xff);
            cuda_synchronize();
            patched.patch_and_launch(capture, execution);
            verify("node patch");
            dout.fill(0xff);
            cuda_synchronize();
            updated.update_whole_graph_and_launch(capture, execution);
            verify("whole-graph update");
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
        DeviceCache standalone(initial, mapping, table_capacity);
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
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes ||
        (graph_buckets.empty() && workspace.peak_used() != workspace_bytes)) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_cached_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const AttentionCase& test_case, MappingPattern mapping,
                    std::span<const int> oracle_queries) {
    const std::int32_t total          = test_case.base + test_case.tokens;
    const std::int32_t table_capacity = std::max(total + 3, test_case.table_capacity);
    const std::size_t q_elements      = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const float amplitude = test_case.qk_amplitude;
    std::vector<float> q  = make_bf16_values(q_elements, test_case.seed, -amplitude, amplitude);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host =
        make_cache(geometry, storage, total + 3, test_case.seed + 10u, amplitude);
    const std::vector<double> reference =
        ideal_attention(select_query_columns(q, kHeadDim * geometry.q_heads, oracle_queries),
                        cache_host, select_query_columns(positions, 1, oracle_queries));
    DeviceCache cache(cache_host, mapping, table_capacity);

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
        op_geometry(geometry), storage, test_case.bucket, 1, test_case.tokens, test_case.tokens,
        execution.multiprocessor_count());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    MemoryArena workspace(workspace_buffer.span());

    launch_attention_case(
        [&](CudaExecutionView cuda) {
            ops::causal_softmax_attention_cached(tq, tp, op_geometry(geometry), kAttentionScale,
                                                 cache.view(), test_case.bucket, workspace, tout,
                                                 cuda);
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
                   const BatchAttentionCase& test_case,
                   std::optional<ContextBucket> bucket_override,
                   std::span<const ContextBucket> graph_buckets) {
    const int batch = test_case.contexts.size(), width = test_case.width;
    const int pool_rows = std::max(
        1, *std::max_element(test_case.table_rows.begin(), test_case.table_rows.end()) + 1);
    const int phases = !graph_buckets.empty()   ? int(graph_buckets.size())
                       : test_case.graph_replay ? 3
                                                : 1;

    struct Phase {
        const std::vector<int>& contexts;
        std::vector<int> valid;
        std::vector<int> lanes;
    };

    std::vector<Phase> metadata;
    std::vector<int> row_contexts(pool_rows);
    auto next_valid = test_case.valid_columns, next_lanes = test_case.table_rows;
    for (int phase = 0; phase < phases; ++phase) {
        const auto& contexts = test_case.replay_contexts.empty()
                                   ? test_case.contexts
                                   : test_case.replay_contexts.at(phase);
        metadata.push_back({contexts, next_valid, next_lanes});
        for (int b = 0; b < batch; ++b)
            if (next_valid[b] > 0) {
                auto& context = row_contexts[next_lanes[b]];
                context       = std::max(context, contexts[b] + next_valid[b]);
            }
        std::rotate(next_lanes.begin(), next_lanes.begin() + 1, next_lanes.end());
        for (auto& count : next_valid) count = (count + 1) % (width + 1);
    }
    const int maximum_visible = *std::max_element(row_contexts.begin(), row_contexts.end());
    const auto bucket         = bucket_override.value_or(select_context_bucket(maximum_visible));
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
            make_cache(geometry, storage, row_contexts[row] + 3, test_case.seed + 20u + 3u * row));
    auto expected = initial;
    BatchDeviceCache cache(initial, test_case.mapping, maximum_visible);
    std::optional<BatchDeviceCache> control;
    const bool replay = test_case.graph_replay || !graph_buckets.empty();
    if (replay) control.emplace(initial, test_case.mapping, maximum_visible);
    GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), dp(columns * 4),
        dvalid(batch * 4), dlanes(batch * 4), dout(q.size() * 2);
    Tensor tq(dq.span(), DType::BF16, {kHeadDim, geometry.q_heads, width, batch});
    Tensor tk(dk.span(), DType::BF16, {kHeadDim, geometry.kv_heads, width, batch}),
        tv(dv.span(), DType::BF16, {kHeadDim, geometry.kv_heads, width, batch});
    Tensor tp(dp.span(), DType::I32, {width, batch}), tvalid(dvalid.span(), DType::I32, {batch}),
        tlanes(dlanes.span(), DType::I32, {batch});
    Tensor tout(dout.span(), DType::BF16, {kHeadDim, geometry.q_heads, width, batch});
    const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, replay_range(bucket, graph_buckets), batch, width, width,
        execution.multiprocessor_count());
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(scratch.span());
    const bool masked =
        replay || std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                              [&](int count) { return count != width; });
    const auto launch = [&](ContextBucket launch_bucket, CudaExecutionView cuda) {
        ops::causal_softmax_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, tlanes,
                                      op_geometry(geometry), kAttentionScale, cache.view(),
                                      launch_bucket, workspace, tout, cuda);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    std::vector<CapturedKernelSequence> captures;
    std::optional<FlatGraphPatchHarness> patched, updated;
    if (!graph_buckets.empty()) {
        captures = capture_bucket_sequence(execution, geometry, storage, width, batch, masked,
                                           graph_buckets, causal_verify_nodes(storage, width),
                                           launch, failures);
        patched.emplace(captures.front());
        updated.emplace(captures.front());
    } else if (test_case.graph_replay) {
        definition.capture(execution.stream(), [&] { launch(bucket, execution); });
        graph.instantiate(definition);
    }
    std::vector<int> positions(columns);
    for (int phase = 0; phase < phases; ++phase) {
        const auto& [contexts, valid, lanes] = metadata[phase];
        if (phase) {
            for (auto& x : q) x *= -0.5f;
            for (auto& x : k) x *= -0.5f;
            for (auto& x : v) x *= -0.5f;
            round_to_bf16(q);
            round_to_bf16(k);
            round_to_bf16(v);
        }
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < width; ++j)
                positions[b * width + j] =
                    valid[b] == 0 ? -1 : contexts[b] + std::min(j, valid[b] - 1);
        const auto q_bits = to_bf16_bits(q), k_bits = to_bf16_bits(k), v_bits = to_bf16_bits(v);
        dq.copy_from_host(q_bits.data(), q_bits.size() * 2);
        dk.copy_from_host(k_bits.data(), k_bits.size() * 2);
        dv.copy_from_host(v_bits.data(), v_bits.size() * 2);
        dp.copy_from_host(positions.data(), positions.size() * 4);
        dvalid.copy_from_host(valid.data(), batch * 4);
        auto selected_lanes = lanes;
        for (int b = 0; b < batch; ++b)
            if (valid[b] == 0) selected_lanes[b] = -1;
        dlanes.copy_from_host(selected_lanes.data(), batch * 4);
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
        const std::string label = std::string("causal batch ") + geometry.name + " " +
                                  cache_name(storage) + " W=" + std::to_string(width) +
                                  " B=" + std::to_string(batch) + " phase=" + std::to_string(phase);
        const auto verify_output = [&](const std::string& check) {
            cuda_synchronize(execution.stream());
            const auto output = copy_from_guarded<std::uint16_t>(dout, q.size());
            return verify_attention(check, bf16_bits_to_double(output), reference,
                                    attention_criterion(storage)) +
                   verify_invalid_columns_zero(check, output, geometry, width, valid);
        };
        if (patched) {
            patched->patch_and_launch(captures[phase], execution);
            failures += verify_output(label + " node patch");
            cache.copy_from(*control);
            dout.fill(0xff);
            scratch.fill(0xa7);
            cuda_synchronize();
            updated->update_whole_graph_and_launch(captures[phase], execution);
            failures += verify_output(label + " whole-graph update");
        } else {
            if (test_case.graph_replay)
                graph.launch(execution.stream());
            else
                launch(bucket, execution);
            failures += verify_output(label);
        }
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
                    verify_positions(label + " lanes unchanged", dlanes, selected_lanes);
        failures += dout.verify_guards(label) + scratch.verify_guards(label);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) {
            std::cerr << label << ": workspace mismatch\n";
            ++failures;
        }
    }
    return failures;
}

int run_ragged_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const RaggedAttentionCase& test_case) {
    RaggedFixture ragged(test_case.lengths);
    const int n = ragged.tokens(), rows = ragged.segments();
    const int width       = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int table_slots = rows + 2;
    const std::size_t qw = kHeadDim * geometry.q_heads, kvw = kHeadDim * geometry.kv_heads;
    std::vector<int> table_rows(rows), frontiers            = test_case.histories;
    if (frontiers.empty()) frontiers.assign(rows, 0);
    for (int s = 0; s < rows; ++s) table_rows[s] = (rows + 1 - s);
    table_rows.back() = 0; // Sparse, reversed, with interior holes.

    std::vector<int> capacity(table_slots, std::max(1, test_case.shared_prefix));
    for (int s = 0; s < rows; ++s) capacity[table_rows[s]] = frontiers[s] + ragged.lengths[s] + 1;
    if (test_case.replay) {
        const int largest = *std::max_element(capacity.begin(), capacity.end());
        std::fill(capacity.begin(), capacity.end(), largest + kPagedKVPageSize);
    }
    if (test_case.shared_prefix) {
        const int largest = *std::max_element(capacity.begin(), capacity.end());
        std::fill(capacity.begin(), capacity.end(), largest);
    }
    std::vector<HostCache> initial;
    for (int i = 0; i < table_slots; ++i)
        initial.push_back(
            make_cache(geometry, storage, capacity[i], 393 + (test_case.shared_prefix ? 0 : i)));
    BatchDeviceCache cache(initial, MappingPattern::Fragmented, 0, test_case.shared_prefix);
    const auto device         = execution.device();
    const std::size_t q_bytes = n * qw * 2, kv_bytes = n * kvw * 2;
    const std::size_t positions_bytes  = std::size_t(n) * sizeof(int);
    const std::size_t table_rows_bytes = std::size_t(rows) * sizeof(int);
    const std::size_t offset           = test_case.offset_bf16 ? sizeof(std::uint16_t) : 0;
    const auto bytes = ops::causal_softmax_attention_ragged_workspace_capacity_bytes(
        op_geometry(geometry), storage, test_case.bucket, n, n, rows, rows,
        execution.multiprocessor_count());
    const auto prefix_capacity = test_case.arena_operands
                                     ? 2 * (q_bytes + kv_bytes) +
                                           2 * (positions_bytes + table_rows_bytes) + sizeof(int) +
                                           9 * 256
                                     : 0;
    GuardedDeviceBuffer scratch(prefix_capacity + bytes);
    MemoryArena arena(scratch.span());
    const auto separate = [&](std::size_t size) { return test_case.arena_operands ? 0 : size; };
    DeviceBuffer dq(device, separate(q_bytes + offset)), dk(device, separate(kv_bytes + offset)),
        dv(device, separate(kv_bytes + offset)), dp(device, separate(positions_bytes)),
        dr(device, separate(table_rows_bytes));
    GuardedDeviceBuffer output(separate(q_bytes), 256 + offset);
    const auto operand = [&](MemorySpan standalone, std::size_t size, std::size_t start = 0) {
        if (test_case.arena_operands) return arena.alloc_bytes(size + start).subspan(start);
        return standalone.subspan(start);
    };
    const Tensor tq(operand(dq.span(), q_bytes, offset), DType::BF16, {256, geometry.q_heads, n});
    const Tensor tk(operand(dk.span(), kv_bytes, offset), DType::BF16, {256, geometry.kv_heads, n}),
        tv(operand(dv.span(), kv_bytes, offset), DType::BF16, {256, geometry.kv_heads, n});
    const Tensor tp(operand(dp.span(), positions_bytes), DType::I32, {n}),
        tr(operand(dr.span(), table_rows_bytes), DType::I32, {rows});
    Tensor out(operand(output.span(), q_bytes), DType::BF16, {256, geometry.q_heads, n});
    const auto upload = [](const Tensor& dst, const void* src) {
        cuda_check(cudaMemcpy(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice), "ragged upload");
    };
    if (test_case.arena_operands) {
        const auto metadata = [&](const std::vector<int>& host) {
            const auto tensor = arena.alloc(DType::I32, {static_cast<int>(host.size())});
            upload(tensor, host.data());
            return tensor;
        };
        ragged.view = {metadata(ragged.prefix), metadata(ragged.lengths), metadata(ragged.mapping),
                       ragged.prefix};
    }
    const auto prefix_used = arena.used();
    const auto launch      = [&] {
        ops::causal_softmax_attention(tq, tk, tv, ragged.view, tp, tr, op_geometry(geometry),
                                           kAttentionScale, cache.view(), test_case.bucket, arena, out,
                                           execution);
    };

    DecodeGraphDefinition graph;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int replay = 0; replay < (test_case.replay ? 2 : 1); ++replay) {
        const auto label = std::string("ragged ") + test_case.name + " " + geometry.name + " " +
                           cache_name(storage) + " replay=" + std::to_string(replay);
        if (replay) {
            std::reverse(table_rows.begin(), table_rows.end());
            for (int s = 0; s < rows; ++s) frontiers[s] = (frontiers[s] + 63 + s % 4) % 79;
        }
        const unsigned seed = 325 + 100 * replay;
        const auto q        = make_bf16_values(n * qw, seed, -.25f, .25f);
        auto k              = make_bf16_values(n * kvw, seed + 2, -.25f, .25f);
        auto v              = make_bf16_values(n * kvw, seed + 4, -1.f, 1.f);
        // Adjacent packed rows have sharply different values, including at partial Q-tile tails.
        for (int s = 0; s < rows; ++s) {
            const int begin  = ragged.prefix[s];
            const float sign = (s % 2 == 0) ? 1.f : -1.f;
            for (std::size_t d = 0; d < kvw; ++d) {
                k[begin * kvw + d] = sign * 0.5f;
                v[begin * kvw + d] = sign * (d % 3 == 0 ? 1.f : 0.375f);
            }
        }
        std::vector<int> positions(n);
        for (int s = 0; s < rows; ++s)
            std::iota(positions.begin() + ragged.prefix[s],
                      positions.begin() + ragged.prefix[s + 1], frontiers[s]);
        const auto upload_bf16 = [&](const Tensor& destination, const std::vector<float>& values) {
            const auto bits = to_bf16_bits(values);
            upload(destination, bits.data());
        };
        upload_bf16(tq, q);
        upload_bf16(tk, k);
        upload_bf16(tv, v);
        upload(tp, positions.data());
        upload(tr, table_rows.data());
        cuda_check(cudaMemsetAsync(out.data, 0xff, out.bytes(), execution.stream()),
                   "initialize ragged output");
        const auto before = cache.snapshot_bytes();
        if (test_case.replay) {
            if (replay == 0) {
                graph.capture(execution.stream(), launch);
                executable.instantiate(graph);
            }
            executable.launch(execution.stream());
        } else {
            launch();
        }
        cuda_synchronize(execution.stream());
        const auto actual = from_device_bf16(out.data, n * qw);
        if (std::any_of(actual.begin(), actual.end(),
                        [](double value) { return !std::isfinite(value); })) {
            std::cerr << label << ": packed output contains an unwritten or non-finite element\n";
            ++failures;
        }

        // Keep previous replay writes in the oracle. Only this replay's append range may change.
        for (int s = 0; s < rows; ++s) {
            const int begin = ragged.prefix[s], len = ragged.lengths[s];
            const auto columns = [&](const auto& data, std::size_t stride) {
                using T = typename std::decay_t<decltype(data)>::value_type;
                return std::vector<T>(data.begin() + begin * stride,
                                      data.begin() + (begin + len) * stride);
            };
            const auto pos = columns(positions, 1);
            auto& history  = initial[table_rows[s]];
            append_cache(history, columns(k, kvw), columns(v, kvw), pos);
            std::vector<int> queries;
            if (len > 96 || frontiers[s] >= 1024) {
                queries = {0, 1, 63, 64, 127, 128, 255, 256, len / 2, len - 1};
                std::erase_if(queries, [=](int t) { return t >= len; });
                std::sort(queries.begin(), queries.end());
                queries.erase(std::unique(queries.begin(), queries.end()), queries.end());
            }
            const auto expected = ideal_attention(select_query_columns(columns(q, qw), qw, queries),
                                                  history, select_query_columns(pos, 1, queries));
            failures += verify_attention(label + " segment " + std::to_string(s),
                                         select_query_columns(columns(actual, qw), qw, queries),
                                         expected, attention_criterion(storage));
        }
        failures += cache.verify(label, initial);
        failures += cache.verify_untouched(label, before, ragged.unpack(positions, width, 1),
                                           table_rows, ragged.lengths, width);
        failures += scratch.verify_guards(label) + output.verify_guards(label);
    }
    const auto scratch_begin = (prefix_used + 255) / 256 * 256;
    if (arena.used() != prefix_used || arena.peak_used() > scratch_begin + bytes) {
        std::cerr << test_case.name << ": ragged workspace exceeded its envelope\n";
        ++failures;
    }
    return failures;
}

int verify_ragged_workspace_capacity(CudaExecutionView execution, const Geometry& geometry,
                                     KvCacheStorage storage) {
    constexpr int limit = 263;
    constexpr ContextBucketRange buckets{ContextBucket::K2, ContextBucket::K8};
    const auto dev = execution.device();
    const auto g   = op_geometry(geometry);
    const int sms  = execution.multiprocessor_count();
    DeviceBuffer q(dev, std::size_t(limit) * 256 * g.query_heads * 2);
    DeviceBuffer k(dev, std::size_t(limit) * 256 * g.kv_heads * 2);
    DeviceBuffer v(dev, k.size()), out_buf(dev, q.size());
    auto positions  = to_device(std::vector<int>(limit));
    auto table_rows = to_device(std::vector<int>{0, 1, 2});
    DeviceBuffer cu(dev, 4 * sizeof(int)), lengths(dev, 3 * sizeof(int)),
        mapping(dev, limit * sizeof(int));
    const std::vector<HostCache> initial(3, make_cache(geometry, storage, limit, 823));
    BatchDeviceCache cache(initial, MappingPattern::Identity, limit);
    int failures = 0;
    for (const auto& range :
         {std::array{1, 20, 1, 3}, std::array{31, 40, 2, 3}, std::array{18, 21, 2, 2},
          std::array{119, 132, 2, 2}, std::array{255, limit, 2, 2}}) {
        const auto [first_t, last_t, first_s, last_s] = range;
        const auto capacity = ops::causal_softmax_attention_ragged_workspace_capacity_bytes(
            g, storage, buckets, first_t, last_t, first_s, last_s, sms);
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena arena(scratch.span());
        std::vector<int> partition;
        DecodeGraphDefinition graph;
        const auto capture = [&](int total, ContextBucket bucket) {
            const int rows = partition.size();
            std::vector<int> prefix(rows + 1), segment_lengths(rows), token_to_seq(total);
            ops::populate_ragged_host_metadata(partition, prefix, segment_lengths, token_to_seq);
            cu.copy_from_host(prefix.data(), prefix.size() * sizeof(int));
            lengths.copy_from_host(segment_lengths.data(), segment_lengths.size() * sizeof(int));
            mapping.copy_from_host(token_to_seq.data(), token_to_seq.size() * sizeof(int));
            std::vector<int> consecutive(total);
            for (int s = 0; s < rows; ++s)
                std::iota(consecutive.begin() + prefix[s], consecutive.begin() + prefix[s + 1], 0);
            positions.copy_from_host(consecutive.data(), consecutive.size() * sizeof(int));
            // These captures witness host allocation without duplicating the numerical cases.
            const ops::RaggedSequenceView ragged{Tensor(cu.span(), DType::I32, {rows + 1}),
                                                 Tensor(lengths.span(), DType::I32, {rows}),
                                                 Tensor(mapping.span(), DType::I32, {total}),
                                                 prefix};
            Tensor out(out_buf.span(), DType::BF16, {256, g.query_heads, total});
            graph.capture(execution.stream(), [&] {
                ops::causal_softmax_attention(
                    Tensor(q.span(), DType::BF16, {256, g.query_heads, total}),
                    Tensor(k.span(), DType::BF16, {256, g.kv_heads, total}),
                    Tensor(v.span(), DType::BF16, {256, g.kv_heads, total}), ragged,
                    Tensor(positions.span(), DType::I32, {total}),
                    Tensor(table_rows.span(), DType::I32, {rows}), g, kAttentionScale, cache.view(),
                    bucket, arena, out, execution);
            });
        };
        const auto enumerate = [&](auto&& self, int total, int remaining, int rows, int minimum,
                                   ContextBucket bucket) -> void {
            if (rows == 0) {
                if (remaining == 0) capture(total, bucket);
                return;
            }
            // Permutations have the same allocation, so enumerate sorted partitions.
            for (int len = minimum; len <= remaining / rows; ++len) {
                partition.push_back(len);
                self(self, total, remaining - len, rows - 1, len, bucket);
                partition.pop_back();
            }
        };
        buckets.for_each([&](ContextBucket bucket) {
            for (int total = first_t; total <= last_t; ++total)
                for (int rows = first_s; rows <= last_s; ++rows)
                    enumerate(enumerate, total, total, rows, 1, bucket);
        });
        if (arena.used() || arena.peak_used() != capacity) {
            std::cerr << geometry.name << ' ' << cache_name(storage) << ": ragged capacity "
                      << capacity << " vs witnessed " << arena.peak_used() << '\n';
            ++failures;
        }
    }
    // A packed total above INT_MAX - 1 is possible even with fewer than INT_MAX rows.
    // With one token per row the launch slices share a single metadata allocation.
    const int largest = std::numeric_limits<int>::max();
    const auto huge   = ops::causal_softmax_attention_ragged_workspace_capacity_bytes(
        g, storage, ContextBucket::K2, largest - 1, largest - 1, largest - 1, largest - 1, sms);
    constexpr int full_rows = 65535;
    const int final_rows    = (largest - 1) % full_rows;
    // At width one the only additional allocation is an I32 segment index. An exact
    // full-row replay-sized envelope witnesses the full slice's high water.
    const auto exact = [&](int rows) {
        return rows <= 1 ? std::size_t{0}
                         : ops::causal_softmax_attention_ragged_workspace_capacity_bytes(
                               g, storage, ContextBucket::K2, rows, rows, rows, rows, sms);
    };
    if (huge != std::max(exact(full_rows), exact(final_rows))) {
        std::cerr << "ragged causal workspace overflow or incorrect slicing\n";
        ++failures;
    }
    return failures;
}

int verify_ragged_workspace_synthetic() {
    constexpr ops::AttentionHeadGeometry geometry{256, 24, 4};
    constexpr auto storage = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    constexpr auto bucket  = ContextBucket::K32;
    int failures           = 0;
    const auto check       = [&](int sms, std::array<int, 4> envelope, std::size_t witnessed) {
        const auto [first_n, last_n, first_s, last_s] = envelope;
        const auto actual = ops::causal_softmax_attention_ragged_workspace_capacity_bytes(
            geometry, storage, bucket, first_n, last_n, first_s, last_s, sms);
        if (actual != witnessed) {
            std::cerr << "FP8 ragged workspace at " << sms << " SMs, N=[" << first_n << ", "
                      << last_n << "], S=[" << first_s << ", " << last_s << "]: " << actual
                      << " bytes vs witnessed " << witnessed << '\n';
            ++failures;
        }
    };
    // On RTX, twelve 256-token segments use 24 tiles and four splits at 70 SMs.
    // The Spark cap selects direct output here. Its high water is instead the
    // dense fallback for a 3061-token segment and eleven one-token segments.
    check(70, {3072, 3072, 12, 12}, NINFER_TARGET_SPARK ? 303259520 : 304349184);

    const auto align256 = [](std::size_t size) { return (size + 255) / 256 * 256; };
    for (const int sms : {48, 70, 96, 108, 132, 160, 170, 192, std::numeric_limits<int>::max()}) {
        const auto dense = [&](int width, int rows) {
            return ops::causal_softmax_attention_workspace_capacity_bytes(geometry, storage, bucket,
                                                                          rows, width, width, sms);
        };
        for (const auto envelope : {std::array{17, 24, 1, 3}, std::array{125, 132, 2, 4},
                                    std::array{252, 259, 2, 3}, std::array{65, 70, 65, 67}}) {
            const auto [first_n, last_n, first_s, last_s] = envelope;
            std::vector<int> part;
            std::size_t witnessed = 0;
            const auto measure    = [&] {
                const auto longs      = std::upper_bound(part.begin(), part.end(), 16);
                const int short_count = longs - part.begin(), long_count = part.end() - longs;
                std::size_t short_bytes = 0, long_bytes = 0;
                if (short_count) {
                    const auto partials = dense(longs[-1], short_count);
                    const auto map      = short_count > 64 ? short_count * sizeof(int) : 0;
                    short_bytes         = partials ? align256(map) + partials : map;
                }
                if (long_count == 1) long_bytes = dense(part.back(), 1);
                if (long_count >= 2) {
                    const int tile_rows = part.back() <= 64 ? 64 : 128;
                    const auto ctas =
                        24LL * std::accumulate(longs, part.end(), 0, [&](int count, int len) {
                            return count + (len + tile_rows - 1) / tile_rows;
                        });
                    const int waves_for_direct = tile_rows == 64 ? 4
                                                    : NINFER_TARGET_SPARK
                                                        ? 8
                                                        : std::numeric_limits<int>::max();
                    const int limit            = ctas >= waves_for_direct * std::int64_t(sms) ? 1
                                                    : ctas >= sms                                ? 4
                                                                  : 8;
                    int splits                 = 1;
                    for (int s = 2; s <= limit; ++s) {
                        const auto waves = [&](int count) { return (ctas * count - 1) / sms + 1; };
                        if (waves(s) * splits < waves(splits) * s) splits = s;
                    }
                    const auto map = long_count > 64 ? long_count * 3 * sizeof(int) : 0;
                    const auto stats =
                        24 * sizeof(float) * splits * std::accumulate(longs, part.end(), 0);
                    long_bytes =
                        splits == 1 ? map : align256(map) + 256 * stats + align256(stats) + stats;
                }
                witnessed = std::max({witnessed, short_bytes, long_bytes});
            };
            const auto enumerate = [&](auto&& self, int remaining, int rows, int min_len) -> void {
                if (rows == 0) {
                    if (remaining == 0) measure();
                    return;
                }
                for (int len = min_len; len <= remaining / rows; ++len) {
                    part.push_back(len);
                    self(self, remaining - len, rows - 1, len);
                    part.pop_back();
                }
            };
            for (int n = first_n; n <= last_n; ++n)
                for (int rows = first_s; rows <= last_s; ++rows) enumerate(enumerate, n, rows, 1);
            check(sms, envelope, witnessed);
        }
    }
    return failures;
}

int verify_workspace_capacity_contract(CudaExecutionView execution, KvCacheStorage storage) {
    int failures = 0;
    constexpr ContextBucketRange buckets{ContextBucket::K2, ContextBucket::K256};
    constexpr ops::AttentionHeadGeometry geometry{kHeadDim, 16, 2};
    const std::size_t interval = ops::causal_softmax_attention_workspace_capacity_bytes(
        geometry, storage, buckets, 1, 1, 17, execution.multiprocessor_count());
    std::size_t witness = 0;
    buckets.for_each([&](ContextBucket bucket) {
        for (int tokens = 1; tokens <= 17; ++tokens)
            witness = std::max(witness, ops::causal_softmax_attention_workspace_capacity_bytes(
                                            geometry, storage, bucket, 1, tokens, tokens,
                                            execution.multiprocessor_count()));
    });
    if (interval != witness) {
        std::cerr << "causal_softmax_attention interval capacity has no exact route witness\n";
        ++failures;
    }

    {
        // Prefill split counts can decrease as query width grows. The interval
        // query must still cover every supported point, including before a drop.
        for (const auto& item : kGeometries) {
            constexpr auto prefill_bucket = ContextBucket::K128;
            std::size_t largest           = 0;
            for (int width = 17; width <= 1025; ++width)
                largest = std::max(largest, ops::causal_softmax_attention_workspace_capacity_bytes(
                                                op_geometry(item), storage, prefill_bucket, 1,
                                                width, width, execution.multiprocessor_count()));
            const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
                op_geometry(item), storage, prefill_bucket, 1, 17, 1025,
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
        constexpr auto capacity_bucket = ContextBucket::K128;
        for (const int batch : {1, 3, 17, 65537}) {
            const int max_width = batch == 1 ? 257 : 16;
            std::size_t largest = 0;
            for (int width = 1; width <= max_width; ++width)
                largest = std::max(largest, ops::causal_softmax_attention_workspace_capacity_bytes(
                                                capacity_geometry, storage, capacity_bucket, batch,
                                                width, width, sm_count));
            const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
                capacity_geometry, storage, capacity_bucket, batch, 1, max_width, sm_count);
            if (capacity != largest) {
                std::cerr << "causal_softmax_attention capacity interval mismatch for SM="
                          << sm_count << " B=" << batch << '\n';
                ++failures;
            }
        }
    }
    const auto reject = [&](std::string_view label, auto call) {
        try {
            call();
            std::cerr << "causal attention accepted " << label << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    const int sms          = execution.multiprocessor_count();
    constexpr auto invalid = static_cast<ContextBucket>(255);
    for (const auto range :
         {ContextBucketRange{ContextBucket::K8, ContextBucket::K2},
          ContextBucketRange{invalid, ContextBucket::K8},
          ContextBucketRange{ContextBucket::K2, invalid}, ContextBucketRange{ContextBucket::K512},
          ContextBucketRange{ContextBucket::M1}}) {
        reject("invalid workspace bucket range", [&] {
            (void)ops::causal_softmax_attention_workspace_capacity_bytes(geometry, storage, range,
                                                                         1, 1, 17, sms);
        });
    }
    const auto profile = [&](ContextBucket bucket, int batch, int width, int sms,
                             bool append = true, bool masked = false) {
        return ops::causal_softmax_attention_graph_profile(geometry, storage, bucket, batch, width,
                                                           sms, append, masked);
    };
    for (const auto bucket : {ContextBucket::K512, ContextBucket::M1, invalid})
        reject("unsupported graph bucket", [&] { (void)profile(bucket, 1, 1, sms); });
    for (const int count : {0, -1}) {
        reject("nonpositive workspace SM count", [&] {
            (void)ops::causal_softmax_attention_workspace_capacity_bytes(geometry, storage, buckets,
                                                                         1, 1, 17, count);
        });
        reject("nonpositive graph SM count", [&] { (void)profile(buckets.first, 1, 1, count); });
    }
    reject("batched width above 16", [&] { (void)profile(buckets.first, 2, 17, sms); });
    reject("batched cached graph", [&] { (void)profile(buckets.first, 2, 1, sms, false); });
    reject("masked cached graph", [&] { (void)profile(buckets.first, 1, 1, sms, false, true); });
    reject("unsupported graph geometry", [&] {
        (void)ops::causal_softmax_attention_graph_profile({128, 16, 2}, storage, buckets.first, 1,
                                                          1, sms);
    });
    reject("unsupported graph storage", [&] {
        (void)ops::causal_softmax_attention_graph_profile(
            geometry, static_cast<KvCacheStorage>(255), buckets.first, 1, 1, sms);
    });

    constexpr int width = int(ops::context_bucket_tokens(ContextBucket::K2)) + 1;
    reject("unmasked graph width above the bucket",
           [&] { (void)profile(ContextBucket::K2, 1, width, sms); });
    // This width is legal with a short live prefix.
    (void)profile(ContextBucket::K2, 1, width, sms, true, true);

    const auto& item   = kGeometries[1];
    const auto initial = make_cache(item, storage, width, 1481u);
    DeviceCache cache(initial, MappingPattern::Identity, width);
    const auto q_bytes  = std::size_t(width) * kHeadDim * item.q_heads * 2;
    const auto kv_bytes = std::size_t(width) * kHeadDim * item.kv_heads * 2;
    GuardedDeviceBuffer dq(q_bytes), dk(kv_bytes), dv(kv_bytes), dout(q_bytes),
        dp(width * sizeof(int)), drow(sizeof(int));
    dq.fill(0);
    dk.fill(0);
    dv.fill(0);
    drow.fill(0);
    std::vector<int> positions(width);
    for (int i = 0; i < width; ++i) positions[i] = i;
    dp.copy_from_host(positions.data(), positions.size() * sizeof(int));
    Tensor tq(dq.span(), DType::BF16, {kHeadDim, item.q_heads, width});
    Tensor tk(dk.span(), DType::BF16, {kHeadDim, item.kv_heads, width});
    Tensor tv(dv.span(), DType::BF16, {kHeadDim, item.kv_heads, width});
    Tensor out(dout.span(), DType::BF16, {kHeadDim, item.q_heads, width});
    Tensor tp(dp.span(), DType::I32, {width}), row(drow.span(), DType::I32, {1});
    const auto bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        geometry, storage, buckets, 1, width, width, sms);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    MemoryArena workspace(scratch.span());
    for (const auto bucket : {ContextBucket::K2, ContextBucket::K512, ContextBucket::M1, invalid}) {
        reject("invalid append extent or bucket", [&] {
            ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, row, geometry, kAttentionScale,
                                          cache.batch_view(), bucket, workspace, out, execution);
            cuda_synchronize(execution.stream());
        });
        reject("invalid cached extent or bucket", [&] {
            ops::causal_softmax_attention_cached(tq, tp, geometry, kAttentionScale, cache.view(),
                                                 bucket, workspace, out, execution);
            cuda_synchronize(execution.stream());
        });
    }
    return failures;
}
} // namespace ninfer::test::causal_attention
