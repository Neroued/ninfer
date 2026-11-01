#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/transfer.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/kda_replay.h"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr float scale = 0.08838834764831845F;

std::vector<float> values(std::size_t n, unsigned seed, float amplitude = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

int replay_case(int layers, int heads, int width, int batch, bool alias_records, bool in_place,
                bool graph, int qheads = 0) {
    auto& context = cuda_environment().context;
    CudaStream stream(context);
    const auto cuda = context.execution_view(stream.stream());
    if (!qheads) qheads = heads;
    const int channels = (2 * qheads + heads) * 128, capacity = 8;
    StateImageBuilder sb;
    const auto guard        = sb.add(DType::I32, {64}, 1, "unrelated field");
    const auto state_layout = plan_linear_attention_state(
        sb, {static_cast<unsigned>(layers), channels, 3, heads, 128, 128, DType::BF16});
    const auto layout = sb.finish();
    std::vector<std::int64_t> sources(batch), destinations(batch);
    std::size_t cursor = 256;
    for (int b = 0; b < batch; ++b) {
        sources[b] = cursor;
        cursor += layout.image_bytes + (b + 1) * 256;
        destinations[b] = in_place ? sources[b] : static_cast<std::int64_t>(cursor);
        if (!in_place) cursor += layout.image_bytes + (b + 2) * 256;
    }
    DeviceBuffer arena(cuda.device(), cursor);
    CUDA_CHECK(cudaMemsetAsync(arena.data(), 0x5a, arena.size(), cuda.stream()));
    StateImageStorage storage(arena.span(), layout);
    LinearAttentionStateView states(storage, state_layout);
    auto source_buffer = to_device(sources), destination_buffer = to_device(destinations);
    Tensor source_table(source_buffer.span(), DType::I64, {batch});
    Tensor destination_table(destination_buffer.span(), DType::I64, {batch});
    std::vector<int> valid(batch), accepted(batch);
    for (int b = 0; b < batch; ++b) {
        valid[b]    = std::max(1, width - b % width);
        accepted[b] = batch == 1 ? valid[b] : (b == batch - 1 ? valid[b] : std::min(b, valid[b]));
    }
    auto valid_buffer = to_device(valid), accepted_buffer = to_device(accepted);
    Tensor valid_table(valid_buffer.span(), DType::I32, {batch});
    Tensor accepted_table(accepted_buffer.span(), DType::I32, {batch});
    LayoutBuilder rb;
    const auto rl =
        plan_kda_replay_records(rb, {layers, capacity, width, channels, qheads, heads, 128, 128});
    DeviceBuffer record_buffer(cuda.device(), rb.finish(256));
    CUDA_CHECK(cudaMemsetAsync(record_buffer.data(), 0x35, record_buffer.size(), cuda.stream()));
    KdaReplayRecords records(record_buffer.span(), rl);
    std::vector<std::vector<float>> expected(layers * batch);
    std::vector<std::vector<std::uint16_t>> expected_conv(layers * batch);
    int failures = 0;
    for (int l = 0; l < layers; ++l) {
        const std::size_t vectors  = std::size_t(128) * heads * width * batch;
        const std::size_t qvectors = std::size_t(128) * qheads * width * batch;
        auto qbuf                  = to_device_bf16(values(qvectors, 101 + l));
        auto kbuf                  = to_device_bf16(values(qvectors, 201 + l));
        auto vbuf                  = to_device_bf16(values(vectors, 301 + l));
        auto gbuf                  = to_device_bf16(values(vectors, 401 + l, 4));
        auto bbuf   = to_device_bf16(values(std::size_t(heads) * width * batch, 501 + l));
        auto abuf   = to_device_f32(values(heads, 601 + l, 0.5F));
        auto biases = values(128 * heads, 701 + l);
        biases.insert(biases.begin(), 0.0F);
        auto dbuf = to_device_f32(biases);
        DeviceBuffer output(cuda.device(), vectors * 2);
        Tensor q(qbuf.span(), DType::BF16, {128, qheads, width, batch});
        Tensor k(kbuf.span(), DType::BF16, {128, qheads, width, batch});
        Tensor v(vbuf.span(), DType::BF16, {128, heads, width, batch});
        Tensor g(gbuf.span(), DType::BF16, {128, heads, width, batch});
        Tensor beta(bbuf.span(), DType::BF16, {heads, width, batch});
        Tensor a(abuf.span(), DType::FP32, {heads}),
            dt(dbuf.span().subspan(sizeof(float)), DType::FP32, {128, heads});
        Tensor out(output.span(), DType::BF16, {128, heads, width, batch});
        auto rec               = records.layer(l, batch);
        const auto conv_values = values(std::size_t(channels) * width * batch, 801 + l);
        auto convbuf           = to_device_bf16(conv_values);
        copy_bytes_async({rec.conv.data, rec.conv.bytes(), cuda.device()}, convbuf.span(), cuda);
        if (alias_records) {
            copy_bytes_async({rec.key.data, rec.key.bytes(), cuda.device()}, kbuf.span(), cuda);
            copy_bytes_async({rec.value.data, rec.value.bytes(), cuda.device()}, vbuf.span(), cuda);
            k = rec.key;
            v = rec.value;
        }
        std::vector<std::vector<float>> initial(batch);
        for (int b = 0; b < batch; ++b) {
            initial[b]       = values(128 * 128 * heads, 901 + l * 11 + b, 0.02F);
            const auto field = states.recurrent_tensor(l, sources[b]);
            copy_bytes({field.data, field.bytes(), cuda.device()},
                       {initial[b].data(), field.bytes(), Device::cpu()}, cuda);
            const auto old = values(channels * 3, 1001 + l * 11 + b);
            std::vector<std::uint16_t> old_bits(old.size());
            std::transform(old.begin(), old.end(), old_bits.begin(), f32_to_bf16);
            const auto conv = states.conv_tensor(l, sources[b]);
            copy_bytes({conv.data, conv.bytes(), cuda.device()},
                       {old_bits.data(), conv.bytes(), Device::cpu()}, cuda);
            auto& tail = expected_conv[l * batch + b];
            tail.resize(channels * 3);
            for (int j = 0; j < 3; ++j)
                for (int c = 0; c < channels; ++c) {
                    const int t = accepted[b] - 3 + j;
                    tail[j * channels + c] =
                        t < 0
                            ? old_bits[(t + 3) * channels + c]
                            : f32_to_bf16(conv_values[(std::size_t(b) * width + t) * channels + c]);
                }
        }
        const auto launch = [&] {
            ops::kimi_delta_attention_replay_record(
                q, k, v, g, beta, a, dt, -5, scale, states.layer_view(l, source_table).recurrent,
                valid_table, {rec.key, rec.value, rec.log_decay, rec.beta}, out, cuda);
        };
        cuda_synchronize();
        launch();
        if (graph) {
            cudaGraph_t gr;
            cudaGraphExec_t ex;
            CUDA_CHECK(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
            launch();
            CUDA_CHECK(cudaStreamEndCapture(cuda.stream(), &gr));
            CUDA_CHECK(cudaGraphInstantiate(&ex, gr, 0));
            std::vector<int> one(batch, 1);
            valid_buffer.copy_from_host(one.data(), valid_buffer.size());
            CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            // Move the source record without changing the captured field views.
            for (int b = 0; b < batch; ++b) {
                const auto src = states.recurrent_tensor(l, sources[b]);
                const auto dst = states.recurrent_tensor(l, destinations[b]);
                copy_bytes({dst.data, dst.bytes(), cuda.device()},
                           {src.data, src.bytes(), cuda.device()}, cuda);
            }
            source_buffer.copy_from_host(destinations.data(), source_buffer.size());
            valid_buffer.copy_from_host(valid.data(), valid_buffer.size());
            CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            source_buffer.copy_from_host(sources.data(), source_buffer.size());
            CUDA_CHECK(cudaGraphExecDestroy(ex));
            CUDA_CHECK(cudaGraphDestroy(gr));
        }
        cuda_synchronize();
        const auto got        = from_device<std::uint16_t>(output, vectors);
        const auto keys       = from_device<std::uint16_t>(rec.key.data, qvectors);
        const auto vals       = from_device<std::uint16_t>(rec.value.data, vectors);
        const auto input_keys = from_device<std::uint16_t>(kbuf, qvectors);
        const auto input_vals = from_device<std::uint16_t>(vbuf, vectors);
        const auto logs       = from_device<std::uint32_t>(rec.log_decay.data, vectors);
        const auto betas      = from_device<std::uint32_t>(rec.beta.data, heads * width * batch);
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < width; ++t) {
                const auto column = std::size_t(b) * width + t;
                for (int c = 0; c < 128 * qheads; ++c) {
                    const auto i    = column * 128 * qheads + c;
                    const auto want = t < valid[b] || alias_records ? input_keys[i] : 0x3535;
                    if (keys[i] != want) throw std::runtime_error("key record words changed");
                }
                for (int c = 0; c < 128 * heads; ++c) {
                    const auto i    = column * 128 * heads + c;
                    const auto want = t < valid[b] || alias_records ? input_vals[i] : 0x3535;
                    if (vals[i] != want) throw std::runtime_error("value record words changed");
                    if (t >= valid[b] && logs[i] != 0x35353535)
                        throw std::runtime_error("decay record suffix changed");
                }
                if (t >= valid[b])
                    for (int h = 0; h < heads; ++h)
                        if (betas[column * heads + h] != 0x35353535)
                            throw std::runtime_error("beta record suffix changed");
            }
        for (int b = 0; b < batch; ++b) {
            auto running = to_device_f32(initial[b]);
            Tensor running_state(running.span(), DType::FP32, {128, 128, heads});
            DeviceBuffer scratch(cuda.device(), 256), one_output(cuda.device(), 128 * heads * 2);
            MemoryArena workspace(scratch.span());
            Tensor y(one_output.span(), DType::BF16, {128, heads, 1});
            expected[l * batch + b] = initial[b];
            for (int t = 0; t < valid[b]; ++t) {
                const auto column = [&](const Tensor& x) {
                    return x.slice(3, b, 1).slice(2, t, 1);
                };
                ops::kimi_delta_attention(column(q), column(k), column(v), column(g),
                                          beta.slice(2, b, 1).slice(1, t, 1), a, dt, -5, scale,
                                          workspace, running_state, y, cuda);
                cuda_synchronize();
                const auto step = from_device<std::uint16_t>(one_output, 128 * heads);
                failures += verify_exact(
                    "record readout",
                    std::vector<std::uint16_t>(got.begin() + (b * width + t) * 128 * heads,
                                               got.begin() + (b * width + t + 1) * 128 * heads),
                    step);
                if (t + 1 == accepted[b])
                    expected[l * batch + b] = from_device<float>(running, initial[b].size());
            }
            for (int t = valid[b]; t < width; ++t)
                for (int c = 0; c < 128 * heads; ++c)
                    if (got[(b * width + t) * 128 * heads + c] != 0)
                        throw std::runtime_error("record output suffix is not positive zero");
            const auto untouched = states.recurrent_tensor(l, sources[b]);
            failures += verify_exact(
                "record source", from_device<float>(untouched.data, initial[b].size()), initial[b]);
        }
    }
    ops::KdaReplayFoldPlan fold(records, states.all_layers_view());
    // Zero rows have no payload addresses, including under graph replay.
    auto fold_sources = sources, fold_destinations = destinations;
    for (int b = 0; b < batch; ++b)
        if (!accepted[b]) fold_sources[b] = fold_destinations[b] = -1;
    source_buffer.copy_from_host(fold_sources.data(), source_buffer.size());
    destination_buffer.copy_from_host(fold_destinations.data(), destination_buffer.size());
    if (graph) {
        cudaGraph_t gr;
        cudaGraphExec_t ex;
        CUDA_CHECK(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
        fold.execute(source_table, destination_table, accepted_table, cuda);
        CUDA_CHECK(cudaStreamEndCapture(cuda.stream(), &gr));
        CUDA_CHECK(cudaGraphInstantiate(&ex, gr, 0));
        const std::vector<int> zero(batch, 0);
        const std::vector<std::int64_t> invalid(batch, -1);
        accepted_buffer.copy_from_host(zero.data(), accepted_buffer.size());
        source_buffer.copy_from_host(invalid.data(), source_buffer.size());
        destination_buffer.copy_from_host(invalid.data(), destination_buffer.size());
        CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        accepted_buffer.copy_from_host(accepted.data(), accepted_buffer.size());
        source_buffer.copy_from_host(fold_sources.data(), source_buffer.size());
        destination_buffer.copy_from_host(fold_destinations.data(), destination_buffer.size());
        CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        CUDA_CHECK(cudaGraphExecDestroy(ex));
        CUDA_CHECK(cudaGraphDestroy(gr));
    } else {
        std::vector<ops::KdaReplayFoldRow> rows;
        for (int b = 0; b < batch; ++b)
            rows.push_back({fold_sources[b], fold_destinations[b], accepted[b]});
        fold.execute(rows, cuda);
        cuda_synchronize();
    }
    for (int b = 0; b < batch; ++b) {
        const auto offset = accepted[b] ? destinations[b] : sources[b];
        for (int l = 0; l < layers; ++l) {
            const auto recurrent = states.recurrent_tensor(l, offset),
                       conv      = states.conv_tensor(l, offset);
            failures += verify_exact("fold recurrent",
                                     from_device<float>(recurrent.data, 128 * 128 * heads),
                                     expected[l * batch + b]);
            failures += verify_exact("fold convolution",
                                     from_device<std::uint16_t>(conv.data, channels * 3),
                                     expected_conv[l * batch + b]);
        }
        const auto field = storage.tensor(guard, offset);
        const auto words = from_device<std::uint32_t>(field.data, 64);
        if (std::any_of(words.begin(), words.end(), [](auto x) { return x != 0x5a5a5a5aU; }))
            throw std::runtime_error("fold changed unrelated field");
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("kda/replay/mixed", [] { return replay_case(2, 3, 7, 8, false, false, false, 1); });
    suite.add("kda/replay/alias", [] { return replay_case(8, 32, 4, 2, true, true, false); });
    suite.add("kda/replay/graph", [] { return replay_case(2, 32, 17, 8, true, false, true); });
    suite.add("kda/replay/full_layers",
              [] { return replay_case(34, 32, 4, 1, false, false, true); });
    suite.add("kda/replay/pair", [] { return replay_case(1, 32, 2, 2, false, true, false); });
    suite.add("kda/replay/single", [] { return replay_case(1, 1, 1, 1, false, false, false); });
    return suite.main(argc, argv);
}
