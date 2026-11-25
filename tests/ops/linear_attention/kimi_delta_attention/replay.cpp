#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/transfer.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/kda_replay.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear_attention/kda_ref.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr float scale = 0.08838834764831845F;

constexpr ReductionCriterion output_criterion{6.0e-3, 1.0e-5, 1.5e-2};
constexpr ReductionCriterion state_criterion_for(DType state_dtype = DType::FP32) {
    return state_dtype == DType::BF16 ? ReductionCriterion{7.5e-3, 2.0e-5, 1.6e-2}
                                      : ReductionCriterion{4.0e-3, 1.0e-5, 1.2e-2};
}
using RecordAccess = ninfer::detail::KdaReplayRecordAccess;

template <class T>
void require_words(const void* device, const std::vector<T>& expected, const char* label) {
    const auto actual = from_device<T>(device, expected.size());
    if (std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(T)) != 0)
        throw std::runtime_error(label);
}

std::vector<std::uint16_t> bf16_words(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), f32_to_bf16);
    return out;
}

void upload_state(const Tensor& field, const std::vector<float>& values, CudaExecutionView cuda) {
    if (field.dtype == DType::BF16) {
        const auto words = bf16_words(values);
        copy_bytes({field.data, field.bytes(), cuda.device()},
                   {words.data(), field.bytes(), Device::cpu()}, cuda);
        return;
    }
    copy_bytes({field.data, field.bytes(), cuda.device()},
               {values.data(), field.bytes(), Device::cpu()}, cuda);
}

void require_state_words(const Tensor& field, const std::vector<float>& expected,
                         const char* label) {
    if (field.dtype == DType::BF16) {
        require_words(field.data, bf16_words(expected), label);
        return;
    }
    require_words(field.data, expected, label);
}

std::vector<double> read_state(const Tensor& tensor) {
    if (tensor.dtype == DType::BF16) return from_device_bf16(tensor.data, tensor.numel());
    const auto fp32 = from_device<float>(tensor.data, tensor.numel());
    return {fp32.begin(), fp32.end()};
}

template <class T>
void require_fill(const void* device, std::size_t n, T expected, const char* label) {
    if (!n) return;
    const auto actual = from_device<T>(device, n);
    if (std::any_of(actual.begin(), actual.end(), [=](T x) { return x != expected; }))
        throw std::runtime_error(label);
}

void require_state_poison(const Tensor& field, const char* label) {
    if (field.dtype == DType::BF16) {
        require_fill(field.data, field.numel(), std::uint16_t{0x5a5a}, label);
        return;
    }
    require_fill(field.data, field.numel(), 0x5a5a5a5au, label);
}

template <class Fn>
void reject(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(label);
}

kda_ref::Inputs reference_input(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                const Tensor& beta, const Tensor& a, const Tensor& bias, int b,
                                int n, const std::vector<float>& s0) {
    const int width = q.ne[2], qheads = q.ne[1], heads = v.ne[1];
    const auto row = [&](const Tensor& tensor, int stride) {
        const auto* base = static_cast<const std::uint16_t*>(tensor.data);
        const auto data =
            from_device_bf16(base + std::size_t(b) * width * stride, std::size_t(n) * stride);
        return std::vector<float>(data.begin(), data.end());
    };
    return {qheads,
            heads,
            n,
            row(q, 128 * qheads),
            row(k, 128 * qheads),
            row(v, 128 * heads),
            row(g, 128 * heads),
            row(beta, heads),
            from_device<float>(a.data, heads),
            from_device<float>(bias.data, 128 * heads),
            s0};
}

void append_prefix(kda_ref::Inputs& sequence, const kda_ref::Inputs& next, int count) {
    const auto append = [count](auto& target, const auto& source, int stride) {
        target.insert(target.end(), source.begin(), source.begin() + std::size_t(count) * stride);
    };
    append(sequence.q, next.q, 128 * next.qk_heads);
    append(sequence.k, next.k, 128 * next.qk_heads);
    append(sequence.v, next.v, 128 * next.value_heads);
    append(sequence.g, next.g, 128 * next.value_heads);
    append(sequence.beta, next.beta, next.value_heads);
    sequence.tokens += count;
}

kda_ref::Result evaluate_prefix(kda_ref::Inputs input, int count, double lower_bound = -5.0) {
    input.tokens = count;
    input.q.resize(std::size_t(count) * 128 * input.qk_heads);
    input.k.resize(input.q.size());
    input.v.resize(std::size_t(count) * 128 * input.value_heads);
    input.g.resize(input.v.size());
    input.beta.resize(std::size_t(count) * input.value_heads);
    return kda_ref::evaluate(input, lower_bound, scale);
}

std::vector<float> values(std::size_t n, unsigned seed, float amplitude = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

int check_replay(const KdaReplayRecordLayout& rl, DeviceBuffer& record_buffer, int batch,
                 bool in_place, bool graph, bool projected, bool inactive,
                 DType state_dtype = DType::FP32) {
    auto& context = cuda_environment().context;
    CudaStream stream(context);
    const auto cuda  = context.execution_view(stream.stream());
    const auto& spec = rl.spec;
    const int layers = spec.layers, heads = spec.value_heads, qheads = spec.qk_heads,
              width = spec.width, channels = spec.conv_channels;
    StateImageBuilder sb;
    const auto guard        = sb.add(DType::I32, {64}, 1, "unrelated field");
    const auto state_layout = plan_linear_attention_state(
        sb, {static_cast<unsigned>(layers), channels, 3, heads, 128, 128, DType::BF16, state_dtype});
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
        valid[b] = projected ? width : std::max(1, width - b % width);
        if (inactive && b % 3 == 0) valid[b] = 0;
        accepted[b] = batch == 1 ? valid[b] : (b == batch - 1 ? valid[b] : std::min(b, valid[b]));
    }
    std::vector<int> selected(batch);
    auto record_sources = sources;
    for (int b = 0; b < batch; ++b) {
        selected[b]       = graph ? batch - 1 - b : b;
        record_sources[b] = valid[b] ? sources[selected[b]] : -1;
    }
    source_buffer.copy_from_host(record_sources.data(), source_buffer.size());
    auto valid_buffer = to_device(valid), accepted_buffer = to_device(accepted);
    Tensor valid_table(valid_buffer.span(), DType::I32, {batch});
    Tensor accepted_table(accepted_buffer.span(), DType::I32, {batch});
    CUDA_CHECK(cudaMemsetAsync(record_buffer.data(), 0x35, record_buffer.size(), cuda.stream()));
    const KdaReplayRecords records(record_buffer.span(), rl);
    std::vector<kda_ref::Inputs> inputs(layers * batch);
    std::vector<std::vector<float>> initial(layers * batch);
    std::vector<std::vector<std::uint16_t>> initial_conv(layers * batch), physical_conv(layers);
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
        for (int b = 0; b < batch; ++b) {
            auto& s0 = initial[l * batch + b];
            s0       = values(128 * 128 * heads, 901 + l * 11 + b, 0.02F);
            if (state_dtype == DType::BF16) round_to_bf16(s0);
            const auto field = states.recurrent_tensor(l, sources[b]);
            upload_state(field, s0, cuda);
            const auto old = values(channels * 3, 1001 + l * 11 + b);
            auto& old_bits = initial_conv[l * batch + b];
            old_bits.resize(old.size());
            std::transform(old.begin(), old.end(), old_bits.begin(), f32_to_bf16);
            const auto conv = states.conv_tensor(l, sources[b]);
            copy_bytes({conv.data, conv.bytes(), cuda.device()},
                       {old_bits.data(), conv.bytes(), Device::cpu()}, cuda);
        }
        if (projected) {
            input_projection::DevicePackedWeight weight(
                quantized_weight::make_patterned_weight(QType::MXFP8, 12288, 4096, 1801));
            const ops::SingleProjectionWeight projection{weight.view(), ops::LinearPolicy::AllowA8};
            auto xb = to_device_bf16(values(std::size_t(4096) * width * batch, 1803, 0.01F));
            auto wb = to_device_bf16(values(12288 * 4, 1805, 0.1F));
            Tensor x(xb.span(), DType::BF16, {4096, width, batch}),
                w(wb.span(), DType::BF16, {12288, 4});
            const auto bytes = ops::kda_input_proj_conv_record_workspace_capacity_bytes(
                linear_profile(projection), batch, width, width);
            DeviceBuffer scratch(cuda.device(), std::max<std::size_t>(bytes, 256));
            MemoryArena workspace(scratch.span());
            ops::kda_input_proj_conv_record(
                x, projection, w, {states.layer_view(l, source_table).conv}, valid_table,
                {q.view({4096, width, batch}), k.view({4096, width, batch}),
                 v.view({4096, width, batch})},
                rec.conv, workspace, cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        }
        physical_conv[l]  = from_device<std::uint16_t>(rec.conv.data, rec.conv.numel());
        const auto launch = [&] {
            ops::kimi_delta_attention_replay_record(q, k, v, g, beta, a, dt, -5, scale,
                                                    states.layer_view(l, source_table).recurrent,
                                                    valid_table, rec.recurrence, out, cuda);
        };
        const auto [key, delta, alpha] = RecordAccess::planes(rec.recurrence);
        if (l == 0 && !projected && heads == qheads && width >= 2 && batch >= 2) {
            const auto attempt = [&](const Tensor& kk, const Tensor& vv, const Tensor& counts,
                                     Tensor& result) {
                ops::kimi_delta_attention_replay_record(
                    q, kk, vv, g, beta, a, dt, -5, scale,
                    states.layer_view(l, source_table).recurrent, counts, rec.recurrence, result,
                    cuda);
            };
            reject([&] { attempt(key, v, valid_table, out); }, "key record aliases K");
            reject([&] { attempt(k, key, valid_table, out); }, "record storage aliases V");
            const Tensor aliased_v(delta.data, cuda.device(), DType::BF16,
                                   {128, heads, width, batch});
            reject([&] { attempt(k, aliased_v, valid_table, out); }, "delta record aliases V");
            const Tensor aliased_valid(alpha.data, cuda.device(), DType::I32, {batch});
            reject([&] { attempt(k, v, aliased_valid, out); }, "alpha record aliases lengths");
            Tensor aliased_out = key;
            reject([&] { attempt(k, v, valid_table, aliased_out); }, "key record aliases output");
            reject([&] { attempt(k, v, valid_table.slice(0, 0, 1), out); },
                   "record accepts malformed lengths");
        }
        if (graph) {
            // Capture with different source offsets and lengths at the same table addresses.
            std::vector<int> one(batch, 1);
            source_buffer.copy_from_host(sources.data(), source_buffer.size());
            valid_buffer.copy_from_host(one.data(), valid_buffer.size());
            cudaGraph_t gr;
            cudaGraphExec_t ex;
            CUDA_CHECK(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
            launch();
            CUDA_CHECK(cudaStreamEndCapture(cuda.stream(), &gr));
            CUDA_CHECK(cudaGraphInstantiate(&ex, gr, 0));
            source_buffer.copy_from_host(record_sources.data(), source_buffer.size());
            valid_buffer.copy_from_host(valid.data(), valid_buffer.size());
            CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            CUDA_CHECK(cudaGraphExecDestroy(ex));
            CUDA_CHECK(cudaGraphDestroy(gr));
        } else {
            launch();
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        }
        const auto out_words = from_device<std::uint16_t>(output, vectors);
        const auto key_words = from_device<std::uint16_t>(key.data, qvectors);
        const auto raw_keys  = from_device<std::uint16_t>(kbuf, qvectors);
        const auto deltas    = from_device<std::uint32_t>(delta.data, vectors);
        const auto decays    = from_device<std::uint32_t>(alpha.data, vectors);
        for (int b = 0; b < batch; ++b) {
            for (int t = 0; t < width; ++t) {
                const auto column = std::size_t(b) * width + t;
                for (int c = 0; c < 128 * qheads; ++c) {
                    const auto i = column * 128 * qheads + c;
                    if (key_words[i] != (t < valid[b] ? raw_keys[i] : 0x3535))
                        throw std::runtime_error("incorrect raw key record words");
                }
                if (t >= valid[b])
                    for (int c = 0; c < 128 * heads; ++c) {
                        const auto i = column * 128 * heads + c;
                        if (out_words[i] != 0 || deltas[i] != 0x35353535 || decays[i] != 0x35353535)
                            throw std::runtime_error("invalid output or transition record suffix");
                    }
            }
            if (valid[b]) {
                auto& ref           = inputs[l * batch + b];
                ref                 = reference_input(q, k, v, g, beta, a, dt, b, valid[b],
                                                      initial[l * batch + selected[b]]);
                const auto expected = kda_ref::evaluate(ref, -5, scale);
                const auto actual   = from_device_bf16(static_cast<const std::uint16_t*>(out.data) +
                                                           std::size_t(b) * width * 128 * heads,
                                                       expected.out.size());
                failures +=
                    verify_reduction("record readout", actual, expected.out, output_criterion);
            }
            require_state_words(states.recurrent_tensor(l, sources[b]), initial[l * batch + b],
                                "record changed source recurrent state");
            require_words(states.conv_tensor(l, sources[b]).data, initial_conv[l * batch + b],
                          "record changed source convolution history");
        }
    }

    const auto [key, delta, alpha] = RecordAccess::planes(records);
    for (const auto* plane : {&records.conv, &key, &delta, &alpha}) {
        const auto* data     = static_cast<const std::byte*>(plane->data);
        const auto row_bytes = plane->bytes() / (layers * spec.record_capacity);
        for (int l = 0; l < layers; ++l)
            require_fill(data + (l * spec.record_capacity + batch) * row_bytes,
                         (spec.record_capacity - batch) * row_bytes, std::uint8_t{0x35},
                         "record wrote inactive capacity rows");
    }
    const auto frame_before = from_device<std::uint8_t>(record_buffer, record_buffer.size());
    ops::KdaReplayFoldPlan fold(records, states.all_layers_view());
    auto submit = [&](const std::vector<int>& counts, bool capture) {
        std::vector<ops::KdaReplayFoldRow> rows(batch);
        for (int b = 0; b < batch; ++b)
            rows[b] = {counts[b] ? record_sources[b] : -1,
                       counts[b] ? (in_place ? record_sources[b] : destinations[b]) : -1,
                       counts[b]};
        if (capture) {
            const std::vector<std::int64_t> invalid(batch, -1);
            const std::vector<int> zeros(batch);
            source_buffer.copy_from_host(invalid.data(), source_buffer.size());
            destination_buffer.copy_from_host(invalid.data(), destination_buffer.size());
            accepted_buffer.copy_from_host(zeros.data(), accepted_buffer.size());
            cudaGraph_t gr;
            cudaGraphExec_t ex;
            CUDA_CHECK(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeThreadLocal));
            fold.execute(source_table, destination_table, accepted_table, cuda);
            CUDA_CHECK(cudaStreamEndCapture(cuda.stream(), &gr));
            CUDA_CHECK(cudaGraphInstantiate(&ex, gr, 0));
            CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            for (int l = 0; l < layers; ++l)
                for (int b = 0; b < batch; ++b) {
                    require_state_words(states.recurrent_tensor(l, sources[b]),
                                        initial[l * batch + b], "zero fold changed source");
                    require_words(states.conv_tensor(l, sources[b]).data,
                                  initial_conv[l * batch + b], "zero fold changed source history");
                    if (!in_place) {
                        require_state_poison(states.recurrent_tensor(l, destinations[b]),
                                             "zero fold changed destination");
                        require_fill(states.conv_tensor(l, destinations[b]).data, channels * 3,
                                     std::uint16_t{0x5a5a},
                                     "zero fold changed destination history");
                    }
                }
            std::vector<std::int64_t> so, ds;
            for (const auto& row : rows) {
                so.push_back(row.source_offset);
                ds.push_back(row.destination_offset);
            }
            source_buffer.copy_from_host(so.data(), source_buffer.size());
            destination_buffer.copy_from_host(ds.data(), destination_buffer.size());
            accepted_buffer.copy_from_host(counts.data(), accepted_buffer.size());
            CUDA_CHECK(cudaGraphLaunch(ex, cuda.stream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            require_words(source_table.data, so, "fold modified source metadata");
            require_words(destination_table.data, ds, "fold modified destination metadata");
            require_words(accepted_table.data, counts, "fold modified commit metadata");
            CUDA_CHECK(cudaGraphExecDestroy(ex));
            CUDA_CHECK(cudaGraphDestroy(gr));
        } else {
            fold.execute(rows, cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        }
    };

    if (batch >= 2 && width >= 2 && !in_place && !inactive) {
        const ops::KdaReplayFoldRow out_of_range{sources[0], destinations[0], width + 1};
        reject([&] { fold.execute({&out_of_range, 1}, cuda); }, "fold accepts oversized prefix");
        const std::array duplicate{ops::KdaReplayFoldRow{sources[0], destinations[0], 1},
                                   ops::KdaReplayFoldRow{sources[1], destinations[0], 1}};
        reject([&] { fold.execute(duplicate, cuda); }, "fold accepts duplicate destination");
        auto overwrites                  = duplicate;
        overwrites[1].destination_offset = sources[0];
        reject([&] { fold.execute(overwrites, cuda); }, "fold overwrites another active source");
        const Tensor aliased_columns(records.conv.data, cuda.device(), DType::I32, {batch});
        reject([&] { fold.execute(source_table, destination_table, aliased_columns, cuda); },
               "fold accepts metadata overlapping records");
    }

    const auto verify_fold = [&](const std::vector<int>& counts) {
        for (int b = 0; b < batch; ++b) {
            const int slot  = selected[b];
            const auto dest = in_place ? sources[slot] : destinations[b];
            for (int l = 0; l < layers; ++l) {
                const auto recurrent = states.recurrent_tensor(l, dest);
                const auto conv      = states.conv_tensor(l, dest);
                const auto& old      = initial_conv[l * batch + slot];
                if (const int count = counts[b]) {
                    const auto expected = evaluate_prefix(inputs[l * batch + b], count);
                    failures +=
                        verify_reduction("fold prefix state", read_state(recurrent),
                                         expected.final_state, state_criterion_for(state_dtype));
                    std::vector<std::uint16_t> tail(3 * channels);
                    for (int j = 0; j < 3; ++j) {
                        const int t      = count - 3 + j;
                        const auto* from = t < 0 ? old.data() + (t + 3) * channels
                                                 : physical_conv[l].data() +
                                                       (std::size_t(b) * width + t) * channels;
                        std::copy_n(from, channels, tail.data() + j * channels);
                    }
                    require_words(conv.data, tail, "fold convolution tail differs");
                } else if (!in_place) {
                    require_state_poison(recurrent, "zero commit changed recurrent destination");
                    require_fill(conv.data, conv.numel(), std::uint16_t{0x5a5a},
                                 "zero commit changed convolution destination");
                }
                if (!counts[b] || !in_place) {
                    require_state_words(states.recurrent_tensor(l, sources[slot]),
                                        initial[l * batch + slot], "fold changed untouched source");
                    require_words(states.conv_tensor(l, sources[slot]).data, old,
                                  "fold changed untouched source history");
                }
            }
            for (const auto offset : {sources[slot], dest})
                require_fill(storage.tensor(guard, offset).data, 64, 0x5a5a5a5au,
                             "fold changed unrelated field");
        }
    };
    submit(accepted, graph);
    verify_fold(accepted);
    if (projected) {
        for (int shift = 0; shift <= width; ++shift) {
            for (int b = 0; b < batch; ++b)
                for (int l = 0; l < layers; ++l) {
                    const int slot       = selected[b];
                    const auto dest      = in_place ? sources[slot] : destinations[b];
                    const auto recurrent = states.recurrent_tensor(l, dest);
                    const auto conv      = states.conv_tensor(l, dest);
                    if (in_place) {
                        upload_state(recurrent, initial[l * batch + slot], cuda);
                        copy_bytes(
                            {conv.data, conv.bytes(), cuda.device()},
                            {initial_conv[l * batch + slot].data(), conv.bytes(), Device::cpu()},
                            cuda);
                    } else {
                        CUDA_CHECK(cudaMemsetAsync(recurrent.data, 0x5a, recurrent.bytes(),
                                                   cuda.stream()));
                        CUDA_CHECK(cudaMemsetAsync(conv.data, 0x5a, conv.bytes(), cuda.stream()));
                    }
                }
            std::vector<int> counts(batch);
            for (int b = 0; b < batch; ++b) counts[b] = (shift + b) % (width + 1);
            submit(counts, false);
            verify_fold(counts);
        }
    }
    require_words(record_buffer.data(), frame_before, "fold modified records");
    return failures;
}

int replay_case(int layers, int heads, int width, int batch, bool in_place, bool graph,
                int qheads = 0, bool projected = false, bool inactive = false,
                DType state_dtype = DType::FP32) {
    if (!qheads) qheads = heads;
    LayoutBuilder builder;
    const auto layout =
        plan_kda_replay_records(builder, {layers, batch > 8 ? batch + 3 : 8, width,
                                          (2 * qheads + heads) * 128, qheads, heads, 128, 128});
    DeviceBuffer storage(cuda_environment().context.device(), builder.finish(256));
    return check_replay(layout, storage, batch, in_place, graph, projected, inactive, state_dtype);
}

int switching_widths(DType state_dtype = DType::FP32) {
    LayoutBuilder builder;
    const auto maximum = plan_kda_replay_records(builder, {3, 8, 8, 5 * 128, 1, 3, 128, 128});
    DeviceBuffer storage(cuda_environment().context.device(), builder.finish(256));
    int failures = 0;
    for (int width : {8, 5, 2, 8}) {
        const auto layout = maximum.with_width(width);
        failures += check_replay(layout, storage, 5, false, false, false, true, state_dtype);
    }
    return failures;
}

int cancellation(int width, DType state_dtype = DType::FP32) {
    constexpr int dim = 128, channels = 3 * dim;
    auto& context = cuda_environment().context;
    CudaStream stream(context);
    const auto cuda = context.execution_view(stream.stream());
    StateImageBuilder state_builder;
    const auto fields = plan_linear_attention_state(
        state_builder, {1, channels, 3, 1, dim, dim, DType::BF16, state_dtype});
    const auto layout = state_builder.finish();
    DeviceBuffer memory(cuda.device(), layout.image_bytes);
    StateImageStorage storage(memory.span(), layout);
    LinearAttentionStateView states(storage, fields);

    kda_ref::Inputs inputs{.qk_heads    = 1,
                           .value_heads = 1,
                           .tokens      = width,
                           .q           = std::vector<float>(width * dim),
                           .k           = std::vector<float>(width * dim),
                           .v           = std::vector<float>(width * dim),
                           .g           = std::vector<float>(width * dim),
                           // BF16 beta logits of 64 yield sigmoid(beta) = 1 in FP32 and FP64.
                           .beta    = std::vector<float>(width, 64.f),
                           .a_log   = {0.f},
                           .dt_bias = std::vector<float>(dim),
                           .state   = std::vector<float>(dim * dim)};
    inputs.state[0] = 0x1p20f;
    inputs.k[0] = inputs.k[dim] = 32.f;
    inputs.v[dim]               = 0x1p-6f;
    const auto state            = states.recurrent_tensor(0, 0);
    upload_state(state, inputs.state, cuda);
    auto offsets_buffer = to_device(std::vector<std::int64_t>{0});
    const Tensor offsets(offsets_buffer.span(), DType::I64, {1});

    LayoutBuilder builder;
    const auto record_layout =
        plan_kda_replay_records(builder, {1, 1, width, channels, 1, 1, dim, dim});
    DeviceBuffer frame(cuda.device(), builder.finish(256));
    const KdaReplayRecords records(frame.span(), record_layout);
    const auto rec = records.layer(0, 1);
    CUDA_CHECK(cudaMemsetAsync(rec.conv.data, 0, rec.conv.bytes(), cuda.stream()));
    auto qb = to_device_bf16(inputs.q), kb = to_device_bf16(inputs.k);
    auto vb = to_device_bf16(inputs.v), gb = to_device_bf16(inputs.g);
    auto bb = to_device_bf16(inputs.beta);
    auto ab = to_device_f32(inputs.a_log), db = to_device_f32(inputs.dt_bias);
    DeviceBuffer output(cuda.device(), width * dim * sizeof(std::uint16_t));
    const Tensor q(qb.span(), DType::BF16, {dim, 1, width, 1});
    const Tensor k(kb.span(), DType::BF16, {dim, 1, width, 1});
    const Tensor v(vb.span(), DType::BF16, {dim, 1, width, 1});
    const Tensor g(gb.span(), DType::BF16, {dim, 1, width, 1});
    const Tensor beta(bb.span(), DType::BF16, {1, width, 1});
    const Tensor a(ab.span(), DType::FP32, {1}), bias(db.span(), DType::FP32, {dim, 1});
    Tensor out(output.span(), DType::BF16, {dim, 1, width, 1});

    constexpr float lower_bound = 0.f;
    ops::kimi_delta_attention_replay_record(q, k, v, g, beta, a, bias, lower_bound, scale,
                                            states.layer_view(0, offsets).recurrent, {},
                                            rec.recurrence, out, cuda);
    const ops::KdaReplayFoldPlan fold(records, states.all_layers_view());
    const ops::KdaReplayFoldRow row{0, 0, width};
    fold.execute({&row, 1}, cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));

    const auto expected = kda_ref::evaluate(inputs, lower_bound, scale);
    return verify_reduction("fold preserves small correction", read_state(state),
                            expected.final_state, state_criterion_for(state_dtype));
}

int chained(DType state_dtype = DType::FP32) {
    constexpr float lower_bound = -0.25F;
    constexpr int heads = 2, qheads = 1, channels = (heads + 2 * qheads) * 128;
    auto& context = cuda_environment().context;
    CudaStream stream(context);
    const auto cuda = context.execution_view(stream.stream());
    StateImageBuilder state_builder;
    const auto fields = plan_linear_attention_state(
        state_builder, {1, channels, 3, heads, 128, 128, DType::BF16, state_dtype});
    const auto state_layout = state_builder.finish();
    DeviceBuffer memory(cuda.device(), state_layout.image_bytes);
    StateImageStorage storage(memory.span(), state_layout);
    LinearAttentionStateView states(storage, fields);
    const auto state = states.recurrent_tensor(0, 0);
    auto initial     = values(state.numel(), 19, 0.1F);
    if (state_dtype == DType::BF16) round_to_bf16(initial);
    upload_state(state, initial, cuda);
    CUDA_CHECK(cudaMemsetAsync(states.conv_tensor(0, 0).data, 0, channels * 3 * 2, cuda.stream()));
    auto sb = to_device(std::vector<std::int64_t>{0});
    Tensor offsets(sb.span(), DType::I64, {1});
    LayoutBuilder builder;
    const auto maximum =
        plan_kda_replay_records(builder, {1, 1, 35, channels, qheads, heads, 128, 128});
    DeviceBuffer frame(cuda.device(), builder.finish(256));
    const auto a_log = values(heads, 61, 0.5F), bias = values(128 * heads, 71);
    auto ab = to_device_f32(a_log), db = to_device_f32(bias);
    const Tensor a(ab.span(), DType::FP32, {heads}), dt(db.span(), DType::FP32, {128, heads});
    kda_ref::Inputs trajectory{.qk_heads    = qheads,
                               .value_heads = heads,
                               .a_log       = a_log,
                               .dt_bias     = bias,
                               .state       = initial};
    std::vector<float> committed_state = initial;
    int failures = 0, seed = 100;
    for (const auto [width, count] :
         {std::pair{1, 1}, {8, 4}, {20, 17}, {35, 34}, {4, 0}, {8, 8}}) {
        const KdaReplayRecords records(frame.span(), maximum.with_width(width));
        const auto rec = records.layer(0, 1);
        auto qb        = to_device_bf16(values(128 * qheads * width, ++seed));
        auto kb        = to_device_bf16(values(128 * qheads * width, ++seed));
        auto vb        = to_device_bf16(values(128 * heads * width, ++seed));
        auto gb        = to_device_bf16(values(128 * heads * width, ++seed, 4));
        auto bb        = to_device_bf16(values(heads * width, ++seed));
        DeviceBuffer output(cuda.device(), 128 * heads * width * 2);
        const Tensor q(qb.span(), DType::BF16, {128, qheads, width, 1});
        const Tensor k(kb.span(), DType::BF16, {128, qheads, width, 1});
        const Tensor v(vb.span(), DType::BF16, {128, heads, width, 1});
        const Tensor g(gb.span(), DType::BF16, {128, heads, width, 1});
        const Tensor beta(bb.span(), DType::BF16, {heads, width, 1});
        Tensor out(output.span(), DType::BF16, {128, heads, width, 1});
        CUDA_CHECK(cudaMemsetAsync(rec.conv.data, 0, rec.conv.bytes(), cuda.stream()));
        const auto state_before =
            count == 0 ? from_device<std::uint8_t>(state.data, state.bytes())
                       : std::vector<std::uint8_t>{};
        ops::kimi_delta_attention_replay_record(q, k, v, g, beta, a, dt, lower_bound, scale,
                                                states.layer_view(0, offsets).recurrent, {},
                                                rec.recurrence, out, cuda);
        ops::KdaReplayFoldPlan fold(records, states.all_layers_view());
        const ops::KdaReplayFoldRow row{count ? 0 : -1, count ? 0 : -1, count};
        fold.execute({&row, 1}, cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));

        const auto round = reference_input(q, k, v, g, beta, a, dt, 0, width, committed_state);
        if (state_dtype == DType::BF16) {
            const auto verify = kda_ref::evaluate(round, lower_bound, scale);
            failures += verify_reduction("chained readouts", from_device_bf16(output.data(), out.numel()),
                                         verify.out, output_criterion);
            if (count > 0) {
                const auto committed = evaluate_prefix(round, count, lower_bound);
                failures += verify_reduction("chained prefix state", read_state(state),
                                             committed.final_state,
                                             state_criterion_for(state_dtype));
                committed_state.resize(committed.final_state.size());
                for (std::size_t i = 0; i < committed_state.size(); ++i)
                    committed_state[i] = static_cast<float>(committed.final_state[i]);
                round_to_bf16(committed_state);
            } else {
                require_words(state.data, state_before, "zero chained fold changed state");
            }
        } else {
            auto candidates = trajectory;
            append_prefix(candidates, round, width);
            const auto verify = kda_ref::evaluate(candidates, lower_bound, scale);
            failures += verify_reduction(
                "chained readouts", from_device_bf16(output.data(), out.numel()),
                std::span(verify.out).subspan(std::size_t(trajectory.tokens) * 128 * heads),
                output_criterion);
            append_prefix(trajectory, round, count);
            const auto committed = kda_ref::evaluate(trajectory, lower_bound, scale);
            failures += verify_reduction("chained prefix state", read_state(state),
                                         committed.final_state, state_criterion_for(state_dtype));
        }
    }
    return failures;
}

int contract_state_dtype_rejection() {
    constexpr int dim = 128, heads = 2, channels = 3 * heads * dim, width = 2;
    auto& context = cuda_environment().context;
    CudaStream stream(context);
    const auto cuda = context.execution_view(stream.stream());
    StateImageBuilder sb;
    const auto fields =
        plan_linear_attention_state(sb, {1, channels, 3, heads, dim, dim, DType::BF16, DType::FP32});
    const auto state_layout = sb.finish();
    DeviceBuffer memory(cuda.device(), state_layout.image_bytes);
    StateImageStorage storage(memory.span(), state_layout);
    LinearAttentionStateView states(storage, fields);
    auto offsets_buf = to_device(std::vector<std::int64_t>{0});
    Tensor offsets(offsets_buf.span(), DType::I64, {1});

    LayoutBuilder builder;
    const auto rl = plan_kda_replay_records(builder, {1, 1, width, channels, heads, heads, dim, dim});
    DeviceBuffer frame(cuda.device(), builder.finish(256));
    const KdaReplayRecords records(frame.span(), rl);
    const auto rec = records.layer(0, 1);

    DeviceBuffer qkv(cuda.device(), std::size_t(dim) * heads * width * sizeof(std::uint16_t));
    DeviceBuffer beta_b(cuda.device(), std::size_t(heads) * width * sizeof(std::uint16_t));
    DeviceBuffer a_b(cuda.device(), std::size_t(heads) * sizeof(float));
    DeviceBuffer dt_b(cuda.device(), std::size_t(dim) * heads * sizeof(float));
    DeviceBuffer out_b(cuda.device(), std::size_t(dim) * heads * width * sizeof(std::uint16_t));
    Tensor q(qkv.span(), DType::BF16, {dim, heads, width, 1});
    Tensor beta(beta_b.span(), DType::BF16, {heads, width, 1});
    Tensor a(a_b.span(), DType::FP32, {heads});
    Tensor dt(dt_b.span(), DType::FP32, {dim, heads});
    Tensor out(out_b.span(), DType::BF16, {dim, heads, width, 1});

    auto bad_src                    = states.layer_view(0, offsets).recurrent;
    bad_src.field.tensor.dtype      = DType::FP16;
    reject([&] {
        ops::kimi_delta_attention_replay_record(q, q, q, q, beta, a, dt, -5, scale, bad_src, {},
                                                rec.recurrence, out, cuda);
    }, "kda replay record accepts FP16 source state");

    auto bad_all                        = states.all_layers_view();
    bad_all.spec.recurrent_dtype        = DType::FP16;
    bad_all.recurrent.field.tensor.dtype = DType::FP16;
    reject([&] { (void)ops::KdaReplayFoldPlan(records, bad_all); },
           "kda replay fold accepts FP16 state");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("kda/replay/contract_state_dtype", contract_state_dtype_rejection);
    suite.add("kda/replay/switch_widths", [] { return switching_widths(); });
    suite.add("kda/replay/mixed", [] { return replay_case(2, 3, 7, 8, false, false, 1); });
    suite.add("kda/replay/inplace", [] { return replay_case(8, 32, 4, 2, true, false); });
    suite.add("kda/replay/graph",
              [] { return replay_case(2, 8, 20, 8, false, true, 4, false, true); });
    suite.add("kda/replay/t20_inplace", [] { return replay_case(2, 4, 20, 4, true, false, 2); });
    suite.add("kda/replay/t20_out_of_place",
              [] { return replay_case(2, 4, 20, 2, false, false, 2); });
    suite.add("kda/replay/three_chunks", [] { return replay_case(2, 4, 35, 2, true, false, 2); });
    suite.add("kda/replay/t35_out_of_place",
              [] { return replay_case(2, 4, 35, 2, false, false, 2); });
    suite.add("kda/replay/cancellation_short", [] { return cancellation(2); });
    suite.add("kda/replay/cancellation_chunk", [] { return cancellation(5); });
    suite.add("kda/replay/full_layers", [] { return replay_case(34, 32, 4, 1, false, true); });
    suite.add("kda/replay/pair", [] { return replay_case(1, 32, 2, 2, true, false); });
    suite.add("kda/replay/single", [] { return replay_case(1, 1, 1, 1, false, false); });
    suite.add("kda/replay/single_mixed",
              [] { return replay_case(2, 4, 1, 4, true, true, 2, false, true); });
    suite.add("kda/replay/projected_t8b8",
              [] { return replay_case(1, 32, 8, 8, false, true, 32, true); });
    suite.add("kda/replay/projected_t4b8",
              [] { return replay_case(1, 32, 4, 8, true, true, 32, true); });
    suite.add("kda/replay/projected_t5b4",
              [] { return replay_case(1, 32, 5, 4, true, true, 32, true); });
    suite.add("kda/replay/projected_t16b4",
              [] { return replay_case(1, 32, 16, 4, false, true, 32, true); });
    suite.add("kda/replay/mixed_multilayer_t8",
              [] { return replay_case(2, 32, 8, 4, false, true); });
    suite.add("kda/replay/inplace_multilayer_t16",
              [] { return replay_case(2, 32, 16, 8, true, true); });
    suite.add("kda/replay/chained", [] { return chained(); });
    suite.add("kda/replay/expanded_host",
              [] { return replay_case(2, 1, 4, 17, false, false, 1, false, true); },
              {{"suite", "concurrency"}});
    suite.add("kda/replay/expanded_device",
              [] { return replay_case(2, 1, 3, 64, false, true, 1, false, true); },
              {{"suite", "concurrency"}});
    suite.add("kda/replay/grouped_state_tiles",
              [] { return replay_case(1, 32, 4, 17, false, true, 32, false, true); },
              {{"suite", "concurrency"}});
    suite.add("kda/replay/grouped_long_window",
              [] { return replay_case(1, 32, 16, 64, false, true, 32, false, true); },
              {{"suite", "concurrency"}});
    suite.add("kda/replay_bf16/switch_widths", [] { return switching_widths(DType::BF16); });
    suite.add("kda/replay_bf16/mixed",
              [] { return replay_case(2, 3, 7, 8, false, false, 1, false, false, DType::BF16); });
    suite.add("kda/replay_bf16/inplace",
              [] { return replay_case(8, 32, 4, 2, true, false, 0, false, false, DType::BF16); });
    suite.add("kda/replay_bf16/graph",
              [] { return replay_case(2, 8, 20, 8, false, true, 4, false, true, DType::BF16); });
    suite.add("kda/replay_bf16/t20_inplace",
              [] { return replay_case(2, 4, 20, 4, true, false, 2, false, false, DType::BF16); });
    suite.add("kda/replay_bf16/t35_out_of_place",
              [] { return replay_case(2, 4, 35, 2, false, false, 2, false, false, DType::BF16); });
    suite.add("kda/replay_bf16/cancellation_short",
              [] { return cancellation(2, DType::BF16); });
    suite.add("kda/replay_bf16/cancellation_chunk",
              [] { return cancellation(5, DType::BF16); });
    suite.add("kda/replay_bf16/full_layers",
              [] { return replay_case(34, 32, 4, 1, false, true, 0, false, false, DType::BF16); });
    suite.add("kda/replay_bf16/single",
              [] { return replay_case(1, 1, 1, 1, false, false, 0, false, false, DType::BF16); });
    suite.add("kda/replay_bf16/projected_t8b8",
              [] { return replay_case(1, 32, 8, 8, false, true, 32, true, false, DType::BF16); });
    suite.add("kda/replay_bf16/projected_t4b8",
              [] { return replay_case(1, 32, 4, 8, true, true, 32, true, false, DType::BF16); });
    suite.add("kda/replay_bf16/chained", [] { return chained(DType::BF16); });
    suite.add("kda/replay_bf16/expanded_device",
              [] { return replay_case(2, 1, 3, 64, false, true, 1, false, true, DType::BF16); },
              {{"suite", "concurrency"}});
    suite.add("kda/replay_bf16/grouped_state_tiles",
              [] { return replay_case(1, 32, 4, 17, false, true, 32, false, true, DType::BF16); },
              {{"suite", "concurrency"}});
    suite.add("kda/replay_bf16/grouped_long_window",
              [] { return replay_case(1, 32, 16, 64, false, true, 32, false, true, DType::BF16); },
              {{"suite", "concurrency"}});
    return suite.main(argc, argv);
}
