#include "gdn_ref.h"

#include "cuda_fixture.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"
#include "ops/op_tester.h"
#include "ops/suite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim      = 128;
constexpr double kScale = 1.0 / std::sqrt(double(kDim));

struct Geometry {
    int layers, qk_heads, heads, channels;
};

constexpr Geometry k48x48{48, 16, 48, 10240};
constexpr Geometry k30x32{30, 16, 32, 8192};
constexpr Geometry k36x48{36, 16, 48, 10240};
constexpr Geometry k36x24{36, 8, 24, 5120};

enum class Execution { Host, HostGraph, Device, DeviceGraph };

bool device_rows(Execution execution) {
    return execution == Execution::Device || execution == Execution::DeviceGraph;
}

struct Stream {
    cudaStream_t handle{};

    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking)); }

    ~Stream() { cudaStreamDestroy(handle); }
};

template <class Launch, class Prepare>
void run_graph(cudaStream_t stream, Launch launch, Prepare prepare) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    prepare();
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

template <class Fn>
int must_reject(const char* label, Fn fn) {
    try {
        fn();
    } catch (const std::out_of_range&) { return 0; } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "fold accepted " << label << '\n';
    return 1;
}

std::vector<float> random_floats(std::size_t count, unsigned seed, float low, float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    return values;
}

std::vector<std::uint16_t> random_words(std::size_t count, unsigned seed) {
    const auto values = random_floats(count, seed, -.08f, .08f);
    std::vector<std::uint16_t> words(count);
    std::transform(values.begin(), values.end(), words.begin(), f32_to_bf16);
    words[0] = 0x8000U;
    words[1] = 0x7fc1U; // Check preservation of payload words as well as finite values.
    return words;
}

constexpr int kLayerScales = 16;

float layer_scale(int layer) {
    const int index = layer % kLayerScales;
    return std::ldexp(index < 8 ? 1.0f : -1.0f, -(index % 8));
}

std::vector<std::uint16_t> represented_words(const std::vector<float>& values, float scale = 1.0f) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [scale](float v) { return f32_to_bf16(v * scale); });
    return bits;
}

struct Inputs {
    Inputs(Geometry geometry, int width, int batch, unsigned seed)
        : p(geometry), width(width), batch(batch), q(rounded(geometry.qk_heads * kDim, seed)),
          k(rounded(geometry.qk_heads * kDim, seed + 1)),
          v(rounded(geometry.heads * kDim, seed + 2)),
          g(random_floats(gate_count(), seed + 3, -.55f, -.003f)),
          beta(random_floats(gate_count(), seed + 4, .02f, .98f)) {
        for (int b = 0; b < batch; ++b) {
            std::fill_n(k.data() + std::size_t(b * width) * p.qk_heads * kDim, kDim, 0.0f);
            for (const int t : {0, 2, 3})
                if (t < width)
                    g[(b * width + t) * p.heads] = -std::numeric_limits<float>::infinity();
        }
    }

    std::size_t gate_count() const { return std::size_t(width) * batch * p.heads; }

    std::vector<float> rounded(int rows, unsigned seed) const {
        auto result = random_floats(std::size_t(width) * batch * rows, seed, -.11f, .11f);
        round_to_bf16(result);
        return result;
    }

    gdn_ref::Inputs reference(int b, int tokens, std::span<const float> initial) const {
        const auto prefix = [&](const auto& in, int rows) {
            const auto first = in.begin() + std::size_t(b * width) * rows;
            return std::vector<float>(first, first + std::size_t(tokens) * rows);
        };
        return {kDim,
                p.qk_heads,
                p.heads,
                tokens,
                prefix(q, p.qk_heads * kDim),
                prefix(k, p.qk_heads * kDim),
                prefix(v, p.heads * kDim),
                prefix(g, p.heads),
                prefix(beta, p.heads),
                {initial.begin(), initial.end()}};
    }

    Geometry p;
    int width, batch;
    std::vector<float> q, k, v, g, beta;
};

class FoldFixture {
public:
    FoldFixture(Geometry profile, int slots, int batch, int max_width,
                std::optional<float> first_state = std::nullopt, DType state_dtype = DType::FP32)
        : p(profile), slots(slots), batch(batch), state_dtype_(state_dtype),
          state_layout(plan_linear_attention_state(
              state_builder, {static_cast<std::uint32_t>(p.layers), p.channels, 3, p.heads, kDim,
                              kDim, DType::BF16, state_dtype})),
          image_layout(state_builder.finish()), state_storage(image_layout.image_bytes * slots),
          images(state_storage.span(), image_layout), states(images, state_layout),
          maximum(
              plan_gdn_replay_records(record_builder, {p.layers, batch + 1, max_width, p.channels,
                                                       p.qk_heads, p.heads, kDim, kDim})),
          record_storage(record_builder.finish(256)) {
        recurrent.resize(slots);
        histories.resize(slots);
        std::vector<float> scaled(state_elements());
        for (int slot = 0; slot < slots; ++slot) {
            recurrent[slot] = random_floats(state_elements(), 8021U + slot, -.025f, .025f);
            if (state_dtype_ == DType::BF16) round_to_bf16(recurrent[slot]);
            if (first_state) {
                std::fill(recurrent[slot].begin(), recurrent[slot].end(), 0.f);
                recurrent[slot][0] =
                    state_dtype_ == DType::BF16 ? bf16_to_f32(f32_to_bf16(*first_state)) : *first_state;
            } else {
                recurrent[slot][0] = -0.0f;
            }
            histories[slot].resize(p.layers);
            for (int layer = 0; layer < p.layers; ++layer) {
                auto& history = histories[slot][layer];
                history       = random_words(3 * p.channels, 2017U + 121U * slot + layer);
                CUDA_CHECK(cudaMemcpy(states.conv_tensor(layer, offset(slot)).data, history.data(),
                                      history.size() * sizeof(std::uint16_t),
                                      cudaMemcpyHostToDevice));
                const float scale = layer_scale(layer);
                if (state_dtype_ == DType::BF16) {
                    const auto bits = represented_words(recurrent[slot], scale);
                    CUDA_CHECK(cudaMemcpy(states.recurrent_tensor(layer, offset(slot)).data,
                                          bits.data(), bits.size() * sizeof(std::uint16_t),
                                          cudaMemcpyHostToDevice));
                } else {
                    std::transform(recurrent[slot].begin(), recurrent[slot].end(), scaled.begin(),
                                   [scale](float f) { return f * scale; });
                    CUDA_CHECK(cudaMemcpy(states.recurrent_tensor(layer, offset(slot)).data,
                                          scaled.data(), scaled.size() * sizeof(float),
                                          cudaMemcpyHostToDevice));
                }
            }
        }
    }

    std::int64_t offset(int slot) const { return slot < 0 ? -1 : slot * image_layout.image_bytes; }

    std::size_t state_elements() const { return std::size_t(p.heads) * kDim * kDim; }

    float leading_state() const {
        if (state_dtype_ == DType::BF16)
            return bf16_to_f32(
                from_device<std::uint16_t>(states.recurrent_tensor(0, 0).data, 1).front());
        return from_device<float>(states.recurrent_tensor(0, 0).data, 1).front();
    }

    int round(const Inputs& inputs, const std::vector<int>& source_slots,
              const std::vector<int>& dest_slots, const std::vector<int>& commits,
              Execution execution, bool checks = false, std::vector<double>* trajectory = nullptr) {
        const int width = inputs.width;
        auto& context   = cuda_environment().context;
        Stream stream;
        const auto cuda = context.execution_view(stream.handle);
        const GdnReplayRecords records(record_storage.span(), maximum.with_width(width));
        const ops::GdnReplayFoldPlan plan(records, states.all_layers_view());
        record_storage.fill(0xff);

        std::vector<int> valid(batch);
        std::vector<std::int64_t> record_offsets(batch), sources(batch), destinations(batch);
        std::vector<ops::GdnReplayFoldRow> rows(batch);
        for (int b = 0; b < batch; ++b) {
            valid[b] = commits[b] == 0 && b % 2 == 0 ? 0 : std::max(commits[b], width - b % 3);
            record_offsets[b] = valid[b] ? offset(source_slots[b]) : -1;
            sources[b]        = commits[b] ? offset(source_slots[b]) : -1;
            destinations[b]   = commits[b] ? offset(dest_slots[b]) : -1;
            rows[b]           = {sources[b], destinations[b], commits[b]};
        }
        auto q_buffer              = to_device(represented_words(inputs.q));
        auto k_buffer              = to_device(represented_words(inputs.k));
        auto gate_buffer           = to_device(inputs.g);
        auto beta_buffer           = to_device(inputs.beta);
        auto record_offsets_buffer = to_device(record_offsets);
        auto valid_buffer          = to_device(valid);
        const Tensor q(q_buffer.span(), DType::BF16, {kDim, p.qk_heads, width, batch});
        const Tensor k(k_buffer.span(), DType::BF16, {kDim, p.qk_heads, width, batch});
        const Tensor g(gate_buffer.span(), DType::FP32, {p.heads, width, batch});
        const Tensor beta(beta_buffer.span(), DType::FP32, {p.heads, width, batch});
        const Tensor select(record_offsets_buffer.span(), DType::I64, {batch});
        const Tensor valid_tensor(valid_buffer.span(), DType::I32, {batch});
        std::vector<DeviceBuffer> values;
        for (int i = 0; i < kLayerScales; ++i)
            values.push_back(to_device(represented_words(inputs.v, layer_scale(i))));
        const auto output_elements = std::size_t(kDim) * p.heads * width * batch;
        DeviceBuffer out_buffer(context.device(), output_elements * sizeof(std::uint16_t) *
                                                      (trajectory ? p.layers : 1));
        std::vector<std::vector<std::uint16_t>> conv_records(p.layers);
        for (int layer = 0; layer < p.layers; ++layer) {
            const auto record = records.layer(layer, batch);
            auto& conv        = conv_records[layer];
            conv              = random_words(record.conv.numel(), 1709U + layer * 17 + width);
            for (int b = 0; b < batch; ++b)
                std::fill_n(conv.data() + (std::size_t(b) * width + valid[b]) * p.channels,
                            (width - valid[b]) * p.channels, 0xffffU);
            CUDA_CHECK(cudaMemcpy(record.conv.data, conv.data(), record.conv.bytes(),
                                  cudaMemcpyHostToDevice));
        }
        cuda_synchronize();
        for (int layer = 0; layer < p.layers; ++layer) {
            auto buffer = out_buffer.span();
            if (trajectory)
                buffer = buffer.subspan(layer * output_elements * sizeof(std::uint16_t));
            Tensor out(buffer, DType::BF16, {kDim, p.heads, width, batch});
            const Tensor v(values[layer % kLayerScales].span(), DType::BF16,
                           {kDim, p.heads, width, batch});
            ops::gated_delta_net_replay_record(
                q, k, v, g, beta, float(kScale), states.layer_view(layer, select).recurrent,
                valid_tensor, records.layer(layer, batch).recurrence, out, cuda);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream.handle));
        const auto record_before =
            from_device<std::byte>(record_storage.data(), record_storage.bytes());
        if (execution == Execution::DeviceGraph || execution == Execution::HostGraph) {
            // Both graph modes execute twice. Out-of-place state is stable across launches.
            // The device form first exercises dynamic all-zero metadata with invalid offsets.
            if (execution == Execution::HostGraph && source_slots == dest_slots)
                throw std::logic_error("host Graph replay requires an unchanged source");
        }
        auto source_buffer      = to_device(sources);
        auto destination_buffer = to_device(destinations);
        auto commit_buffer      = to_device(commits);
        const Tensor source(source_buffer.span(), DType::I64, {batch});
        const Tensor destination(destination_buffer.span(), DType::I64, {batch});
        const Tensor columns(commit_buffer.span(), DType::I32, {batch});
        int failures = checks ? validate(plan, records, source, destination, columns, cuda) : 0;
        const auto launch = [&] {
            if (device_rows(execution))
                plan.execute(source, destination, columns, cuda);
            else
                plan.execute(rows, cuda);
        };
        if (execution == Execution::DeviceGraph) {
            source_buffer.fill(0xff);
            destination_buffer.fill(0xff);
            commit_buffer.fill(0);
            run_graph(stream.handle, launch, [&] {
                failures += verify_state(inputs, source_slots, dest_slots, std::vector<int>(batch),
                                         conv_records, {}, nullptr);
                source_buffer.copy_from_host(sources.data(), source.bytes());
                destination_buffer.copy_from_host(destinations.data(), destination.bytes());
                commit_buffer.copy_from_host(commits.data(), columns.bytes());
            });
        } else if (execution == Execution::HostGraph) {
            run_graph(stream.handle, launch, [&] {
                for (int b = 0; b < batch; ++b) {
                    if (!rows[b].commit_columns) continue;
                    for (int layer = 0; layer < p.layers; ++layer) {
                        const auto state   = states.recurrent_tensor(layer, destinations[b]);
                        const auto history = states.conv_tensor(layer, destinations[b]);
                        CUDA_CHECK(cudaMemsetAsync(state.data, 0xff, state.bytes(), stream.handle));
                        CUDA_CHECK(
                            cudaMemsetAsync(history.data, 0xff, history.bytes(), stream.handle));
                    }
                    rows[b] = {-1, -1, 0};
                }
            });
        } else {
            launch();
            CUDA_CHECK(cudaStreamSynchronize(stream.handle));
        }
        failures += verify_state(inputs, source_slots, dest_slots, commits, conv_records,
                                 trajectory ? out_buffer.span() : MemorySpan{}, trajectory);
        failures += record_storage.verify_guards("fold records");
        failures += state_storage.verify_guards("fold states");
        if (from_device<std::byte>(record_storage.data(), record_storage.bytes()) !=
                record_before ||
            from_device<std::int64_t>(source_buffer, batch) != sources ||
            from_device<std::int64_t>(destination_buffer, batch) != destinations ||
            from_device<int>(commit_buffer, batch) != commits) {
            std::cerr << "fold modified records or metadata\n";
            ++failures;
        }
        return failures;
    }

private:
    int validate(const ops::GdnReplayFoldPlan& plan, const GdnReplayRecords& records,
                 const Tensor& sources, const Tensor& destinations, const Tensor& commits,
                 CudaExecutionView cuda) {
        int failures   = 0;
        auto malformed = records;
        --malformed.spec.value_dim;
        failures += must_reject("record geometry", [&] {
            (void)ops::GdnReplayFoldPlan(malformed, states.all_layers_view());
        });
        malformed                 = records;
        const auto private_planes = ninfer::detail::GdnReplayRecordAccess::planes(records);
        malformed.conv.data       = private_planes.delta.data;
        failures += must_reject("overlapping record planes", [&] {
            (void)ops::GdnReplayFoldPlan(malformed, states.all_layers_view());
        });
        failures += must_reject("empty rows", [&] { plan.execute({}, cuda); });
        for (const auto row : {ops::GdnReplayFoldRow{0, 0, -1},
                               {0, 0, records.spec.width + 1},
                               {-1, 0, 1},
                               {0, -1, 1},
                               {2, 0, 1}}) {
            failures +=
                must_reject("invalid active row", [&] { plan.execute(std::span(&row, 1), cuda); });
        }
        const ops::GdnReplayFoldRow duplicates[]{{offset(0), offset(1), 1},
                                                 {offset(0), offset(1), 1}};
        const ops::GdnReplayFoldRow cross_sources[]{{offset(0), offset(1), 1},
                                                    {offset(1), offset(1), 1}};
        failures += must_reject("shared destination", [&] { plan.execute(duplicates, cuda); });
        failures +=
            must_reject("another active source", [&] { plan.execute(cross_sources, cuda); });
        malformed           = records;
        malformed.conv.data = states.conv_tensor(0, offset(0)).data;
        const ops::GdnReplayFoldPlan overlapping(malformed, states.all_layers_view());
        failures += must_reject("record plane overlapping a selected state",
                                [&] { overlapping.execute(std::span(duplicates, 1), cuda); });
        failures += must_reject("device commit dtype",
                                [&] { plan.execute(sources, destinations, sources, cuda); });
        auto overlap = commits;
        overlap.data = records.conv.data;
        failures += must_reject("device metadata overlapping records",
                                [&] { plan.execute(sources, destinations, overlap, cuda); });
        overlap.data = states.conv_tensor(0, 0).data;
        failures += must_reject("device metadata overlapping state storage",
                                [&] { plan.execute(sources, destinations, overlap, cuda); });
        auto bad_states                               = states.all_layers_view();
        bad_states.spec.recurrent_dtype               = DType::FP16;
        bad_states.recurrent.field.tensor.dtype       = DType::FP16;
        failures += must_reject("unsupported recurrent state dtype",
                                [&] { (void)ops::GdnReplayFoldPlan(records, bad_states); });
        return failures;
    }

    int verify_state(const Inputs& inputs, const std::vector<int>& sources,
                     const std::vector<int>& destinations, const std::vector<int>& commits,
                     const std::vector<std::vector<std::uint16_t>>& conv_records, MemorySpan output,
                     std::vector<double>* trajectory) {
        std::vector<std::vector<double>> reference(batch), out_reference(batch);
        std::vector<int> target_row(slots, -1);
        for (int b = 0; b < batch; ++b) {
            const auto m = commits[b];
            if (m == 0) continue;
            target_row[destinations[b]] = b;
            auto represented =
                inputs.reference(b, trajectory ? inputs.width : m, recurrent[sources[b]]);
            auto expected = gdn_ref::evaluate(represented, kScale, true, trajectory != nullptr,
                                              trajectory ? std::span<const double>(*trajectory)
                                                         : std::span<const double>{});
            if (trajectory) {
                const auto first = expected.snapshots.begin() + (m - 1) * state_elements();
                reference[b].assign(first, first + state_elements());
                out_reference[b] = std::move(expected.out);
            } else {
                reference[b] = std::move(expected.final_state);
            }
        }
        int failures   = 0;
        double max_abs = 0.0, max_relative_l2 = 0.0;
        const auto& criterion = gdn_ref::state_criterion(state_dtype_);
        std::vector<float> actual(state_elements()), before(state_elements());
        std::vector<std::uint16_t> actual_bf16(state_dtype_ == DType::BF16 ? state_elements() : 0);
        std::vector<double> actual_fp64(kDim * kDim), expected_fp64(kDim * kDim);
        for (int layer = 0; layer < p.layers; ++layer) {
            const float scale = layer_scale(layer);
            for (int slot = 0; slot < slots; ++slot) {
                const int b      = target_row[slot];
                const int m      = b < 0 ? 0 : commits[b];
                const auto state = states.recurrent_tensor(layer, offset(slot));
                if (state_dtype_ == DType::BF16) {
                    CUDA_CHECK(cudaMemcpy(actual_bf16.data(), state.data, state.bytes(),
                                          cudaMemcpyDeviceToHost));
                    std::transform(actual_bf16.begin(), actual_bf16.end(), actual.begin(),
                                   bf16_to_f32);
                } else {
                    CUDA_CHECK(cudaMemcpy(actual.data(), state.data, state.bytes(),
                                          cudaMemcpyDeviceToHost));
                }
                const auto conv     = states.conv_tensor(layer, offset(slot));
                const auto observed = from_device<std::uint16_t>(conv.data, conv.numel());
                if (m == 0) {
                    bool recurrent_same = false;
                    if (state_dtype_ == DType::BF16) {
                        const auto before_bits = represented_words(recurrent[slot], scale);
                        recurrent_same         = actual_bf16 == before_bits;
                    } else {
                        std::transform(recurrent[slot].begin(), recurrent[slot].end(),
                                       before.begin(), [scale](float f) { return f * scale; });
                        recurrent_same =
                            std::memcmp(actual.data(), before.data(), state.bytes()) == 0;
                    }
                    if (!recurrent_same || observed != histories[slot][layer]) {
                        std::cerr << "fold modified inactive layer " << layer << " slot " << slot
                                  << '\n';
                        ++failures;
                    }
                    continue;
                }
                auto expected_history = histories[sources[b]][layer];
                const auto conv_first =
                    conv_records[layer].begin() + std::size_t(b) * inputs.width * p.channels;
                expected_history.insert(expected_history.end(), conv_first,
                                        conv_first + m * p.channels);
                expected_history.erase(expected_history.begin(),
                                       expected_history.begin() + m * p.channels);
                if (observed != expected_history) {
                    std::cerr << "fold conv tail differs at layer " << layer << " row " << b
                              << '\n';
                    ++failures;
                }
                histories[slot][layer] = std::move(expected_history);
                for (int head = 0; head < p.heads; ++head) {
                    const auto head_offset = std::size_t(head) * kDim * kDim;
                    std::copy_n(actual.begin() + head_offset, actual_fp64.size(),
                                actual_fp64.begin());
                    std::transform(reference[b].begin() + head_offset,
                                   reference[b].begin() + head_offset + expected_fp64.size(),
                                   expected_fp64.begin(), [scale](double f) { return f * scale; });
                    const auto stats = compute_reduction_stats(
                        actual_fp64.data(), expected_fp64.data(), expected_fp64.size());
                    max_abs         = std::max(max_abs, stats.maximum_absolute_error);
                    max_relative_l2 = std::max(max_relative_l2, stats.relative_l2);
                    if (!reduction_passes(stats, expected_fp64.size(), criterion)) {
                        std::cerr << "fold state error at layer " << layer << " row " << b
                                  << " head " << head << " max_abs=" << stats.maximum_absolute_error
                                  << " rel_l2=" << stats.relative_l2 << '\n';
                        ++failures;
                    }
                }
                if (trajectory) {
                    const auto count = out_reference[b].size();
                    const auto words = from_device<std::uint16_t>(
                        static_cast<const std::uint16_t*>(output.data) + layer * count, count);
                    const auto unpacked = gdn_ref::represented_bf16(words);
                    std::vector<double> expected(count);
                    std::transform(out_reference[b].begin(), out_reference[b].end(),
                                   expected.begin(), [scale](double f) { return f * scale; });
                    failures +=
                        verify_reduction("chained record readout",
                                         std::vector<double>(unpacked.begin(), unpacked.end()),
                                         expected, gdn_ref::kOutputCriterion);
                }
            }
        }
        if (trajectory) *trajectory = std::move(reference[0]);
        std::cout << "Fold " << p.layers << 'x' << p.heads << " T=" << inputs.width
                  << (state_dtype_ == DType::BF16 ? " bf16" : " fp32") << " max_abs=" << max_abs
                  << " max_rel_l2=" << max_relative_l2 << '\n';
        return failures;
    }

    Geometry p;
    int slots, batch;
    DType state_dtype_;
    StateImageBuilder state_builder;
    LinearAttentionStateLayout state_layout;
    StateImageLayout image_layout;
    GuardedDeviceBuffer state_storage;
    StateImageStorage images;
    LinearAttentionStateView states;
    LayoutBuilder record_builder;
    GdnReplayRecordLayout maximum;
    GuardedDeviceBuffer record_storage;
    std::vector<std::vector<float>> recurrent;
    std::vector<std::vector<std::vector<std::uint16_t>>> histories;
};

int run_case(Geometry p, int width, const std::vector<int>& commits, Execution execution,
             bool out_of_place = false, bool validate = false, DType state_dtype = DType::FP32) {
    const int batch = commits.size();
    std::vector<int> source(batch, 0), dest(batch, 0);
    int count = 0;
    for (int b = 0; b < batch; ++b) {
        if (!commits[b]) continue;
        source[b] = ++count;
        dest[b]   = out_of_place ? ++count : count;
    }
    FoldFixture fixture(p, std::max(count + 1, 3), batch, width, std::nullopt, state_dtype);
    return fixture.round(Inputs(p, width, batch, 1007U + width), source, dest, commits, execution,
                         validate);
}

int cancellation(DType state_dtype = DType::FP32) {
    FoldFixture fixture(k36x24, 1, 1, 2, 0x1p20f, state_dtype);
    Inputs inputs(k36x24, 2, 1, 729U);
    std::fill(inputs.k.begin(), inputs.k.end(), 0.f);
    std::fill(inputs.v.begin(), inputs.v.end(), 0.f);
    std::fill(inputs.g.begin(), inputs.g.end(), 0.f);
    std::fill(inputs.beta.begin(), inputs.beta.end(), 1.f);
    inputs.k[0] = inputs.k[k36x24.qk_heads * kDim] = 32.f;
    inputs.v[k36x24.heads * kDim]                  = 0x1p-6f;
    return fixture.round(inputs, {0}, {0}, {2}, Execution::Host);
}

int extreme_decay(float gate, float initial, DType state_dtype = DType::FP32) {
    FoldFixture fixture(k36x24, 1, 1, 2, initial, state_dtype);
    Inputs inputs(k36x24, 2, 1, 911U);
    std::fill(inputs.k.begin(), inputs.k.end(), 0.f);
    std::fill(inputs.v.begin(), inputs.v.end(), 0.f);
    std::fill(inputs.g.begin(), inputs.g.end(), gate);
    std::fill(inputs.beta.begin(), inputs.beta.end(), 1.f);
    int failures = fixture.round(inputs, {0}, {0}, {2}, Execution::Host);
    const std::array initial_state{initial};
    auto reference = inputs.reference(0, 2, initial_state);
    reference.state.resize(fixture.state_elements());
    const double expected = gdn_ref::evaluate(reference, kScale, true).final_state.front();
    const double actual   = fixture.leading_state();
    const double rel_tol  = state_dtype == DType::BF16 ? 1e-2 : 1e-5;
    if (!std::isfinite(actual) || std::abs(actual - expected) > rel_tol * std::abs(expected)) {
        std::cerr << "finite extreme-decay result: expected " << expected << ", got " << actual
                  << '\n';
        ++failures;
    }
    return failures;
}

int chain(DType state_dtype = DType::FP32) {
    FoldFixture fixture(k36x24, 2, 1, 16, std::nullopt, state_dtype);
    int failures = 0;
    std::vector<double> trajectory;
    int round = 0;
    for (const auto [width, commit] : {std::pair{16, 1}, {4, 4}, {9, 2}, {16, 11}, {3, 3}}) {
        failures += fixture.round(Inputs(k36x24, width, 1, 8191U + round++), {0}, {0}, {commit},
                                  Execution::Device, false, &trajectory);
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    constexpr Geometry geometries[]{k48x48, k30x32, k36x48, k36x24};
    for (int i = 0; i < 4; ++i) {
        const auto p     = geometries[i];
        const auto label = "gdn_fold/" + std::to_string(p.layers) + "x" + std::to_string(p.heads);
        suite.add(label + "/host_graph_mixed",
                  [=] { return run_case(p, 6, {0, 1, 2, 6}, Execution::HostGraph, true, i == 0); },
                  {{"execution", "graph"}});
        suite.add(label + "/device_graph_mixed",
                  [=] { return run_case(p, 16, {0, 3, 16}, Execution::DeviceGraph, false); },
                  {{"execution", "graph"}});
    }
    for (int width = 2; width <= 16; ++width) {
        suite.add("gdn_fold/width_" + std::to_string(width), [=] {
            return run_case(geometries[width % 4], width, {width - 1}, Execution::Host);
        });
    }
    suite.add("gdn_fold/device_zero",
              [] { return run_case(k36x48, 2, {0, 0}, Execution::Device); });
    suite.add("gdn_fold/host_zero", [] { return run_case(k36x24, 2, {0}, Execution::Host); });
    for (const int batch : {9, 17}) {
        suite.add("gdn_fold/expanded_" + std::to_string(batch),
                  [=] {
                      std::vector<int> commits(batch, 0);
                      commits.front() = 1;
                      commits.back()  = 4;
                      return run_case(k36x24, 4, commits,
                                      batch == 9 ? Execution::Host : Execution::Device);
                  },
                  {{"suite", "concurrency"}});
    }
    suite.add("gdn_fold/cancellation", [] { return cancellation(); });
    suite.add("gdn_fold/decay_product_overflow", [] { return extreme_decay(50.f, 0.f); });
    suite.add("gdn_fold/decay_product_underflow", [] { return extreme_decay(-50.f, 0x1p126f); });
    suite.add("gdn_fold/chained_trajectory", [] { return chain(); });

    // BF16 recurrent state storage cases
    for (int i = 0; i < 4; ++i) {
        const auto p = geometries[i];
        const auto label =
            "gdn_fold_bf16/" + std::to_string(p.layers) + "x" + std::to_string(p.heads);
        suite.add(label + "/host_graph_mixed",
                  [=] {
                      return run_case(p, 6, {0, 1, 2, 6}, Execution::HostGraph, true, false,
                                      DType::BF16);
                  },
                  {{"execution", "graph"}});
        suite.add(label + "/device_graph_mixed",
                  [=] {
                      return run_case(p, 16, {0, 3, 16}, Execution::DeviceGraph, false, false,
                                      DType::BF16);
                  },
                  {{"execution", "graph"}});
    }
    for (int width : {2, 4, 8, 9, 16}) {
        suite.add("gdn_fold_bf16/width_" + std::to_string(width), [=] {
            return run_case(geometries[width % 4], width, {width - 1}, Execution::Host, false,
                            false, DType::BF16);
        });
    }
    suite.add("gdn_fold_bf16/device_zero",
              [] { return run_case(k36x48, 2, {0, 0}, Execution::Device, false, false, DType::BF16); });
    suite.add("gdn_fold_bf16/host_zero",
              [] { return run_case(k36x24, 2, {0}, Execution::Host, false, false, DType::BF16); });
    suite.add("gdn_fold_bf16/expanded_17",
              [] {
                  std::vector<int> commits(17, 0);
                  commits.front() = 1;
                  commits.back()  = 4;
                  return run_case(k36x24, 4, commits, Execution::Device, false, false, DType::BF16);
              },
              {{"suite", "concurrency"}});
    suite.add("gdn_fold_bf16/cancellation", [] { return cancellation(DType::BF16); });
    suite.add("gdn_fold_bf16/decay_product_overflow",
              [] { return extreme_decay(50.f, 0.f, DType::BF16); });
    suite.add("gdn_fold_bf16/decay_product_underflow",
              [] { return extreme_decay(-50.f, 0x1p126f, DType::BF16); });
    suite.add("gdn_fold_bf16/chained_trajectory", [] { return chain(DType::BF16); });
    return suite.main(argc, argv);
}
