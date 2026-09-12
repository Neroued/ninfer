#include "ninfer/ops/context_kv_materialize.h"

#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kLayers             = static_cast<int>(ops::kContextKVMaterializeLayers);
constexpr int kHidden             = 5120;
constexpr int kRows               = 1024;
constexpr int kHeadDim            = 128;
constexpr int kHeads              = 8;
constexpr int kCapacity           = 2048;
constexpr int kLaneCapacity       = 8;
constexpr int kPaddedCapacity     = 2056;
constexpr std::uint16_t kSentinel = 0xa5a5U;
constexpr double kEpsilon         = 1.0e-6;
constexpr double kTheta           = 1.0e7;

constexpr ReductionCriterion kKeyCriterion{
    3.0e-3,
    1.0e-2,
    6.0e-3,
};

constexpr ReductionCriterion kValueCriterion{
    2.9e-3,
    4.0e-3,
    3.8e-3,
};

std::size_t cache_elements() {
    return static_cast<std::size_t>(kHeadDim) * kPaddedCapacity * kHeads * kLaneCapacity;
}

std::size_t cache_index(int lane, int head, int slot, int dim) {
    return static_cast<std::size_t>(dim) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kPaddedCapacity) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(kHeads) * lane));
}

std::vector<float> make_context(int columns, std::uint32_t seed) {
    std::vector<float> result(static_cast<std::size_t>(kHidden) * columns);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < kHidden; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 29U +
                                             static_cast<std::uint32_t>(column) * 47U + seed * 13U;
            const int centered = static_cast<int>(coordinate % 257U) - 128;
            result[static_cast<std::size_t>(column) * kHidden + row] =
                bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * (1.0F / 192.0F)));
        }
    }
    return result;
}

std::vector<float> make_norm(int layer) {
    std::vector<float> result(kHeadDim);
    for (int dim = 0; dim < kHeadDim; ++dim) {
        const float value = 0.75F + static_cast<float>(layer) * 0.03125F +
                            static_cast<float>((dim * 7 + layer) % 19) * (1.0F / 128.0F);
        result[static_cast<std::size_t>(dim)] = bf16_to_f32(f32_to_bf16(value));
    }
    return result;
}

struct LayerStorage {
    quantized_weight::PackedWeight key_host;
    quantized_weight::PackedWeight value_host;
    DeviceBuffer parent_device;
    std::vector<std::uint8_t> parent_host;
    std::vector<float> norm_host;
    DeviceBuffer norm_device;
    GuardedDeviceBuffer cache_k{cache_elements() * sizeof(std::uint16_t)};
    GuardedDeviceBuffer cache_v{cache_elements() * sizeof(std::uint16_t)};
};

struct Fixture {
    std::array<LayerStorage, kLayers> storage;
    std::array<std::size_t, 9> observed_peak{};
    std::array<ops::ContextKVMaterializeLayerView, kLayers> views;

    Fixture() {
        const quantized_weight::PatternedWeightOptions weight_options{
            quantized_weight::RowSplitScalePattern::Tiny};
        for (int layer = 0; layer < kLayers; ++layer) {
            LayerStorage& target = storage[static_cast<std::size_t>(layer)];
            target.key_host      = quantized_weight::make_patterned_weight(
                QType::W8G32_F16S, kRows, kHidden, 0x310U + 2U * layer, weight_options);
            target.value_host = quantized_weight::make_patterned_weight(
                QType::W8G32_F16S, kRows, kHidden, 0x311U + 2U * layer, weight_options);
            constexpr std::size_t parent_codes = 6144ULL * kHidden;
            target.parent_host.resize(parent_codes + 6144ULL * (kHidden / 32) * 2, 0x63);
            const auto put = [&](const quantized_weight::PackedWeight& weight, int row) {
                std::copy_n(weight.payload.data(), weight.code_plane_bytes,
                            target.parent_host.data() + row * kHidden);
                std::copy_n(weight.payload.data() + weight.scale_plane_offset,
                            weight.scale_plane_bytes,
                            target.parent_host.data() + parent_codes + row * (kHidden / 32) * 2);
            };
            put(target.key_host, 4096);
            put(target.value_host, 5120);
            target.parent_device = to_device(target.parent_host);
            const auto row_view  = [&](const quantized_weight::PackedWeight& weight, int row) {
                auto result = weight.device_weight(target.parent_device.p);
                result.qdata =
                    static_cast<const std::uint8_t*>(target.parent_device.p) + row * kHidden;
                result.scales = static_cast<const std::uint8_t*>(target.parent_device.p) +
                                parent_codes + row * (kHidden / 32) * 2;
                result.payload_bytes = target.parent_host.size();
                return result;
            };
            target.norm_host   = make_norm(layer);
            target.norm_device = to_device_bf16(target.norm_host);
            target.cache_k.fill(0xa5);
            target.cache_v.fill(0xa5);

            views[static_cast<std::size_t>(layer)] = {
                row_view(target.key_host, 4096),
                row_view(target.value_host, 5120),
                Tensor(target.norm_device.p, DType::BF16, {kHeadDim}),
                CyclicKVCacheLayerView{
                    .k               = Tensor(target.cache_k.data(), DType::BF16,
                                              {kHeadDim, kPaddedCapacity, kHeads, kLaneCapacity}),
                    .v               = Tensor(target.cache_v.data(), DType::FP16,
                                              {kHeadDim, kPaddedCapacity, kHeads, kLaneCapacity}),
                    .capacity        = kCapacity,
                    .padded_capacity = kPaddedCapacity,
                    .num_kv_heads    = kHeads,
                    .head_dim        = kHeadDim,
                    .lane_capacity   = kLaneCapacity,
                },
            };
        }
    }

    void reset_cache() {
        for (LayerStorage& layer : storage) {
            layer.cache_k.fill(0xa5);
            layer.cache_v.fill(0xa5);
        }
    }
};

float represented_projection(const quantized_weight::PackedWeight& weight, int row,
                             const float* input) {
    const double dot = quantized_weight::dot_fp64(weight, row, input, kHidden);
    // Round the FP64 mathematical dot directly to the semantic BF16 V boundary.
    // BF16 has eight significand bits and a minimum subnormal quantum of 2^-133.
    int exponent = 0;
    (void)std::frexp(dot, &exponent);
    const double quantum = std::ldexp(1.0, std::max(-133, exponent - 8));
    return static_cast<float>(std::nearbyint(dot / quantum) * quantum);
}

void append_key_oracle(const LayerStorage& layer, const float* input, int head, int position,
                       std::vector<double>& expected) {
    std::array<double, kHeadDim> raw{};
    double square_sum = 0.0;
    for (int dim = 0; dim < kHeadDim; ++dim) {
        const int row = head * kHeadDim + dim;
        raw[static_cast<std::size_t>(dim)] =
            quantized_weight::dot_fp64(layer.key_host, row, input, kHidden);
        square_sum += raw[static_cast<std::size_t>(dim)] * raw[static_cast<std::size_t>(dim)];
    }
    const double inverse = 1.0 / std::sqrt(square_sum / kHeadDim + kEpsilon);
    for (int pair = 0; pair < kHeadDim / 2; ++pair) {
        const double first = raw[static_cast<std::size_t>(pair)] * inverse *
                             layer.norm_host[static_cast<std::size_t>(pair)];
        const double second = raw[static_cast<std::size_t>(pair + kHeadDim / 2)] * inverse *
                              layer.norm_host[static_cast<std::size_t>(pair + kHeadDim / 2)];
        const double angle = static_cast<double>(position) *
                             std::pow(kTheta, -2.0 * static_cast<double>(pair) / kHeadDim);
        const double cosine = std::cos(angle);
        const double sine   = std::sin(angle);
        expected.push_back(first * cosine - second * sine);
        expected.push_back(second * cosine + first * sine);
    }
}

int verify_state_effect(const std::string& label, const Fixture& fixture,
                        const std::vector<int>& positions, const std::vector<int>& counts,
                        const std::vector<int>& state_slots, int width) {
    std::vector<std::set<int>> written_slots(kLaneCapacity);
    for (std::size_t batch = 0; batch < counts.size(); ++batch) {
        for (int index = 0; index < counts[batch]; ++index) {
            written_slots[static_cast<std::size_t>(state_slots[batch])].insert(
                positions[batch * static_cast<std::size_t>(width) + index] & (kCapacity - 1));
        }
    }

    int failures = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        const LayerStorage& source = fixture.storage[static_cast<std::size_t>(layer)];
        const auto cache_k = from_device<std::uint16_t>(source.cache_k.data(), cache_elements());
        const auto cache_v = from_device<std::uint16_t>(source.cache_v.data(), cache_elements());
        int first_bad      = -1;
        for (int lane = 0; lane < kLaneCapacity && first_bad < 0; ++lane) {
            for (int head = 0; head < kHeads && first_bad < 0; ++head) {
                for (int slot = 0; slot < kPaddedCapacity && first_bad < 0; ++slot) {
                    const bool written =
                        written_slots[static_cast<std::size_t>(lane)].contains(slot);
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        const std::size_t index = cache_index(lane, head, slot, dim);
                        const bool k_changed    = cache_k[index] != kSentinel;
                        const bool v_changed    = cache_v[index] != kSentinel;
                        if (k_changed != written || v_changed != written) {
                            first_bad = static_cast<int>(index);
                            break;
                        }
                    }
                }
            }
        }
        if (first_bad >= 0) {
            std::cerr << label << " layer=" << layer
                      << ": invalid state-effect footprint at element " << first_bad << '\n';
            ++failures;
        }
    }
    return failures;
}

int verify_numeric_samples(const std::string& label, const Fixture& fixture,
                           const std::vector<float>& context, const std::vector<int>& positions,
                           const std::vector<int>& counts, const std::vector<int>& state_slots,
                           int width) {
    std::vector<std::pair<int, int>> samples;
    for (int batch = 0; batch < static_cast<int>(counts.size()); ++batch) {
        if (counts[static_cast<std::size_t>(batch)] == 0) continue;
        samples.emplace_back(batch, 0);
        if (counts[static_cast<std::size_t>(batch)] > 1) {
            samples.emplace_back(batch, counts[static_cast<std::size_t>(batch)] - 1);
        }
    }

    if (samples.empty()) return 0;
    int failures = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        const LayerStorage& source = fixture.storage[static_cast<std::size_t>(layer)];
        const auto cache_k_bits =
            from_device<std::uint16_t>(source.cache_k.data(), cache_elements());
        const auto cache_v_bits =
            from_device<std::uint16_t>(source.cache_v.data(), cache_elements());
        std::vector<double> key_got;
        std::vector<double> key_expected;
        std::vector<double> value_got;
        std::vector<double> value_expected;
        const std::vector<int> heads = (width == 1 || width == 8 || width == 16 || width > 16)
                                           ? std::vector<int>{0, 3, 7}
                                           : std::vector<int>{width % 8};
        const std::array<int, 4> value_rows{0, 127, 511, 1023};
        for (const auto [batch, local] : samples) {
            const int column   = batch * width + local;
            const int position = positions[static_cast<std::size_t>(column)];
            const int slot     = position & (kCapacity - 1);
            const int lane     = state_slots[static_cast<std::size_t>(batch)];
            const float* input = context.data() + static_cast<std::size_t>(column) * kHidden;
            for (const int head : heads) {
                std::vector<double> head_expected;
                append_key_oracle(source, input, head, position, head_expected);
                for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                    const int first_dim  = pair;
                    const int second_dim = pair + kHeadDim / 2;
                    key_got.push_back(
                        bf16_to_f32(cache_k_bits[cache_index(lane, head, slot, first_dim)]));
                    key_expected.push_back(head_expected[static_cast<std::size_t>(2 * pair)]);
                    key_got.push_back(
                        bf16_to_f32(cache_k_bits[cache_index(lane, head, slot, second_dim)]));
                    key_expected.push_back(head_expected[static_cast<std::size_t>(2 * pair + 1)]);
                }
            }
            for (const int row : value_rows) {
                const int head          = row / kHeadDim;
                const int dim           = row - head * kHeadDim;
                const float represented = represented_projection(source.value_host, row, input);
                const std::uint16_t expected_bits =
                    quantized_weight::detail::f32_to_f16(represented);
                value_expected.push_back(quantized_weight::detail::f16_to_f32(expected_bits));
                value_got.push_back(quantized_weight::detail::f16_to_f32(
                    cache_v_bits[cache_index(lane, head, slot, dim)]));
            }
        }
        failures += verify_reduction(label + " K layer=" + std::to_string(layer), key_got,
                                     key_expected, kKeyCriterion);
        failures += verify_reduction(label + " V layer=" + std::to_string(layer), value_got,
                                     value_expected, kValueCriterion);
    }
    return failures;
}

int run_case(Fixture& fixture, const std::string& label, int width, int batch,
             const std::vector<int>& counts, const std::vector<int>& state_slots,
             std::vector<int> positions, std::uint32_t input_seed, bool narrow = false,
             bool zero_input = false) {
    const int columns          = width * batch;
    std::vector<float> context = make_context(columns, input_seed);
    if (zero_input) std::fill(context.begin(), context.end(), 0.0f);
    DeviceBuffer context_device   = to_device_bf16(context);
    DeviceBuffer positions_device = to_device_i32(positions);
    DeviceBuffer counts_device    = to_device_i32(counts);
    DeviceBuffer slots_device     = to_device_i32(state_slots);
    const std::size_t workspace_bytes =
        ops::context_kv_materialize_workspace_capacity_bytes(batch, width, width);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    Tensor context_tensor(context_device.p, DType::BF16, {kHidden, width, batch});
    Tensor positions_tensor(positions_device.p, DType::I32, {width, batch});
    Tensor counts_tensor(counts_device.p, DType::I32, {batch});
    Tensor slots_tensor(slots_device.p, DType::I32, {batch});
    const ops::ContextKVMaterializeExecutionEnvelope envelope{
        narrow ? static_cast<std::uint32_t>(*std::min_element(counts.begin(), counts.end())) : 0U,
        narrow ? static_cast<std::uint32_t>(*std::max_element(counts.begin(), counts.end()))
               : static_cast<std::uint32_t>(width)};
    const auto launch = [&](cudaStream_t stream) {
        ops::context_kv_materialize(context_tensor, positions_tensor, counts_tensor, slots_tensor,
                                    fixture.views, envelope, workspace, stream);
    };
    launch(nullptr);
    cuda_synchronize();
    int failures = verify_state_effect(label, fixture, positions, counts, state_slots, width);
    failures += verify_exact("counts readonly", from_device<int>(counts_device, batch), counts);
    failures +=
        verify_exact("positions readonly", from_device<int>(positions_device, columns), positions);
    failures += verify_exact("slots readonly", from_device<int>(slots_device, batch), state_slots);
    failures +=
        verify_numeric_samples(label, fixture, context, positions, counts, state_slots, width);
    failures += scratch.verify_guards(label + " workspace");
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) ++failures;
    fixture.observed_peak[batch] = std::max(fixture.observed_peak[batch], workspace.peak_used());
    if (!narrow && (batch == 8 || (batch == 1 && (width == 1 || width == 8 || width >= 16)))) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "graph stream");
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] { launch(stream); });
            executable.instantiate(definition);
            for (int replay = 0; replay < 3; ++replay) {
                std::vector<int> next_counts(batch), next_slots(batch), next_positions(columns, -1);
                for (int b = 0; b < batch; ++b) {
                    next_counts[b] = replay == 0   ? width
                                     : replay == 1 ? (b + width / 2) % (width + 1)
                                                   : 0;
                    next_slots[b]  = next_counts[b] ? (b * 3 + 1) % 8 : -1;
                    for (int i = 0; i < next_counts[b]; ++i)
                        next_positions[b * width + i] = 262140 + b * 4096 + i;
                }
                fixture.reset_cache();
                cuda_synchronize();
                counts_device.copy_from_host(next_counts.data(), batch * 4);
                slots_device.copy_from_host(next_slots.data(), batch * 4);
                positions_device.copy_from_host(next_positions.data(), columns * 4);
                executable.launch(stream);
                cuda_synchronize(stream);
                failures +=
                    verify_state_effect(label + " replay " + std::to_string(replay), fixture,
                                        next_positions, next_counts, next_slots, width);
                if (replay != 2)
                    failures +=
                        verify_numeric_samples(label + " replay", fixture, context, next_positions,
                                               next_counts, next_slots, width);
                failures += scratch.verify_guards(label + " replay scratch");
                failures += verify_exact("replay counts readonly",
                                         from_device<int>(counts_device, batch), next_counts);
                failures +=
                    verify_exact("replay positions readonly",
                                 from_device<int>(positions_device, columns), next_positions);
                failures += verify_exact("replay slots readonly",
                                         from_device<int>(slots_device, batch), next_slots);
            }
        }
        cuda_check(cudaStreamDestroy(stream), "destroy graph stream");
    }
    failures += verify_exact("context readonly", from_device_bf16(context_device, context.size()),
                             std::vector<double>(context.begin(), context.end()));
    for (const auto& layer : fixture.storage) {
        failures += layer.cache_k.verify_guards(label + " K ring");
        failures += layer.cache_v.verify_guards(label + " V ring");
    }

    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "context_kv_materialize: SKIP (CUDA unavailable)\n";
            return 77;
        }

        Fixture fixture;
        int failures = 0;
        const std::vector<int> widths{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        for (int width : widths)
            for (int batch = 1; batch <= 8; ++batch) {
                std::vector<int> counts(batch), slots(batch), positions(width * batch, -12345);
                for (int b = 0; b < batch; ++b) {
                    counts[b] = b == batch - 1 || b % 3 == 2 ? width
                                : b % 3 == 1                 ? std::max(1, width / 2)
                                                             : 0;
                    slots[b]  = counts[b] ? (b * 3 + 2) % 8 : -1;
                    for (int i = 0; i < counts[b]; ++i)
                        positions[b * width + i] = 2046 + b * 4096 + i;
                }
                fixture.reset_cache();
                failures +=
                    run_case(fixture, "W=" + std::to_string(width) + " B=" + std::to_string(batch),
                             width, batch, counts, slots, positions, 0x701U + width);
            }
        for (int width : widths)
            for (int count : {0, 1, std::max(1, width / 2)}) {
                fixture.reset_cache();
                std::vector<int> positions(width * 8, -1), counts(8, count), slots(8, -1);
                for (int b = 0; b < 8; ++b) {
                    if (count) slots[b] = 7 - b;
                    for (int i = 0; i < count; ++i)
                        positions[b * width + i] = 262140 + b * 4096 + i;
                }
                failures += run_case(fixture,
                                     "narrow W=" + std::to_string(width) +
                                         " count=" + std::to_string(count),
                                     width, 8, counts, slots, positions, 0x704U, true);
            }
        for (int width : {17, 81, 85, 86, 97, 128, 129, 256, 257, 2048}) {
            fixture.reset_cache();
            std::vector<int> positions(width);
            for (int i = 0; i < width; ++i) positions[i] = 262140 + i;
            failures += run_case(fixture, "prefill W=" + std::to_string(width), width, 1, {width},
                                 {7}, positions, 0x702U);
        }
        for (int count : {0, 1, 96, 128, 257}) {
            fixture.reset_cache();
            std::vector<int> positions(2048, -1);
            for (int i = 0; i < count; ++i) positions[i] = 262140 + i;
            failures += run_case(fixture, "wide narrow count=" + std::to_string(count), 2048, 1,
                                 {count}, {count ? 3 : -1}, positions, 0x705U, true);
        }
        fixture.reset_cache();
        failures += run_case(fixture, "zero projection", 3, 3, {0, 1, 3}, {-1, 7, 2},
                             {-1, -1, -1, 2047, -1, -1, 262142, 262143, 262144}, 0, true, true);
        for (int batch = 1; batch <= 8; ++batch) {
            const auto capacity = ops::context_kv_materialize_workspace_capacity_bytes(
                batch, 1, batch == 1 ? 2048 : 16);
            if (capacity != fixture.observed_peak[batch]) {
                std::cerr << "workspace interval peak mismatch B=" << batch << "\n";
                ++failures;
            }
        }
        for (const auto& layer : fixture.storage) {
            failures += verify_exact(
                "QKV parent readonly",
                from_device<std::uint8_t>(layer.parent_device, layer.parent_host.size()),
                layer.parent_host);
            failures +=
                verify_exact("norm readonly", from_device_bf16(layer.norm_device, kHeadDim),
                             std::vector<double>(layer.norm_host.begin(), layer.norm_host.end()));
        }

        if (failures != 0) {
            std::cerr << "context_kv_materialize failures=" << failures << '\n';
            return 1;
        }
        std::cout << "context_kv_materialize: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "context_kv_materialize: " << error.what() << '\n';
        return 1;
    }
}
