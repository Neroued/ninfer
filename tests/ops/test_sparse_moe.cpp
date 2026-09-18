#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// SparseMoe is the exact qwen3_6_35b_a3b post-mixer Op. qwen3_6_27b has a dense SwiGLU
// post-mixer and therefore contributes no second SparseMoe geometry.
constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kTopK           = 8;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kExpertGateRows = 2 * kIntermediate;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertGateRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

// All registered variants consume represented BF16 activations and expose one BF16 destination.
// The bound belongs to that A16 compute profile, not to a codec, T, private route, or schedule.
// The limits retain modest headroom over the measured maxima from the complete case matrix:
// rel-L2 1.117e-2 and pointwise 3.687e-3.
constexpr ReductionCriterion kSparseMoeA16Tolerance{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 0.0,
};

constexpr std::size_t kOutputGuardBytes = 256;
constexpr std::uint8_t kOutputGuardByte = 0xa5;

struct QuantGeometry {
    std::int32_t group;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    case QType::Q8_G32_FP16:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("sparse_moe test: unsupported codec");
    }
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int compare_output(const std::string& label, const std::vector<double>& actual,
                   const std::vector<double>& reference) {
    return verify_reduction(label, actual, reference, kSparseMoeA16Tolerance);
}

class GuardedBf16Output {
public:
    explicit GuardedBf16Output(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte),
          words_(words) {}

    void* data() noexcept { return storage_.data(); }

    const void* data() const noexcept { return storage_.data(); }

    std::vector<double> values() const { return from_device_bf16(data(), words_); }

    int verify_guards(const std::string& label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
    std::size_t words_;
};

class DeviceRowSplit {
public:
    DeviceRowSplit(QType qtype, std::int32_t rows, std::int32_t columns)
        : qtype_(qtype), rows_(rows), columns_(columns), geometry_(quant_geometry(qtype)),
          groups_per_row_(columns / geometry_.group),
          code_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.code_bytes_per_group),
          high_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.high_bytes_per_group),
          scale_row_bytes_(static_cast<std::size_t>(groups_per_row_) * sizeof(std::uint16_t)),
          codes_(static_cast<std::size_t>(rows) * code_row_bytes_),
          scales_(static_cast<std::size_t>(rows) * scale_row_bytes_) {
        codes_.fill(0);
        scales_.fill(0);
        if (high_row_bytes_ != 0) {
            high_ =
                std::make_unique<DeviceBuffer>(static_cast<std::size_t>(rows) * high_row_bytes_);
            high_->fill(0);
        }
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != qtype_ || source.weight.k != columns_ || source_rows < 1 ||
            destination_row < 0 || destination_row > rows_ - source_rows) {
            throw std::invalid_argument("sparse_moe test: invalid packed row copy");
        }
        codes_.copy_from_host(source.payload.data(),
                              static_cast<std::size_t>(source_rows) * code_row_bytes_,
                              static_cast<std::size_t>(destination_row) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            high_->copy_from_host(source.payload.data() + source.high_plane_offset,
                                  static_cast<std::size_t>(source_rows) * high_row_bytes_,
                                  static_cast<std::size_t>(destination_row) * high_row_bytes_);
        }
        scales_.copy_from_host(source.payload.data() + source.scale_plane_offset,
                               static_cast<std::size_t>(source_rows) * scale_row_bytes_,
                               static_cast<std::size_t>(destination_row) * scale_row_bytes_);
    }

    Weight weight() const {
        Weight result{};
        result.payload          = codes_.p;
        result.payload_bytes    = codes_.bytes + (high_ ? high_->bytes : 0) + scales_.bytes;
        result.high_plane_bytes = high_ ? high_->bytes : 0;
        result.qtype            = qtype_;
        result.group_size       = static_cast<std::uint32_t>(geometry_.group);
        result.qdata            = codes_.p;
        result.qhigh            = high_ ? high_->p : nullptr;
        result.scales           = scales_.p;
        result.n                = rows_;
        result.k                = columns_;
        result.group            = geometry_.group;
        result.layout           = QuantLayout::RowSplit;
        result.scale_dtype      = DType::FP16;
        result.ndim             = 2;
        result.shape[0]         = rows_;
        result.shape[1]         = columns_;
        result.padded_shape[0]  = rows_;
        result.padded_shape[1]  = columns_;
        return result;
    }

    int verify_rows(const std::string& label, const quantized_weight::PackedWeight& source,
                    std::int32_t destination_row) const {
        const std::int32_t source_rows = source.weight.n;
        int failures                   = 0;
        failures += verify_plane(label + " code", codes_, source.payload.data(),
                                 static_cast<std::size_t>(destination_row) * code_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            failures += verify_plane(label + " high", *high_,
                                     source.payload.data() + source.high_plane_offset,
                                     static_cast<std::size_t>(destination_row) * high_row_bytes_,
                                     static_cast<std::size_t>(source_rows) * high_row_bytes_);
        }
        failures += verify_plane(label + " scale", scales_,
                                 source.payload.data() + source.scale_plane_offset,
                                 static_cast<std::size_t>(destination_row) * scale_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * scale_row_bytes_);
        return failures;
    }

private:
    static int verify_plane(const std::string& label, const DeviceBuffer& device,
                            const std::uint8_t* expected, std::size_t offset, std::size_t bytes) {
        std::vector<std::uint8_t> actual(bytes);
        device.copy_to_host(actual.data(), bytes, offset);
        if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
        std::cerr << label << ": persistent weight was modified\n";
        return 1;
    }

    QType qtype_;
    std::int32_t rows_;
    std::int32_t columns_;
    QuantGeometry geometry_;
    std::int32_t groups_per_row_;
    std::size_t code_row_bytes_;
    std::size_t high_row_bytes_;
    std::size_t scale_row_bytes_;
    DeviceBuffer codes_;
    std::unique_ptr<DeviceBuffer> high_;
    DeviceBuffer scales_;
};

Weight dense_bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    Weight result{};
    result.payload         = data;
    result.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    result.qtype           = QType::BF16;
    result.qdata           = data;
    result.n               = rows;
    result.k               = columns;
    result.layout          = QuantLayout::Contiguous;
    result.ndim            = 2;
    result.shape[0]        = rows;
    result.shape[1]        = columns;
    result.padded_shape[0] = rows;
    result.padded_shape[1] = columns;
    return result;
}

struct RoutePattern {
    std::array<int, kTopK> selected;
    int tied_excluded = -1;
};

constexpr std::array<RoutePattern, 3> kRoutePatterns{{
    {{{255, 0, 17, 31, 63, 127, 191, 223}}, -1},
    {{{0, 17, 31, 63, 127, 191, 223, 254}}, 255},
    {{{223, 191, 127, 63, 31, 17, 0, 255}}, -1},
}};

std::vector<float> make_input(int pattern) {
    std::vector<float> input(kHidden);
    input[0] = 1.0f;
    for (int column = 1; column < kHidden; ++column) {
        input[column] = 0.025f + static_cast<float>((column * 7 + pattern * 11) % 19) * 0.002f;
    }
    for (std::size_t marker = 0; marker < kRoutePatterns.size(); ++marker) {
        input[kHidden - static_cast<int>(kRoutePatterns.size()) + static_cast<int>(marker)] = 0.0f;
    }
    input[kHidden - static_cast<int>(kRoutePatterns.size()) + pattern] = 1.0f;
    round_to_bf16(input);
    return input;
}

std::vector<float> make_residual(int pattern) {
    std::vector<float> residual(kHidden);
    for (int row = 0; row < kHidden; ++row) {
        residual[row] = 0.125f + static_cast<float>((row * 5 + pattern * 13) % 23) * 0.003f;
    }
    round_to_bf16(residual);
    return residual;
}

std::vector<float> make_router() {
    std::vector<float> router(static_cast<std::size_t>(kExperts + 1) * kHidden);
    for (int row = 0; row < kExperts + 1; ++row) {
        for (int column = 0; column < kHidden; ++column) {
            const int pattern = (column * 13 + 5) % 17 - 8;
            router[static_cast<std::size_t>(row) * kHidden + column] =
                static_cast<float>(pattern) * 0.001f;
        }
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        router[static_cast<std::size_t>(expert) * kHidden] -= 8.0f;
    }
    for (std::size_t pattern = 0; pattern < kRoutePatterns.size(); ++pattern) {
        const int marker =
            kHidden - static_cast<int>(kRoutePatterns.size()) + static_cast<int>(pattern);
        const RoutePattern& route = kRoutePatterns[pattern];
        for (int rank = 0; rank < kTopK; ++rank) {
            const float score =
                rank == kTopK - 1 && route.tied_excluded >= 0 ? 2.0f : 4.0f - 0.25f * rank;
            router[static_cast<std::size_t>(route.selected[rank]) * kHidden + marker] +=
                score + 8.0f;
        }
        if (route.tied_excluded >= 0) {
            router[static_cast<std::size_t>(route.tied_excluded) * kHidden + marker] += 10.0f;
        }
    }
    router[static_cast<std::size_t>(kExperts) * kHidden] += 0.375f;
    round_to_bf16(router);
    return router;
}

std::vector<float> make_gate_up(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    const std::int32_t split = rows / 2;
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias       = row < split ? 0.75f : 1.15f;
        const float row_factor = 1.0f + static_cast<float>((row + seed) % 7) * 0.025f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 11LL + column * 5LL + seed) % 15) - 7;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.008f * expert_factor * row_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

std::vector<float> make_down(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias = 0.45f + static_cast<float>((row + seed) % 5) * 0.08f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 7LL + column * 13LL + seed) % 17) - 8;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.007f * expert_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

struct HostExpert {
    int id;
    quantized_weight::PackedWeight gate_up;
    quantized_weight::PackedWeight down;
};

const HostExpert& find_expert(const std::vector<HostExpert>& experts, int id) {
    const auto found = std::find_if(experts.begin(), experts.end(),
                                    [id](const HostExpert& expert) { return expert.id == id; });
    if (found == experts.end()) {
        throw std::logic_error("sparse_moe oracle selected an unpopulated expert");
    }
    return *found;
}

template <class Function>
void parallel_rows(std::int32_t rows, Function&& function) {
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(rows, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * thread) / threads);
        const std::int32_t end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) { function(row); }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

double dot_fp64(const std::vector<float>& matrix, std::int32_t row, std::int32_t columns,
                const std::vector<double>& input) {
    const float* weights = matrix.data() + static_cast<std::size_t>(row) * columns;
    double result        = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
        result += static_cast<double>(weights[column]) * input[column];
    }
    return result;
}

// The one SparseMoe oracle. It directly evaluates the complete public formula from represented
// BF16 values and independently decoded logical weights, with FP64 accumulation throughout.
// It has no production route, staging cast, workspace dtype, reduction tree, or output rounding.
std::vector<double> sparse_moe_oracle(const std::vector<float>& input,
                                      const std::vector<float>& residual,
                                      const std::vector<float>& router,
                                      const std::vector<HostExpert>& experts,
                                      const quantized_weight::PackedWeight& shared_gate_up,
                                      const quantized_weight::PackedWeight& shared_down,
                                      const RoutePattern& intended_route) {
    const std::vector<double> x(input.begin(), input.end());
    std::vector<double> scores(kExperts + 1);
    parallel_rows(kExperts + 1,
                  [&](std::int32_t row) { scores[row] = dot_fp64(router, row, kHidden, x); });

    std::vector<int> ranked(kExperts);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&](int left, int right) {
        return scores[left] > scores[right] || (scores[left] == scores[right] && left < right);
    });

    std::array<int, kTopK> selected{};
    std::array<double, kTopK> route_weight{};
    double route_denominator = 0.0;
    for (int route = 0; route < kTopK; ++route) {
        selected[route]     = ranked[route];
        route_weight[route] = std::exp(scores[selected[route]] - scores[selected[0]]);
        route_denominator += route_weight[route];
    }
    for (double& weight : route_weight) { weight /= route_denominator; }
    if (selected != intended_route.selected) {
        throw std::logic_error("sparse_moe test router did not create its intended ordered top-8");
    }
    const double shared_scale = 1.0 / (1.0 + std::exp(-scores[kExperts]));

    std::array<std::vector<double>, kTopK> routed_activation;
    for (int route = 0; route < kTopK; ++route) {
        routed_activation[route].resize(kIntermediate);
        const HostExpert& expert = find_expert(experts, selected[route]);
        parallel_rows(kIntermediate, [&](std::int32_t row) {
            const double gate = dot_fp64(expert.gate_up.dequant, row, kHidden, x);
            const double up   = dot_fp64(expert.gate_up.dequant, kIntermediate + row, kHidden, x);
            routed_activation[route][row] = (gate / (1.0 + std::exp(-gate))) * up;
        });
    }

    std::vector<double> shared_activation(kIntermediate);
    parallel_rows(kIntermediate, [&](std::int32_t row) {
        const double gate      = dot_fp64(shared_gate_up.dequant, row, kHidden, x);
        const double up        = dot_fp64(shared_gate_up.dequant, kIntermediate + row, kHidden, x);
        shared_activation[row] = (gate / (1.0 + std::exp(-gate))) * up;
    });

    std::vector<double> output(kHidden);
    parallel_rows(kHidden, [&](std::int32_t row) {
        double value = static_cast<double>(residual[row]);
        for (int route = 0; route < kTopK; ++route) {
            const HostExpert& expert = find_expert(experts, selected[route]);
            value += route_weight[route] *
                     dot_fp64(expert.down.dequant, row, kIntermediate, routed_activation[route]);
        }
        value +=
            shared_scale * dot_fp64(shared_down.dequant, row, kIntermediate, shared_activation);
        output[row] = value;
    });
    return output;
}

struct CodecProfile {
    const char* name;
    QType routed_gate_up;
    QType routed_down;
    std::span<const std::int32_t> token_cases;
    bool verify_graph_replay;
};

class SparseMoeFixture {
public:
    explicit SparseMoeFixture(const CodecProfile& profile)
        : profile_(profile), router_(make_router()), router_bits_(bf16_bits(router_)),
          device_router_(to_device(router_bits_)),
          routed_gate_(profile.routed_gate_up, kRoutedGateRows, kHidden),
          routed_down_(profile.routed_down, kRoutedDownRows, kIntermediate),
          shared_gate_(QType::Q8_G32_FP16, kSharedGateRows, kHidden),
          shared_down_device_(QType::Q8_G32_FP16, kHidden, kIntermediate) {
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            inputs_.push_back(make_input(pattern));
            residuals_.push_back(make_residual(pattern));
        }

        std::vector<int> expert_ids;
        for (const RoutePattern& route : kRoutePatterns) {
            for (int expert : route.selected) {
                if (std::find(expert_ids.begin(), expert_ids.end(), expert) == expert_ids.end()) {
                    expert_ids.push_back(expert);
                }
            }
        }
        std::sort(expert_ids.begin(), expert_ids.end());
        for (int expert : expert_ids) {
            const float factor = 0.8f + static_cast<float>((expert * 3) % 11) * 0.045f;
            auto gate_up       = quantized_weight::pack_row_split_lowbit(
                make_gate_up(kExpertGateRows, kHidden, 100U + static_cast<std::uint32_t>(expert),
                                   factor),
                kExpertGateRows, kHidden, profile.routed_gate_up);
            auto down = quantized_weight::pack_row_split_lowbit(
                make_down(kHidden, kIntermediate, 300U + static_cast<std::uint32_t>(expert),
                          factor),
                kHidden, kIntermediate, profile.routed_down);
            routed_gate_.copy_rows(gate_up, expert * kExpertGateRows);
            routed_down_.copy_rows(down, expert * kHidden);
            experts_.push_back({expert, std::move(gate_up), std::move(down)});
        }

        shared_gate_host_ = quantized_weight::pack_q8_g32_row_split(
            make_gate_up(kSharedGateRows, kHidden, 0x512U, 0.93f), kSharedGateRows, kHidden);
        shared_down_host_ = quantized_weight::pack_q8_g32_row_split(
            make_down(kHidden, kIntermediate, 0x731U, 0.87f), kHidden, kIntermediate);
        shared_gate_.copy_rows(shared_gate_host_, 0);
        shared_down_device_.copy_rows(shared_down_host_, 0);

        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            references_.push_back(sparse_moe_oracle(inputs_[pattern], residuals_[pattern], router_,
                                                    experts_, shared_gate_host_, shared_down_host_,
                                                    kRoutePatterns[pattern]));
        }
    }

    int run(std::int32_t tokens, int first_pattern, bool graph_replay) {
        const std::string label = std::string(profile_.name) + " T=" + std::to_string(tokens);
        std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const int pattern = (first_pattern + token) % static_cast<int>(kRoutePatterns.size());
            std::copy(inputs_[pattern].begin(), inputs_[pattern].end(),
                      input.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(residuals_[pattern].begin(), residuals_[pattern].end(),
                      residual.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(references_[pattern].begin(), references_[pattern].end(),
                      reference.begin() + static_cast<std::size_t>(token) * kHidden);
        }

        const std::vector<std::uint16_t> input_bits    = bf16_bits(input);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        DeviceBuffer device_input                      = to_device(input_bits);
        DeviceBuffer residual_seed                     = to_device(residual_bits);
        GuardedBf16Output destination_storage(residual.size());
        cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                              cudaMemcpyDeviceToDevice),
                   "seed SparseMoe destination");

        const ops::SparseMoeWeights weights{
            dense_bf16_weight(device_router_.p, kExperts + 1, kHidden),
            routed_gate_.weight(),
            routed_down_.weight(),
            shared_gate_.weight(),
            shared_down_device_.weight(),
        };
        Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
        Tensor destination(destination_storage.data(), DType::BF16, {kHidden, tokens});
        const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(
            weights.routed_gate_up.qtype, weights.routed_down.qtype, tokens, tokens);
        WorkspaceArena workspace(workspace_bytes);

        if (graph_replay) {
            cudaStream_t stream  = nullptr;
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create SparseMoe graph stream");
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "begin SparseMoe graph capture");
            cuda_check(cudaMemcpyAsync(destination.data, residual_seed.p, destination.bytes(),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture SparseMoe residual seed");
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            stream);
            cuda_check(cudaStreamEndCapture(stream, &graph), "end SparseMoe graph capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                       "instantiate SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "launch SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "replay SparseMoe graph");
            cuda_check(cudaStreamSynchronize(stream), "synchronize SparseMoe graph");
            cuda_check(cudaGraphExecDestroy(exec), "destroy SparseMoe graph executable");
            cuda_check(cudaGraphDestroy(graph), "destroy SparseMoe graph");
            cuda_check(cudaStreamDestroy(stream), "destroy SparseMoe graph stream");
        } else {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            nullptr);
            cuda_synchronize();
        }

        int failures = compare_output(label, destination_storage.values(), reference);
        failures += destination_storage.verify_guards(label);
        failures +=
            verify_exact((label + " input preservation").c_str(),
                         from_device<std::uint16_t>(device_input, input_bits.size()), input_bits);
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        return failures;
    }

    int verify_persistent_inputs() const {
        int failures = 0;
        failures += verify_exact((std::string(profile_.name) + " router preservation").c_str(),
                                 from_device<std::uint16_t>(device_router_, router_bits_.size()),
                                 router_bits_);
        for (const HostExpert& expert : experts_) {
            failures += routed_gate_.verify_rows(
                std::string(profile_.name) + " routed gate expert " + std::to_string(expert.id),
                expert.gate_up, expert.id * kExpertGateRows);
            failures += routed_down_.verify_rows(
                std::string(profile_.name) + " routed down expert " + std::to_string(expert.id),
                expert.down, expert.id * kHidden);
        }
        failures += shared_gate_.verify_rows(std::string(profile_.name) + " shared gate",
                                             shared_gate_host_, 0);
        failures += shared_down_device_.verify_rows(std::string(profile_.name) + " shared down",
                                                    shared_down_host_, 0);
        return failures;
    }

private:
    const CodecProfile& profile_;
    std::vector<float> router_;
    std::vector<std::uint16_t> router_bits_;
    DeviceBuffer device_router_;
    DeviceRowSplit routed_gate_;
    DeviceRowSplit routed_down_;
    DeviceRowSplit shared_gate_;
    DeviceRowSplit shared_down_device_;
    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> residuals_;
    std::vector<HostExpert> experts_;
    quantized_weight::PackedWeight shared_gate_host_;
    quantized_weight::PackedWeight shared_down_host_;
    std::vector<std::vector<double>> references_;
};

int run_profile(const CodecProfile& profile) {
    SparseMoeFixture fixture(profile);
    int failures        = 0;
    std::size_t witness = 0;
    for (std::size_t index = 0; index < profile.token_cases.size(); ++index) {
        const std::int32_t tokens = profile.token_cases[index];
        witness =
            std::max(witness, ops::sparse_moe_workspace_capacity_bytes(
                                  profile.routed_gate_up, profile.routed_down, tokens, tokens));
        // Decode starts with the exact top-8 boundary tie; multi-token cases cycle the tie,
        // high/low expert ids, and a different ordering of the same experts.
        failures +=
            fixture.run(tokens, index == 0 ? 1 : 0, profile.verify_graph_replay && index == 1);
    }
    const std::size_t interval = ops::sparse_moe_workspace_capacity_bytes(
        profile.routed_gate_up, profile.routed_down, 1, profile.token_cases.back());
    if (interval != witness) {
        std::cerr << profile.name << ": interval workspace capacity missed a route witness\n";
        ++failures;
    }
    failures += fixture.verify_persistent_inputs();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // These are public-behavior cases, not route assertions. They exercise decode (T=1), the
    // Small-T supported-domain edges, each profile's first prefill T, the wide-prefill boundary,
    // and one call crossing the 4096-token internal slice without observing any private plan.
    constexpr std::array<std::int32_t, 6> kQ4Q5Tokens{{1, 2, 46, 47, 768, 4097}};
    constexpr std::array<std::int32_t, 5> kQ4Q6Tokens{{1, 2, 46, 47, 768}};
    constexpr std::array<std::int32_t, 5> kQ8Q8Tokens{{1, 2, 19, 20, 768}};
    const std::array<CodecProfile, 3> profiles{{
        {"sparse_moe q4+q5 a16", QType::Q4_G64_FP16, QType::Q5_G64_FP16, kQ4Q5Tokens, true},
        {"sparse_moe q4+q6 a16", QType::Q4_G64_FP16, QType::Q6_G64_FP16, kQ4Q6Tokens, false},
        {"sparse_moe q8+q8 a16", QType::Q8_G32_FP16, QType::Q8_G32_FP16, kQ8Q8Tokens, false},
    }};

    int failures = 0;
    for (const CodecProfile& profile : profiles) { failures += run_profile(profile); }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe correctness\n";
    return failures == 0 ? 0 : 1;
}
