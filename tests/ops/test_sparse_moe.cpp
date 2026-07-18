#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/row_split_pack.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kTopK           = 8;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kExpertGateRows = 1024;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertGateRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

struct QuantGeometry {
    int group;
    int code_bytes_per_group;
    int high_bytes_per_group;
};

QuantGeometry geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {64, 32, 0};
    case QType::Q5G64_F16S:
        return {64, 32, 8};
    case QType::Q6G64_F16S:
        return {64, 32, 16};
    case QType::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("unsupported test codec");
    }
}

class DeviceRowSplit {
public:
    DeviceRowSplit(QType qtype, std::int32_t rows, std::int32_t columns)
        : qtype_(qtype), rows_(rows), columns_(columns), geometry_(geometry(qtype)),
          groups_per_row_(columns / geometry_.group),
          code_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.code_bytes_per_group),
          high_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.high_bytes_per_group),
          scale_row_bytes_(static_cast<std::size_t>(groups_per_row_) * 2),
          codes_(static_cast<std::size_t>(rows) * code_row_bytes_),
          scales_(static_cast<std::size_t>(rows) * scale_row_bytes_) {
        if (high_row_bytes_ != 0) {
            high_ = std::make_unique<DBuf>(static_cast<std::size_t>(rows) * high_row_bytes_);
        }
        cudaMemset(codes_.p, 0, codes_.bytes);
        if (high_) { cudaMemset(high_->p, 0, high_->bytes); }
        cudaMemset(scales_.p, 0, scales_.bytes);
    }

    void copy_rows(const row_split::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != qtype_ || source.weight.k != columns_ || destination_row < 0 ||
            source_rows <= 0 || destination_row > rows_ - source_rows) {
            throw std::invalid_argument("invalid packed test row copy");
        }
        const std::size_t code_bytes  = static_cast<std::size_t>(source_rows) * code_row_bytes_;
        const std::size_t high_bytes  = static_cast<std::size_t>(source_rows) * high_row_bytes_;
        const std::size_t scale_bytes = static_cast<std::size_t>(source_rows) * scale_row_bytes_;
        cudaMemcpy(static_cast<std::uint8_t*>(codes_.p) +
                       static_cast<std::size_t>(destination_row) * code_row_bytes_,
                   source.payload.data(), code_bytes, cudaMemcpyHostToDevice);
        if (high_bytes != 0) {
            cudaMemcpy(static_cast<std::uint8_t*>(high_->p) +
                           static_cast<std::size_t>(destination_row) * high_row_bytes_,
                       source.payload.data() + source.high_plane_offset, high_bytes,
                       cudaMemcpyHostToDevice);
        }
        cudaMemcpy(static_cast<std::uint8_t*>(scales_.p) +
                       static_cast<std::size_t>(destination_row) * scale_row_bytes_,
                   source.payload.data() + source.scale_plane_offset, scale_bytes,
                   cudaMemcpyHostToDevice);
    }

    Weight weight() const {
        Weight out{};
        out.payload          = codes_.p;
        out.payload_bytes    = codes_.bytes + (high_ ? high_->bytes : 0) + scales_.bytes;
        out.high_plane_bytes = high_ ? high_->bytes : 0;
        out.qtype            = qtype_;
        out.group_size       = static_cast<std::uint32_t>(geometry_.group);
        out.qdata            = codes_.p;
        out.qhigh            = high_ ? high_->p : nullptr;
        out.scales           = scales_.p;
        out.n                = rows_;
        out.k                = columns_;
        out.group            = geometry_.group;
        out.layout           = QuantLayout::RowSplit;
        out.scale_dtype      = DType::FP16;
        out.ndim             = 2;
        out.shape[0]         = rows_;
        out.shape[1]         = columns_;
        out.padded_shape[0]  = rows_;
        out.padded_shape[1]  = columns_;
        return out;
    }

private:
    QType qtype_;
    std::int32_t rows_;
    std::int32_t columns_;
    QuantGeometry geometry_;
    std::int32_t groups_per_row_;
    std::size_t code_row_bytes_;
    std::size_t high_row_bytes_;
    std::size_t scale_row_bytes_;
    DBuf codes_;
    std::unique_ptr<DBuf> high_;
    DBuf scales_;
};

Weight dense_bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    Weight out{};
    out.payload         = data;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2ULL;
    out.qtype           = QType::BF16_CTRL;
    out.qdata           = data;
    out.n               = rows;
    out.k               = columns;
    out.layout          = QuantLayout::Contiguous;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

std::vector<float> make_input(int variant) {
    std::vector<float> input(kHidden);
    input[0] = 1.0f;
    for (int k = 1; k < kHidden; ++k) {
        input[k] = 0.025f + static_cast<float>((k * 7 + variant * 11) % 19) * 0.002f;
    }
    // Reserve eight one-hot marker coordinates so a shared router matrix can produce different
    // selected experts for adjacent Small-T columns without changing the represented BF16 domain.
    for (int marker = 0; marker < 8; ++marker) { input[kHidden - 8 + marker] = 0.0f; }
    input[kHidden - 8 + variant] = 1.0f;
    round_to_bf16(input);
    return input;
}

std::vector<float> make_residual(int variant) {
    std::vector<float> residual(kHidden);
    for (int row = 0; row < kHidden; ++row) {
        residual[row] = 0.125f + static_cast<float>((row * 5 + variant * 13) % 23) * 0.003f;
    }
    round_to_bf16(residual);
    return residual;
}

std::vector<float> make_gate_up(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    const std::int32_t split = rows / 2;
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias       = row < split ? 0.75f : 1.15f;
        const float row_factor = 1.0f + static_cast<float>((row + seed) % 7) * 0.025f;
        float* destination     = source.data() + static_cast<std::size_t>(row) * columns;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 11LL + column * 5LL + seed) % 15) - 7;
            destination[column] =
                0.008f * expert_factor * row_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

std::vector<float> make_down(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias   = 0.45f + static_cast<float>((row + seed) % 5) * 0.08f;
        float* destination = source.data() + static_cast<std::size_t>(row) * columns;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern   = static_cast<int>((row * 7LL + column * 13LL + seed) % 17) - 8;
            destination[column] = 0.007f * expert_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

std::vector<float> make_router_pattern(const std::vector<std::array<int, kTopK>>& selected_columns,
                                       const std::vector<int>& tie_excluded_columns) {
    if (selected_columns.empty() || selected_columns.size() > 8 ||
        tie_excluded_columns.size() != selected_columns.size()) {
        throw std::invalid_argument("invalid router test pattern");
    }
    std::vector<float> router(static_cast<std::size_t>(kExperts + 1) * kHidden);
    // Give every router row the same dense background. It cancels from the
    // routing order while exercising every S1 K partition against the oracle.
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
    for (std::size_t column = 0; column < selected_columns.size(); ++column) {
        const int marker       = kHidden - 8 + static_cast<int>(column);
        const auto& selected   = selected_columns[column];
        const int tie_excluded = tie_excluded_columns[column];
        for (int rank = 0; rank < kTopK; ++rank) {
            const float score = rank == kTopK - 1 && tie_excluded >= 0 ? 2.0f : 4.0f - 0.25f * rank;
            router[static_cast<std::size_t>(selected[rank]) * kHidden + marker] += score + 8.0f;
        }
        if (tie_excluded >= 0) {
            router[static_cast<std::size_t>(tie_excluded) * kHidden + marker] += 10.0f;
        }
    }
    router[static_cast<std::size_t>(kExperts) * kHidden] += 0.375f;
    round_to_bf16(router);
    return router;
}

std::vector<float> make_router(const std::array<int, kTopK>& selected, int tie_excluded) {
    return make_router_pattern({selected}, {tie_excluded});
}

struct HostExpert {
    int id;
    row_split::PackedWeight gate_up;
    row_split::PackedWeight down;
};

const HostExpert& find_expert(const std::vector<HostExpert>& experts, int id) {
    const auto it = std::find_if(experts.begin(), experts.end(),
                                 [id](const HostExpert& expert) { return expert.id == id; });
    if (it == experts.end()) { throw std::logic_error("oracle selected an unpopulated expert"); }
    return *it;
}

double dot(const std::vector<float>& matrix, std::int32_t row, std::int32_t columns,
           const std::vector<double>& input) {
    const float* weights = matrix.data() + static_cast<std::size_t>(row) * columns;
    double result        = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
        result += static_cast<double>(weights[column]) * input[column];
    }
    return result;
}

// The one SparseMoe oracle. It independently evaluates the complete logical formula from
// represented BF16 public values and exact test-only row-split decode; it never observes or
// reproduces a D1-D4 workspace value or production reduction tree.
std::vector<double> sparse_moe_oracle(const std::vector<float>& input,
                                      const std::vector<float>& residual,
                                      const std::vector<float>& router,
                                      const std::vector<HostExpert>& experts,
                                      const row_split::PackedWeight& shared_gate_up,
                                      const row_split::PackedWeight& shared_down,
                                      const std::array<int, kTopK>& expected_selected) {
    std::vector<double> x(input.begin(), input.end());
    std::vector<double> scores(kExperts + 1, 0.0);
    for (int row = 0; row < kExperts + 1; ++row) {
        const float* weights = router.data() + static_cast<std::size_t>(row) * kHidden;
        for (int column = 0; column < kHidden; ++column) {
            scores[row] += static_cast<double>(weights[column]) * x[column];
        }
    }

    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    });
    std::array<int, kTopK> ids{};
    std::array<double, kTopK> alpha{};
    double denominator = 0.0;
    for (int route = 0; route < kTopK; ++route) {
        ids[route]   = order[route];
        alpha[route] = std::exp(scores[ids[route]] - scores[ids[0]]);
        denominator += alpha[route];
    }
    for (double& value : alpha) { value /= denominator; }
    if (ids != expected_selected) {
        throw std::logic_error("test router did not produce the intended ordered top-8");
    }
    const double shared_scale = 1.0 / (1.0 + std::exp(-scores[kExperts]));

    std::array<std::vector<double>, kTopK + 1> activation;
    for (auto& values : activation) { values.resize(kIntermediate); }
    for (int route = 0; route < kTopK; ++route) {
        const HostExpert& expert = find_expert(experts, ids[route]);
        for (int j = 0; j < kIntermediate; ++j) {
            const double gate    = dot(expert.gate_up.dequant, j, kHidden, x);
            const double up      = dot(expert.gate_up.dequant, kIntermediate + j, kHidden, x);
            activation[route][j] = (gate / (1.0 + std::exp(-gate))) * up;
        }
    }
    for (int j = 0; j < kIntermediate; ++j) {
        const double gate    = dot(shared_gate_up.dequant, j, kHidden, x);
        const double up      = dot(shared_gate_up.dequant, kIntermediate + j, kHidden, x);
        activation[kTopK][j] = (gate / (1.0 + std::exp(-gate))) * up;
    }

    std::vector<double> output(kHidden);
    for (int row = 0; row < kHidden; ++row) {
        double value = static_cast<double>(residual[row]);
        for (int route = 0; route < kTopK; ++route) {
            const HostExpert& expert = find_expert(experts, ids[route]);
            value += alpha[route] * dot(expert.down.dequant, row, kIntermediate, activation[route]);
        }
        value += shared_scale * dot(shared_down.dequant, row, kIntermediate, activation[kTopK]);
        output[row] = static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
    }
    return output;
}

struct CodecProfile {
    const char* name;
    QType routed_gate_up;
    QType routed_down;
    Tolerance tolerance;
};

struct RouteCase {
    std::array<int, kTopK> selected;
    int tie_excluded;
};

int expect_invalid(const char* label, const auto& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": expected std::invalid_argument\n";
    return 1;
}

int run_case(const CodecProfile& profile, const RouteCase& route, const char* case_name, int tokens,
             int unique_columns, bool graph_replay, bool validate_contract,
             const std::vector<RouteCase>& route_columns = {}) {
    if (!route_columns.empty() && static_cast<int>(route_columns.size()) != unique_columns) {
        throw std::invalid_argument("route column count must match unique input columns");
    }
    std::vector<std::vector<float>> input_columns;
    std::vector<std::vector<float>> residual_columns;
    input_columns.reserve(unique_columns);
    residual_columns.reserve(unique_columns);
    for (int column = 0; column < unique_columns; ++column) {
        input_columns.push_back(make_input(column));
        residual_columns.push_back(make_residual(column));
    }
    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
    for (int token = 0; token < tokens; ++token) {
        const int pattern = token % unique_columns;
        std::copy(input_columns[pattern].begin(), input_columns[pattern].end(),
                  input.begin() + static_cast<std::size_t>(token) * kHidden);
        std::copy(residual_columns[pattern].begin(), residual_columns[pattern].end(),
                  residual.begin() + static_cast<std::size_t>(token) * kHidden);
    }
    std::vector<std::array<int, kTopK>> selected_columns;
    std::vector<int> tie_excluded_columns;
    selected_columns.reserve(unique_columns);
    tie_excluded_columns.reserve(unique_columns);
    for (int column = 0; column < unique_columns; ++column) {
        const RouteCase& column_route = route_columns.empty() ? route : route_columns[column];
        selected_columns.push_back(column_route.selected);
        tie_excluded_columns.push_back(column_route.tie_excluded);
    }
    const std::vector<float> router = make_router_pattern(selected_columns, tie_excluded_columns);

    DBuf device_input         = to_device_bf16(input);
    DBuf device_residual_seed = to_device_bf16(residual);
    DBuf device_destination(residual.size() * sizeof(std::uint16_t));
    DBuf device_router = to_device_bf16(router);
    cudaMemcpy(device_destination.p, device_residual_seed.p, device_destination.bytes,
               cudaMemcpyDeviceToDevice);

    DeviceRowSplit routed_gate(profile.routed_gate_up, kRoutedGateRows, kHidden);
    DeviceRowSplit routed_down(profile.routed_down, kRoutedDownRows, kIntermediate);
    DeviceRowSplit shared_gate(QType::W8G32_F16S, kSharedGateRows, kHidden);
    DeviceRowSplit shared_down_device(QType::W8G32_F16S, kHidden, kIntermediate);

    std::vector<HostExpert> host_experts;
    std::vector<int> populated_experts;
    for (const auto& selected : selected_columns) {
        for (int expert : selected) {
            if (std::find(populated_experts.begin(), populated_experts.end(), expert) ==
                populated_experts.end()) {
                populated_experts.push_back(expert);
            }
        }
    }
    host_experts.reserve(populated_experts.size());
    for (int expert : populated_experts) {
        const float factor = 0.8f + static_cast<float>((expert * 3) % 11) * 0.045f;
        auto gate_up       = row_split::pack_row_split_lowbit(
            make_gate_up(kExpertGateRows, kHidden, 100u + static_cast<std::uint32_t>(expert),
                               factor),
            kExpertGateRows, kHidden, profile.routed_gate_up);
        auto down = row_split::pack_row_split_lowbit(
            make_down(kHidden, kIntermediate, 300u + static_cast<std::uint32_t>(expert), factor),
            kHidden, kIntermediate, profile.routed_down);
        routed_gate.copy_rows(gate_up, expert * kExpertGateRows);
        routed_down.copy_rows(down, expert * kHidden);
        host_experts.push_back({expert, std::move(gate_up), std::move(down)});
    }

    auto host_shared_gate = row_split::pack_w8g32_row_split(
        make_gate_up(kSharedGateRows, kHidden, 0x512u, 0.93f), kSharedGateRows, kHidden);
    auto host_shared_down = row_split::pack_w8g32_row_split(
        make_down(kHidden, kIntermediate, 0x731u, 0.87f), kHidden, kIntermediate);
    shared_gate.copy_rows(host_shared_gate, 0);
    shared_down_device.copy_rows(host_shared_down, 0);

    ops::SparseMoeWeights weights{
        dense_bf16_weight(device_router.p, kExperts + 1, kHidden),
        routed_gate.weight(),
        routed_down.weight(),
        shared_gate.weight(),
        shared_down_device.weight(),
    };
    Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
    Tensor destination(device_destination.p, DType::BF16, {kHidden, tokens});
    const std::size_t workspace_bytes = ops::sparse_moe_workspace_bytes(tokens);
    WorkspaceArena workspace(workspace_bytes);

    std::vector<std::vector<double>> reference_columns;
    reference_columns.reserve(unique_columns);
    for (int column = 0; column < unique_columns; ++column) {
        const RouteCase& column_route = route_columns.empty() ? route : route_columns[column];
        reference_columns.push_back(
            sparse_moe_oracle(input_columns[column], residual_columns[column], router, host_experts,
                              host_shared_gate, host_shared_down, column_route.selected));
    }
    std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
    for (int token = 0; token < tokens; ++token) {
        const auto& column = reference_columns[token % unique_columns];
        std::copy(column.begin(), column.end(),
                  reference.begin() + static_cast<std::size_t>(token) * kHidden);
    }

    if (graph_replay) {
        cudaStream_t stream  = nullptr;
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        cudaMemcpyAsync(destination.data, device_residual_seed.p, destination.bytes(),
                        cudaMemcpyDeviceToDevice, stream);
        ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                        stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(exec, stream);
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                        nullptr);
        cudaDeviceSynchronize();
    }

    const std::vector<double> actual = from_device_bf16(device_destination, input.size());
    int failures                     = verify(case_name, actual, reference, profile.tolerance);
    bool changed_residual            = false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != static_cast<double>(residual[index])) {
            changed_residual = true;
            break;
        }
    }
    if (!changed_residual) {
        std::cerr << case_name << ": sparse-MoE term did not change the nonzero residual\n";
        ++failures;
    }

    if (validate_contract) {
        failures += expect_invalid("sparse_moe zero max_tokens",
                                   [] { (void)ops::sparse_moe_workspace_bytes(0); });
        const std::size_t decode_bytes  = ops::sparse_moe_workspace_bytes(1);
        const std::size_t small_t_bytes = ops::sparse_moe_workspace_bytes(8);
        if (small_t_bytes <= decode_bytes ||
            ops::sparse_moe_workspace_bytes(std::numeric_limits<std::int32_t>::max()) !=
                small_t_bytes) {
            std::cerr << "sparse_moe: workspace range maximum does not cover Small-T\n";
            ++failures;
        }
        WorkspaceArena too_small(workspace_bytes - 1);
        failures += expect_invalid("sparse_moe workspace capacity", [&] {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, too_small,
                            nullptr);
        });
        Tensor mismatched_destination(device_destination.p, DType::BF16, {kHidden, tokens + 1});
        failures += expect_invalid("sparse_moe token mismatch", [&] {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, mismatched_destination,
                            workspace, nullptr);
        });
        Tensor no_tokens = x;
        no_tokens.ne[1]  = 0;
        failures += expect_invalid("sparse_moe positive T", [&] {
            ops::sparse_moe(no_tokens, weights, ops::SparseMoeEpilogue::AddResidual, destination,
                            workspace, nullptr);
        });
        ops::SparseMoeWeights bad_shape = weights;
        bad_shape.shared_down.n         = kHidden - 1;
        failures += expect_invalid("sparse_moe weight shape", [&] {
            ops::sparse_moe(x, bad_shape, ops::SparseMoeEpilogue::AddResidual, destination,
                            workspace, nullptr);
        });
        ops::SparseMoeWeights bad_format = weights;
        bad_format.routed_down.qtype     = QType::W8G32_F16S;
        failures += expect_invalid("sparse_moe weight format", [&] {
            ops::sparse_moe(x, bad_format, ops::SparseMoeEpilogue::AddResidual, destination,
                            workspace, nullptr);
        });
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    const std::array<CodecProfile, 3> profiles = {{
        {"sparse_moe q4+q5", QType::Q4G64_F16S, QType::Q5G64_F16S, Tolerance::sparse_moe_q4_q5()},
        {"sparse_moe q4+q6", QType::Q4G64_F16S, QType::Q6G64_F16S, Tolerance::sparse_moe_q4_q6()},
        {"sparse_moe w8+w8", QType::W8G32_F16S, QType::W8G32_F16S, Tolerance::sparse_moe_w8_w8()},
    }};
    const RouteCase ordinary_route{{255, 0, 17, 31, 63, 127, 191, 223}, -1};
    const RouteCase boundary_tie{{0, 17, 31, 63, 127, 191, 223, 254}, 255};
    const std::vector<RouteCase> correlated_routes = {
        {{{0, 32, 64, 96, 128, 160, 224, 255}}, -1}, {{{0, 32, 64, 96, 129, 161, 225, 254}}, -1},
        {{{0, 32, 65, 97, 129, 161, 225, 253}}, -1}, {{{1, 33, 65, 97, 129, 162, 226, 253}}, -1},
        {{{1, 33, 65, 98, 130, 162, 226, 252}}, -1}, {{{1, 34, 66, 98, 130, 162, 227, 252}}, -1},
    };

    int failures = 0;
    for (std::size_t index = 0; index < profiles.size(); ++index) {
        failures += run_case(profiles[index], ordinary_route, profiles[index].name, 1, 1, false,
                             index == 0);
    }
    // Codec decoding, routing tie behavior, exact-T dispatch, and per-token routing are
    // orthogonal test dimensions.
    failures += run_case(profiles[0], boundary_tie, "sparse_moe boundary tie", 1, 1, false, false);
    for (int tokens = 2; tokens <= 8; ++tokens) {
        const std::string name = "sparse_moe small-T" + std::to_string(tokens);
        failures +=
            run_case(profiles[0], ordinary_route, name.c_str(), tokens, tokens, tokens == 8, false);
    }
    failures +=
        run_case(profiles[1], ordinary_route, "sparse_moe q4+q6 small-T", 6, 6, false, false);
    failures +=
        run_case(profiles[2], ordinary_route, "sparse_moe w8+w8 small-T", 6, 6, false, false);
    failures += run_case(profiles[0], ordinary_route, "sparse_moe correlated routes", 6, 6, true,
                         false, correlated_routes);
    failures += run_case(profiles[0], ordinary_route, "sparse_moe serial T9", 9, 8, false, false);
    std::cout << (failures ? "FAIL" : "OK") << " sparse_moe correctness\n";
    return failures ? 1 : 0;
}
