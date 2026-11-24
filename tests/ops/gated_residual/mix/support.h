#pragma once

#include "ninfer/ops/gated_residual.h"
#include "ops/direct_bf16_weight.h"
#include "core/device.h"

namespace ninfer::test::hc::mix {
inline constexpr int kWidth = 2560, kStreams = 4, kRank = 320, kTotal = kWidth * kStreams;
inline constexpr int kCatalogColumns = 17;
enum class Kind { Block, Final };
enum class Pattern { Dense, Zero, InjectionOnly, RowMapping, Saturation };
const char* name(Kind kind);

struct Weights {
    Weights(Kind kind, Pattern pattern);
    int check() const;
    Kind kind;
    Pattern pattern;
    direct_bf16_weight::HostWeight host_down, host_up;
    std::vector<float> down_values, up_values;
    GuardedDeviceBuffer device_down, device_up;
    Weight down, up;
};

struct Fixture {
    Fixture(Weights& weights, int columns);
    void upload(std::span<const float> catalog, cudaStream_t stream);
    void invoke(CudaExecutionView execution);
    int check(const std::string& label) const;
    std::size_t query(int first, int last) const;

    Weights& weights;
    int columns;
    std::size_t capacity;
    std::vector<std::uint16_t> input_words;
    GuardedDeviceBuffer input, output, injection, scratch;
    MemoryArena workspace;
    Tensor z, x, j;
};

std::vector<float> input_catalog(Pattern pattern, int phase = 0);

struct Column {
    std::vector<double> output, injection;
};

Column reference_column(const Weights& weights, std::span<const float> input);
std::vector<Column> reference(const Weights& weights, std::span<const float> catalog);
int verify(const Fixture& fixture, std::span<const Column> reference, const std::string& label);
int validate(Weights& weights, std::span<const Column> reference, CudaExecutionView execution);
} // namespace ninfer::test::hc::mix
