#include "ninfer/ops/linear.h"
#include "ops/linear/q4/q4_rowsplit_plan.h"
#include "ops/op_tester.h"
#include "ops/row_split_pack.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

using S = ops::detail::Q4ScheduleId;
using V = ops::detail::Q4KernelVariant;

int failures = 0;

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        const cudaError_t error = cudaMalloc(&data_, bytes);
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc(") + std::to_string(bytes) +
                                     ") failed: " + cudaGetErrorString(error));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) { cudaFree(data_); }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() const noexcept { return data_; }

    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* data_        = nullptr;
    std::size_t bytes_ = 0;
};

void require_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

test::row_split::PackedWeight make_deterministic_q4(std::int32_t rows, std::int32_t k,
                                                    std::uint32_t seed) {
    const std::int32_t groups = k / 64;
    const std::size_t group_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(groups);
    const std::size_t code_bytes   = group_count * 32u;
    const std::size_t scale_offset = align_up(code_bytes, 256u);
    const std::size_t scale_bytes  = group_count * sizeof(std::uint16_t);

    test::row_split::PackedWeight packed;
    packed.nibble_plane_bytes = code_bytes;
    packed.high_plane_offset  = scale_offset;
    packed.high_plane_bytes   = 0;
    packed.scale_plane_offset = scale_offset;
    packed.scale_plane_bytes  = scale_bytes;
    packed.payload.resize(scale_offset + scale_bytes);

    constexpr std::size_t kPatternBytes = 1u << 20;
    std::vector<std::uint8_t> pattern(std::min(kPatternBytes, code_bytes));
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        std::uint32_t value = static_cast<std::uint32_t>(i) + seed * 0x9e3779b9u;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        pattern[i] = static_cast<std::uint8_t>(value ^ (value >> 8));
    }
    for (std::size_t offset = 0; offset < code_bytes; offset += pattern.size()) {
        const std::size_t count = std::min(pattern.size(), code_bytes - offset);
        std::memcpy(packed.payload.data() + offset, pattern.data(), count);
    }
    std::fill(packed.payload.begin() + code_bytes, packed.payload.begin() + scale_offset, 0);

    constexpr std::array<std::uint16_t, 4> kScales{{0x2800u, 0x2a00u, 0x2c00u, 0x2e00u}};
    for (std::size_t group = 0; group < group_count; ++group) {
        const std::uint16_t scale  = kScales[(group + seed) % kScales.size()];
        const std::size_t offset   = scale_offset + group * sizeof(std::uint16_t);
        packed.payload[offset]     = static_cast<std::uint8_t>(scale & 0xffu);
        packed.payload[offset + 1] = static_cast<std::uint8_t>(scale >> 8);
    }

    packed.weight.qtype            = QType::Q4G64_F16S;
    packed.weight.layout           = QuantLayout::RowSplit;
    packed.weight.scale_dtype      = DType::FP16;
    packed.weight.payload          = packed.payload.data();
    packed.weight.payload_bytes    = packed.payload.size();
    packed.weight.high_plane_bytes = 0;
    packed.weight.qdata            = packed.payload.data();
    packed.weight.qhigh            = nullptr;
    packed.weight.scales           = packed.payload.data() + scale_offset;
    packed.weight.group_size       = 64;
    packed.weight.group            = 64;
    packed.weight.ndim             = 2;
    packed.weight.shape[0]         = rows;
    packed.weight.shape[1]         = k;
    packed.weight.shape[2]         = 1;
    packed.weight.shape[3]         = 1;
    packed.weight.padded_shape[0]  = rows;
    packed.weight.padded_shape[1]  = k;
    packed.weight.padded_shape[2]  = 1;
    packed.weight.padded_shape[3]  = 1;
    packed.weight.n                = rows;
    packed.weight.k                = k;
    return packed;
}

void fill_device_input(void* device, std::size_t count, std::uint32_t seed) {
    constexpr std::array<float, 16> kValues{{
        -1.0f,
        -0.75f,
        -0.5f,
        -0.25f,
        -0.125f,
        -0.0625f,
        -0.03125f,
        -0.015625f,
        0.015625f,
        0.03125f,
        0.0625f,
        0.125f,
        0.25f,
        0.5f,
        0.75f,
        1.0f,
    }};
    constexpr std::size_t kChunkWords = 1u << 20;
    std::vector<std::uint16_t> host(std::min(kChunkWords, count));
    for (std::size_t offset = 0; offset < count; offset += host.size()) {
        const std::size_t words = std::min(host.size(), count - offset);
        for (std::size_t i = 0; i < words; ++i) {
            const std::size_t global = offset + i;
            const std::size_t index =
                (global * 13u + (global >> 7u) + static_cast<std::size_t>(seed)) % kValues.size();
            host[i] = test::f32_to_bf16(kValues[index]);
        }
        require_cuda(cudaMemcpy(static_cast<std::uint8_t*>(device) + offset * sizeof(std::uint16_t),
                                host.data(), words * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
                     "input cudaMemcpy");
    }
}

int compare_device_bf16(const std::string& label, const void* public_out, const void* fixed_out,
                        std::size_t count) {
    constexpr std::size_t kChunkWords = 1u << 20;
    std::vector<std::uint16_t> public_bits(std::min(kChunkWords, count));
    std::vector<std::uint16_t> fixed_bits(public_bits.size());

    for (std::size_t offset = 0; offset < count; offset += public_bits.size()) {
        const std::size_t words = std::min(public_bits.size(), count - offset);
        require_cuda(cudaMemcpy(public_bits.data(),
                                static_cast<const std::uint8_t*>(public_out) +
                                    offset * sizeof(std::uint16_t),
                                words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                     "public output cudaMemcpy");
        require_cuda(
            cudaMemcpy(fixed_bits.data(),
                       static_cast<const std::uint8_t*>(fixed_out) + offset * sizeof(std::uint16_t),
                       words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "fixed output cudaMemcpy");
        for (std::size_t i = 0; i < words; ++i) {
            if (public_bits[i] != fixed_bits[i]) {
                std::cerr << "FAIL " << label << ": BF16 word mismatch at " << offset + i
                          << " public=0x" << std::hex << public_bits[i] << " fixed=0x"
                          << fixed_bits[i] << std::dec << '\n';
                return 1;
            }
        }
    }
    return 0;
}

std::string plan_name(const ops::detail::Q4Plan& plan) {
    return std::string(ops::detail::q4_schedule_name(plan.schedule)) + "." +
           ops::detail::q4_kernel_variant_name(plan.variant);
}

struct RoutePoint {
    std::int32_t cols;
    S schedule;
    V variant;
};

constexpr std::array<RoutePoint, 4> k1024Routes{{
    {1, S::GemvR1W8Direct, V::None},
    {2, S::SimtR8C4, V::Predicated},
    {15, S::SimtR8C4, V::Predicated},
    {16, S::SimtR8C8, V::Full},
}};

constexpr std::array<RoutePoint, 5> k4096Routes{{
    {1, S::GemvR1W8Direct, V::None},
    {2, S::SimtR8C4, V::Predicated},
    {4, S::SimtR8C4, V::Full},
    {5, S::SimtR8C8, V::Predicated},
    {16, S::SimtR8C8, V::Full},
}};

constexpr std::array<RoutePoint, 5> k6144Routes{{
    {1, S::GemvR1W8Direct, V::None},
    {2, S::SimtR8C4, V::Predicated},
    {7, S::SimtR8C4, V::Predicated},
    {8, S::SimtR8C8, V::Full},
    {16, S::SimtR8C8, V::Full},
}};

constexpr std::array<RoutePoint, 7> k7168Routes{{
    {1, S::GemvR1W8Direct, V::None},
    {2, S::SimtR8C4, V::Predicated},
    {7, S::SimtR8C4, V::Predicated},
    {8, S::SimtR8C8, V::Full},
    {9, S::SimtR8C4, V::Predicated},
    {15, S::SimtR8C4, V::Predicated},
    {16, S::SimtR8C8, V::Full},
}};

constexpr std::array<RoutePoint, 6> k34816Routes{{
    {1, S::GemvR1W8Direct, V::None},
    {2, S::SimtR8C4, V::Predicated},
    {4, S::SimtR8C4, V::Full},
    {5, S::SimtR8C8, V::Predicated},
    {16, S::SimtR8C8, V::Full},
    {17, S::MmaR64C128, V::Predicated},
}};

constexpr std::array<RoutePoint, 1> k131072Routes{{
    {1, S::GemvR4W1Direct, V::None},
}};

constexpr std::array<RoutePoint, 6> k3456Routes{{
    {4, S::SimtR8C4, V::Predicated},
    {36, S::SimtR8C4, V::Predicated},
    {40, S::MmaR64C64, V::Predicated},
    {320, S::MmaR64C64, V::Full},
    {324, S::MmaR64C128, V::Predicated},
    {131072, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 9> k4304Routes{{
    {4, S::SimtR8C4, V::Predicated},
    {8, S::SimtR8C8, V::Predicated},
    {12, S::SimtR8C4, V::Predicated},
    {16, S::SimtR8C8, V::Predicated},
    {24, S::SimtR8C8, V::Predicated},
    {28, S::MmaR64C64, V::Predicated},
    {320, S::MmaR64C64, V::Predicated},
    {324, S::MmaR64C128, V::Predicated},
    {131072, S::MmaR64C128, V::Predicated},
}};

struct SupportCase {
    const char* label;
    std::int32_t rows;
    std::int32_t k;
    const RoutePoint* routes;
    std::size_t route_count;
    std::uint32_t seed;
};

std::array<bool, 6> schedules_seen{};
std::array<bool, 3> variants_seen{};

void record_coverage(S schedule, V variant) {
    switch (schedule) {
    case S::GemvR4W1Direct:
        schedules_seen[0] = true;
        break;
    case S::GemvR1W8Direct:
        schedules_seen[1] = true;
        break;
    case S::SimtR8C4:
        schedules_seen[2] = true;
        break;
    case S::SimtR8C8:
        schedules_seen[3] = true;
        break;
    case S::MmaR64C64:
        schedules_seen[4] = true;
        break;
    case S::MmaR64C128:
        schedules_seen[5] = true;
        break;
    }
    switch (variant) {
    case V::None:
        variants_seen[0] = true;
        break;
    case V::Full:
        variants_seen[1] = true;
        break;
    case V::Predicated:
        variants_seen[2] = true;
        break;
    }
}

void expect_public_rejection(const std::string& label, const Tensor& x, const Weight& weight,
                             Tensor& out, WorkspaceArena& workspace) {
    const ops::detail::Q4Problem problem = ops::detail::q4_rowsplit_problem(x, weight, out);
    if (ops::detail::q4_rowsplit_admits(problem)) {
        std::cerr << "FAIL " << label << ": internal admission unexpectedly accepted the problem\n";
        ++failures;
    }
    try {
        ops::linear(x, weight, out, workspace, nullptr);
        require_cuda(cudaDeviceSynchronize(), "unexpected public launch synchronize");
        std::cerr << "FAIL " << label << ": public linear accepted an unregistered Q4 problem\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << label << ": wrong exception: " << error.what() << '\n';
        ++failures;
    }
}

void run_support(const SupportCase& support) {
    std::int32_t max_cols = 0;
    for (std::size_t i = 0; i < support.route_count; ++i) {
        max_cols = std::max(max_cols, support.routes[i].cols);
    }
    test::row_split::PackedWeight packed =
        make_deterministic_q4(support.rows, support.k, support.seed);
    DeviceBuffer device_weight(packed.payload.size());
    require_cuda(cudaMemcpy(device_weight.data(), packed.payload.data(), packed.payload.size(),
                            cudaMemcpyHostToDevice),
                 "weight cudaMemcpy");
    const Weight weight = packed.device_weight(device_weight.data());
    std::vector<std::uint8_t>().swap(packed.payload);

    const std::size_t input_words =
        static_cast<std::size_t>(support.k) * static_cast<std::size_t>(max_cols);
    const std::size_t output_words =
        static_cast<std::size_t>(support.rows) * static_cast<std::size_t>(max_cols);
    DeviceBuffer input(input_words * sizeof(std::uint16_t));
    DeviceBuffer public_output(output_words * sizeof(std::uint16_t));
    DeviceBuffer fixed_output(output_words * sizeof(std::uint16_t));
    fill_device_input(input.data(), input_words, support.seed + 100u);
    WorkspaceArena workspace(256);

    for (std::size_t i = 0; i < support.route_count; ++i) {
        const RoutePoint& route = support.routes[i];
        const std::string label = std::string(support.label) + " C=" + std::to_string(route.cols);
        Tensor x(input.data(), DType::BF16, {support.k, route.cols});
        Tensor public_out(public_output.data(), DType::BF16, {support.rows, route.cols});
        Tensor fixed_out(fixed_output.data(), DType::BF16, {support.rows, route.cols});
        const ops::detail::Q4Problem problem{support.rows, support.k, support.k, route.cols};

        ops::detail::Q4Plan plan{};
        try {
            plan = ops::detail::q4_rowsplit_resolve_plan(problem);
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << label << ": resolver threw: " << error.what() << '\n';
            ++failures;
            continue;
        }
        if (plan.schedule != route.schedule || plan.variant != route.variant) {
            const ops::detail::Q4Plan expected{route.schedule, route.variant};
            std::cerr << "FAIL " << label << ": expected plan " << plan_name(expected) << ", got "
                      << plan_name(plan) << '\n';
            ++failures;
        }
        record_coverage(plan.schedule, plan.variant);

        const std::size_t bytes =
            static_cast<std::size_t>(support.rows) * route.cols * sizeof(std::uint16_t);
        require_cuda(cudaMemset(public_output.data(), 0xa5, bytes), "public output cudaMemset");
        require_cuda(cudaMemset(fixed_output.data(), 0x5a, bytes), "fixed output cudaMemset");
        try {
            ops::linear(x, weight, public_out, workspace, nullptr);
            require_cuda(cudaDeviceSynchronize(), "public linear synchronize");
            ops::detail::q4_rowsplit_launch_candidate(plan.schedule, plan.variant, x, weight,
                                                      fixed_out, nullptr);
            require_cuda(cudaDeviceSynchronize(), "fixed candidate synchronize");
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << label << ": launch failed: " << error.what() << '\n';
            ++failures;
            continue;
        }
        failures += compare_device_bf16(label, public_output.data(), fixed_output.data(),
                                        static_cast<std::size_t>(support.rows) *
                                            static_cast<std::size_t>(route.cols));
    }

    if (support.rows == 3456) {
        Tensor x(input.data(), DType::BF16, {support.k, 5});
        Tensor out(public_output.data(), DType::BF16, {support.rows, 5});
        expect_public_rejection(std::string(support.label) + " rejected non-step C=5", x, weight,
                                out, workspace);
    }
}

void standalone_public_rejection(const char* label, std::int32_t rows, std::int32_t k,
                                 std::int32_t cols, std::uint32_t seed) {
    test::row_split::PackedWeight packed = make_deterministic_q4(rows, k, seed);
    DeviceBuffer device_weight(packed.payload.size());
    require_cuda(cudaMemcpy(device_weight.data(), packed.payload.data(), packed.payload.size(),
                            cudaMemcpyHostToDevice),
                 "rejection weight cudaMemcpy");
    const Weight weight = packed.device_weight(device_weight.data());
    std::vector<std::uint8_t>().swap(packed.payload);

    const std::size_t input_words = static_cast<std::size_t>(k) * static_cast<std::size_t>(cols);
    const std::size_t output_words =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    DeviceBuffer input(input_words * sizeof(std::uint16_t));
    DeviceBuffer output(output_words * sizeof(std::uint16_t));
    fill_device_input(input.data(), input_words, seed + 100u);
    WorkspaceArena workspace(256);
    Tensor x(input.data(), DType::BF16, {k, cols});
    Tensor out(output.data(), DType::BF16, {rows, cols});
    expect_public_rejection(label, x, weight, out, workspace);
}

void verify_declared_coverage() {
    for (std::size_t i = 0; i < schedules_seen.size(); ++i) {
        if (!schedules_seen[i]) {
            std::cerr << "FAIL dispatch test did not exercise production schedule index " << i
                      << '\n';
            ++failures;
        }
    }
    for (std::size_t i = 0; i < variants_seen.size(); ++i) {
        if (!variants_seen[i]) {
            std::cerr << "FAIL dispatch test did not exercise variant index " << i << '\n';
            ++failures;
        }
    }
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    constexpr std::array<SupportCase, 9> supports{{
        {"Q4 [1024,5120]", 1024, 5120, k1024Routes.data(), k1024Routes.size(), 11u},
        {"Q4 [4096,5120]", 4096, 5120, k4096Routes.data(), k4096Routes.size(), 13u},
        {"Q4 [6144,5120]", 6144, 5120, k6144Routes.data(), k6144Routes.size(), 17u},
        {"Q4 [7168,5120]", 7168, 5120, k7168Routes.data(), k7168Routes.size(), 37u},
        {"Q4 [34816,5120]", 34816, 5120, k34816Routes.data(), k34816Routes.size(), 19u},
        {"Q4 [131072,5120]", 131072, 5120, k131072Routes.data(), k131072Routes.size(), 23u},
        {"Q4 [131072,2048]", 131072, 2048, k131072Routes.data(), k131072Routes.size(), 27u},
        {"Q4 [3456,1152]", 3456, 1152, k3456Routes.data(), k3456Routes.size(), 29u},
        {"Q4 [4304,1152]", 4304, 1152, k4304Routes.data(), k4304Routes.size(), 31u},
    }};

    try {
        for (const SupportCase& support : supports) { run_support(support); }
        verify_declared_coverage();
    } catch (const std::exception& error) {
        std::cerr << "FAIL Q4 dispatch test infrastructure: " << error.what() << '\n';
        return 1;
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4 Linear public dispatch\n";
    return failures == 0 ? 0 : 1;
}
