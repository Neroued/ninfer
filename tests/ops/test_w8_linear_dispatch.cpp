#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ops/linear/w8/w8_rowsplit_plan.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"
#include "ops/op_tester.h"

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

using S = ops::detail::W8ScheduleId;
using V = ops::detail::W8KernelVariant;
using P = ops::detail::W8TailPolicy;

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

private:
    void* data_        = nullptr;
    std::size_t bytes_ = 0;
};

struct PackedW8 {
    std::vector<std::uint8_t> payload;
    Weight weight;
};

void require_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

PackedW8 make_deterministic_w8(std::int32_t rows, std::int32_t k, std::uint32_t seed) {
    const std::int32_t groups = k / 32;
    const std::size_t group_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(groups);
    const std::size_t code_bytes   = static_cast<std::size_t>(rows) * k;
    const std::size_t scale_offset = align_up(code_bytes, 256u);
    const std::size_t scale_bytes  = group_count * sizeof(std::uint16_t);

    PackedW8 packed;
    packed.payload.resize(scale_offset + scale_bytes);
    constexpr std::size_t kPatternBytes = 1u << 20;
    std::vector<std::uint8_t> pattern(std::min(kPatternBytes, code_bytes));
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const std::uint32_t mixed =
            static_cast<std::uint32_t>((i * 17u + seed * 131u + (i >> 7u)) % 255u);
        const std::int32_t signed_value = static_cast<std::int32_t>(mixed) - 127;
        pattern[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(signed_value));
    }
    for (std::size_t offset = 0; offset < code_bytes; offset += pattern.size()) {
        const std::size_t count = std::min(pattern.size(), code_bytes - offset);
        std::memcpy(packed.payload.data() + offset, pattern.data(), count);
    }
    std::fill(packed.payload.begin() + code_bytes, packed.payload.begin() + scale_offset, 0);

    constexpr std::array<std::uint16_t, 4> kScales{{0x2400u, 0x2600u, 0x2800u, 0x2a00u}};
    for (std::size_t group = 0; group < group_count; ++group) {
        const std::uint16_t scale  = kScales[(group + seed) % kScales.size()];
        const std::size_t offset   = scale_offset + group * sizeof(std::uint16_t);
        packed.payload[offset]     = static_cast<std::uint8_t>(scale & 0xffu);
        packed.payload[offset + 1] = static_cast<std::uint8_t>(scale >> 8);
    }

    Weight& weight          = packed.weight;
    weight.qtype            = QType::W8G32_F16S;
    weight.layout           = QuantLayout::RowSplit;
    weight.scale_dtype      = DType::FP16;
    weight.payload          = packed.payload.data();
    weight.payload_bytes    = packed.payload.size();
    weight.high_plane_bytes = 0;
    weight.qdata            = packed.payload.data();
    weight.qhigh            = nullptr;
    weight.scales           = packed.payload.data() + scale_offset;
    weight.group_size       = 32;
    weight.group            = 32;
    weight.ndim             = 2;
    weight.shape[0]         = rows;
    weight.shape[1]         = k;
    weight.shape[2]         = 1;
    weight.shape[3]         = 1;
    weight.padded_shape[0]  = rows;
    weight.padded_shape[1]  = k;
    weight.padded_shape[2]  = 1;
    weight.padded_shape[3]  = 1;
    weight.n                = rows;
    weight.k                = k;
    return packed;
}

Weight device_weight(const PackedW8& packed, void* device) {
    Weight weight         = packed.weight;
    const auto* host_base = static_cast<const std::uint8_t*>(packed.weight.payload);
    const std::size_t scale_offset =
        static_cast<const std::uint8_t*>(packed.weight.scales) - host_base;
    weight.payload = device;
    weight.qdata   = device;
    weight.scales  = static_cast<std::uint8_t*>(device) + scale_offset;
    return weight;
}

Weight w8_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    Weight view                      = parent;
    const std::size_t code_row_bytes = static_cast<std::size_t>(parent.padded_shape[1]);
    const std::size_t scale_row_bytes =
        static_cast<std::size_t>(parent.padded_shape[1] / parent.group) * 2u;
    view.qdata = static_cast<const std::uint8_t*>(parent.qdata) +
                 static_cast<std::size_t>(row_begin) * code_row_bytes;
    view.scales = static_cast<const std::uint8_t*>(parent.scales) +
                  static_cast<std::size_t>(row_begin) * scale_row_bytes;
    view.n               = rows;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    return view;
}

void fill_device_input(void* device, std::size_t count, std::uint32_t seed) {
    constexpr std::array<float, 8> kValues{{
        -1.0f,
        -0.5f,
        -0.125f,
        -0.03125f,
        0.03125f,
        0.125f,
        0.5f,
        1.0f,
    }};
    constexpr std::size_t kChunkWords = 1u << 20;
    std::vector<std::uint16_t> host(std::min(kChunkWords, count));
    for (std::size_t offset = 0; offset < count; offset += host.size()) {
        const std::size_t words = std::min(host.size(), count - offset);
        for (std::size_t i = 0; i < words; ++i) {
            const std::size_t global = offset + i;
            host[i] = test::f32_to_bf16(kValues[(global * 13u + seed) % kValues.size()]);
        }
        require_cuda(cudaMemcpy(static_cast<std::uint8_t*>(device) + offset * sizeof(std::uint16_t),
                                host.data(), words * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
                     "input cudaMemcpy");
    }
}

int compare_device_bf16(const std::string& label, const void* lhs, const void* rhs,
                        std::size_t count) {
    constexpr std::size_t kChunkWords = 1u << 20;
    std::vector<std::uint16_t> a(std::min(kChunkWords, count));
    std::vector<std::uint16_t> b(a.size());
    for (std::size_t offset = 0; offset < count; offset += a.size()) {
        const std::size_t words = std::min(a.size(), count - offset);
        require_cuda(
            cudaMemcpy(a.data(),
                       static_cast<const std::uint8_t*>(lhs) + offset * sizeof(std::uint16_t),
                       words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "lhs cudaMemcpy");
        require_cuda(
            cudaMemcpy(b.data(),
                       static_cast<const std::uint8_t*>(rhs) + offset * sizeof(std::uint16_t),
                       words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "rhs cudaMemcpy");
        for (std::size_t i = 0; i < words; ++i) {
            if (a[i] != b[i]) {
                std::cerr << "FAIL " << label << ": BF16 mismatch at " << offset + i << " 0x"
                          << std::hex << a[i] << " != 0x" << b[i] << std::dec << '\n';
                return 1;
            }
        }
    }
    return 0;
}

struct RoutePoint {
    std::int32_t cols;
    S schedule;
    V variant;
    P tail_policy = P::Homogeneous;
};

constexpr std::array<RoutePoint, 6> kDefault17Routes{{
    {1, S::SimtR8C4, V::Predicated},
    {4, S::SimtR8C4, V::Full},
    {5, S::SimtR8C8, V::Predicated},
    {16, S::SimtR8C8, V::Full},
    {17, S::MmaR64C128, V::Predicated},
    {128, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 4> kEarly9Routes{{
    {4, S::SimtR8C4, V::Full},
    {8, S::SimtR8C8, V::Full},
    {9, S::MmaR64C128, V::Predicated},
    {128, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 4> kR32Routes{{
    {4, S::SimtR8C4, V::Full},
    {16, S::SimtR8C8, V::Full},
    {17, S::MmaR32C128, V::Predicated},
    {128, S::MmaR32C128, V::Full},
}};

constexpr std::array<RoutePoint, 9> kVision4608Routes{{
    {1, S::SimtR8C4, V::Predicated},
    {8, S::SimtR8C4, V::Full},
    {9, S::MmaR32C128, V::Predicated},
    {11, S::MmaR32C128, V::Predicated},
    {12, S::SimtR8C4, V::Full},
    {13, S::MmaR32C128, V::Predicated},
    {256, S::MmaR32C128, V::Full},
    {257, S::MmaR64C128, V::Predicated},
    {32768, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 4> kVision5120Routes{{
    {4, S::SimtR8C4, V::Full},
    {5, S::SimtR8C8, V::Predicated},
    {6, S::MmaR64C128, V::Predicated},
    {128, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 20> kVision2048Routes{{
    {1, S::SimtR8C4, V::Predicated},     {14, S::SimtR8C4, V::Predicated},
    {15, S::MmaR32C128, V::Predicated},  {16, S::SimtR8C4, V::Full},
    {17, S::MmaR32C128, V::Predicated},  {19, S::MmaR32C128, V::Predicated},
    {20, S::SimtR8C4, V::Full},          {21, S::MmaR32C128, V::Predicated},
    {23, S::MmaR32C128, V::Predicated},  {24, S::SimtR8C4, V::Full},
    {25, S::MmaR32C128, V::Predicated},  {27, S::MmaR32C128, V::Predicated},
    {28, S::SimtR8C4, V::Full},          {29, S::MmaR32C128, V::Predicated},
    {31, S::MmaR32C128, V::Predicated},  {32, S::SimtR8C4, V::Full},
    {33, S::MmaR32C128, V::Predicated},  {871, S::MmaR32C128, V::Predicated},
    {872, S::MmaR64C128, V::Predicated}, {32768, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 6> kText2048x4096Routes{{
    {1, S::SimtR8C4, V::Predicated},
    {56, S::SimtR8C4, V::Full},
    {57, S::MmaR32C128, V::Predicated},
    {895, S::MmaR32C128, V::Predicated},
    {896, S::MmaR64C128, V::Full},
    {1024, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 6> kText12288x2048Routes{{
    {1, S::SimtR8C4, V::Predicated},
    {16, S::SimtR8C4, V::Full},
    {17, S::MmaR64C128, V::Predicated},
    {127, S::MmaR64C128, V::Predicated},
    {128, S::MmaR64C128, V::Full},
    {1024, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 8> kText9216x2048Routes{{
    {1, S::SimtR8C4, V::Predicated},
    {13, S::SimtR8C4, V::Predicated},
    {14, S::MmaR32C128, V::Predicated},
    {127, S::MmaR32C128, V::Predicated},
    {128, S::MmaR32C128, V::Full},
    {129, S::MmaR64C128, V::Predicated},
    {255, S::MmaR64C128, V::Predicated},
    {1024, S::MmaR64C128, V::Full},
}};

constexpr std::array<RoutePoint, 75> kConditioning2048x16384Routes{{
    {1, S::DecodeR4, V::None},
    {2, S::SplitKMmaExactT, V::None},
    {32, S::SplitKMmaExactT, V::None},
    {33, S::SplitKMma32PlusTail, V::None},
    {88, S::SplitKMma32PlusTail, V::None},
    {89, S::SplitKMediumC96, V::None},
    {96, S::SplitKMediumC96, V::None},
    {97, S::SplitKMediumC128, V::None},
    {128, S::SplitKMediumC128, V::None},
    {129, S::SplitKMediumC144, V::None},
    {144, S::SplitKMediumC144, V::None},
    {145, S::MmaR32C128, V::Predicated},
    {255, S::MmaR32C128, V::Predicated},
    {256, S::MmaR32C64, V::Full},
    {384, S::MmaR32C64, V::Full},
    {385, S::MmaR32C96, V::Predicated},
    {480, S::MmaR32C96, V::Full},
    {481, S::MmaR32C96, V::Predicated, P::ConditioningExact},
    {482, S::MmaR32C128, V::Predicated},
    {640, S::MmaR32C128, V::Full},
    {641, S::MmaR32C128, V::Predicated, P::ConditioningExact},
    {668, S::MmaR32C128, V::Predicated, P::ConditioningExact},
    {669, S::MmaR48C96, V::Predicated},
    {672, S::MmaR48C96, V::Predicated},
    {673, S::MmaR48C96, V::Predicated, P::ConditioningExact},
    {674, S::MmaR48C64, V::Predicated},
    {704, S::MmaR48C64, V::Predicated},
    {705, S::MmaR48C112, V::Predicated},
    {784, S::MmaR48C112, V::Predicated},
    {785, S::MmaR48C128, V::Predicated},
    {896, S::MmaR48C128, V::Predicated},
    {897, S::MmaR48C128, V::Predicated, P::ConditioningExact},
    {912, S::MmaR48C128, V::Predicated, P::ConditioningExact},
    {913, S::MmaR64C96, V::Predicated},
    {960, S::MmaR64C96, V::Full},
    {961, S::MmaR64C96, V::Predicated, P::ConditioningExact},
    {1007, S::MmaR64C96, V::Predicated, P::ConditioningExact},
    {1008, S::MmaR64C112, V::Full},
    {1009, S::MmaR64C128, V::Predicated},
    {1023, S::MmaR64C128, V::Predicated},
    {1024, S::MmaR64C128, V::Full},
    {1025, S::MmaR64C128, V::Predicated},
    {1119, S::MmaR64C128, V::Predicated},
    {1120, S::MmaR64C112, V::Full},
    {1121, S::MmaR64C128, V::Predicated},
    {1152, S::MmaR64C128, V::Full},
    {1280, S::MmaR64C128, V::Full},
    {1281, S::MmaR64C128, V::Predicated, P::ConditioningExact},
    {1313, S::MmaR64C128, V::Predicated, P::ConditioningExact},
    {1314, S::MmaR128C64, V::Predicated},
    {1344, S::MmaR128C64, V::Full},
    {1345, S::MmaR96C96, V::Predicated},
    {1440, S::MmaR96C96, V::Predicated},
    {1441, S::MmaR96C96, V::Predicated, P::ConditioningExact},
    {1500, S::MmaR96C96, V::Predicated, P::ConditioningExact},
    {1501, S::MmaR128C80, V::Predicated},
    {1680, S::MmaR128C80, V::Full},
    {1681, S::MmaR128C80, V::Predicated, P::ConditioningExact},
    {1745, S::MmaR128C80, V::Predicated, P::ConditioningExact},
    {1746, S::MmaR48C128, V::Predicated},
    {1791, S::MmaR48C128, V::Predicated},
    {1792, S::MmaR64C128, V::Full},
    {1793, S::MmaR48C128, V::Predicated},
    {1919, S::MmaR48C128, V::Predicated},
    {1920, S::MmaR64C128, V::Full},
    {1921, S::MmaR64C128, V::Predicated, P::ConditioningExact},
    {1953, S::MmaR64C128, V::Predicated, P::ConditioningExact},
    {1954, S::MmaR64C96, V::Predicated},
    {2016, S::MmaR64C96, V::Full},
    {2017, S::MmaR64C96, V::Predicated, P::ConditioningExact},
    {2048, S::MmaR64C96, V::Predicated, P::ConditioningExact},
    {2049, S::MmaR96C96, V::Predicated},
    {2112, S::MmaR96C96, V::Predicated},
    {2113, S::MmaR64C128, V::Predicated},
    {2176, S::MmaR64C128, V::Full},
}};

struct SupportCase {
    const char* label;
    std::int32_t rows;
    std::int32_t k;
    const RoutePoint* routes;
    std::size_t route_count;
    std::uint32_t seed;
};

void run_support(const SupportCase& support) {
    std::int32_t max_cols = 0;
    for (std::size_t i = 0; i < support.route_count; ++i) {
        max_cols = std::max(max_cols, support.routes[i].cols);
    }

    PackedW8 packed = make_deterministic_w8(support.rows, support.k, support.seed);
    DeviceBuffer weight_buffer(packed.payload.size());
    require_cuda(cudaMemcpy(weight_buffer.data(), packed.payload.data(), packed.payload.size(),
                            cudaMemcpyHostToDevice),
                 "weight cudaMemcpy");
    const Weight weight = device_weight(packed, weight_buffer.data());
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
        const RoutePoint route  = support.routes[i];
        const std::string label = std::string(support.label) + " C=" + std::to_string(route.cols);
        Tensor x(input.data(), DType::BF16, {support.k, route.cols});
        Tensor public_out(public_output.data(), DType::BF16, {support.rows, route.cols});
        Tensor fixed_out(fixed_output.data(), DType::BF16, {support.rows, route.cols});

        const ops::detail::W8Plan plan =
            ops::detail::w8_rowsplit_resolve_plan({support.rows, support.k, support.k, route.cols});
        if (plan.schedule != route.schedule || plan.variant != route.variant ||
            plan.tail_policy != route.tail_policy) {
            std::cerr << "FAIL " << label << ": unexpected resolved plan\n";
            ++failures;
        }

        const std::size_t bytes =
            static_cast<std::size_t>(support.rows) * route.cols * sizeof(std::uint16_t);
        require_cuda(cudaMemset(public_output.data(), 0xa5, bytes), "public output cudaMemset");
        require_cuda(cudaMemset(fixed_output.data(), 0x5a, bytes), "fixed output cudaMemset");
        try {
            ops::linear(x, weight, public_out, workspace, nullptr);
            require_cuda(cudaDeviceSynchronize(), "public linear synchronize");
            ops::detail::w8_rowsplit_launch_candidate(plan.schedule, plan.variant, x, weight,
                                                      fixed_out, nullptr, plan.tail_policy);
            require_cuda(cudaDeviceSynchronize(), "fixed linear synchronize");
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << label << ": launch failed: " << error.what() << '\n';
            ++failures;
            continue;
        }
        failures += compare_device_bf16(label, public_output.data(), fixed_output.data(),
                                        static_cast<std::size_t>(support.rows) * route.cols);
    }
}

void pair_routes() {
    constexpr std::int32_t rows     = 1024;
    constexpr std::int32_t k        = 5120;
    constexpr std::int32_t max_cols = 128;
    PackedW8 first_packed           = make_deterministic_w8(rows, k, 211u);
    PackedW8 second_packed          = make_deterministic_w8(rows, k, 223u);
    DeviceBuffer first_weight_buffer(first_packed.payload.size());
    DeviceBuffer second_weight_buffer(second_packed.payload.size());
    require_cuda(cudaMemcpy(first_weight_buffer.data(), first_packed.payload.data(),
                            first_packed.payload.size(), cudaMemcpyHostToDevice),
                 "first pair weight cudaMemcpy");
    require_cuda(cudaMemcpy(second_weight_buffer.data(), second_packed.payload.data(),
                            second_packed.payload.size(), cudaMemcpyHostToDevice),
                 "second pair weight cudaMemcpy");
    const Weight first_weight  = device_weight(first_packed, first_weight_buffer.data());
    const Weight second_weight = device_weight(second_packed, second_weight_buffer.data());

    DeviceBuffer input(static_cast<std::size_t>(k) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer public_first(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer public_second(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer fixed_first(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer fixed_second(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    fill_device_input(input.data(), static_cast<std::size_t>(k) * max_cols, 227u);
    WorkspaceArena workspace(256);

    for (const std::int32_t cols : {4, 5, 56, 57, 128}) {
        const std::string label = "W8 pair C=" + std::to_string(cols);
        Tensor x(input.data(), DType::BF16, {k, cols});
        Tensor public_a(public_first.data(), DType::BF16, {rows, cols});
        Tensor public_b(public_second.data(), DType::BF16, {rows, cols});
        Tensor fixed_a(fixed_first.data(), DType::BF16, {rows, cols});
        Tensor fixed_b(fixed_second.data(), DType::BF16, {rows, cols});
        const ops::detail::W8PairPlan plan = ops::detail::w8_pair_resolve_plan({rows, k, k, cols});

        ops::linear_pair(x, first_weight, second_weight, public_a, public_b, workspace, nullptr);
        require_cuda(cudaDeviceSynchronize(), "public pair synchronize");
        ops::detail::w8_pair_execute_plan(plan, x, first_weight, second_weight, fixed_a, fixed_b,
                                          nullptr);
        require_cuda(cudaDeviceSynchronize(), "fixed pair synchronize");
        const std::size_t words = static_cast<std::size_t>(rows) * cols;
        failures +=
            compare_device_bf16(label + " first", public_first.data(), fixed_first.data(), words);
        failures += compare_device_bf16(label + " second", public_second.data(),
                                        fixed_second.data(), words);
    }

    Tensor x(input.data(), DType::BF16, {k, 57});
    Tensor out_a(public_first.data(), DType::BF16, {rows, 57});
    Tensor out_b(public_second.data(), DType::BF16, {rows, 57});
    Weight misaligned = first_weight;
    misaligned.scales = static_cast<const std::uint8_t*>(misaligned.scales) + 4;
    try {
        ops::linear(x, misaligned, out_a, workspace, nullptr);
        std::cerr << "FAIL W8 MMA accepted a 4-byte-only aligned scale plane\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        ops::linear_pair(x, misaligned, second_weight, out_a, out_b, workspace, nullptr);
        std::cerr << "FAIL W8 pair MMA accepted a 4-byte-only aligned scale plane\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

void dflash_pair_routes() {
    constexpr std::int32_t rows        = 1024;
    constexpr std::int32_t k           = 2048;
    constexpr std::int32_t parent_rows = 6144;
    constexpr std::int32_t max_cols    = 2271;
    PackedW8 packed                    = make_deterministic_w8(parent_rows, k, 229u);
    DeviceBuffer weight_buffer(packed.payload.size());
    require_cuda(cudaMemcpy(weight_buffer.data(), packed.payload.data(), packed.payload.size(),
                            cudaMemcpyHostToDevice),
                 "dFlash pair parent weight cudaMemcpy");
    const Weight parent        = device_weight(packed, weight_buffer.data());
    const Weight first_weight  = w8_row_view(parent, 4096, rows);
    const Weight second_weight = w8_row_view(parent, 5120, rows);

    DeviceBuffer input(static_cast<std::size_t>(k) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer public_first(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer public_second(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer fixed_first(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    DeviceBuffer fixed_second(static_cast<std::size_t>(rows) * max_cols * sizeof(std::uint16_t));
    fill_device_input(input.data(), static_cast<std::size_t>(k) * max_cols, 233u);
    WorkspaceArena workspace(256);

    for (const std::int32_t cols :
         {1,    2,    16,   32,   33,   48,   49,   64,   65,   80,   81,   88,   89,   96,   97,
          104,  105,  112,  113,  128,  129,  160,  161,  192,  193,  384,  385,  480,  481,  640,
          641,  642,  672,  673,  680,  681,  784,  785,  896,  897,  960,  961,  976,  977,  1280,
          1281, 1316, 1317, 1344, 1345, 1346, 1440, 1441, 1466, 1467, 1680, 1681, 1708, 1709, 1920,
          1921, 1922, 1923, 2016, 2017, 2018, 2019, 2048, 2049, 2208, 2209, 2270, 2271}) {
        const std::string label = "W8 dFlash context pair C=" + std::to_string(cols);
        Tensor x(input.data(), DType::BF16, {k, cols});
        Tensor public_a(public_first.data(), DType::BF16, {rows, cols});
        Tensor public_b(public_second.data(), DType::BF16, {rows, cols});
        Tensor fixed_a(fixed_first.data(), DType::BF16, {rows, cols});
        Tensor fixed_b(fixed_second.data(), DType::BF16, {rows, cols});
        const ops::detail::W8PairPlan plan = ops::detail::w8_pair_resolve_plan({rows, k, k, cols});

        ops::linear_pair(x, first_weight, second_weight, public_a, public_b, workspace, nullptr);
        require_cuda(cudaDeviceSynchronize(), "public dFlash pair synchronize");
        ops::detail::w8_pair_execute_plan(plan, x, first_weight, second_weight, fixed_a, fixed_b,
                                          nullptr);
        require_cuda(cudaDeviceSynchronize(), "fixed dFlash pair synchronize");
        const std::size_t words = static_cast<std::size_t>(rows) * cols;
        failures +=
            compare_device_bf16(label + " first", public_first.data(), fixed_first.data(), words);
        failures += compare_device_bf16(label + " second", public_second.data(),
                                        fixed_second.data(), words);
    }

    Tensor x(input.data(), DType::BF16, {k, 16});
    Tensor out_a(public_first.data(), DType::BF16, {rows, 16});
    Tensor out_b(public_second.data(), DType::BF16, {rows, 16});
    Weight wrong_first = w8_row_view(parent, 4088, rows);
    try {
        ops::linear_pair(x, wrong_first, second_weight, out_a, out_b, workspace, nullptr);
        std::cerr << "FAIL W8 dFlash pair accepted a wrong parent K row offset\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    Weight wrong_second = w8_row_view(parent, 5112, rows);
    try {
        ops::linear_pair(x, first_weight, wrong_second, out_a, out_b, workspace, nullptr);
        std::cerr << "FAIL W8 dFlash pair accepted a wrong parent V row offset\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    Weight wrong_scale = first_weight;
    wrong_scale.scales = static_cast<const std::uint8_t*>(wrong_scale.scales) + 4;
    try {
        ops::linear_pair(x, wrong_scale, second_weight, out_a, out_b, workspace, nullptr);
        std::cerr << "FAIL W8 dFlash pair accepted a wrong parent K scale offset\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    Weight detached_second  = second_weight;
    detached_second.payload = second_weight.qdata;
    try {
        ops::linear_pair(x, first_weight, detached_second, out_a, out_b, workspace, nullptr);
        std::cerr << "FAIL W8 dFlash pair accepted detached V parent metadata\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        ops::linear_pair(x, first_weight, second_weight, out_a, out_a, workspace, nullptr);
        std::cerr << "FAIL W8 dFlash pair accepted overlapping outputs\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

void rejection_contract() {
    PackedW8 packed = make_deterministic_w8(64, 256, 251u);
    DeviceBuffer weight_buffer(packed.payload.size());
    require_cuda(cudaMemcpy(weight_buffer.data(), packed.payload.data(), packed.payload.size(),
                            cudaMemcpyHostToDevice),
                 "rejection weight cudaMemcpy");
    const Weight weight = device_weight(packed, weight_buffer.data());
    DeviceBuffer input(256u * sizeof(std::uint16_t));
    DeviceBuffer output(64u * sizeof(std::uint16_t));
    fill_device_input(input.data(), 256, 257u);
    Tensor x(input.data(), DType::BF16, {256, 1});
    Tensor out(output.data(), DType::BF16, {64, 1});
    WorkspaceArena workspace(256);
    try {
        ops::linear(x, weight, out, workspace, nullptr);
        require_cuda(cudaDeviceSynchronize(), "unexpected rejection synchronize");
        std::cerr << "FAIL public W8 Linear accepted an unregistered problem\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    constexpr std::array<SupportCase, 15> supports{{
        {"W8 [5120,10240]", 5120, 10240, kDefault17Routes.data(), kDefault17Routes.size(), 11u},
        {"W8 [14336,5120]", 14336, 5120, kEarly9Routes.data(), kEarly9Routes.size(), 13u},
        {"W8 [1024,5120]", 1024, 5120, kR32Routes.data(), kR32Routes.size(), 17u},
        {"W8 [1024,2048]", 1024, 2048, kR32Routes.data(), kR32Routes.size(), 18u},
        {"W8 [6144,5120]", 6144, 5120, kDefault17Routes.data(), kDefault17Routes.size(), 19u},
        {"W8 [5120,6144]", 5120, 6144, kDefault17Routes.data(), kDefault17Routes.size(), 23u},
        {"W8 [34816,5120]", 34816, 5120, kEarly9Routes.data(), kEarly9Routes.size(), 29u},
        {"W8 [5120,17408]", 5120, 17408, kDefault17Routes.data(), kDefault17Routes.size(), 31u},
        {"W8 [4608,4608]", 4608, 4608, kVision4608Routes.data(), kVision4608Routes.size(), 37u},
        {"W8 [5120,4608]", 5120, 4608, kVision5120Routes.data(), kVision5120Routes.size(), 41u},
        {"W8 [2048,4608]", 2048, 4608, kVision2048Routes.data(), kVision2048Routes.size(), 43u},
        {"W8 [2048,4096]", 2048, 4096, kText2048x4096Routes.data(), kText2048x4096Routes.size(),
         47u},
        {"W8 [12288,2048]", 12288, 2048, kText12288x2048Routes.data(), kText12288x2048Routes.size(),
         53u},
        {"W8 [9216,2048]", 9216, 2048, kText9216x2048Routes.data(), kText9216x2048Routes.size(),
         59u},
        {"W8 conditioning [2048,16384]", 2048, 16384, kConditioning2048x16384Routes.data(),
         kConditioning2048x16384Routes.size(), 61u},
    }};

    try {
        for (const SupportCase& support : supports) { run_support(support); }
        pair_routes();
        dflash_pair_routes();
        rejection_contract();
    } catch (const std::exception& error) {
        std::cerr << "FAIL W8 dispatch test infrastructure: " << error.what() << '\n';
        return 1;
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " W8 Linear public dispatch\n";
    return failures == 0 ? 0 : 1;
}
