#include "ninfer/ops/linear.h"
#include "ops/linear/q6/q6_rowsplit_plan.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

using S = ops::detail::Q6ScheduleId;
using V = ops::detail::Q6KernelVariant;

int failures = 0;

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        const cudaError_t error = cudaMalloc(&data_, bytes_);
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc(") + std::to_string(bytes_) +
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

void require_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

struct Q6Layout {
    std::int32_t padded_k;
    std::size_t high_offset;
    std::size_t high_bytes;
    std::size_t scale_offset;
    std::size_t payload_bytes;
};

Q6Layout q6_layout(std::int32_t rows, std::int32_t k) {
    const std::int32_t padded_k = static_cast<std::int32_t>(align_up(k, 128));
    const std::size_t groups =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(padded_k / 64);
    const std::size_t code_bytes   = groups * 32u;
    const std::size_t high_offset  = align_up(code_bytes, 256u);
    const std::size_t high_bytes   = groups * 16u;
    const std::size_t scale_offset = high_offset + align_up(high_bytes, 256u);
    return {padded_k, high_offset, high_bytes, scale_offset,
            scale_offset + groups * sizeof(std::uint16_t)};
}

class DeviceQ6Weight {
public:
    DeviceQ6Weight(std::int32_t rows, std::int32_t k, std::uint8_t pattern)
        : layout_(q6_layout(rows, k)), storage_(layout_.payload_bytes) {
        require_cuda(cudaMemset(storage_.data(), pattern, layout_.payload_bytes),
                     "weight cudaMemset");

        weight_.payload          = storage_.data();
        weight_.payload_bytes    = layout_.payload_bytes;
        weight_.high_plane_bytes = layout_.high_bytes;
        weight_.qtype            = QType::Q6G64_F16S;
        weight_.layout           = QuantLayout::RowSplit;
        weight_.scale_dtype      = DType::FP16;
        weight_.group_size       = 64;
        weight_.shape[0]         = rows;
        weight_.shape[1]         = k;
        weight_.shape[2]         = 1;
        weight_.shape[3]         = 1;
        weight_.padded_shape[0]  = rows;
        weight_.padded_shape[1]  = layout_.padded_k;
        weight_.padded_shape[2]  = 1;
        weight_.padded_shape[3]  = 1;
        weight_.ndim             = 2;
        weight_.qdata            = storage_.data();
        weight_.qhigh  = static_cast<const std::uint8_t*>(storage_.data()) + layout_.high_offset;
        weight_.scales = static_cast<const std::uint8_t*>(storage_.data()) + layout_.scale_offset;
        weight_.n      = rows;
        weight_.k      = k;
        weight_.group  = 64;
    }

    const Weight& get() const noexcept { return weight_; }

private:
    Q6Layout layout_;
    DeviceBuffer storage_;
    Weight weight_{};
};

void fill_input(void* device, std::size_t words, std::uint8_t pattern) {
    require_cuda(cudaMemset(device, pattern, words * sizeof(std::uint16_t)), "input cudaMemset");
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

std::string plan_name(ops::detail::Q6Plan plan) {
    return std::string(ops::detail::q6_schedule_name(plan.schedule)) + "." +
           ops::detail::q6_kernel_variant_name(plan.variant);
}

struct RoutePoint {
    std::int32_t cols;
    S schedule;
    V variant;
};

constexpr std::array<RoutePoint, 3> kHeadRoutes{{
    {1, S::SimtR8C4, V::None},
    {6, S::SimtR8C4, V::None},
    {7, S::MmaR64C128, V::Predicated},
}};

constexpr std::array<RoutePoint, 5> kHead2048Routes{{
    {1, S::SimtR8C4, V::None},
    {4, S::SimtR8C4, V::None},
    {5, S::SimtR8C8, V::None},
    {6, S::SimtR8C8, V::None},
    {7, S::MmaR64C128, V::Predicated},
}};

constexpr std::array<RoutePoint, 18> kVisionRoutes{{
    {4, S::SimtR8C4, V::None},
    {96, S::SimtR8C4, V::None},
    {100, S::MmaR64C64, V::Predicated},
    {704, S::MmaR64C64, V::Full},
    {708, S::MmaR64C128, V::Predicated},
    {768, S::MmaR64C128, V::Full},
    {828, S::MmaR64C128, V::Predicated},
    {832, S::MmaR64C64, V::Full},
    {836, S::MmaR64C128, V::Predicated},
    {896, S::MmaR64C128, V::Full},
    {900, S::MmaR64C64, V::Predicated},
    {960, S::MmaR64C64, V::Full},
    {964, S::MmaR64C128, V::Predicated},
    {1024, S::MmaR64C128, V::Full},
    {1028, S::MmaR64C64, V::Predicated},
    {1088, S::MmaR64C64, V::Full},
    {1092, S::MmaR64C128, V::Predicated},
    {131072, S::MmaR64C128, V::Full},
}};

std::array<bool, 4> schedules_seen{};
std::array<bool, 3> variants_seen{};

void record_coverage(S schedule, V variant) {
    switch (schedule) {
    case S::SimtR8C4:
        schedules_seen[0] = true;
        break;
    case S::SimtR8C8:
        schedules_seen[1] = true;
        break;
    case S::MmaR64C64:
        schedules_seen[2] = true;
        break;
    case S::MmaR64C128:
        schedules_seen[3] = true;
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
    const ops::detail::Q6Problem problem = ops::detail::q6_rowsplit_problem(x, weight, out);
    if (ops::detail::q6_rowsplit_admits(problem)) {
        std::cerr << "FAIL " << label << ": internal admission unexpectedly accepted the problem\n";
        ++failures;
    }
    try {
        ops::linear(x, weight, out, workspace, nullptr);
        require_cuda(cudaDeviceSynchronize(), "unexpected public launch synchronize");
        std::cerr << "FAIL " << label << ": public linear accepted an unregistered Q6 problem\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << label << ": wrong exception: " << error.what() << '\n';
        ++failures;
    }
}

template <std::size_t N>
void run_support(const char* label, std::int32_t rows, std::int32_t k,
                 const std::array<RoutePoint, N>& routes, std::uint8_t pattern) {
    DeviceQ6Weight device_weight(rows, k, pattern);
    WorkspaceArena workspace(256);

    for (const RoutePoint& route : routes) {
        const std::string case_label = std::string(label) + " C=" + std::to_string(route.cols);
        const std::size_t input_words =
            static_cast<std::size_t>(k) * static_cast<std::size_t>(route.cols);
        const std::size_t output_words =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(route.cols);
        DeviceBuffer input(input_words * sizeof(std::uint16_t));
        DeviceBuffer public_output(output_words * sizeof(std::uint16_t));
        DeviceBuffer fixed_output(output_words * sizeof(std::uint16_t));
        fill_input(input.data(), input_words, 0x3fu);

        Tensor x(input.data(), DType::BF16, {k, route.cols});
        Tensor public_out(public_output.data(), DType::BF16, {rows, route.cols});
        Tensor fixed_out(fixed_output.data(), DType::BF16, {rows, route.cols});
        const ops::detail::Q6Problem problem{rows, k, device_weight.get().padded_shape[1],
                                             route.cols};

        ops::detail::Q6Plan plan{};
        try {
            plan = ops::detail::q6_rowsplit_resolve_plan(problem);
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << case_label << ": resolver threw: " << error.what() << '\n';
            ++failures;
            continue;
        }
        if (plan.schedule != route.schedule || plan.variant != route.variant) {
            std::cerr << "FAIL " << case_label << ": expected "
                      << plan_name({route.schedule, route.variant}) << ", got " << plan_name(plan)
                      << '\n';
            ++failures;
        }
        record_coverage(plan.schedule, plan.variant);

        require_cuda(cudaMemset(public_output.data(), 0xa5, output_words * sizeof(std::uint16_t)),
                     "public output cudaMemset");
        require_cuda(cudaMemset(fixed_output.data(), 0x5a, output_words * sizeof(std::uint16_t)),
                     "fixed output cudaMemset");
        try {
            ops::linear(x, device_weight.get(), public_out, workspace, nullptr);
            require_cuda(cudaDeviceSynchronize(), "public linear synchronize");
            ops::detail::q6_rowsplit_execute_plan(plan, x, device_weight.get(), fixed_out, nullptr);
            require_cuda(cudaDeviceSynchronize(), "fixed launch synchronize");
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << case_label << ": launch failed: " << error.what() << '\n';
            ++failures;
            continue;
        }
        failures += compare_device_bf16(case_label, public_output.data(), fixed_output.data(),
                                        output_words);
    }
}

void rejection_checks() {
    WorkspaceArena workspace(256);

    DeviceQ6Weight draft(65536, 5120, 0x2fu);
    DeviceBuffer draft_input(5120u * sizeof(std::uint16_t));
    DeviceBuffer draft_output(65536u * sizeof(std::uint16_t));
    Tensor draft_x(draft_input.data(), DType::BF16, {5120, 1});
    Tensor draft_out(draft_output.data(), DType::BF16, {65536, 1});
    expect_public_rejection("unregistered N=65536", draft_x, draft.get(), draft_out, workspace);

    DeviceQ6Weight vision(1152, 1536, 0x35u);
    DeviceBuffer vision_input(1536u * 5u * sizeof(std::uint16_t));
    DeviceBuffer vision_output(1152u * 5u * sizeof(std::uint16_t));
    Tensor vision_x(vision_input.data(), DType::BF16, {1536, 5});
    Tensor vision_out(vision_output.data(), DType::BF16, {1152, 5});
    expect_public_rejection("vision non-step C=5", vision_x, vision.get(), vision_out, workspace);
}

void verify_coverage() {
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

    try {
        run_support("Q6 [248320,5120]", 248320, 5120, kHeadRoutes, 0x2bu);
        run_support("Q6 [248320,2048]", 248320, 2048, kHead2048Routes, 0x2du);
        run_support("Q6 [1152,1536]", 1152, 1536, kVisionRoutes, 0x31u);
        rejection_checks();
        verify_coverage();
    } catch (const std::exception& error) {
        std::cerr << "FAIL Q6 dispatch test infrastructure: " << error.what() << '\n';
        ++failures;
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6 Linear public dispatch\n";
    return failures == 0 ? 0 : 1;
}
