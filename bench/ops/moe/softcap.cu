#include "core/buffer.h"
#include "ninfer/ops/moe.h"
#include "op_measurement.cuh"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/softcap_routes.h"

#include <array>
#include <sstream>

namespace {
using namespace ninfer;
namespace meter  = bench::measurement;
namespace routed = ops::detail::moe_routed;
using ops::LinearPolicy;
using routed::softcap::Schedule;

const char* schedule_name(Schedule schedule) {
    switch (schedule) {
    case Schedule::VectorComplete:
        return "vector-complete";
    case Schedule::VectorDecodeSplit:
        return "vector-decode-split";
    case Schedule::DecodeA4Reuse:
        return "decode-a4-reuse";
    case Schedule::Bf16Bucketed:
        return "bf16-bucketed";
    case Schedule::Bf16Short16:
        return "bf16-short16";
    case Schedule::Bf16Mma:
        return "bf16-mma";
    case Schedule::Fp4Short:
        return "fp4-short";
    case Schedule::Fp4Mma:
        return "fp4-mma";
    case Schedule::Sm120Tma64:
        return "sm120-tma-red64";
    case Schedule::Sm120TmaDense:
        return "sm120-tma-dense";
    }
    throw std::logic_error("invalid softcap schedule");
}

const char* schedule_name(const routed::mxfp8_softcap::Plan& plan) {
    using S = routed::mxfp8_softcap::Schedule;
    switch (plan.schedule) {
    case S::Vector: return "vector";
    case S::Grouped16: return plan.gate_a8 ? "mxfp8-grouped16" : "bf16-grouped16";
    case S::Grouped32: return plan.gate_a8 ? "mxfp8-grouped32" : "bf16-grouped32";
    case S::Grouped64: return "bf16-grouped64";
    case S::Grouped128: return "mxfp8-grouped128";
    case S::Wide128:
        return "mxfp8-wide128";
    case S::Wide128Transposed:
        return "mxfp8-wide128-transposed-down";
    }
    throw std::logic_error("invalid MXFP8 softcap schedule");
}

constexpr int hidden = 2304, intermediate = 1152, experts = 128, top_k = 8;

struct Options {
    meter::Options timing;
    std::vector<int> tokens;
    std::string format = "nvfp4-mul", policy, distribution = "all";
};

Options parse(int argc, char** argv) {
    Options o;
    for (int a = 1; a < argc; ++a) {
        const std::string_view flag = argv[a];
        if (flag == "--help") {
            std::cout << "ninfer_moe_softcap_bench\n"
                         "  --format nvfp4-mul --policy a16|a4\n"
                         "  --format mxfp8 --policy a16|a8\n"
                         "  --distribution uniform|reuse|pool|tail|spread|all\n"
                         "  --tokens T[,T...] --sweep A:B[:S]\n"
                         "  CSV first_slice_plan describes the host schedule of the first slice.\n";
            meter::help();
            std::exit(0);
        }
        const auto value = meter::take(argc, argv, a);
        if (meter::parse(o.timing, flag, value)) continue;
        if (flag == "--format")
            o.format = value;
        else if (flag == "--policy")
            o.policy = value;
        else if (flag == "--distribution")
            o.distribution = value;
        else if (flag == "--tokens" || flag == "--sweep") {
            const auto points = flag == "--tokens" ? meter::integers(value) : meter::sweep(value);
            o.tokens.insert(o.tokens.end(), points.begin(), points.end());
        } else {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
    }
    const std::string_view quantized = o.format == "mxfp8" ? "a8" : "a4";
    if (o.policy.empty()) o.policy = quantized;
    if ((o.format != "nvfp4-mul" && o.format != "mxfp8") ||
        (o.policy != "a16" && o.policy != quantized) ||
        (o.distribution != "all" &&
         std::find(test::softcap::distributions.begin(), test::softcap::distributions.end(),
                   o.distribution) == test::softcap::distributions.end()))
        throw std::invalid_argument("invalid format, policy or distribution");
    if (o.tokens.empty()) o.tokens = {1, 4, 8, 16, 64, 256, 512, 1024, 4096};
    std::sort(o.tokens.begin(), o.tokens.end());
    o.tokens.erase(std::unique(o.tokens.begin(), o.tokens.end()), o.tokens.end());
    return o;
}

__global__ void fill_scales(std::uint8_t* scales, std::size_t bytes, unsigned seed) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += gridDim.x * blockDim.x)
        scales[i] = 0x20 + bench::fixture::bits(i, seed) % 24;
}

struct Matrix {
    DeviceBuffer codes, scales, multipliers;

    Matrix(int rows, int columns, unsigned seed)
        : codes(bench::make_buffer(std::size_t(experts) * rows * columns / 2)),
          scales(bench::make_buffer(std::size_t(experts) * rows * columns / 16)),
          multipliers(bench::make_buffer(experts * sizeof(float))) {
        CUDA_CHECK(bench::fixture::fill_bytes(codes.data(), codes.size(), seed));
        fill_scales<<<1024, 256>>>(static_cast<std::uint8_t*>(scales.data()), scales.size(), seed);
        CUDA_CHECK(cudaGetLastError());
        std::array<float, experts> values;
        for (int e = 0; e < experts; ++e) values[e] = .07F + .001F * (e % 17);
        multipliers.copy_from_host(values.data(), sizeof(values));
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView v;
        v.codes               = static_cast<const std::byte*>(codes.data());
        v.scales              = static_cast<const std::byte*>(scales.data());
        v.code_stride         = codes.size() / experts;
        v.scale_stride        = scales.size() / experts;
        v.matrix_scales       = static_cast<const float*>(multipliers.data());
        v.matrix_scale_stride = 1;
        v.format              = QType::NVFP4_MUL;
        v.layout              = QuantLayout::BlockScaleK16M128x4;
        v.physical_count      = experts;
        return v;
    }
};

struct Nvfp4Fixture {
    Matrix gate_up{2 * intermediate, hidden, 39}, down{hidden, intermediate, 83};

    ops::RoutedSwiGLUProfile profile(LinearPolicy policy) const {
        constexpr auto q      = QType::NVFP4_MUL;
        constexpr auto layout = QuantLayout::BlockScaleK16M128x4;
        return {{hidden,
                 intermediate,
                 q,
                 q,
                 layout,
                 layout,
                 policy,
                 policy,
                 {},
                 ops::SwiGLUSoftcap{4.F, 25.F}},
                experts,
                top_k};
    }

    ops::CudaSwiGLUExperts bank(LinearPolicy policy) const {
        ops::CudaSwiGLUExperts weights{
            .bank = {gate_up.view(), down.view(), experts, gate_up.codes.device()},
            .activation_calibration = {}};
        if (policy == LinearPolicy::AllowA4) {
            using ops::ActivationCalibrationView;
            using ops::ActivationScaleOperation;
            weights.activation_calibration = {
                ActivationCalibrationView::broadcast(ActivationScaleOperation::Multiply, .02F),
                ActivationCalibrationView::broadcast(ActivationScaleOperation::Multiply, .05F)};
        }
        return weights;
    }

    std::size_t expert_bytes() const {
        return gate_up.view().code_stride + gate_up.view().scale_stride + down.view().code_stride +
               down.view().scale_stride + 2 * sizeof(float);
    }

    double compute_rate(LinearPolicy policy) const {
        return policy == LinearPolicy::A16Only ? bench::kRoofline.bf16_f32_tflops
                                               : bench::kRoofline.nvfp4_f32_tflops;
    }

    const char* plan_name(const routed::Plan& plan) const {
        return schedule_name(plan.softcap.schedule);
    }
};

__global__ void fill_mxfp8(std::uint8_t* values, std::size_t size, unsigned seed, bool scales) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += gridDim.x * blockDim.x) {
        const unsigned sample = bench::fixture::bits(i, seed);
        values[i]             = scales ? 112 + sample % 5 : (sample & 128) | (sample % 127);
    }
}

struct Mxfp8Matrix {
    DeviceBuffer codes, scales;

    Mxfp8Matrix(int rows, int columns, unsigned seed)
        : codes(bench::make_buffer(std::size_t(experts) * rows * columns)),
          scales(bench::make_buffer(codes.size() / 32)) {
        fill_mxfp8<<<1024, 256>>>(static_cast<std::uint8_t*>(codes.data()), codes.size(), seed,
                                  false);
        fill_mxfp8<<<1024, 256>>>(static_cast<std::uint8_t*>(scales.data()), scales.size(),
                                  seed + 1, true);
        CUDA_CHECK(cudaGetLastError());
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView v;
        v.codes          = static_cast<const std::byte*>(codes.data());
        v.scales         = static_cast<const std::byte*>(scales.data());
        v.code_stride    = codes.size() / experts;
        v.scale_stride   = scales.size() / experts;
        v.format         = QType::MXFP8;
        v.layout         = QuantLayout::RowGroupScale;
        v.physical_count = experts;
        return v;
    }
};

struct Mxfp8Fixture {
    Mxfp8Matrix gate_up{2 * intermediate, hidden, 39}, down{hidden, intermediate, 83};

    ops::RoutedSwiGLUProfile profile(LinearPolicy policy) const {
        constexpr auto q      = QType::MXFP8;
        constexpr auto layout = QuantLayout::RowGroupScale;
        return {{hidden,
                 intermediate,
                 q,
                 q,
                 layout,
                 layout,
                 policy,
                 policy,
                 {},
                 ops::SwiGLUSoftcap{4.F, 25.F}},
                experts,
                top_k};
    }

    ops::CudaSwiGLUExperts bank(LinearPolicy) const {
        return {{gate_up.view(), down.view(), experts, gate_up.codes.device()}, {}};
    }

    std::size_t expert_bytes() const {
        return gate_up.view().code_stride + gate_up.view().scale_stride + down.view().code_stride +
               down.view().scale_stride;
    }

    double compute_rate(LinearPolicy policy) const {
        return policy == LinearPolicy::A16Only ? bench::kRoofline.bf16_f32_tflops
                                               : bench::kRoofline.mxfp8_f32_tflops;
    }

    const char* plan_name(const routed::Plan& plan) const {
        return schedule_name(plan.mxfp8);
    }
};

template <class Fixture>
void run(const Options& o, const Fixture& fixture) {
    meter::Csv csv(o.timing.csv_out,
                   "distribution,unique_experts,assignments,unique_operand_bytes,useful_flops,"
                   "routed_memory_bound_us,sustained_memory_bound_us,allowed_compute_bound_us,"
                   "ideal_roofline_us,roofline_efficiency_pct,logical_gb_s,useful_tflops,"
                   "first_slice_plan,min_expert_load,max_expert_load");
    bench::L2FlushBuffer flush(o.timing.flush_bytes);
    const auto cuda     = bench::device_context().execution_view();
    const auto policy   = o.policy == "a16"  ? LinearPolicy::A16Only
                          : o.policy == "a8" ? LinearPolicy::AllowA8
                                             : LinearPolicy::AllowA4;
    const auto profile  = fixture.profile(policy);
    const auto bank     = fixture.bank(policy);
    constexpr auto mode = ops::MoeEpilogue::Write;
    const ops::ExpertSubset subset{};
    const auto& shape = routed::shape_dispatch(profile);

    for (const int tokens : o.tokens) {
        const auto first_slice =
            shape.select(profile.ffn, std::min(tokens, shape.slice_tokens(subset.kind)), 0);
        const char* plan_name = fixture.plan_name(first_slice);
        const int assignments = tokens * top_k;
        auto x                = bench::make_bf16(std::size_t(tokens) * hidden, 784, -1.F, 1.F);
        auto output           = bench::make_buffer(x.size());
        const Tensor input(x.span(), DType::BF16, {hidden, tokens});
        Tensor y(output.span(), DType::BF16, {hidden, tokens});
        std::vector<float> coefficients(assignments);
        for (int a = 0; a < assignments; ++a) coefficients[a] = .05F + (a % top_k) * .017F;
        auto c = bench::make_buffer(coefficients.size() * sizeof(float));
        c.copy_from_host(coefficients.data(), c.size());
        for (const std::string distribution : test::softcap::distributions) {
            if (o.distribution != "all" && distribution != o.distribution) continue;
            const auto ids = test::softcap::make_routes(distribution, tokens, experts, top_k);
            std::array<int, experts> loads{};
            for (const int e : ids) ++loads[e];
            const int unique = std::count_if(loads.begin(), loads.end(), [](int n) { return n; });
            int minimum = tokens, maximum = 0;
            for (const int n : loads)
                if (n) {
                    minimum = std::min(minimum, n);
                    maximum = std::max(maximum, n);
                }
            auto ids_device = meter::make_i32(ids);
            const ops::ExpertRoutes routes{Tensor(ids_device.span(), DType::I32, {top_k, tokens}),
                                           Tensor(c.span(), DType::FP32, {top_k, tokens})};
            const std::size_t operands = unique * fixture.expert_bytes() + x.size() +
                                         output.size() + ids_device.size() + c.size();
            const std::uint64_t flops = std::uint64_t(assignments) * 6 * hidden * intermediate;
            const double memory       = operands / (bench::kRoofline.memory_gbs * 1000);
            const double sustained    = operands / (bench::kRoofline.sustained_read_gbs * 1000);
            const double compute      = flops / (fixture.compute_rate(policy) * 1e6);
            const double ideal        = std::max(memory, compute);
            const auto bytes          = ops::routed_swiglu_workspace_capacity_bytes(
                profile, subset.kind, y.dtype, mode, tokens, tokens);
            auto workspace = bench::make_buffer(bytes);
            MemoryArena arena(workspace.span());
            const auto launch = [&](CudaExecutionView execution) {
                ops::routed_swiglu(input, profile, bank, routes, subset, mode, y, arena, execution);
            };
            for (const auto& execution : meter::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : meter::selections(o.timing.cache, "cold", "warm")) {
                    const auto result = meter::measure(
                        launch, [](CudaExecutionView) {}, flush, cuda, o.timing, execution, cache);
                    const double us = result.gpu.median_us;
                    std::ostringstream extra;
                    extra << distribution << ',' << unique << ',' << assignments << ',' << operands
                          << ',' << flops << ',' << memory << ',' << sustained << ',' << compute
                          << ',' << ideal << ',' << 100 * ideal / us << ','
                          << operands / (us * 1000) << ',' << flops / (us * 1e6) << ',' << plan_name
                          << ',' << minimum << ',' << maximum;
                    meter::Row row{.op              = "routed_swiglu_softcap",
                                   .format          = o.format,
                                   .policy          = o.policy,
                                   .n               = hidden,
                                   .k               = intermediate,
                                   .tokens          = tokens,
                                   .workspace_bytes = bytes,
                                   .workspace_peak  = arena.peak_used()};
                    csv.write(row, result, execution, cache, extra.str());
                }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.format == "mxfp8")
            run(options, Mxfp8Fixture{});
        else
            run(options, Nvfp4Fixture{});
    } catch (const std::exception& e) {
        std::cerr << "ninfer_moe_softcap_bench: " << e.what() << '\n';
        return 1;
    }
}
