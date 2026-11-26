#include "ninfer/ops/logit_softcap.h"
#include "common/gpu_activity.cuh"
#include "op_measurement.cuh"

#include <cmath>
#include <optional>

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) try {
    std::vector<int> vocab{201024}, batches{1, 8, 64};
    std::string format = "both", alias = "separate";
    float softcap = 30.f;
    measurement::Options timing;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Logit softcap: --v V[,V...] --batch B[,B...] --softcap F\n"
                         "               --dtype bf16|fp32|both --alias separate|inplace|both\n"
                         "               Fixture logits are uniform on [-32,32].\n";
            measurement::help();
            return 0;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--v")
            vocab = measurement::integers(value);
        else if (flag == "--batch")
            batches = measurement::integers(value);
        else if (flag == "--softcap") {
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), softcap);
            if (error != std::errc{} || end != value.data() + value.size() || !(softcap > 0.f) ||
                !std::isfinite(softcap))
                throw std::invalid_argument("softcap must be positive and finite");
        } else if (flag == "--dtype" && (value == "bf16" || value == "fp32" || value == "both"))
            format = value;
        else if (flag == "--alias" &&
                 (value == "separate" || value == "inplace" || value == "both"))
            alias = value;
        else if (!measurement::parse(timing, flag, value))
            throw std::invalid_argument("invalid option " + std::string(flag));
    }

    char cap_buffer[32];
    const auto [cap_end, cap_error] =
        std::to_chars(cap_buffer, cap_buffer + sizeof(cap_buffer), softcap);
    if (cap_error != std::errc{}) throw std::runtime_error("cannot format softcap");
    const std::string cap_text(cap_buffer, cap_end);

    constexpr int condition_us = 200;
    measurement::hardware();
    std::cerr << "GPU activity prelude " << condition_us << " us outside timing\n";
    measurement::LogicalBenchmark benchmark(timing);
    for (int v : vocab)
        for (int batch : batches)
            for (const auto& dtype : measurement::selections(format, "bf16", "fp32")) {
                const bool bf16  = dtype == "bf16";
                const DType type = bf16 ? DType::BF16 : DType::FP32;
                const auto count = std::size_t(v) * batch;
                auto x =
                    bf16 ? make_bf16(count, 101, -32.f, 32.f) : make_f32(count, 101, -32.f, 32.f);
                Tensor input(x.span(), type, {v, batch});
                for (const auto& placement :
                     measurement::selections(alias, "separate", "inplace")) {
                    const bool inplace = placement == "inplace";
                    DeviceBuffer dst;
                    std::optional<SavedBuffer> initial;
                    if (inplace)
                        initial.emplace(x);
                    else
                        dst = DeviceBuffer(x.device(), x.size());
                    Tensor out(inplace ? x.span() : dst.span(), type, {v, batch});
                    benchmark.run(
                        {.op     = "logit-softcap",
                         .format = dtype,
                         .policy = placement + "/cap=" + cap_text,
                         .n      = v,
                         .batch  = batch},
                        2 * x.size(),
                        [&](CudaExecutionView e) { ops::logit_softcap(input, softcap, out, e); },
                        [&](CudaExecutionView e) {
                            if (initial) initial->restore(e);
                            condition_gpu(e, condition_us);
                        });
                    if (initial) {
                        initial->restore(device_context().execution_view());
                        CUDA_CHECK(cudaStreamSynchronize(device_context().stream()));
                    }
                }
            }
    return 0;
} catch (const std::exception& error) {

    std::cerr << "ninfer_logit_softcap_bench: " << error.what() << '\n';
    return 2;
}
