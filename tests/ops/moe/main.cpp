#include "fixture.h"

#include <tuple>

namespace ninfer::test::moe {
int test_weights();
int test_complete(Fixture&, CudaExecutionView);
int test_route(Fixture&, CudaExecutionView);
int test_routed(Fixture&, CudaExecutionView);
} // namespace ninfer::test::moe

int main() {
    using namespace ninfer;
    using namespace ninfer::test;
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        auto& context = cuda_environment().context;
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "create MoE test stream");
        const auto execution = context.execution_view(stream);
        int failures         = moe::test_weights();
        for (const auto [gu, down, name] :
             {std::tuple{QType::Q4_G64_FP16, QType::Q5_G64_FP16, "q4-q5"},
              std::tuple{QType::Q4_G64_FP16, QType::Q6_G64_FP16, "q4-q6"},
              std::tuple{QType::Q8_G32_FP16, QType::Q8_G32_FP16, "q8-q8"}}) {
            std::cout << "MoE profile " << name << std::endl;
            moe::Fixture fixture(gu, down);
            failures += moe::test_complete(fixture, execution);
            if (down == QType::Q5_G64_FP16) { failures += moe::test_route(fixture, execution); }
            failures += moe::test_routed(fixture, execution);
            failures += fixture.verify_preserved();
        }
        cuda_check(cudaStreamDestroy(stream), "destroy MoE test stream");
        std::cout << (failures ? "FAIL" : "OK") << " MoE contracts\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
