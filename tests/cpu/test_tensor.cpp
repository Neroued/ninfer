#include "core/arena.h"
#include "core/buffer.h"
#include "core/cpu/executor.h"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
} // namespace

int main() {
    using namespace ninfer;
    try {
        HostBuffer owner(4096);
        MemoryArena arena(owner.span());
        auto matrix = arena.alloc(DType::FP32, {8, 4});
        auto* data  = static_cast<float*>(matrix.data);
        for (int i = 0; i < 32; ++i) { data[i] = static_cast<float>(i); }

        CpuTopology topology;
        CpuExecutor executor(topology, plan_cpu_team(topology.info(), 1));
        {
            auto scope     = arena.scope();
            Tensor scratch = arena.alloc(DType::FP32, {32});

            struct Work {
                float* input;
                float* temporary;
            } work{data, static_cast<float*>(scratch.data)};

            executor.execution_view().run(
                [](void* pointer, const CpuWorkerContext& worker) noexcept {
                    auto& work       = *static_cast<Work*>(pointer);
                    const auto range = worker.range(32);
                    for (auto i = range.begin; i < range.end; ++i)
                        work.temporary[i] = work.input[i] * 2;
                    worker.barrier();
                    for (auto i = range.begin; i < range.end; ++i)
                        work.input[i] = work.temporary[31 - i];
                },
                &work);
        }
        for (int i = 0; i < 32; ++i) require(data[i] == 2.0F * (31 - i), "CPU tensor computation");

        std::cout << "OK CPU tensor computation\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
