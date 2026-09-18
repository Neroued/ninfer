#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"
#include <cstdlib>

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    namespace qwen       = ninfer::models::qwen3_5;
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) { return 77; }
    static_assert(ninfer::models::qwen3_5::kMaximumVisionItemTokens == 16384);
    // The current Vision route fits below 827 MiB, including arena alignment and empty-scratch
    // backing. The context growth check below protects the single-item lifetime bound.
    constexpr std::size_t kWorkspaceCeiling = 827ULL << 20;
    try {
        int devices       = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            return 77;
        }
        CUDA_CHECK(status);
        ninfer::DeviceContext device;
        ninfer::models::LoadOptions selected;
        selected.vision        = true;
        selected.speculative   = ninfer::SpeculativeBackend::Mtp;
        selected.proposal_head = ninfer::ProposalHead::Optimized;
        auto model             = qwen::load_model(artifact, selected, device);
        const qwen::execution::Parameters parameters(*model);
        const auto capacity = [&](std::uint32_t max_context) {
            ninfer::EngineOptions options;
            options.max_context         = max_context;
            options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
            options.prefill_chunk       = 1024;
            options.kv_cache            = ninfer::KvCacheStorage::Fp8E4M3Row256;
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens         = 3;
            options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
            options.enable_vision                    = true;
            options.use_cuda_graph                   = false;
            options.context_cache.device_state_slots = 1;
            auto planner     = qwen::make_sequence_planner(parameters, device, options);
            const auto pages = planner.capacity_curve().minimum_main_page_groups;
            return std::move(planner).finalize(pages).workspace_capacity_bytes();
        };
        // Increasing total context cannot exceed the single Vision item's 16K workspace.
        const auto at_item_limit    = capacity(16384);
        const auto above_item_limit = capacity(131072);
        if (at_item_limit == 0 || at_item_limit > kWorkspaceCeiling ||
            above_item_limit != at_item_limit) {
            throw std::runtime_error("Vision workspace exceeded the single-item bound: at=" +
                                     std::to_string(at_item_limit) +
                                     " above=" + std::to_string(above_item_limit));
        }
        std::cout << "Vision workspace remains bounded for long text context\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
