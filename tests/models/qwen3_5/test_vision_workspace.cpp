#include "models/common/qwen/vision/prefill.h"
#include "core/device.h"
#include "core/buffer.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/common/qwen/vision/vision.h"
#include "models/qwen3_5/planning.h"
#include <cstdlib>

#include "models/common/qwen/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_handoff_lifetime(ninfer::DeviceContext& device,
                           const q35::execution::Parameters& parameters) {
    const auto& config     = parameters.profile->config.vision.value();
    const auto merge       = static_cast<std::int32_t>(config.spatial_merge_size);
    const auto patches     = static_cast<std::size_t>(merge) * merge;
    const auto make_prompt = [&](std::uint32_t seed) {
        ninfer::models::qwen::PreparedPromptData prompt;
        prompt.token_ids.resize(5);
        prompt.token_types = {0, static_cast<std::uint8_t>(ninfer::models::qwen::Modality::Image),
                              0, static_cast<std::uint8_t>(ninfer::models::qwen::Modality::Image),
                              0};
        prompt.prepare.media_items   = 2;
        prompt.prepare.raw_patches   = 2 * patches;
        prompt.prepare.vision_tokens = 2;
        for (std::size_t item = 0; item != 2; ++item) {
            prompt.vision_items.push_back(ninfer::models::qwen::VisionItem{
                .modality    = ninfer::models::qwen::Modality::Image,
                .grid        = {.temporal = 1, .height = merge, .width = merge},
                .patch_begin = item * patches,
                .patch_count = patches,
                .token_spans = {{.begin = 2 * item + 1, .count = 1}},
            });
            auto payload = std::make_shared<ninfer::models::qwen::PreparedMediaPayload>();
            payload->patch_elements = patches * config.patch_width();
            payload->patches        = std::make_unique<std::uint16_t[]>(payload->patch_elements);
            for (auto& value : payload->mutable_span()) {
                seed  = seed * 1664525U + 1013904223U;
                value = static_cast<std::uint16_t>(0x3f00U | ((seed >> 24) & 0x7fU) |
                                                   ((seed >> 16) & 0x8000U));
            }
            prompt.media_payloads.push_back(std::move(payload));
        }
        return prompt;
    };
    auto prompt_a = make_prompt(1);
    auto prompt_b = make_prompt(3);
    ninfer::models::qwen::VisionPrefillPlan plan;
    plan.control_plan = std::make_shared<ninfer::models::qwen::VisionControlPlan>(
        ninfer::models::qwen::plan_vision_control(prompt_a, config));
    plan.control = std::make_shared<ninfer::models::qwen::VisionControl>(
        ninfer::models::qwen::build_vision_control(prompt_a, *plan.control_plan, 0));
    plan.uses             = {{.begin = 1, .end = 2, .prepared_item_index = 0, .control_index = 0},
                             {.begin = 3, .end = 4, .prepared_item_index = 1, .control_index = 1}};
    plan.max_merged_count = 1;
    const auto workspace_plan =
        ninfer::models::qwen::VisionContext::plan_workspace(config, *parameters.vision, 1, 256);
    ninfer::DeviceBuffer workspace(device.device(), workspace_plan.capacity_bytes);
    const auto backing = workspace.span();
    ninfer::models::qwen::VisionHandoffState handoff;
    std::size_t peak_bytes    = 0;
    const auto make_execution = [&](const ninfer::models::qwen::PreparedPromptData& prompt) {
        auto input            = std::make_shared<ninfer::models::qwen::VisionExecutionInput>();
        input->prompt_tokens  = static_cast<std::uint32_t>(prompt.token_ids.size());
        input->vision_items   = prompt.vision_items;
        input->media_payloads = prompt.media_payloads;
        input->plan           = plan;
        return std::make_unique<ninfer::models::qwen::VisionPrefillExecution>(
            device, config, *parameters.vision, backing, workspace_plan, std::move(input), handoff,
            peak_bytes);
    };
    const auto read = [&](const ninfer::models::qwen::VisionChunk& chunk) {
        require(chunk.control != nullptr, "Vision replay omitted its item");
        std::vector<std::uint16_t> output(chunk.embeddings.bytes() / sizeof(std::uint16_t));
        CUDA_CHECK(cudaMemcpyAsync(output.data(), chunk.embeddings.data, chunk.embeddings.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream()));
        device.synchronize();
        return output;
    };
    auto execution_a    = make_execution(prompt_a);
    auto execution_b    = make_execution(prompt_b);
    const auto original = read(execution_a->prepare_chunk(0, 2));
    require(read(execution_a->prepare_chunk(2, 2)) != original,
            "Vision handoff fixture needs distinguishable items");
    require(read(execution_a->prepare_chunk(0, 2)) == original,
            "Vision replay failed to revisit an earlier item");
    require(execution_a->prepare_chunk(4, 1).control == nullptr,
            "text after the last Vision item retained a scatter binding");
    require(read(execution_a->prepare_chunk(0, 2)) == original,
            "Vision replay failed after all item spans were consumed");
    const auto other = read(execution_b->prepare_chunk(0, 2));
    require(other != original, "Vision handoff fixture needs distinguishable requests");
    require(execution_a->active_handoff_bytes() == 0 && execution_b->active_handoff_bytes() != 0,
            "Vision handoff remained owned by an overwritten execution");
    require(read(execution_a->prepare_chunk(1, 1)) == original,
            "Vision request failed to reconstruct an overwritten handoff");
    execution_b->retire_handoff();
    require(execution_a->active_handoff_bytes() != 0,
            "retiring an overwritten execution released the current Vision handoff");

    // Text/decode may replace all general scratch while a Vision item remains live.
    CUDA_CHECK(cudaMemsetAsync(workspace.data(), 0xa5, workspace_plan.general_capacity_bytes,
                               device.stream()));
    require(read(execution_a->prepare_chunk(1, 1)) == original,
            "general workspace use changed the live Vision handoff");
    { auto unused_execution = make_execution(prompt_b); }
    require(execution_a->active_handoff_bytes() != 0,
            "destroying an unused execution released another Vision handoff");
    require(read(execution_b->prepare_chunk(0, 2)) == other,
            "retired Vision request could not reconstruct its handoff");
    execution_a.reset();
    require(execution_b->active_handoff_bytes() != 0,
            "destroying an overwritten execution released the current Vision handoff");
    require(prompt_a.media_payloads[0] && prompt_a.media_payloads[1] &&
                prompt_b.media_payloads[0] && prompt_b.media_payloads[1],
            "Vision execution released durable prepared media");
}

} // namespace

int main() {
    namespace q35        = ninfer::models::qwen3_5;
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) { return 77; }
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
        ninfer::artifact::Reader reader(artifact);
        auto definition = q35::read_definition(reader, selected);
        auto profile    = q35::execution::describe_model(*definition);
        auto model =
            q35::materialize_model(q35::resolve_load(reader, definition, device.device()), device);
        const q35::execution::Parameters parameters(*model, profile);
        const auto capacity = [&](std::uint32_t max_context) {
            ninfer::EngineOptions options;
            options.max_context         = max_context;
            options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
            options.prefill_chunk       = 1024;
            options.kv_cache            = ninfer::KvCacheStorage::Fp8E4M3Row256;
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens        = 3;
            options.speculative.proposal_head       = ninfer::ProposalHead::Optimized;
            options.enable_vision                   = true;
            options.use_cuda_graph                  = false;
            options.context_cache.extra_state_slots = 1;
            auto planner     = q35::make_sequence_planner(parameters, device, options);
            const auto pages = planner.capacity_curve().minimum_primary_page_groups;
            auto plan        = std::move(planner).finalize(
                {pages,
                        options.max_concurrency + options.context_cache.extra_state_slots.value_or(0)});
            const auto session = q35::plan_session(plan);
            require(session.vision_workspace.has_value(), "Vision session lost its workspace plan");
            const auto vision = *session.vision_workspace;
            require(plan.workspace_capacity_bytes() >= vision.capacity_bytes &&
                        vision.capacity_bytes >= vision.encode_peak_bytes &&
                        vision.handoff_offset_bytes >= vision.general_capacity_bytes &&
                        vision.handoff_offset_bytes + vision.handoff_capacity_bytes <=
                            vision.capacity_bytes,
                    "Vision workspace does not cover encoding and the retained handoff");
            return vision;
        };
        // A longer text context does not enlarge the one-item Vision handoff.
        const auto item_tokens      = ninfer::models::qwen::kMaximumVisionItemTokens;
        const auto at_item_limit    = capacity(item_tokens);
        const auto above_item_limit = capacity(131072);
        require(at_item_limit.max_merged_tokens == item_tokens &&
                    above_item_limit.max_merged_tokens == item_tokens &&
                    at_item_limit.handoff_capacity_bytes == above_item_limit.handoff_capacity_bytes,
                "Long text context increased the retained Vision item capacity");
        test_handoff_lifetime(device, parameters);
        std::cout << "Vision handoff capacity remains one item for long text context\n";
        std::cout << "Vision handoff survives request rotation and replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
