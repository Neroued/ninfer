#include "models/qwen4_exp/startup.h"
#include "models/common/qwen/frontend/digest.h"

#include <nlohmann/json.hpp>

namespace ninfer::models::qwen4_exp {
Startup::Startup(const artifact::Reader& reader, const EngineOptions& options, Device device)
    : reader_(reader), selection_(load_options(options)), source_(plan_load(reader, selection_)),
      table_(source_->ple_table().geometry()) {
    profile = execution::describe_model(*source_);
    placements_.push_back(
        {WeightPlacement::Resident, source_->placement(device, device).device_capacity_bytes});
    offload_ = describe_moe_plan(*profile, options.moe);
    if (offload_)
        placements_.push_back({WeightPlacement::HostExperts,
                               source_->placement(device, Device::cpu()).device_capacity_bytes});
}

std::optional<Startup::ExecutionPlan> Startup::plan(DeviceContext& device,
                                                    const EngineOptions& options,
                                                    WeightPlacement placement,
                                                    std::size_t available) const {
    const bool host = placement == WeightPlacement::HostExperts;
    if (host && !offload_) return {};
    auto planned =
        plan_execution(profile, table_, options, device, available, host ? offload_ : nullptr);
    if (!planned) return {};
    const auto capacity = planned->capacity;
    const auto chunk    = planned->shape.prefill_chunk;
    const auto moe      = planned->moe;
    return ExecutionPlan{std::move(*planned), capacity, placement, chunk, {moe}};
}

std::unique_ptr<Model> Startup::materialize(DeviceContext& device, WeightPlacement placement,
                                            const StartupObserver& observer) {
    if (!source_) source_.emplace(plan_load(reader_, selection_));
    auto resolved =
        resolve_load(std::move(*source_), device.device(),
                     placement == WeightPlacement::HostExperts ? Device::cpu() : device.device());
    source_.reset();
    return materialize_model(std::move(resolved), device, &observer);
}

qwen::Frontend Startup::frontend(const Model& model, const EngineOptions& o) const {
    const auto generation = nlohmann::json::parse(model.resources().generation_config_json);
    SamplingPreset preset;
    preset.temperature = generation.value("temperature", 1.0F);
    preset.top_k       = generation.value("top_k", 20);
    preset.top_p       = generation.value("top_p", 0.95F);
    preset.min_p       = generation.value("min_p", 0.0F);
    return qwen::make_frontend(model.resources(),
                               {.chat_template_path       = o.chat_template_path,
                                .grammar_cache_bytes      = o.grammar_cache_bytes,
                                .vision_enabled           = o.enable_vision,
                                .max_context              = o.max_context,
                                .media_cache_bytes        = o.media_cache_bytes,
                                .media_live_bytes         = o.media_live_bytes,
                                .media_preprocess_threads = o.media_preprocess_threads},
                               {preset, preset});
}

std::unique_ptr<Program> create_program(const execution::Parameters& p,
                                        qwen4_exp::ExecutionPlan&& plan, DeviceContext& device,
                                        const StartupObserver&) {
    return std::make_unique<Program>(p, std::move(plan), device);
}

GraphPreparationPlan Startup::graph_plan(const EngineOptions& options,
                                         const ExecutionPreparation& preparation) const {
    return describe_graph_preparation(options, preparation.experts.get());
}

Startup::Calibration Startup::calibrate(const execution::Parameters& parameters,
                                        DeviceContext& device, const EngineOptions& options,
                                        const ExecutionPreparation& preparation) const {
    return calibrate_graphs(parameters, device, options, preparation.experts);
}

std::string prefill_signature(const Model& model, const moe::Plan* moe, std::uint32_t chunk) {
    using Json    = nlohmann::json;
    const auto& c = model.config().text;
    Json facts{{"implementation", "qwen4-exp-eager-1"},
               {"chunk", chunk},
               {"hidden", c.hidden},
               {"vocabulary", c.vocabulary},
               {"hc", {c.hc.streams, c.hc.lowrank}},
               {"gdn", {c.gdn.key_heads, c.gdn.value_heads, c.gdn.key_dim, c.gdn.value_dim}},
               {"qsa",
                {c.qsa.query_heads, c.qsa.kv_heads, c.qsa.head_dim, c.qsa.compression,
                 c.qsa.block_budget}},
               {"ple", {c.ple.layer, c.ple.shards, c.ple.rows_per_shard}}};
    auto& layers = facts["layers"] = Json::array();
    if (const auto& vision = model.config().vision)
        facts["vision"] = {vision->depth,
                           vision->hidden_size,
                           vision->intermediate_size,
                           vision->num_heads,
                           vision->patch_size,
                           vision->temporal_patch_size,
                           vision->spatial_merge_size,
                           vision->num_position_embeddings};
    for (const auto& layer : c.layers) layers.push_back(static_cast<int>(layer.mixer));
    auto& weights = facts["weights"] = Json::array();
    for (const auto& w : model.weight_data()) {
        Json item{{"role", w.name},
                  {"shape", w.view.shape},
                  {"parts", Json::array()},
                  {"uses", Json::array()}};
        for (const auto& part : w.view.parts)
            item["parts"].push_back(
                {part.parent->geometry.format, part.parent->geometry.layout, part.begin, part.end});
        for (const auto& use : w.uses) item["uses"].push_back({use.input, use.policy});
        weights.push_back(std::move(item));
    }
    if (moe)
        facts["moe"] = {moe->description().cache_bytes, moe->description().fixed_bytes,
                        moe->description().window_count, moe->description().cpu_workers};
    return qwen::frontend::sha256_hex(qwen::frontend::sha256(facts.dump()));
}

std::string Startup::prefill_signature(const Model& model, const ExecutionPreparation& preparation,
                                       std::uint32_t chunk) const {
    return qwen4_exp::prefill_signature(model, preparation.experts.get(), chunk);
}

} // namespace ninfer::models::qwen4_exp
