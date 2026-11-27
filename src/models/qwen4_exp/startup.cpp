#include "models/qwen4_exp/startup.h"
#include "models/common/frontend/digest.h"

#include <nlohmann/json.hpp>
#include <iostream>

namespace ninfer::models::qwen4_exp {
Startup::Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
                 const EngineOptions& options, Device device)
    : reader_(rank_source), definition_(read_definition(definition_source, load_options(options))) {
    const auto layouts = ple::placements(options.ple, definition_->config().text.ple.shards, 1);
    if (options.deployment &&
        plan_parallel(*definition_, *options.deployment, layouts).ranks.size() != 1)
        throw std::invalid_argument("Qwen4Exp local startup requires one rank");
    batches_ = std::make_shared<const BatchExecutionPlan>(make_batch_execution_spec(
        options.max_concurrency, 1,
        options.use_cuda_graph && options.purpose == EnginePurpose::Generation));
    for (const auto layout : layouts) {
        try {
            auto table = plan_ple_table(reader_, *definition_, layout, {}, device);
            tables_.push_back({layout, table.geometry(), table.device_bytes(), table.host_bytes(),
                               execution::describe_model(*definition_, layout)});
        } catch (const artifact::ArtifactIoError& error) {
            if (layouts.size() == 1 || layout != ple::Placement::Disk) throw;
            std::clog << "[PLE] " << ple::name(layout) << ": " << error.what() << '\n';
        }
    }
    const auto add = [&](WeightPlacement placement, Device experts) {
        // Only the immutable model weights enter this materialization plan.
        auto base =
            resolve_load(reader_, definition_, device, experts, {}, ple::Placement::Replicated);
        weights_.push_back({placement, base.materialization().device_capacity_bytes,
                            base.materialization().host_capacity_bytes});
    };
    add(WeightPlacement::Resident, device);
    offload_ = describe_moe_plan(*tables_.front().profile, options.moe);
    if (offload_) add(WeightPlacement::HostExperts, Device::cpu());
}

const Startup::TableChoice& Startup::table(ple::Placement placement) const {
    const auto found = std::ranges::find(tables_, placement, &TableChoice::placement);
    if (found == tables_.end()) throw std::logic_error("PLE layout was not offered");
    return *found;
}

std::vector<Startup::Candidate> Startup::placements(DeviceContext& device,
                                                    const EngineOptions& options,
                                                    std::size_t available, std::size_t host,
                                                    bool shared) const {
    std::vector<Candidate> result;
    std::string failure;
    for (const auto& weights : weights_) {
        std::vector<std::pair<std::size_t, Candidate>> ranked;
        for (std::size_t i = 0; i < tables_.size(); ++i) {
            const auto& table = tables_[i];
            Candidate choice{{weights.placement, table.placement},
                             weights.device_bytes + table.device_bytes,
                             weights.host_bytes + table.host_bytes,
                             i,
                             0};
            const auto charge = choice.device_bytes + (shared ? choice.host_bytes : 0);
            try {
                if (charge > available || choice.host_bytes > host)
                    throw runtime::ContextBudgetError(
                        "weight and table backing exceed budget by " +
                        std::to_string(
                            std::max(charge - std::min(charge, available),
                                     choice.host_bytes - std::min(choice.host_bytes, host))) +
                        " bytes");
                const auto remaining      = available - charge,
                           host_remaining = host - choice.host_bytes;
                auto configured =
                    plan(device, options, choice.placement, remaining, host_remaining);
                if (!configured)
                    throw runtime::ContextBudgetError(
                        "insufficient capacity for the configured Context working set");
                choice.upper_blocks =
                    capacity_blocks(*configured, options, remaining, host_remaining);
                ranked.emplace_back(choice.upper_blocks, choice);
            } catch (const runtime::ContextBudgetError& error) {
                failure += std::string(ple::name(table.placement)) + ": " + error.what() + "; ";
            }
        }
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& entry : ranked) result.push_back(entry.second);
    }
    if (result.empty()) throw runtime::ContextBudgetError("Qwen4Exp placement: " + failure);
    if (!failure.empty()) std::clog << "[Qwen4Exp placement] " << failure << '\n';
    return result;
}

std::optional<Startup::ExecutionPlan> Startup::plan(DeviceContext& device,
                                                    const EngineOptions& options,
                                                    LocalPlacement placement, std::size_t available,
                                                    std::size_t available_host_bytes) const {
    const auto& choice = table(placement.table);
    const bool host    = placement.weights == WeightPlacement::HostExperts;
    if (host && !offload_) return {};
    auto planned = plan_execution(choice.profile, choice.geometry, options, batches_, device,
                                  available, host ? offload_ : nullptr, true, available_host_bytes);
    if (!planned) return {};
    const auto capacity = planned->capacity;
    const auto chunk    = planned->shape.prefill_chunk;
    const auto moe      = planned->moe;
    return ExecutionPlan{std::move(*planned), capacity, placement, chunk, {moe}};
}

std::size_t Startup::capacity_blocks(const ExecutionPlan& plan, const EngineOptions& options,
                                     std::size_t available, std::size_t host) const {
    if (options.ple.preference != PlePreference::Capacity || tables_.size() == 1) return 0;
    const auto budget = describe_execution_capacity(plan.sequence, options);
    const auto capacity =
        runtime::resolve_context_capacity(ContextCapacityPolicy::automatic(), budget, available,
                                          host, plan.sequence.shared_physical_memory);
    return capacity.device_bytes / capacity.block_bytes;
}

void Startup::report_placement(std::span<const Candidate> candidates, std::size_t selected,
                               std::size_t blocks, std::string_view reasons) const {
    const auto& winner = candidates[selected];
    std::clog << "[PLE] selected " << ple::name(winner.placement.table);
    if (blocks) std::clog << " available_blocks=" << blocks;
    for (const auto& candidate : candidates) {
        if (candidate.placement.weights != winner.placement.weights ||
            candidate.placement.table == winner.placement.table)
            continue;
        std::clog << "; alternative " << ple::name(candidate.placement.table);
        if (candidate.upper_blocks)
            std::clog << " upper_bound_blocks=" << candidate.upper_blocks;
        else
            std::clog << " preference=" << candidate.preference;
    }
    if (!reasons.empty()) std::clog << "; " << reasons;
    std::clog << '\n';
}

void Startup::release_other_weights(LocalPlacement next) {
    if (current_weights_ != next.weights) retained_.reset();
}

void Startup::retain_weights(std::unique_ptr<ModelShard> model) {
    model->release_ple();
    retained_ = std::move(model);
}

Startup::Candidate Startup::outstanding(Candidate candidate) const {
    if (retained_ && candidate.placement.weights == current_weights_) {
        const auto& choice     = table(candidate.placement.table);
        candidate.device_bytes = choice.device_bytes;
        candidate.host_bytes   = choice.host_bytes;
    }
    return candidate;
}

std::unique_ptr<ModelShard> Startup::materialize(DeviceContext& device, LocalPlacement placement,
                                                 const StartupObserver& observer) {
    if (current_weights_ != placement.weights) retained_.reset();
    current_weights_  = placement.weights;
    const bool reused = bool(retained_);
    if (!retained_) {
        auto base = resolve_load(
            reader_, definition_, device.device(),
            placement.weights == WeightPlacement::HostExperts ? Device::cpu() : device.device(), {},
            ple::Placement::Replicated);
        retained_ = materialize_weights(std::move(base), device, &observer);
    }
    try {
        auto table_plan =
            plan_ple_table(reader_, *definition_, placement.table, {}, device.device());
        retained_->prepare_ple(std::move(table_plan), device, &observer);
    } catch (const artifact::ArtifactIoError& error) {
        throw runtime::ContextBudgetError(std::string(ple::name(placement.table)) + ": " +
                                          error.what());
    }
    last_load_ = retained_->storage_stats();
    if (reused) {
        last_load_ = {};
        if (const auto* table = std::get_if<DevicePleTable>(&retained_->ple_table())) {
            last_load_ = table->backing.stats();
            last_load_.h2d_bytes += table->decode.size();
        }
    }
    profile = table(placement.table).profile;
    return std::move(retained_);
}

qwen::Frontend Startup::frontend(const EngineOptions& o) const {
    return create_frontend(*definition_, o);
}

qwen::Frontend create_frontend(const ModelDefinition& definition, const EngineOptions& o) {
    const auto generation =
        nlohmann::json::parse(definition.resources().text.generation_config_json);
    SamplingPreset preset;
    preset.temperature = generation.value("temperature", 1.0F);
    preset.top_k       = generation.value("top_k", 20);
    preset.top_p       = generation.value("top_p", 0.95F);
    preset.min_p       = generation.value("min_p", 0.0F);
    return qwen::make_frontend(definition.resources(),
                               {.chat_template_path       = o.chat_template_path,
                                .grammar_cache_bytes      = o.grammar_cache_bytes,
                                .vision_enabled           = o.enable_vision,
                                .max_context              = o.max_context,
                                .media_cache_bytes        = o.media_cache_bytes,
                                .media_live_bytes         = o.media_live_bytes,
                                .media_preprocess_threads = o.media_preprocess_threads},
                               {preset, preset});
}

std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters& p,
                                                 qwen4_exp::ExecutionPlan&& plan,
                                                 DeviceContext& device,
                                                 const StartupObserver& observer,
                                                 GraphPreparationMode mode) {
    auto program = std::make_unique<RankProgram>(p, std::move(plan), device, observer);
    program->prepare_execution(mode);
    return program;
}

GraphPreparationPlan Startup::graph_plan(const EngineOptions& options,
                                         const ExecutionPreparation& preparation) const {
    return describe_graph_preparation(options, *batches_, preparation.experts.get());
}

Startup::Calibration Startup::calibrate(const execution::Parameters& parameters,
                                        DeviceContext& device, const EngineOptions& options,
                                        const ExecutionPreparation& preparation) const {
    return calibrate_graphs(parameters, device, options, batches_, preparation.experts);
}

std::string prefill_signature(const ModelDefinition& model, const moe::Plan* moe,
                              std::uint32_t chunk) {
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
    return models::frontend::sha256_hex(models::frontend::sha256(facts.dump()));
}

std::string Startup::prefill_signature(const ExecutionPreparation& preparation,
                                       std::uint32_t chunk) const {
    return qwen4_exp::prefill_signature(*definition_, preparation.experts.get(), chunk) +
           ";ple=" + ple::name(profile->ple_placement);
}

} // namespace ninfer::models::qwen4_exp
