#include "models/qwen3_5/measurement.h"

#include "models/qwen3_5/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <vector>

namespace ninfer::models::qwen3_5 {

std::string prefill_signature(const Model& model) {
    using Json         = nlohmann::json;
    const auto& config = model.config();
    const auto& t      = config.text;
    Json facts{{"implementation", "qwen3_5-prefill-1"},
               {"architecture", architecture_name(t.architecture)},
               {"hidden_size", t.hidden_size},
               {"vocab_size", t.vocab_size},
               {"public_token_count", model.resources().public_token_count},
               {"num_hidden_layers", t.num_hidden_layers},
               {"rms_norm_eps", t.rms_norm_eps},
               {"layer_types", t.layer_types}};
    if (t.attention) {
        const auto& a      = *t.attention;
        facts["attention"] = {a.num_attention_heads, a.num_key_value_heads, a.head_dim};
    }
    if (t.rope_parameters) {
        const auto& r = *t.rope_parameters;
        facts["rope"] = {r.rope_theta, r.rotary_dim, r.pair_axes};
    }
    if (t.gdn) {
        const auto& g = *t.gdn;
        facts["gdn"]  = {g.linear_num_key_heads, g.linear_key_head_dim, g.linear_num_value_heads,
                         g.linear_value_head_dim, g.linear_conv_kernel_dim};
    }
    if (const auto* d = std::get_if<DenseConfig>(&t.ffn)) {
        facts["ffn"] = {d->intermediate_size};
    } else {
        const auto& m = std::get<MoeConfig>(t.ffn);
        facts["ffn"]  = {m.num_experts, m.num_experts_per_tok, m.moe_intermediate_size,
                         m.shared_expert_intermediate_size};
    }
    if (config.vision) {
        const auto& v   = *config.vision;
        facts["vision"] = {
            v.depth,      v.hidden_size,         v.intermediate_size,  v.num_heads,
            v.patch_size, v.temporal_patch_size, v.spatial_merge_size, v.num_position_embeddings};
    }
    std::vector<const BoundWeight*> weights;
    // PrefillWork prices primary Text/Vision reconstruction. The coefficients were measured
    // for that work; optional draft/proposal calls do not change the baseline's applicability.
    for (const auto& weight : model.weight_data()) {
        if (weight.name.starts_with("text/") || weight.name.starts_with("vision/")) {
            weights.push_back(&weight);
        }
    }
    std::sort(weights.begin(), weights.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });
    std::map<const WeightParent*, std::size_t> parents;
    auto& inventory = facts["weights"] = Json::array();
    for (const auto* weight : weights) {
        Json item{{"role", weight->name},
                  {"shape", weight->view.shape},
                  {"parts", Json::array()},
                  {"uses", Json::array()}};
        for (const auto& part : weight->view.parts) {
            const auto [it, inserted] = parents.emplace(part.parent, parents.size());
            const auto& geometry      = part.parent->geometry;
            item["parts"].push_back({it->second, part.begin, part.end, geometry.format,
                                     geometry.layout, geometry.shape, geometry.padded_columns,
                                     geometry.group_size});
        }
        for (const auto& use : weight->uses) {
            if (!use.input.starts_with("text/") && !use.input.starts_with("vision/")) { continue; }
            item["uses"].push_back(
                {use.input, use.policy, use.activation_input_divisor.has_value()});
        }
        inventory.push_back(std::move(item));
    }
    return frontend::sha256_hex(frontend::sha256(facts.dump()));
}

} // namespace ninfer::models::qwen3_5
