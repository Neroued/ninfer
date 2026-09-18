#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using ninfer::test::artifact_fixture::require;

struct Sample {
    std::uint64_t offset;
    std::vector<std::byte> bytes;
};

void check_selection(const qwen::LoadPlan& plan, const LoadOptions& options) {
    const auto& weights = plan.weights();
    require(weights.vision.has_value() == options.vision,
            "Vision demand differs from startup selection");
    require(weights.mtp.has_value() == (options.speculative == SpeculativeBackend::Mtp),
            "MTP selection differs");
    require(weights.draft.has_value() == (options.speculative == SpeculativeBackend::DFlash ||
                                          options.speculative == SpeculativeBackend::DFlash2),
            "draft selection differs");
    require(weights.proposal.has_value() == options.proposal_enabled(),
            "proposal selection differs");
    require(weights.text.layers.size() == plan.config().text.num_hidden_layers,
            "layer data differs from instance config");
    const auto check_head = [&](qwen::WeightId head, qwen::WeightUseId use,
                                const std::string& input) {
        require(use.parameter == head && plan.uses(head)[use.use_index].input == input,
                "shared output head selected the wrong mathematical Use");
    };
    check_head(weights.text.output_head, weights.text.output_head_use, "text/final_hidden");
    if (weights.mtp) {
        check_head(weights.mtp->output_head, weights.mtp->output_head_use, "mtp/final_hidden");
        require(weights.mtp->token_embedding == weights.text.token_embedding,
                "MTP embedding did not share target");
    }
    if (weights.draft) {
        check_head(weights.draft->output_head, weights.draft->output_head_use,
                   std::string(options.speculative_component()) + "/final_hidden");
        require(weights.draft->token_embedding == weights.text.token_embedding,
                "draft embedding did not share target");
    }
    if (!options.vision) {
        require(plan.resources().preprocessor_config_json.empty() &&
                    plan.resources().video_preprocessor_config_json.empty(),
                "Text-only loading retained Vision resources");
    }
    std::set<std::size_t> parents;
    std::uint64_t end = 0;
    for (const auto& item : plan.materialization().device_objects) {
        require(parents.insert(item.object.index).second,
                "device parent was planned more than once");
        require(item.offset >= end && item.offset % item.alignment == 0,
                "device placement overlaps or is unaligned");
        end = item.offset + item.bytes;
    }
    require(end == plan.materialization().device_capacity_bytes,
            "weight capacity differs from actual placements");
}

// Exercise native grouping after Reader destruction, independently of Program selection.
void check_native_inputs(const qwen::Model& model) {
    const auto linear = [&](qwen::WeightId id) {
        return ops::prepare_linear_weight(model.input(id));
    };
    const auto dense = [&](const qwen::DenseWeights& weights) {
        (void)ops::prepare_linear_swiglu_weight(model.input(weights.gate), model.input(weights.up));
        (void)linear(weights.down);
    };
    const auto block = [&](const qwen::BlockWeights& weights, bool mtp = false) {
        if (const auto* attention = std::get_if<qwen::AttentionWeights>(&weights.mixer)) {
            if (mtp) {
                const std::array bank{model.input(attention->query), model.input(attention->key),
                                      model.input(attention->gate), model.input(attention->value)};
                (void)ops::prepare_linear_weight(bank);
                for (const auto& input : bank) { (void)ops::prepare_linear_weight(input); }
            } else {
                (void)ops::prepare_attn_input_proj_weights(
                    model.input(attention->query), model.input(attention->key),
                    model.input(attention->gate), model.input(attention->value));
            }
            (void)linear(attention->output);
        } else {
            const auto& gdn = std::get<qwen::GdnWeights>(weights.mixer);
            (void)ops::prepare_gdn_input_proj_weights(model.input(gdn.query), model.input(gdn.key),
                                                      model.input(gdn.value), model.input(gdn.z));
            (void)ops::prepare_gdn_gating_proj_weights(model.input(gdn.a_projection),
                                                       model.input(gdn.b_projection));
            (void)linear(gdn.output);
        }
        if (const auto* mlp = std::get_if<qwen::DenseWeights>(&weights.ffn)) {
            dense(*mlp);
        } else {
            const auto& moe = std::get<qwen::MoeWeights>(weights.ffn);
            std::vector<ops::WeightInput> gate_up, down;
            for (const auto& expert : moe.experts) {
                gate_up.push_back(model.input(expert.gate));
                gate_up.push_back(model.input(expert.up));
                down.push_back(model.input(expert.down));
            }
            (void)ops::prepare_sparse_moe_weights(
                model.input(moe.router), model.input(moe.shared_score), gate_up, down,
                model.input(moe.shared.gate), model.input(moe.shared.up),
                model.input(moe.shared.down));
        }
    };
    const auto& weights = model.weights();
    (void)native_weight(model.weight(weights.text.token_embedding).view);
    (void)ops::prepare_linear_weight(model.input(weights.text.output_head_use));
    for (const auto& layer : weights.text.layers) { block(layer); }
    if (weights.vision) {
        const auto& vision = *weights.vision;
        (void)linear(vision.patch_embedding);
        for (const auto& layer : vision.layers) {
            const std::array qkv{model.input(layer.query), model.input(layer.key),
                                 model.input(layer.value)};
            (void)ops::prepare_linear_weight(qkv);
            (void)linear(layer.output);
            (void)linear(layer.fc1);
            (void)linear(layer.fc2);
        }
        (void)linear(vision.merger_fc1);
        (void)linear(vision.merger_fc2);
    }
    if (weights.mtp) {
        (void)linear(weights.mtp->input_projection);
        block(weights.mtp->layer, true);
        (void)ops::prepare_linear_weight(model.input(weights.mtp->output_head_use));
    }
    if (weights.draft) {
        const auto& draft = *weights.draft;
        (void)linear(draft.feature_projection);
        for (const auto& layer : draft.layers) {
            const auto& a = layer.attention;
            (void)ops::prepare_attn_input_proj_weights(model.input(a.query), model.input(a.key),
                                                       model.input(a.value));
            const std::array kv{model.input(a.context_key), model.input(a.context_value)};
            (void)ops::prepare_linear_weight(kv);
            (void)linear(a.output);
            dense(layer.mlp);
            if (layer.attention_conv) { (void)linear(layer.attention_conv->kernel_projection); }
            if (layer.mlp_conv) { (void)linear(layer.mlp_conv->kernel_projection); }
        }
        if (draft.selector) { (void)linear(draft.selector->hidden_projection); }
        (void)ops::prepare_linear_weight(model.input(draft.output_head_use));
    }
    if (model.weight(weights.text.output_head).uses.size() > 1) {
        ninfer::test::artifact_fixture::rejects<std::invalid_argument>(
            [&] { (void)model.input(weights.text.output_head); },
            "ambiguous shared output-head Use was silently selected");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        LoadOptions options;
        bool host_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--vision") {
                options.vision = true;
            } else if (arg == "--host-only") {
                host_only = true;
            } else if (arg == "--scoring") {
                options.purpose = EnginePurpose::CausalScoring;
            } else if (arg == "--speculative") {
                const auto backend = value();
                if (backend == "none") {
                    options.speculative = SpeculativeBackend::None;
                } else if (backend == "mtp") {
                    options.speculative = SpeculativeBackend::Mtp;
                } else if (backend == "dflash") {
                    options.speculative = SpeculativeBackend::DFlash;
                } else if (backend == "dflash2") {
                    options.speculative = SpeculativeBackend::DFlash2;
                } else {
                    throw std::invalid_argument("unknown speculative backend");
                }
            } else if (arg == "--proposal") {
                const auto head = value();
                if (head == "full") {
                    options.proposal_head = ProposalHead::Full;
                } else if (head == "optimized") {
                    options.proposal_head = ProposalHead::Optimized;
                } else {
                    throw std::invalid_argument("unknown proposal head");
                }
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--vision] [--speculative none|mtp|dflash|dflash2]\n"
                             "[--proposal full|optimized] [--host-only] [--scoring]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device;
        std::unique_ptr<qwen::Model> model;
        std::map<std::string, std::vector<Sample>> samples;
        std::uint64_t expected_h2d = 0;
        {
            artifact::Reader reader(path);
            auto plan = qwen::plan_load(reader, options);
            check_selection(plan, options);
            if (host_only) {
                std::cout << path.filename().string() << ": Host binding passed, architecture="
                          << architecture_name(plan.config().text.architecture)
                          << " V=" << plan.resources().public_token_count
                          << " device_bytes=" << plan.materialization().device_capacity_bytes
                          << '\n';
                return 0;
            }
            for (const auto& item : plan.materialization().device_objects) {
                const auto& object   = reader.directory().tensor(item.object);
                const auto& geometry = reader.geometry(item.object);
                expected_h2d += object.bytes;
                const auto size = std::min<std::uint64_t>(64, object.bytes);
                std::set<std::uint64_t> offsets{0, (object.bytes - size) / 2, object.bytes - size};
                if (geometry.high_bytes) { offsets.insert(geometry.high_offset); }
                if (geometry.scale_bytes) { offsets.insert(geometry.scale_offset); }
                for (const auto offset : offsets) {
                    const auto count = std::min<std::uint64_t>(size, object.bytes - offset);
                    samples[object.id].push_back(
                        {offset, reader.read_range(object.offset + offset, count)});
                }
            }
            device = std::make_unique<DeviceContext>();
            model  = qwen::materialize_model(std::move(plan), *device);
        }
        require(model->storage_stats().h2d_bytes == expected_h2d,
                "H2D did not cover all selected parent bytes");
        std::set<const WeightParent*> visited;
        for (const auto& weight : model->weight_data()) {
            require(weight.view.parts.size() == weight.source_objects.size(),
                    "diagnostic parent association lost");
            for (std::size_t i = 0; i < weight.view.parts.size(); ++i) {
                const auto* parent = weight.view.parts[i].parent;
                if (!visited.insert(parent).second) { continue; }
                for (const auto& sample : samples.at(weight.source_objects[i])) {
                    std::vector<std::byte> received(sample.bytes.size());
                    CUDA_CHECK(cudaMemcpy(received.data(), parent->data + sample.offset,
                                          received.size(), cudaMemcpyDeviceToHost));
                    require(received == sample.bytes,
                            "resident parent differs from original v3 bytes");
                }
            }
        }
        require(visited.size() == samples.size(),
                "some resident parents are not reachable from model weights");
        check_native_inputs(*model);
        const auto ids = model->resources().tokenizer->encode("Hello");
        require(model->resources().tokenizer->decode(ids) == "Hello",
                "tokenizer did not survive Reader destruction");
        require(!model->resources().chat_template_jinja.empty(),
                "template bytes were lost after loading");
        std::cout << path.filename().string()
                  << ": GPU loading and native inputs passed, parents=" << visited.size()
                  << " device_bytes=" << model->storage_stats().device_capacity_bytes
                  << " V=" << model->resources().public_token_count << '\n';
        model.reset();
        device->synchronize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
