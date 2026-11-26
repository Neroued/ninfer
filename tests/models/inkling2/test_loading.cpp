#include "models/inkling2/load.h"

#include "artifact/reader.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models;
namespace ink = models::inkling2;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Fn>
void require_throws(Fn&& fn, const std::string& message) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected exception: " + message);
}

void require_contiguous(std::span<const BoundWeight> weights, std::initializer_list<WeightId> ids,
                        const std::string& label) {
    const WeightParent* parent = nullptr;
    std::uint64_t end          = 0;
    for (const auto id : ids) {
        const auto region = contiguous_weight_region(weights[id.index].view);
        require(!parent || (parent == region.parent && region.begin == end),
                label + ": merged projection has a gap or different parent");
        parent = region.parent;
        end    = region.end;
    }
}

void require_same_storage(std::span<const BoundWeight> weights, WeightId a, WeightId b,
                          const std::string& label) {
    const auto ra = contiguous_weight_region(weights[a.index].view);
    const auto rb = contiguous_weight_region(weights[b.index].view);
    require(ra.parent == rb.parent && ra.begin == rb.begin && ra.end == rb.end,
            label + ": expected identical shared storage");
}

void verify_host(const std::filesystem::path& artifact_path) {
    std::shared_ptr<const ink::ModelDefinition> full;
    {
        artifact::Reader reader(artifact_path);

        // Reject unsupported speculative backends and optimized proposal head.
        require_throws(
            [&] {
                (void)ink::read_definition(reader, {.speculative = SpeculativeBackend::Mtp});
            },
            "MTP backend rejected");
        require_throws(
            [&] {
                (void)ink::read_definition(reader, {.speculative = SpeculativeBackend::DFlash});
            },
            "DFlash backend rejected");
        require_throws(
            [&] {
                (void)ink::read_definition(reader, {.speculative = SpeculativeBackend::DFlash2});
            },
            "DFlash2 backend rejected");
        require_throws(
            [&] {
                (void)ink::read_definition(reader, {.speculative   = SpeculativeBackend::DSpark,
                                                    .proposal_head = ProposalHead::Optimized});
            },
            "Optimized proposal head rejected for DSpark");

        // 1. Text only
        const auto text_only = ink::read_definition(reader, {});
        require(!text_only->config().vision && !text_only->config().dspark &&
                    !text_only->weights().vision && !text_only->weights().dspark,
                "text-only definition should not bind vision or dspark");
        for (const auto& p : text_only->parameters()) {
            require(!p.name.starts_with("vision/") && !p.name.starts_with("dspark/"),
                    "text-only bound optional parameter: " + p.name);
        }

        // 2. Text + Vision
        const auto with_vision = ink::read_definition(reader, {.vision = true});
        require(with_vision->config().vision && with_vision->weights().vision &&
                    !with_vision->config().dspark && !with_vision->weights().dspark,
                "text+vision selection mismatch");

        // 3. Text + DSpark
        const auto with_dspark =
            ink::read_definition(reader, {.speculative = SpeculativeBackend::DSpark});
        require(!with_dspark->config().vision && !with_dspark->weights().vision &&
                    with_dspark->config().dspark && with_dspark->weights().dspark,
                "text+dspark selection mismatch");

        // 4. Text + Vision + DSpark
        full = ink::read_definition(
            reader, {.vision = true, .speculative = SpeculativeBackend::DSpark});

        const auto& cfg = full->config();
        const auto& tc  = cfg.text;
        require(tc.hidden == 2304 && tc.num_layers == 24 && tc.vocabulary == 200064 &&
                    tc.unpadded_vocabulary == 200058 && tc.attn_res_block_size == 6 &&
                    tc.sconv_kernel_size == 4 && tc.d_rel == 16 && tc.rel_extent == 1024 &&
                    tc.sliding_window == 1024,
                "Inkling-2-Mini text geometry mismatch");
        require(tc.full_attention.heads == 24 && tc.full_attention.kv_heads == 8 &&
                    tc.full_attention.head_dim == 128 && tc.swa_attention.heads == 24 &&
                    tc.swa_attention.kv_heads == 8 && tc.swa_attention.head_dim == 128,
                "Inkling-2-Mini attention geometry mismatch");
        require(tc.moe.experts == 128 && tc.moe.top_k == 8 && tc.moe.intermediate == 1152 &&
                    tc.moe.gate_softcap == 4.0F && tc.moe.up_softcap == 25.0F &&
                    tc.moe.route_scale == 8.0F,
                "Inkling-2-Mini MoE geometry mismatch");
        require(tc.loop && tc.loop->core_start == 6 && tc.loop->core_end == 18 &&
                    tc.loop->num_passes == 3 && tc.loop->gate_rank == 64,
                "Inkling-2-Mini loop_config mismatch");
        require(tc.logical_layers.size() == 48 && tc.swa_layers == 40 && tc.full_layers == 8,
                "Inkling-2-Mini logical layer count mismatch");

        // Verify logical layer schedule and DSpark aux taps [5, 23, 41, 48]
        for (std::uint32_t v = 0; v < 48; ++v) {
            const auto& ll = tc.logical_layers[v];
            require(ll.logical_index == v, "logical_index mismatch");
            require(ll.starts_new_block == (v > 0 && (v % 6) == 0), "starts_new_block mismatch");
            if (v < 6) {
                require(ll.physical_layer == v && ll.pass_index == 0 && !ll.in_core,
                        "prelude schedule mismatch");
            } else if (v < 42) {
                const auto core_off = v - 6;
                require(ll.physical_layer == 6 + (core_off % 12) &&
                            ll.pass_index == (core_off / 12) && ll.in_core,
                        "core schedule mismatch");
            } else {
                require(ll.physical_layer == 18 + (v - 42) && ll.pass_index == 0 && !ll.in_core,
                        "coda schedule mismatch");
            }
        }
        require(cfg.dspark && cfg.dspark->num_layers == 3 && cfg.dspark->swa_layers == 2 &&
                    cfg.dspark->full_layers == 1 && cfg.dspark->block_size == 8 &&
                    cfg.dspark->markov_rank == 256,
                "Inkling-2-Mini DSpark config mismatch");
        require(tc.logical_layers[5].capture_aux_slot == std::optional<std::uint32_t>{0} &&
                    tc.logical_layers[23].capture_aux_slot == std::optional<std::uint32_t>{1} &&
                    tc.logical_layers[41].capture_aux_slot == std::optional<std::uint32_t>{2} &&
                    cfg.dspark->final_aux_slot == std::optional<std::uint32_t>{3},
                "DSpark aux capture slots mismatch");

        require(cfg.vision && cfg.vision->depth == 23 && cfg.vision->hidden == 1024 &&
                    cfg.vision->heads == 16 && cfg.vision->head_dim == 64 &&
                    cfg.vision->intermediate == 4096 && cfg.vision->native_patch_size == 20 &&
                    cfg.vision->merge_size == 2 && cfg.vision->posemb_grid_size == 32 &&
                    cfg.vision->projection_hidden_dim == 4096 && !cfg.vision->projector_bias,
                "Inkling-2-Mini Vision config mismatch");

        // Verify weight bindings, packing groups, per-pass uses, and expert banks
        const auto& w    = full->weights();
        const auto data  = full->weight_data();
        const auto banks = full->expert_banks();
        require(banks.size() == 24 && w.text.layers.size() == 24, "physical layer count mismatch");

        for (std::uint32_t l = 0; l < 24; ++l) {
            const auto& phys = tc.physical_layers[l];
            const auto& lw   = w.text.layers[l];
            require(lw.attention.attn_res_query.has_value() == (l > 0),
                    "attn_res_query presence mismatch");
            require(lw.loop.has_value() == phys.in_core, "loop weights presence mismatch");
            require(lw.attention.k_conv_proj.has_value() == !phys.is_swa &&
                        lw.attention.v_conv_proj.has_value() == !phys.is_swa,
                    "dynamic conv proj presence mismatch");
            if (phys.is_swa) {
                require_contiguous(data,
                                   {lw.attention.key, lw.attention.value, lw.attention.query,
                                    lw.attention.output_gate, lw.attention.relative},
                                   "SWA attention layer " + std::to_string(l));
            } else {
                require_contiguous(
                    data, {lw.attention.query, lw.attention.output_gate, lw.attention.relative},
                    "Full attention layer " + std::to_string(l));
            }

            const auto expected_uses = phys.num_passes;
            require(full->uses(lw.attention.query).size() == expected_uses &&
                        full->uses(lw.attention.output).size() == expected_uses &&
                        full->uses(lw.moe.router).size() == expected_uses &&
                        full->uses(lw.moe.gate).size() == expected_uses &&
                        full->uses(lw.moe.up).size() == expected_uses &&
                        full->uses(lw.moe.down).size() == expected_uses,
                    "per-pass use count mismatch at physical layer " + std::to_string(l));
            for (std::uint32_t pass = 0; pass < expected_uses; ++pass) {
                const auto gate_in = full->input(WeightUseId{lw.moe.gate, pass});
                const auto down_in = full->input(WeightUseId{lw.moe.down, pass});
                if (l < 20) {
                    const bool valid_a16 = gate_in.policy == ops::LinearPolicy::A16Only &&
                                           down_in.policy == ops::LinearPolicy::A16Only;
                    const bool valid_a4 =
                        gate_in.policy == ops::LinearPolicy::AllowA4 &&
                        down_in.policy == ops::LinearPolicy::AllowA4 &&
                        gate_in.calibration.operation == ops::ActivationScaleOperation::Multiply &&
                        down_in.calibration.operation == ops::ActivationScaleOperation::Multiply &&
                        !gate_in.calibration.values().empty() &&
                        !down_in.calibration.values().empty();
                    require(valid_a16 || valid_a4,
                            "MoE linear policy mismatch at physical layer " + std::to_string(l));
                } else {
                    require(gate_in.policy == ops::LinearPolicy::AllowA8 &&
                                down_in.policy == ops::LinearPolicy::AllowA8,
                            "MoE linear policy mismatch at physical layer " + std::to_string(l));
                }
            }

            const auto gu_geom   = reader.geometry(banks[l].gate_up.object);
            const auto down_geom = reader.geometry(banks[l].down.object);
            const auto expected_qtype = l < 20 ? QType::NVFP4_MUL : QType::MXFP8;
            require(gu_geom.format == expected_qtype && down_geom.format == expected_qtype,
                    "expert bank format mismatch at physical layer " + std::to_string(l));
        }

        // Vision packing groups
        for (std::uint32_t l = 0; l < cfg.vision->depth; ++l) {
            const auto& vl = w.vision->layers[l];
            require_contiguous(data, {vl.query, vl.key, vl.value},
                               "Vision QKV layer " + std::to_string(l));
            require_contiguous(data, {vl.query_bias, vl.key_bias, vl.value_bias},
                               "Vision QKV bias layer " + std::to_string(l));
        }

        // DSpark packing groups & shared bindings
        for (std::uint32_t l = 0; l < cfg.dspark->num_layers; ++l) {
            const auto& dl = w.dspark->layers[l];
            require_contiguous(data, {dl.key, dl.value, dl.query},
                               "DSpark QKV layer " + std::to_string(l));
            require_contiguous(data, {dl.context_key, dl.context_value},
                               "DSpark context KV layer " + std::to_string(l));
            require_contiguous(data, {dl.mlp.gate, dl.mlp.up},
                               "DSpark MLP gate/up layer " + std::to_string(l));
        }

        // Planning & storage deduplication
        const auto rank = ink::plan_single_rank(*full);
        auto load       = ink::resolve_rank_load(reader, full, rank, Device::cuda(0));
        const auto ldata = load.weight_data();
        require_same_storage(ldata, w.text.embedding, w.dspark->embedding,
                             "planned dspark/token_embedding");
        require_same_storage(ldata, w.text.embed_norm, w.dspark->embed_norm,
                             "planned dspark/embed_norm");
        require_same_storage(ldata, w.text.head, w.dspark->output_head,
                             "planned dspark/output_head");
        for (const auto& weight : ldata)
            for (const auto& part : weight.view.parts)
                require(!part.parent->data, "resolve_rank_load must not allocate payload memory");

        // Verify plan_parallel accepts 1-rank and rejects multi-rank
        DeploymentOptions single_dep;
        single_dep.control_host = "127.0.0.1";
        single_dep.parallel     = {.tp = 1, .ep = 1, .dp = 1, .pp = 1, .adp = 1};
        single_dep.nodes        = {{0, {}, "/worker", artifact_path, {}}};
        single_dep.ranks        = {{0, 0, 0}};
        const auto single_plan  = ink::plan_parallel(*full, single_dep);
        require(single_plan.ranks.size() == 1 && single_plan.topology.ranks.size() == 1,
                "single-rank parallel plan mismatch");

        DeploymentOptions two_dep = single_dep;
        two_dep.parallel.tp       = 2;
        two_dep.ranks.push_back({1, 0, 1});
        require_throws([&] { (void)ink::plan_parallel(*full, two_dep); },
                       "multi-rank parallel plan should be rejected");
    }

    // Verify resources survive after Reader destruction
    require(!full->resources().text.chat_template_jinja.empty() && full->resources().processor &&
                full->resources().text.public_token_count == 200058,
            "ModelDefinition lost resources after Reader destruction");
    std::cout << "PASS Inkling2 host loading & planning\n";
}

int verify_gpu(const std::filesystem::path& artifact_path) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;

    DeviceContext device;
    std::unique_ptr<ink::ModelShard> shard;
    {
        artifact::Reader reader(artifact_path);
        auto definition = ink::read_definition(
            reader, {.vision = true, .speculative = SpeculativeBackend::DSpark});
        auto plan = ink::plan_single_rank(*definition);
        auto load = ink::resolve_rank_load(reader, definition, std::move(plan), device.device());
        shard     = ink::materialize_model(std::move(load), device);
    }

    const auto& w = shard->definition().weights();
    require(shard->storage_stats().device_capacity_bytes > 0, "expected device weight allocation");
    require(shard->expert_data().size() == 24, "expected 24 materialized expert banks");
    for (std::uint32_t l = 0; l < 24; ++l) {
        const auto& bank = shard->experts({l}).view();
        require(bank.count == 128 && bank.gate_up.codes != nullptr && bank.down.codes != nullptr,
                "materialized expert bank invalid at layer " + std::to_string(l));
    }
    require_same_storage(shard->weight_data(), w.text.embedding, w.dspark->embedding,
                         "resident dspark/token_embedding");
    require_same_storage(shard->weight_data(), w.text.embed_norm, w.dspark->embed_norm,
                         "resident dspark/embed_norm");
    require_same_storage(shard->weight_data(), w.text.head, w.dspark->output_head,
                         "resident dspark/output_head");
    for (std::uint32_t pass = 0; pass < 3; ++pass) {
        const auto inp = shard->input(WeightUseId{w.text.layers[6].attention.query, pass});
        require(inp.weight.parts.front().parent->data != nullptr,
                "resident core layer query pointer null");
    }
    std::cout << "PASS Inkling2 GPU materialization ("
              << shard->storage_stats().device_capacity_bytes << " bytes resident)\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool gpu_mode = false;
        std::filesystem::path artifact_path;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--gpu")
                gpu_mode = true;
            else if (arg == "--host")
                gpu_mode = false;
            else
                artifact_path = arg;
        }
        if (artifact_path.empty()) {
            if (const char* env = std::getenv("NINFER_INKLING2_ARTIFACT"); env && *env)
                artifact_path = env;
            else
                artifact_path =
                    std::filesystem::path(NINFER_SOURCE_DIR) / "out/inkling2_mini.ninfer";
        }
        if (!std::filesystem::exists(artifact_path)) {
            std::cout << "SKIP Inkling2 loading test: artifact not found at " << artifact_path
                      << '\n';
            return 77;
        }
        if (gpu_mode) return verify_gpu(artifact_path);
        verify_host(artifact_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
