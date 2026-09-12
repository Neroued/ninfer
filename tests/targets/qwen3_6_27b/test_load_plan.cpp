#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

ninfer::targets::qwen3_6::StartupFeatures
features(ninfer::SpeculativeBackend backend,
         ninfer::ProposalHead proposal_head = ninfer::ProposalHead::Full) {
    return {
        .vision        = false,
        .speculative   = backend,
        .proposal_head = proposal_head,
    };
}

bool is_device_object(const ninfer::artifact::MaterializationPlan& plan,
                      ninfer::artifact::ObjectHandle handle) {
    return std::ranges::any_of(plan.device_objects, [handle](const auto& object) {
        return object.object.index == handle.index;
    });
}

std::size_t dflash2_device_objects(const ninfer::artifact::Reader& reader,
                                   const ninfer::artifact::MaterializationPlan& plan) {
    return static_cast<std::size_t>(
        std::ranges::count_if(plan.device_objects, [&](const auto& item) {
            return ninfer::artifact::object_name(reader.objects().at(item.object.index))
                .starts_with("dflash2/");
        }));
}

bool valid_divisors(const WeightPlan& weight) {
    if (weight.format != NumericFormat::NVFP4) { return false; }
    const float weight_divisor = std::bit_cast<float>(weight.weight_scale_divisor_bits);
    const float input_divisor  = std::bit_cast<float>(weight.input_scale_divisor_bits);
    return std::isfinite(weight_divisor) && weight_divisor > 0.0F && std::isfinite(input_divisor) &&
           input_divisor > 0.0F;
}

int verify_groupwise(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen36GroupwiseInt) {
        std::cerr << "groupwise identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen36GroupwiseInt, all_features());
    if (plan.materialization.object_count != 1124 ||
        plan.materialization.device_objects.size() != 1118 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "groupwise materialization plan is incomplete\n";
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::Q6G64_F16S ||
        plan.bindings.output_head.format != NumericFormat::Q6G64_F16S) {
        std::cerr << "groupwise vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (layer.is_full_attention) {
            if (!std::holds_alternative<SplitAttentionProjectionPlan>(layer.attention.projection)) {
                std::cerr << "groupwise attention parent boundary changed\n";
                return 1;
            }
        } else if (!std::holds_alternative<SplitGdnInputProjectionPlan>(
                       layer.gdn.input_projection)) {
            std::cerr << "groupwise GDN parent boundary changed\n";
            return 1;
        }
        if (layer.mlp.gate_up.format != NumericFormat::Q4G64_F16S ||
            layer.mlp.down.format != NumericFormat::Q5G64_F16S) {
            std::cerr << "groupwise MLP storage profile changed\n";
            return 1;
        }
    }
    return 0;
}

int verify_nvfp4(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen36Nvfp4) {
        std::cerr << "NVFP4 identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen36Nvfp4, all_features());
    if (plan.materialization.object_count != 1307 ||
        plan.materialization.device_objects.size() != 1054 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.object_count - plan.materialization.device_objects.size() -
                plan.materialization.host_objects.size() !=
            247 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "NVFP4 materialization plan is incomplete: objects="
                  << plan.materialization.object_count
                  << " device=" << plan.materialization.device_objects.size()
                  << " host=" << plan.materialization.host_objects.size() << '\n';
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::W8G32_F16S ||
        plan.bindings.output_head.format != NumericFormat::W8G32_F16S) {
        std::cerr << "NVFP4 vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }

    std::size_t nvfp4_weights          = 0;
    std::size_t bf16_attention_inputs  = 0;
    std::size_t bf16_attention_outputs = 0;
    std::size_t bf16_gdn_outputs       = 0;
    const auto count_weight            = [&](const WeightPlan& weight) {
        if (weight.format == NumericFormat::NVFP4) {
            ++nvfp4_weights;
            return valid_divisors(weight);
        }
        return true;
    };
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (!count_weight(layer.mlp.gate_up) || !count_weight(layer.mlp.down)) {
            std::cerr << "NVFP4 MLP divisor is invalid\n";
            return 1;
        }
        if (layer.is_full_attention) {
            const auto* fused =
                std::get_if<FusedAttentionProjectionPlan>(&layer.attention.projection);
            if (fused == nullptr || !count_weight(fused->query_key_gate_value) ||
                !count_weight(layer.attention.output)) {
                std::cerr << "NVFP4 attention binding is invalid\n";
                return 1;
            }
            bf16_attention_inputs +=
                fused->query_key_gate_value.format == NumericFormat::BF16 ? 1 : 0;
            bf16_attention_outputs += layer.attention.output.format == NumericFormat::BF16 ? 1 : 0;
        } else {
            const auto* fused =
                std::get_if<FusedGdnInputProjectionPlan>(&layer.gdn.input_projection);
            if (fused == nullptr || !count_weight(fused->query_key_value_z) ||
                !count_weight(layer.gdn.output)) {
                std::cerr << "NVFP4 GDN binding is invalid\n";
                return 1;
            }
            bf16_gdn_outputs += layer.gdn.output.format == NumericFormat::BF16 ? 1 : 0;
        }
    }
    if (nvfp4_weights != 247 || bf16_attention_inputs != 6 || bf16_attention_outputs != 2 ||
        bf16_gdn_outputs != 1) {
        std::cerr << "NVFP4 Text inventory has the wrong storage profile: nvfp4=" << nvfp4_weights
                  << " bf16_attention_input=" << bf16_attention_inputs
                  << " bf16_attention_output=" << bf16_attention_outputs
                  << " bf16_gdn_output=" << bf16_gdn_outputs << '\n';
        return 1;
    }
    return 0;
}

int verify_legacy_dflash2_compatibility(const std::filesystem::path& path, WeightsProfile profile) {
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan =
            bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::None));
        if (plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 0) {
            std::cerr << "legacy artifact unexpectedly bound DFlash2: " << path << '\n';
            return 1;
        }
    }
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan =
            bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::Mtp));
        if (plan.bindings.dflash2 ||
            !is_device_object(plan.materialization, plan.bindings.mtp.input_projection)) {
            std::cerr << "legacy artifact did not preserve MTP-only binding: " << path << '\n';
            return 1;
        }
    }
    try {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        (void)bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::DFlash2));
    } catch (const ninfer::artifact::ArtifactError& error) {
        if (std::string(error.what()).find("no DFlash2 weight bundle") != std::string::npos) {
            return 0;
        }
    }
    std::cerr << "legacy artifact did not reject selected DFlash2: " << path << '\n';
    return 1;
}

int verify_dflash2_bundle(const std::filesystem::path& path, WeightsProfile profile) {
    for (const ninfer::SpeculativeBackend backend :
         {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp}) {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan = bind_artifact(binder, profile, features(backend));
        if (!plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 0) {
            std::cerr << "inactive DFlash2 bundle was not validate-only: " << path << '\n';
            return 1;
        }
        const bool mtp_is_device =
            is_device_object(plan.materialization, plan.bindings.mtp.input_projection);
        if (mtp_is_device != (backend == ninfer::SpeculativeBackend::Mtp)) {
            std::cerr << "MTP placement does not match backend selection: " << path << '\n';
            return 1;
        }
    }

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(
        binder, profile,
        features(ninfer::SpeculativeBackend::DFlash2, ninfer::ProposalHead::Optimized));
    if (!plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 66 ||
        is_device_object(plan.materialization, plan.bindings.mtp.input_projection) ||
        !is_device_object(plan.materialization, plan.bindings.draft_head) ||
        !is_device_object(plan.materialization, plan.bindings.draft_head_token_ids)) {
        std::cerr << "selected DFlash2 bundle has the wrong placement: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_rejection() {
    try {
        (void)Package::resolve_weights({"qwen3.6-27b", "unknown"});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("qwen3.6-27b/unknown") != std::string::npos) { return 0; }
    }
    std::cerr << "unknown weights identity was not rejected with the full identity\n";
    return 1;
}

int verify_profile_mismatch_rejection() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context                      = 128;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk                    = 128;
    options.use_cuda_graph                   = false;
    options.context_cache.device_state_slots = options.max_concurrency;
    auto planner =
        Package::make_sequence_planner(device, options, WeightsProfile::Qwen36GroupwiseInt);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    auto sequence             = std::move(planner).finalize(pages);
    RuntimeModelView empty_model;
    try {
        (void)ninfer::targets::qwen3_6::create_program<Variant>(
            empty_model, WeightsProfile::Qwen36Nvfp4, std::move(sequence), device,
            ninfer::StartupObserver{});
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("weights profile") != std::string::npos) { return 0; }
    }
    std::cerr << "mismatched load/sequence weights profiles were not rejected\n";
    return 1;
}

int verify_vision_workspace_planning() {
    static_assert(ninfer::targets::qwen3_6::kMaximumPromptVisionTokens == 32768);
    static_assert(ninfer::targets::qwen3_6::kMaximumVisionItemTokens == 16384);
    constexpr std::size_t kExpectedMaximumItemWorkspace = 866'648'064;

    ninfer::DeviceContext device(0);
    const auto workspace_capacity = [&](std::uint32_t max_context) {
        ninfer::EngineOptions options;
        options.max_context              = max_context;
        options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
        options.prefill_chunk            = 1024;
        options.kv_cache                 = ninfer::KvCacheStorage::Fp8E4M3Row256;
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
        options.enable_vision                    = true;
        options.use_cuda_graph                   = false;
        options.context_cache.device_state_slots = 1;
        auto planner = Package::make_sequence_planner(device, options, WeightsProfile::Qwen36Nvfp4);
        const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
        return std::move(planner).finalize(pages).workspace_capacity_bytes();
    };

    const std::size_t at_item_limit    = workspace_capacity(16384);
    const std::size_t above_item_limit = workspace_capacity(131072);
    if (at_item_limit != kExpectedMaximumItemWorkspace ||
        above_item_limit != kExpectedMaximumItemWorkspace) {
        std::cerr << "Vision workspace does not clamp Device execution at the 16K item bound: "
                  << "at_limit=" << at_item_limit << " above_limit=" << above_item_limit
                  << " expected=" << kExpectedMaximumItemWorkspace << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path nvfp4 =
        artifact_path("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS", "qwen3_6_27b_nvfp4.ninfer");
    const std::filesystem::path qwen38_groupwise =
        artifact_path("NINFER_QWEN3_8_27B_OLD_WEIGHTS", "qwen3_8_27b_old.ninfer");
    const std::filesystem::path qwen38_nvfp4 =
        artifact_path("NINFER_QWEN3_8_27B_NVFP4_OLD_WEIGHTS", "qwen3_8_27b_nvfp4_old.ninfer");
    const std::filesystem::path qwen38_groupwise_dflash2 =
        artifact_path("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS", "qwen3_8_27b.ninfer");
    const std::filesystem::path qwen38_nvfp4_dflash2 = artifact_path(
        "NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS", "qwen3_8_27b_nvfp4.ninfer");
    if (!std::filesystem::is_regular_file(groupwise) || !std::filesystem::is_regular_file(nvfp4)) {
        std::cerr << "skip: both real 27B artifacts are required: groupwise=" << groupwise
                  << " nvfp4=" << nvfp4 << '\n';
        return 77;
    }
    if (const int result = verify_vision_workspace_planning(); result != 0) { return result; }
    if (const int result = verify_rejection(); result != 0) { return result; }
    if (const int result = verify_profile_mismatch_rejection(); result != 0) { return result; }
    if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
    if (const int result = verify_nvfp4(nvfp4); result != 0) { return result; }
    if (const int result =
            verify_legacy_dflash2_compatibility(groupwise, WeightsProfile::Qwen36GroupwiseInt);
        result != 0) {
        return result;
    }
    const std::array dflash2_artifacts{qwen38_groupwise, qwen38_nvfp4, qwen38_groupwise_dflash2,
                                       qwen38_nvfp4_dflash2};
    if (!std::ranges::all_of(dflash2_artifacts, [](const auto& path) {
            return std::filesystem::is_regular_file(path);
        })) {
        std::cerr << "skip DFlash2 binding matrix: old and new Qwen3.8 artifacts are required\n";
        return 0;
    }
    if (const int result = verify_legacy_dflash2_compatibility(qwen38_groupwise,
                                                               WeightsProfile::Qwen38GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result =
            verify_legacy_dflash2_compatibility(qwen38_nvfp4, WeightsProfile::Qwen38Nvfp4);
        result != 0) {
        return result;
    }
    if (const int result =
            verify_dflash2_bundle(qwen38_groupwise_dflash2, WeightsProfile::Qwen38GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result = verify_dflash2_bundle(qwen38_nvfp4_dflash2, WeightsProfile::Qwen38Nvfp4);
        result != 0) {
        return result;
    }
    return 0;
}
