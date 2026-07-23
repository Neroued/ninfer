#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_rule.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/gqa_attention.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const std::size_t columns = plan.draft_window + 1ULL;
    const std::size_t slots   = columns + 1ULL;
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers = TextConfig::full_attention_layers(),
                     .mtp_layers            = TextConfig::mtp_layers,
                     .capacity              = plan.capacity,
                     .kv_heads              = TextConfig::kv_heads,
                     .attention_head_dim    = TextConfig::head_dim,
                     .kv_dtype              = plan.kv_dtype,
                     .kv_quant_group        = plan.kv_quant_group,
                     .enable_mtp            = plan.draft_window != 0,
                     .gdn =
                         {
                             .layers         = TextConfig::gdn_layers(),
                             .conv_dim       = TextConfig::convolution_dim,
                             .conv_width     = TextConfig::gdn_conv_state_width,
                             .value_heads    = TextConfig::gdn_value_heads,
                             .value_head_dim = TextConfig::gdn_value_head_dim,
                             .key_head_dim   = TextConfig::gdn_key_head_dim,
                             .snapshot_slots = static_cast<std::int32_t>(slots),
                             .conv_dtype     = DType::BF16,
                         },
                 });

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden       = TextConfig::hidden,
                                         .output_rows  = TextConfig::output_rows,
                                         .draft_window = plan.draft_window,
                                         .enable_mtp   = plan.draft_window != 0});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.prefill_chunk)},
        "step prefill hidden");
    qwen3_6::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts        = i32(TextConfig::token_domain, "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(builder, DType::I32, {config_words}, "sampling config");
    out.tail_hidden     = add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "tail hidden");
    out.boundary_hidden =
        add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "boundary hidden");
    out.bytes            = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes = out.decoder.kv_payload_bytes();
    return out;
}

std::size_t workspace_bytes(const SequencePlanImpl& plan) {
    if (plan.prefill_chunk > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("prefill_chunk exceeds int32 workspace dimensions");
    }
    const auto prefill_tokens = static_cast<std::int32_t>(plan.prefill_chunk);
    const auto verify_tokens  = static_cast<std::int32_t>(plan.draft_window + 1);

    const auto matrix      = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        matrix(layout, DType::I32, 1, tokens);
        matrix(layout, DType::I32, 1, tokens);
        matrix(layout, DType::I32, 3, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::I32, 1, tokens);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        auto scope = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        layout.alloc_bytes(ops::gqa_attention_workspace_bytes(TextConfig::query_heads, tokens));
        layout.alloc_bytes(
            ops::linear_add_workspace_bytes(TextConfig::hidden, TextConfig::query_size, tokens));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens, bool verify) {
        auto scope = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::FP32, TextConfig::gdn_value_heads, tokens);
        matrix(layout, DType::FP32, TextConfig::gdn_value_heads, tokens);
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(Variant::gdn_norm_control_projection_workspace_bytes(tokens));
        }
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::key_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::key_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        if (verify) {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(Variant::gdn_input_projection_snapshot_workspace_bytes(tokens));
        } else {
            matrix(layout, DType::BF16, TextConfig::convolution_dim, tokens);
            {
                auto operator_scope = layout.scope();
                layout.alloc_bytes(Variant::gdn_input_projection_workspace_bytes(tokens));
            }
            matrix(layout, DType::BF16, TextConfig::convolution_dim, tokens);
        }
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(ops::gated_delta_rule_workspace_bytes(
                TextConfig::gdn_value_head_dim, TextConfig::gdn_key_heads,
                TextConfig::gdn_value_heads, tokens, /*normalize_qk=*/true));
        }
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(Variant::gdn_output_gate_projection_workspace_bytes(tokens));
        }
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        layout.alloc_bytes(
            ops::linear_add_workspace_bytes(TextConfig::hidden, TextConfig::value_dim, tokens));
    };
    const auto mlp_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        auto scope = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        layout.alloc_bytes(Variant::post_mixer_workspace_bytes(tokens));
    };

    WorkspaceLayoutBuilder prefill;
    common_root(prefill, prefill_tokens);
    attention_stage(prefill, prefill_tokens);
    gdn_stage(prefill, prefill_tokens, false);
    mlp_stage(prefill, prefill_tokens);
    {
        auto final_scope = prefill.scope();
        matrix(prefill, DType::BF16, TextConfig::hidden, 1);
        prefill.alloc_bytes(ops::sampling_workspace_bytes(TextConfig::token_domain, 1));
    }

    WorkspaceLayoutBuilder verify;
    matrix(verify, DType::I32, 1, verify_tokens);
    matrix(verify, DType::BF16, TextConfig::hidden, verify_tokens);
    attention_stage(verify, verify_tokens);
    gdn_stage(verify, verify_tokens, true);
    mlp_stage(verify, verify_tokens);
    {
        auto final_scope = verify.scope();
        matrix(verify, DType::BF16, TextConfig::hidden, verify_tokens);
    }

    const auto variant_scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        auto scope = layout.scope();
        layout.alloc_bytes(bytes);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, 1);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        if (!preembedded) { matrix(layout, DType::BF16, TextConfig::hidden, tokens); }
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::mtp_input_rows, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        auto core_scope = layout.scope();
        mtp_stem(layout, tokens, false);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        variant_scratch(layout, Variant::mtp_attention_workspace_bytes(tokens));
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        matrix(layout, DType::BF16, TextConfig::query_size, tokens);
        variant_scratch(layout,
                        ops::gqa_attention_workspace_bytes(TextConfig::query_heads, tokens));
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        variant_scratch(layout, Variant::mtp_post_mixer_workspace_bytes(tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   bool build_proposal) {
        auto position_scope = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens);
        if (build_proposal) {
            auto proposal_scope = layout.scope();
            proposal_scratch(layout);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        auto scratch_scope = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk_scope = layout.scope();
            mtp_stem(layout, tokens, true);
            matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
            matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
            variant_scratch(layout, Variant::mtp_kv_workspace_bytes(tokens));
            matrix(layout, DType::BF16, TextConfig::kv_size, tokens);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        variant_scratch(layout, Variant::mtp_q_gate_workspace_bytes(1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        variant_scratch(layout, ops::gqa_attention_workspace_bytes(TextConfig::query_heads, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        variant_scratch(layout, Variant::mtp_post_mixer_workspace_bytes(1));
        proposal_scratch(layout);
    };

    WorkspaceLayoutBuilder mtp_prefill;
    WorkspaceLayoutBuilder mtp_full;
    if (plan.features.mtp) {
        common_root(mtp_prefill, prefill_tokens);
        matrix(mtp_prefill, DType::I32, 1, prefill_tokens);
        matrix(mtp_prefill, DType::BF16, TextConfig::hidden, prefill_tokens);
        matrix(mtp_prefill, DType::I32, 1, prefill_tokens);
        mtp_prefill_chunk(mtp_prefill, prefill_tokens);
        for (std::uint32_t i = 1; i < plan.draft_window; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, true);
        }

        mtp_full_call(mtp_full, verify_tokens, false);
        mtp_full_call(mtp_full, 1, true);
        {
            auto proposal_scope = mtp_full.scope();
            proposal_scratch(mtp_full);
        }
    }

    WorkspaceLayoutBuilder decision;
    decision.alloc_bytes(
        ops::sampling_workspace_bytes(TextConfig::token_domain, std::max(1, verify_tokens)));

    std::size_t maximum =
        std::max({prefill.peak_bytes(), verify.peak_bytes(), mtp_prefill.peak_bytes(),
                  mtp_full.peak_bytes(), decision.peak_bytes()});
    if (plan.features.vision) {
        maximum = std::max(maximum, schedule::VisionContext::maximum_workspace_bytes());
    }
    return maximum;
}

} // namespace

std::size_t target_speculative_workspace_bytes(std::uint32_t prefill_chunk,
                                               std::uint32_t draft_window) {
    if (prefill_chunk == 0 || draft_window == 0 ||
        prefill_chunk > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("target speculative workspace dimensions are invalid");
    }
    SequencePlanImpl plan;
    plan.prefill_chunk = prefill_chunk;
    plan.draft_window  = draft_window;
    plan.proposal_head = ProposalHead::Full;
    return workspace_bytes(plan);
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
        throw std::invalid_argument("MTP draft window must be in [0,5]");
    }
    if (options.speculative.draft_tokens == 0 &&
        options.speculative.proposal_head == ProposalHead::Optimized) {
        throw std::invalid_argument("optimized proposal head requires MTP");
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> plan_sequence_impl(DeviceContext& device,
                                                     const EngineOptions& options) {
    validate_target_options(device, options);

    auto impl             = std::make_unique<SequencePlanImpl>();
    impl->capacity        = options.max_context;
    impl->prefill_chunk   = options.prefill_chunk;
    impl->draft_window    = options.speculative.draft_tokens;
    impl->proposal_head   = options.speculative.proposal_head;
    impl->features        = qwen3_6::startup_features(options);
    impl->use_cuda_graph  = options.use_cuda_graph;
    impl->device          = options.device;
    impl->kv_dtype        = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8;
    impl->kv_quant_group  = impl->kv_dtype == DType::I8 ? kKvQuantGroup : 0;
    impl->persistent      = persistent_layout(*impl);
    impl->workspace_bytes = workspace_bytes(*impl);
    if (impl->use_cuda_graph) {
        const std::size_t ordinary_variants = ordinary_graph_ranges(impl->capacity).size();
        const std::size_t ordinary_graphs =
            ordinary_variants * (impl->draft_window == 0 ? 1ULL : 2ULL);
        // Cold full-model capture also materializes lazy CUDA module state. The 35B
        // K=3/C=4096 public benchmark measured 123,277,312 bytes across its 12 ordinary/aligned
        // and short-window executables. Keep one conservative family allowance of 12 MiB per
        // executable. Long MTP executables also trigger substantially larger driver allocations.
        impl->graph_allowance_bytes = ordinary_graphs * 12ULL * kMiB;
        for (const GraphFrontierRange range :
             mtp_graph_ranges(impl->capacity, impl->draft_window)) {
            const std::uint64_t final_visible =
                static_cast<std::uint64_t>(range.max) + 2ULL * impl->draft_window;
            impl->graph_allowance_bytes += (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace_bytes, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
