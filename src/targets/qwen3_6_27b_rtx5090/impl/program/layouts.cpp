#include "targets/qwen3_6_27b_rtx5090/impl/program/layouts.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_rule.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "ninfer/ops/gqa_attention.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "targets/qwen3_6_27b_rtx5090/impl/program/graph_policy.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(label);
    }
    return (value + mask) & ~mask;
}

std::size_t alloc_after(std::size_t cursor, std::size_t bytes, const char* label) {
    return checked_add(align_up(cursor, kArenaAlign, label), bytes, label);
}

std::size_t sequence_bytes(std::size_t cursor, std::initializer_list<std::size_t> sizes,
                           const char* label) {
    for (const std::size_t bytes : sizes) { cursor = alloc_after(cursor, bytes, label); }
    return cursor;
}

std::size_t tensor_bytes(std::size_t elements, DType dtype, const char* label) {
    return checked_mul(elements, dtype_size(dtype), label);
}

std::size_t matrix_bytes(std::size_t rows, std::size_t tokens, DType dtype, const char* label) {
    return tensor_bytes(checked_mul(rows, tokens, label), dtype, label);
}

std::size_t gdn_stage_bytes(std::size_t tokens) {
    if (tokens == 0) { return 0; }
    constexpr std::size_t kChunk = 64;
    constexpr std::size_t kS     = 128;
    constexpr std::size_t kHv    = 48;
    const std::size_t chunks     = (tokens + kChunk - 1) / kChunk;
    return sequence_bytes(0,
                          {
                              matrix_bytes(kHv, tokens, DType::FP32, "GDN workspace"),
                              matrix_bytes(kHv * kS, tokens, DType::BF16, "GDN workspace"),
                              matrix_bytes(kHv * kS, tokens, DType::BF16, "GDN workspace"),
                              matrix_bytes(kHv * kS, tokens, DType::BF16, "GDN workspace"),
                              tensor_bytes(chunks * kHv * kS * kS, DType::BF16, "GDN workspace"),
                          },
                          "GDN workspace");
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlan::Impl& plan) {
    const std::size_t columns = plan.mtp_k + 1ULL;
    const std::size_t drafts  = std::max<std::size_t>(1, plan.mtp_k);
    const std::size_t slots   = columns + 1ULL;
    LayoutBuilder builder;
    PersistentLayout out;
    out.text_kv = plan_kv_cache(builder, TextConfig::full_attention_layers(), plan.capacity,
                                TextConfig::kv_heads, TextConfig::head_dim, plan.kv_dtype,
                                plan.kv_quant_group);
    if (plan.mtp_k != 0) {
        out.mtp_kv =
            plan_kv_cache(builder, TextConfig::mtp_layers, plan.capacity, TextConfig::kv_heads,
                          TextConfig::head_dim, plan.kv_dtype, plan.kv_quant_group);
    }
    out.gdn = plan_gdn_state(builder, TextConfig::gdn_layers(), TextConfig::convolution_dim,
                             TextConfig::gdn_conv_state_width, TextConfig::gdn_value_heads,
                             TextConfig::gdn_value_head_dim, TextConfig::gdn_key_head_dim,
                             static_cast<std::int32_t>(slots));

    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.io.token      = i32(1, "step token");
    out.io.pos        = i32(1, "step position");
    out.io.rope_pos   = i32(1, "step rope position");
    out.io.rope_delta = i32(1, "step rope delta");
    out.io.logits =
        add_tensor(builder, DType::BF16,
                   {TextConfig::output_rows, static_cast<std::int32_t>(columns)}, "step logits");
    out.io.verify_hidden =
        add_tensor(builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(columns)},
                   "step verify hidden");
    out.io.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.prefill_chunk)},
        "step prefill hidden");
    out.io.target_tokens    = i32(columns, "step target tokens");
    out.io.drafts           = i32(drafts, "step drafts");
    out.io.sampled_out      = i32(columns, "step sampled output");
    out.io.num_sampled      = i32(1, "step sampled count");
    out.io.verify_ids       = i32(columns, "step verify ids");
    out.io.shifted_ids      = i32(columns, "step shifted ids");
    out.io.positions        = i32(columns, "step positions");
    out.io.window_base      = i32(1, "step window base");
    out.io.accepted         = i32(1, "step accepted drafts");
    out.io.gdn_initial_slot = i32(1, "step GDN initial slot");
    out.io.ar_pos           = i32(1, "step MTP autoregressive position");
    out.io.mtp_ar_hidden =
        add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "step MTP autoregressive hidden");
    out.io.stats            = add_tensor(builder, DType::I64, {kStepStatsCounters}, "step stats");
    out.token_counts        = i32(TextConfig::token_domain, "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(builder, DType::I32, {config_words}, "sampling config");
    out.tail_hidden     = add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "tail hidden");
    out.boundary_hidden =
        add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "boundary hidden");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.text_kv.payload_bytes() + (out.mtp_kv ? out.mtp_kv->payload_bytes() : 0);
    return out;
}

std::size_t workspace_bytes(const SequencePlan::Impl& plan) {
    if (plan.prefill_chunk > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("prefill_chunk exceeds int32 workspace dimensions");
    }
    const auto prefill_tokens = static_cast<std::int32_t>(plan.prefill_chunk);
    const auto verify_tokens  = static_cast<std::int32_t>(plan.mtp_k + 1);

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
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        auto scope = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::convolution_dim, tokens);
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(ops::gdn_input_proj_workspace_bytes(2 * TextConfig::key_dim,
                                                                   TextConfig::value_dim, tokens));
        }
        matrix(layout, DType::BF16, TextConfig::convolution_dim, tokens);
        matrix(layout, DType::FP32, TextConfig::gdn_value_heads, tokens);
        matrix(layout, DType::FP32, TextConfig::gdn_value_heads, tokens);
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(ops::gdn_gating_proj_workspace_bytes(tokens));
        }
        matrix(layout, DType::BF16, TextConfig::key_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::key_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::key_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::key_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(ops::gated_delta_rule_workspace_bytes(
                TextConfig::gdn_value_head_dim, TextConfig::gdn_key_heads,
                TextConfig::gdn_value_heads, tokens));
        }
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        matrix(layout, DType::BF16, TextConfig::value_dim, tokens);
        layout.alloc_bytes(
            ops::linear_add_workspace_bytes(TextConfig::hidden, TextConfig::value_dim, tokens));
    };
    const auto mlp_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        auto scope = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        matrix(layout, DType::BF16, TextConfig::intermediate, tokens);
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(
                ops::linear_swiglu_workspace_bytes(2 * TextConfig::intermediate, tokens));
        }
        {
            auto operator_scope = layout.scope();
            layout.alloc_bytes(ops::linear_add_workspace_bytes(TextConfig::hidden,
                                                               TextConfig::intermediate, tokens));
        }
    };

    WorkspaceLayoutBuilder prefill;
    common_root(prefill, prefill_tokens);
    attention_stage(prefill, prefill_tokens);
    gdn_stage(prefill, prefill_tokens);
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
    gdn_stage(verify, verify_tokens);
    mlp_stage(verify, verify_tokens);
    {
        auto final_scope = verify.scope();
        matrix(verify, DType::BF16, TextConfig::hidden, verify_tokens);
    }

    WorkspaceLayoutBuilder mtp;
    if (plan.mtp_k != 0) {
        common_root(mtp, prefill_tokens);
        matrix(mtp, DType::I32, 1, prefill_tokens);
        matrix(mtp, DType::BF16, TextConfig::hidden, prefill_tokens);
        matrix(mtp, DType::I32, 1, prefill_tokens);
        matrix(mtp, DType::BF16, TextConfig::hidden, 1);
        matrix(mtp, DType::BF16, TextConfig::hidden, 1);
        for (std::uint32_t i = 1; i < plan.mtp_k; ++i) {
            matrix(mtp, DType::BF16, TextConfig::hidden, 1);
        }
        {
            auto bulk_scope = mtp.scope();
            matrix(mtp, DType::BF16, TextConfig::hidden, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::hidden, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::hidden, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::mtp_input_rows, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::hidden, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::kv_size, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::kv_size, prefill_tokens);
            matrix(mtp, DType::BF16, TextConfig::kv_size, prefill_tokens);
        }
        {
            auto tail_scope = mtp.scope();
            matrix(mtp, DType::BF16, TextConfig::query_size, 1);
            matrix(mtp, DType::BF16, TextConfig::query_size, 1);
            matrix(mtp, DType::BF16, TextConfig::query_size, 1);
            matrix(mtp, DType::BF16, TextConfig::query_size, 1);
            mtp.alloc_bytes(ops::gqa_attention_workspace_bytes(TextConfig::query_heads, 1));
            matrix(mtp, DType::BF16, TextConfig::hidden, 1);
            matrix(mtp, DType::BF16, TextConfig::hidden, 1);
            matrix(mtp, DType::BF16, TextConfig::mtp_mlp_gate_up_rows, 1);
            matrix(mtp, DType::BF16, TextConfig::intermediate, 1);
            matrix(mtp, DType::BF16, TextConfig::hidden, 1);
            matrix(mtp, DType::BF16, TextConfig::output_rows, 1);
        }
    }

    WorkspaceLayoutBuilder decision;
    decision.alloc_bytes(
        ops::sampling_workspace_bytes(TextConfig::token_domain, std::max(1, verify_tokens)));

    return std::max(
        {prefill.peak_bytes(), verify.peak_bytes(), mtp.peak_bytes(), decision.peak_bytes()});
}

} // namespace

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 ||
        options.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("max_context must be in [1, INT32_MAX]");
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
    if (device.sm() != 120 || std::string_view(device.props.name) != "NVIDIA GeForce RTX 5090") {
        throw std::invalid_argument("qwen3_6_27b_rtx5090 requires NVIDIA GeForce RTX 5090");
    }
}

std::unique_ptr<SequencePlan::Impl> plan_sequence_impl(DeviceContext& device,
                                                       const EngineOptions& options) {
    validate_target_options(device, options);

    auto impl             = std::make_unique<SequencePlan::Impl>();
    impl->capacity        = options.max_context;
    impl->prefill_chunk   = options.prefill_chunk;
    impl->mtp_k           = options.speculative.draft_tokens;
    impl->proposal_head   = options.speculative.proposal_head;
    impl->use_cuda_graph  = options.use_cuda_graph;
    impl->device          = options.device;
    impl->kv_dtype        = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8;
    impl->kv_quant_group  = impl->kv_dtype == DType::I8 ? kKvQuantGroup : 0;
    impl->persistent      = persistent_layout(*impl);
    impl->workspace_bytes = workspace_bytes(*impl);
    if (impl->use_cuda_graph) {
        const std::size_t ordinary_variants = ordinary_graph_ranges(impl->capacity).size();
        const std::size_t ordinary_graphs   = ordinary_variants * (impl->mtp_k == 0 ? 1ULL : 2ULL);
        // Full-model capture measures 5-5.5 MiB per ordinary or short-window executable. Long MTP
        // executables also trigger substantially larger driver allocations: repeated cold 256K
        // K=5 construction peaks at 512.32 MiB against this 548 MiB target-private allowance.
        impl->graph_allowance_bytes = ordinary_graphs * 6ULL * kMiB;
        for (const GraphFrontierRange range : mtp_graph_ranges(impl->capacity, impl->mtp_k)) {
            const std::uint64_t final_visible =
                static_cast<std::uint64_t>(range.max) + 2ULL * impl->mtp_k;
            impl->graph_allowance_bytes += (final_visible <= 4096 ? 6ULL : 82ULL) * kMiB;
        }
    }

    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t required = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace_bytes, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    if (required > free_bytes) {
        throw std::invalid_argument("requested context requires " + std::to_string(required) +
                                    " bytes of sequence/workspace memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free after loading weights");
    }
    return impl;
}

SequencePlan::SequencePlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept            = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;
SequencePlan::~SequencePlan()                                  = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
