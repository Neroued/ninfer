#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/qwen4_draft_head.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/tp2/split.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/workspace.h"
#include "core/device.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/moe_predict.h"
#include "ninfer/ops/moe_slots.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include <algorithm>
#include <string_view>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

// The routed-expert banks of the offload backend's cache layers (Text, then MTP under MTP).
std::vector<execution::Qwen4ExpertBank> gguf_bank_list(const execution::Parameters& parameters,
                                                       SpeculativeBackend speculative) {
    std::vector<execution::Qwen4ExpertBank> banks;
    if (!parameters.qwen4 || !parameters.model.config().text.qwen4_routed_experts()) {
        return banks;
    }
    for (const auto& layer : parameters.qwen4->layers) { banks.push_back(layer.moe().experts); }
    if (speculative == SpeculativeBackend::Mtp && parameters.qwen4_mtp) {
        banks.push_back(parameters.qwen4_mtp->layer.moe().experts);
    }
    return banks;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

// At tensor-parallel width 2 (program/tp2/startup_tp2.inc): one rank's scratch of the DFlash2
// candidate ranking over `columns` final drafter columns, split by vocabulary or whole on rank 0;
// whether the DFlash2 drafter is split across the ranks.
std::size_t dflash2_candidates_workspace_tp2(const SequencePlanImpl& plan, std::int32_t columns);
bool dflash2_drafter_split_tp2(const SequencePlanImpl& plan);

// Active KV in RAM: the 4-token groups of one column's QSA selection.
std::int32_t qsa_selection_groups(const TextConfig& config) {
    if (!config.qwen4() || !config.sparse_attention) { return 0; }
    return static_cast<std::int32_t>((config.sparse_attention->max_selected_tokens() + 3) / 4);
}

// NINFER_KVRAM_REGION_TOKENS: the tokens of one layer the active KV region holds for prefill
// (0 = default 262,144); longer contexts stage the union of each chunk's selections.
std::uint32_t staging_region_tokens_override() {
    const char* env = std::getenv("NINFER_KVRAM_REGION_TOKENS");
    if (env == nullptr || *env == '\0') { return 0; }
    char* end         = nullptr;
    const long tokens = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || tokens <= 0 || tokens > (1L << 30)) {
        throw std::invalid_argument("NINFER_KVRAM_REGION_TOKENS must be a positive token count");
    }
    return static_cast<std::uint32_t>(tokens);
}

// NINFER_KVRAM_CACHE_BLOCKS: the active KV cache's blocks per attention layer (0 = default).
std::uint32_t staging_cache_blocks_override() {
    const char* env = std::getenv("NINFER_KVRAM_CACHE_BLOCKS");
    if (env == nullptr || *env == '\0') { return 0; }
    char* end         = nullptr;
    const long blocks = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || blocks <= 0 || blocks > (1L << 24)) {
        throw std::invalid_argument("NINFER_KVRAM_CACHE_BLOCKS must be a positive block count");
    }
    return static_cast<std::uint32_t>(blocks);
}

// Rank `rank_index`'s persistent layout. At tp 2 the ranks differ only in the masked drafter's
// state (its context, pending and prefill features and the DFlash local slots of each StateImage):
// the drafter runs on rank 0 alone, so rank 1 holds none of it.
PersistentLayout persistent_layout(const SequencePlanImpl& plan, int rank_index) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    // One rank's KV heads and GDN channels/value heads; the whole config at tp 1. The round
    // state keeps the complete vocabulary: rank 0 gathers the complete logits and samples there.
    // Rank 1's round state has the same layout, so its logits frames are allocated but unused.
    const TextConfig rank = execution::shard_text_config(config, plan.tp);
    // The split DFlash2 drafter at tp 2 keeps its state on both ranks, each ring holding the
    // rank's KV heads (execution/tp2/draft_split.h); otherwise the drafter is rank 0's alone.
    const bool split_drafter = plan.tp != 1 && dflash2_drafter_split_tp2(plan);
    const bool drafter       = plan.features.dflash2() && (rank_index == 0 || split_drafter);

    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.5 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    // T64 elastic KV: the KV page planes and the expert slot pools go to a second layout (the
    // elastic arena, CUDA VMM); everything else stays in the ordinary persistent allocation, so
    // cross-device copies (tp 2) never touch VMM memory.
    LayoutBuilder elastic_builder;
    const bool elastic = plan.elastic_kv.enabled && config.qwen4();
    PersistentLayout out;
    // The hidden a round carries: Qwen4Exp's wide residual (S*H), else the final-normed hidden.
    const std::int32_t carried_hidden =
        config.qwen4() ? dimension(config.residual_width()) : dimension(config.hidden_size);
    out.decoder = qwen3_5::plan_decoder_state(
        builder,
        qwen3_5::DecoderStateSpec{
            .full_attention_layers     = config.full_attention_layers,
            .mtp_layers                = 1,
            .capacity                  = plan.capacity,
            .kv_heads                  = dimension(rank.attention->num_key_value_heads),
            .attention_head_dim        = dimension(config.attention->head_dim),
            .kv_storage                = plan.kv_storage,
            .enable_mtp                = plan.features.mtp(),
            .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
            .text_physical_page_groups = physical_pages,
            .mtp_physical_page_groups  = mtp_physical_pages,
            .qsa_index                 = config.sparse_attention.has_value(),
            .kv_ram_active             = plan.kv_ram_active && config.qwen4(),
            .staging_window            = static_cast<std::int32_t>(
                plan.speculative_backend != SpeculativeBackend::None ? plan.draft_window + 1U : 1U),
            .staging_selection_groups = qsa_selection_groups(config),
            .staging_cache_blocks =
                plan.kv_ram_active && config.qwen4() ? staging_cache_blocks_override() : 0U,
            .staging_region_tokens =
                plan.kv_ram_active && config.qwen4() ? staging_region_tokens_override() : 0U,
        },
        elastic ? &elastic_builder : nullptr);
    qwen3_5::StateImageSpec state_image_spec{
        .linear =
            {
                .layers         = config.linear_attention_layers,
                .conv_channels  = (rank.gdn ? dimension(rank.gdn->conv_channels()) : 0),
                .conv_width     = (rank.gdn ? dimension(rank.gdn->linear_conv_kernel_dim - 1) : 0),
                .value_heads    = (rank.gdn ? dimension(rank.gdn->linear_num_value_heads) : 0),
                .value_head_dim = (rank.gdn ? dimension(rank.gdn->linear_value_head_dim) : 0),
                .key_head_dim   = (rank.gdn ? dimension(rank.gdn->linear_key_head_dim) : 0),
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
            },
        .hidden = carried_hidden,
    };
    if (config.ple) {
        // One PLE layer: (kernel-1)*dilation previous normalized gated columns of the wide width.
        state_image_spec.ple = qwen3_5::PleStateSpec{
            .layers = 1U,
            .rows = checked_i32(std::uint64_t(config.ple->conv_history()) * config.residual_width(),
                                "Qwen4Exp PLE state rows exceed int32")};
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (drafter) {
            state_image_spec.dflash_local = qwen3_5::DFlashLocalStateSpec{
                .layers   = draft->num_hidden_layers,
                .capacity = draft->sliding_window,
                .kv_heads =
                    dimension(draft->attention.num_key_value_heads) / (split_drafter ? plan.tp : 1),
                .head_dim = dimension(draft->attention.head_dim),
            };
        }
    }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder,
            GdnReplayRecordSpec{
                .layers          = dimension(config.linear_attention_layers),
                .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                // One rank's key/value heads and convolution channels, as for the StateImages.
                .conv_channels = (rank.gdn ? dimension(rank.gdn->conv_channels()) : 0),
                .qk_heads      = (rank.gdn ? dimension(rank.gdn->linear_num_key_heads) : 0),
                .value_heads   = (rank.gdn ? dimension(rank.gdn->linear_num_value_heads) : 0),
                .key_dim       = (rank.gdn ? dimension(rank.gdn->linear_key_head_dim) : 0),
                .value_dim     = (rank.gdn ? dimension(rank.gdn->linear_value_head_dim) : 0),
            });
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (drafter) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            const std::int32_t feature_rows =
                dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size()));
            dflash.prefill_features =
                add_tensor(builder, DType::BF16, {feature_rows, effective_prefill_chunk},
                           "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                  static_cast<std::int32_t>(plan.draft_window + 1U),
                  static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash pending target features");
        }
    }

    out.round = qwen3_5::begin_round_state_layout(
        builder, qwen3_5::RoundStateSpec{.hidden         = carried_hidden,
                                         .output_rows    = dimension(config.vocab_size),
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .backend        = plan.speculative_backend,
                                         .causal_scoring = plan.causal_scoring});
    // With lent Qwen4Exp prompt workspace (T42) only the base chunk: a wider chunk keeps its
    // hidden in the lent arena and its last column here.
    const std::int32_t prefill_hidden_columns =
        plan.qwen4_base_chunk > 0
            ? static_cast<std::int32_t>(std::min(plan.qwen4_base_chunk, plan.capacity))
            : effective_prefill_chunk;
    out.prefill_hidden = add_tensor(builder, DType::BF16, {carried_hidden, prefill_hidden_columns},
                                    "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden = add_tensor(builder, DType::BF16,
                                      {carried_hidden, static_cast<std::int32_t>(kCausalScoreTile)},
                                      "causal score hidden staging");
    }
    qwen3_5::complete_round_state_layout(builder, out.round);
    if (!plan.causal_scoring) {
        out.token_counts        = add_tensor(builder, DType::I32,
                                             {dimension(parameters.model.resources().public_token_count),
                                              static_cast<std::int32_t>(plan.max_concurrency)},
                                             "sampling token counts");
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config = add_tensor(
            builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
            "sampling config");
        if (rank_index == 0) {
            // Structured output: one token-mask slot per lane, on the sampling rank only. MTP
            // verifies K+1 columns; ordinary and DFlash2 rounds (constrained rows draft nothing,
            // D3) read one.
            const auto positions = static_cast<std::int32_t>(
                plan.speculative_backend == SpeculativeBackend::Mtp ? plan.draft_window + 1U : 1U);
            out.grammar_masks = add_tensor(
                builder, DType::I32,
                {dimension((parameters.model.resources().public_token_count + 31U) / 32U),
                 positions, static_cast<std::int32_t>(plan.max_concurrency)},
                "structured output token masks");
        }
    }
    {
        // The Text forward's (Qwen4Forward) persistent inputs; a dense FFN and no PLE leave them
        // empty.
        Qwen4PersistentLayout& q = out.qwen4.emplace();
        // A dense FFN plans no expert state: every expert block below is guarded by `routed`.
        const bool routed  = config.qwen4_routed_experts();
        const auto experts = routed ? dimension(std::get<MoeConfig>(config.ffn).num_experts) : 0;
        const auto layers  = dimension(config.num_hidden_layers);
        if (routed) {
            q.expert_table = add_tensor(builder, DType::I32, {experts}, "Qwen4Exp expert table");
        }
        const bool speculative = plan.speculative_backend != SpeculativeBackend::None;
        const auto window = static_cast<std::int32_t>(speculative ? plan.draft_window + 1U : 1U);
        if (config.ple && !plan.causal_scoring) {
            q.decode_ple_rows =
                add_tensor(builder, DType::BF16,
                           {dimension(config.ple->embed_dim),
                            checked_i32(static_cast<std::uint64_t>(window) * plan.max_concurrency,
                                        "Qwen4Exp decode PLE columns exceed int32")},
                           "Qwen4Exp decode PLE rows");
        }
        if (config.ple && speculative) {
            q.ple_record = add_tensor(builder, DType::BF16,
                                      {dimension(config.residual_width()), window,
                                       static_cast<std::int32_t>(plan.max_concurrency)},
                                      "Qwen4Exp PLE speculative record");
        }
        const bool mtp =
            plan.speculative_backend == SpeculativeBackend::Mtp && parameters.qwen4_mtp;
        if (routed && mtp && parameters.qwen4_mtp->uniform_gate_up_divisor > 0.0F) {
            q.mtp_gate_up_divisors =
                add_tensor(builder, DType::FP32, {2 * experts}, "Qwen4Exp MTP gate/up divisors");
            q.mtp_down_divisors =
                add_tensor(builder, DType::FP32, {experts}, "Qwen4Exp MTP down divisors");
        }
        if (routed && plan.qwen4_experts.backend == RoutedExpertBackend::Offload) {
            q.expert_slots    = static_cast<std::int32_t>(plan.qwen4_experts.slots_per_layer);
            q.elastic_columns = q.expert_slots > 0
                                    ? static_cast<std::int32_t>(plan.qwen4_experts.elastic_columns)
                                    : 0;
            // The offload backend serves the MTP layer as one more cache layer.
            q.cache_layers = layers + (mtp ? 1 : 0);
        }
        if (q.expert_slots > 0) {
            const auto cache_layers = q.cache_layers;
            q.cache_table =
                add_tensor(builder, DType::I32,
                           {checked_i32(std::uint64_t(cache_layers) * std::uint64_t(experts),
                                        "Qwen4Exp expert table exceeds int32")},
                           "Qwen4Exp expert slot table");
            q.cache_divisors = add_tensor(
                builder, DType::FP32,
                {checked_i32(std::uint64_t(cache_layers) *
                                 std::uint64_t(execution::Qwen4OffloadExperts::kDivisorsPerLayer),
                             "Qwen4Exp expert divisors exceed int32")},
                "Qwen4Exp expert divisors");
            q.cache_reads =
                add_tensor(builder, DType::I32, {ops::kMoeSlotsTopK * ops::kMoeSlotsMaxTokens},
                           "Qwen4Exp expert device reads");
            // Rank 0's residency mirror (the CPU path's publish) and rank 1's (its miss-split
            // share, T42): ~100 KB each.
            if (plan.tp == 2) {
                q.cache_mirror =
                    add_tensor(builder, DType::I32,
                               {checked_i32(std::uint64_t(cache_layers) * std::uint64_t(experts),
                                            "Qwen4Exp expert mirror exceeds int32")},
                               "Qwen4Exp expert residency mirror");
            }
            q.cache_stream_tables =
                add_tensor(builder, DType::I32,
                           {checked_i32(std::uint64_t(layers) * std::uint64_t(experts),
                                        "Qwen4Exp expert stream tables exceed int32")},
                           "Qwen4Exp expert stream tables");
            if (execution::staged_miss_split(plan.qwen4_experts)) {
                // Staged GPU share of the misses (T18): scratch slots, their table, the pick list.
                q.stage_table =
                    add_tensor(builder, DType::I32, {experts}, "Qwen4Exp miss split table");
                q.stage_picks = add_tensor(builder, DType::I32, {execution::kMissSplitSlots + 1},
                                           "Qwen4Exp miss split picks");
                const auto planes =
                    execution::Qwen4OffloadExperts::pool_plane_bytes(1, execution::kMissSplitSlots);
                for (std::size_t plane = 0; plane < planes.size(); ++plane) {
                    q.stage_pool[plane] =
                        builder.add(static_cast<std::size_t>(planes[plane]), kArenaAlign,
                                    "Qwen4Exp miss split scratch slots");
                }
            }
            // T25 prefetch: staging slots after the cache's slots, and their bookkeeping.
            q.staging_slots = static_cast<std::int32_t>(2U * plan.qwen4_experts.prefetch_slots);
            if (q.staging_slots > 0) {
                q.prefetch_tags  = add_tensor(builder, DType::I32, {q.staging_slots},
                                              "Qwen4Exp prefetch staging tags");
                q.prefetch_epoch = add_tensor(builder, DType::I32, {1}, "Qwen4Exp prefetch epoch");
                q.prefetch_rows =
                    add_tensor(builder, DType::I32,
                               {checked_i32(std::uint64_t(layers) * std::uint64_t(experts),
                                            "Qwen4Exp prefetch rows exceed int32")},
                               "Qwen4Exp prefetch effective slot rows");
                q.prefetch_workspace = builder.add(ops::moe_predict_workspace_bytes(), kArenaAlign,
                                                   "Qwen4Exp prefetch predictor workspace");
            }
            // ggml block banks: one arena of variable-size slots (Qwen4OffloadExperts::gguf_pool).
            std::vector<execution::Qwen4ExpertBank> banks;
            for (const auto& layer : parameters.qwen4->layers) {
                banks.push_back(layer.moe().experts);
            }
            if (mtp) { banks.push_back(parameters.qwen4_mtp->layer.moe().experts); }
            q.gguf = execution::Qwen4OffloadExperts::gguf_banks(banks);
            if (q.gguf) {
                if (q.staging_slots > 0) {
                    throw std::invalid_argument(
                        "Qwen4Exp ggml block banks: the router-ahead prefetch is not implemented");
                }
                std::vector<std::int64_t> bytes;
                for (const auto& b : banks) {
                    bytes.push_back(execution::Qwen4OffloadExperts::gguf_slot_bytes(b));
                }
                const auto layer_base = execution::Qwen4OffloadExperts::layer_slot_profile(
                    cache_layers, layers, q.expert_slots, q.elastic_columns,
                    static_cast<std::int32_t>(plan.tp));
                const auto pool =
                    execution::Qwen4OffloadExperts::gguf_pool(bytes, layer_base, q.elastic_columns);
                // The pools slots, then the prefill rings (Qwen4OffloadExperts::gguf_ring_slots).
                q.cache_slot_offsets =
                    add_tensor(builder, DType::I64,
                               {checked_i32(pool.slot_offset.size() +
                                                static_cast<std::size_t>(
                                                    execution::Qwen4OffloadExperts::gguf_ring_slots(
                                                        static_cast<std::int32_t>(plan.tp))),
                                            "Qwen4Exp GGUF slots")},
                               "Qwen4Exp expert slot offsets");
                for (std::size_t plane = 0; plane < 4; ++plane) {
                    const std::size_t region =
                        plane == 0 ? static_cast<std::size_t>(pool.total_bytes) : kArenaAlign;
                    q.cache_pool[plane] =
                        (elastic ? elastic_builder : builder)
                            .add(region, kArenaAlign, "Qwen4Exp expert slot pool");
                    q.cache_pool_bytes += region;
                    q.elastic_first[plane] =
                        plane == 0 ? static_cast<std::size_t>(pool.base_bytes) : region;
                    q.elastic_bytes[plane] =
                        plane == 0 ? static_cast<std::size_t>(pool.column_bytes) * q.elastic_columns
                                   : 0;
                }
            } else {
                // Base slots, the prefetch staging slots, then the elastic columns.
                const auto planes = execution::Qwen4OffloadExperts::pool_plane_bytes(
                    cache_layers, q.expert_slots + q.elastic_columns, q.staging_slots);
                for (std::size_t plane = 0; plane < planes.size(); ++plane) {
                    q.cache_pool[plane] = (elastic ? elastic_builder : builder)
                                              .add(static_cast<std::size_t>(planes[plane]),
                                                   kArenaAlign, "Qwen4Exp expert slot pool");
                    q.cache_pool_bytes += static_cast<std::size_t>(planes[plane]);
                    const std::int64_t stride = execution::Qwen4OffloadExperts::plane_slot_bytes(
                        static_cast<std::int32_t>(plane));
                    q.elastic_first[plane] = static_cast<std::size_t>(
                        (static_cast<std::int64_t>(cache_layers) * q.expert_slots +
                         q.staging_slots) *
                        stride);
                    q.elastic_bytes[plane] = static_cast<std::size_t>(
                        static_cast<std::int64_t>(cache_layers) * q.elastic_columns * stride);
                }
            }
        }
        // T43: the draft shortlist head: rank 0 alone, or split over both ranks (rank 0's block
        // is a view, rank 1 gathers the other half).
        const bool shortlist = plan.draft_prefix > 0 || !plan.draft_extras.empty();
        const std::int32_t shortlist_rows =
            plan.draft_prefix + static_cast<std::int32_t>(plan.draft_extras.size());
        if (mtp && shortlist && plan.draft_split && rank_index == 0) {
            q.draft_prefix = plan.draft_prefix;
            q.draft_ids    = add_tensor(builder, DType::I32, {shortlist_rows},
                                        "Qwen4Exp draft shortlist token IDs");
        }
        if (mtp && shortlist && plan.draft_split && rank_index == 1) {
            q.draft_extra    = shortlist_rows / 2;
            const auto bytes = execution::draft_row_plane_bytes(parameters.text.output_head.weight);
            for (std::size_t p = 0; p < bytes.size(); ++p) {
                if (bytes[p] == 0) { continue; } // a BF16 head has one plane
                q.draft_planes[p] = builder.add(bytes[p] * static_cast<std::size_t>(q.draft_extra),
                                                kArenaAlign, "Qwen4Exp draft shortlist head rows");
            }
        }
        if (mtp && shortlist && !plan.draft_split && rank_index == 0) {
            q.draft_prefix = plan.draft_prefix;
            q.draft_extra  = static_cast<std::int32_t>(plan.draft_extras.size());
            q.draft_ids    = add_tensor(builder, DType::I32, {q.draft_prefix + q.draft_extra},
                                        "Qwen4Exp draft shortlist token IDs");
            if (q.draft_extra > 0) {
                const auto bytes =
                    execution::draft_row_plane_bytes(parameters.text.output_head.weight);
                for (std::size_t p = 0; p < bytes.size(); ++p) {
                    if (bytes[p] == 0) { continue; } // a BF16 head has one plane
                    q.draft_planes[p] =
                        builder.add(bytes[p] * static_cast<std::size_t>(q.draft_extra), kArenaAlign,
                                    "Qwen4Exp draft shortlist head rows");
                }
            }
        }
    }
    out.bytes            = builder.finish(kArenaAlign, "persistent layout");
    out.elastic_bytes    = elastic ? elastic_builder.finish(kArenaAlign, "elastic layout") : 0;
    out.kv_payload_bytes = out.decoder.kv_payload_bytes();
    return out;
}

void reserve_matrix(WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                    std::int32_t tokens) {
    (void)layout.alloc(dtype, {rows, tokens});
}

void reserve_scratch(WorkspaceLayoutBuilder& layout, std::size_t bytes) {
    if (bytes == 0) { return; }
    auto scope = layout.scope();
    (void)layout.alloc_bytes(bytes);
}

void reserve_linear(WorkspaceLayoutBuilder& layout, const execution::LinearParameters& p, int first,
                    int last) {
    reserve_scratch(layout, ops::linear_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
}

// The drafter's transient phases (execution/draft.cpp). They run on the executing device at tp 1
// and on rank 0 alone at tp 2, where the drafter and its proposal head are whole; both widths plan
// them from rank 0's Parameters.

// One context append over `batch` rows of `width` target columns; `compact_input` adds the compact
// feature matrix the decode round gathers first.
std::size_t dflash_context_workspace(const SequencePlanImpl& plan, std::int32_t width,
                                     std::int32_t batch, bool compact_input) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    const auto& draft      = *parameters.model.config().draft;
    const auto tokens      = width * batch;
    WorkspaceLayoutBuilder layout;
    if (compact_input) {
        reserve_matrix(layout, DType::BF16,
                       dimension(config.hidden_size * std::uint64_t(draft.target_layer_ids.size())),
                       tokens);
    }
    const auto local_width = std::min(width, dimension(draft.sliding_window));
    (void)workspace::dflash_context(layout, config, draft, local_width * batch);
    reserve_linear(layout, parameters.draft->feature_projection, local_width * batch,
                   local_width * batch);
    reserve_scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(batch, local_width,
                                                                                 local_width));
    return layout.peak_bytes(1);
}

// One proposal over `batch` rows of `width` masked-block columns.
std::size_t dflash_proposal_workspace(const SequencePlanImpl& plan, std::int32_t width,
                                      std::int32_t batch) {
    const auto& parameters    = *plan.parameters;
    const auto& config        = parameters.model.config().text;
    const auto& draft         = *parameters.model.config().draft;
    const auto drafts         = static_cast<std::int32_t>(plan.draft_window);
    const std::int32_t tokens = width * batch;
    const auto& head          = plan.proposal_head == ProposalHead::Optimized
                                    ? parameters.proposal->head
                                    : parameters.draft->output_head;
    WorkspaceLayoutBuilder layout;
    reserve_matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
    const auto prepare = [&] {
        (void)workspace::dflash2_branch(layout, config, draft, width, batch);
        reserve_scratch(layout, ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    width, width, batch, batch));
    };
    {
        auto attention = layout.scope();
        prepare();
        reserve_matrix(layout, DType::BF16, dimension(draft.attention.query_width()), tokens);
        reserve_matrix(layout, DType::BF16, dimension(draft.attention.key_width()), tokens);
        reserve_matrix(layout, DType::BF16, dimension(draft.attention.key_width()), tokens);
        reserve_matrix(layout, DType::BF16, dimension(draft.attention.query_width()), tokens);
        reserve_scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                    {dimension(draft.attention.head_dim),
                                     dimension(draft.attention.num_attention_heads),
                                     dimension(draft.attention.num_key_value_heads)},
                                    dimension(draft.sliding_window), {0, plan.capacity}, width,
                                    width, batch));
        reserve_scratch(layout,
                        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                            dimension(draft.attention.query_width()), width, width, batch, batch));
    }
    {
        auto mlp = layout.scope();
        prepare();
        reserve_matrix(layout, DType::BF16, dimension(draft.intermediate_size), tokens);
        for (const auto& block : parameters.draft->layers) {
            const auto& p = block.mlp.gate_up;
            reserve_scratch(layout,
                            ops::linear_swiglu_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, tokens, tokens));
        }
        reserve_scratch(layout,
                        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                            dimension(draft.intermediate_size), width, width, batch, batch));
    }
    const auto mask_columns = drafts * batch;
    reserve_matrix(layout, DType::BF16, dimension(config.hidden_size), mask_columns);
    reserve_matrix(layout, DType::FP32, dimension(draft.dflash2.selector_top_k), mask_columns);
    reserve_scratch(layout, plan.tp != 1 ? dflash2_candidates_workspace_tp2(plan, mask_columns)
                                         : ops::linear_topk_workspace_capacity_bytes(
                                               head.weight.qtype, head.weight.n, head.weight.k,
                                               mask_columns, mask_columns));
    reserve_matrix(layout, DType::BF16, dimension(draft.dflash2.selector_rank), mask_columns);
    reserve_linear(layout, parameters.draft->selector->hidden_projection, mask_columns,
                   mask_columns);
    reserve_scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(drafts, drafts,
                                                                                  batch, batch));
    return layout.peak_bytes(1);
}

// Rank 0's acceptance over `batch` verified rows: sparse over DFlash2's candidate proposals.
std::size_t dflash_accept_workspace(const SequencePlanImpl& plan, std::int32_t batch) {
    const auto& parameters   = *plan.parameters;
    const auto drafts        = static_cast<std::int32_t>(plan.draft_window);
    const auto public_tokens = dimension(parameters.model.resources().public_token_count);
    return ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
        public_tokens, {false}, drafts, drafts, batch, batch);
}

// The merged-token extent one Vision item's encode is planned for.
std::uint32_t vision_item_tokens(const SequencePlanImpl& plan) {
    if (plan.vision_parameters == nullptr || !plan.vision_parameters->vision) {
        throw std::logic_error("Vision workspace planning has no Vision parameters");
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        {plan.capacity, kMaximumVisionItemTokens, plan.max_vision_item_tokens}));
}

// At tensor-parallel width 2 (program/tp2/startup_tp2.inc).
WorkspacePlan build_qwen3_5_head_workspace_plan_tp2(const SequencePlanImpl& plan);
// The Qwen3.5 heads of a dense model (one device below, tp 2 above) and the Qwen4 forward's tail.
WorkspacePlan build_qwen3_5_head_workspace_plan(const SequencePlanImpl& plan);
void finish_qwen4_workspace_plan(const SequencePlanImpl& plan, WorkspacePlan& out);
void validate_tensor_parallel_options(const execution::Parameters& parameters,
                                      const execution::Parameters* peer_parameters,
                                      const EngineOptions& options);

// The Text forward (Qwen4Forward) of Qwen4Exp or of a dense Qwen3.5 model, at one device or at
// tp 2: prefill chunks, ordinary decode rounds, causal-score tiles, the target verification of a
// speculative round (MTP or DFlash2) and, under the Qwen4 MTP layer, the prompt's MTP KV and
// proposals and the round's MTP phases. The forward phases replay
// execution/qwen4_forward.h in planning mode (identical allocations); at tp 2 over both ranks'
// Parameters and shard geometry, one layout per rank, and both ranks allocate the larger (rank
// 1's work arena has rank 0's capacity, PeerRuntime). With hyper-connections the carried hidden is
// the wide residual (S*H), which the output head reads through the Text output mixer; without
// them it is the final-normed hidden.
WorkspacePlan build_qwen4_workspace_plan(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    const bool tp2         = plan.tp == execution::kTensorParallelWidth;
    if (tp2 && (plan.peer_parameters == nullptr || !plan.peer_parameters->qwen4)) {
        throw std::logic_error("Qwen4Exp tp 2 workspace plan has no rank 1 Parameters");
    }
    const TextConfig shard        = execution::shard_text_config(config, plan.tp);
    const int ranks               = tp2 ? 2 : 1;
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    using Layouts              = std::array<WorkspaceLayoutBuilder, 2>;
    const auto chunk           = static_cast<std::int32_t>(chunk_u32);
    const auto H               = dimension(config.hidden_size);
    const auto wide            = dimension(config.residual_width());
    const auto vocab           = dimension(config.vocab_size);
    const auto rank_parameters = [&](int rank) -> const execution::Parameters& {
        return rank == 0 ? parameters : *plan.peer_parameters;
    };
    // `rows` x `tokens` on every rank.
    const auto matrix = [&](Layouts& layout, DType dtype, std::int32_t rows, std::int32_t tokens) {
        for (int r = 0; r < ranks; ++r) {
            (void)layout[static_cast<std::size_t>(r)].alloc(dtype, {rows, tokens});
        }
    };
    const auto finish = [&](const Layouts& layout) {
        std::size_t bytes = 0;
        for (int r = 0; r < ranks; ++r) {
            bytes = std::max(bytes, layout[static_cast<std::size_t>(r)].peak_bytes(1));
        }
        return bytes;
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout,
                                    const execution::LinearParameters& p, int first, int last) {
        reserve_scratch(layout, ops::linear_workspace_capacity_bytes(
                                    p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last));
    };
    // One forward phase over W columns of B lanes (prefill: W = chunk, B = 1) on every rank.
    const auto make = [&](Layouts& layout, bool prefill, std::int32_t width, std::int32_t batch,
                          std::int32_t first, std::int32_t last) {
        execution::Qwen4Columns columns;
        columns.width         = width;
        columns.batch         = batch;
        columns.max_visible   = plan.capacity;
        columns.prefill       = prefill;
        columns.kv_storage    = plan.kv_storage;
        columns.kv_ram_active = plan.kv_ram_active;
        std::array<execution::Qwen4Rank<WorkspaceLayoutBuilder>, 2> rank_operands;
        for (int r = 0; r < ranks; ++r) {
            rank_operands[static_cast<std::size_t>(r)] =
                execution::Qwen4Rank<WorkspaceLayoutBuilder>{
                    .config     = &shard,
                    .parameters = &rank_parameters(r),
                    .work       = &layout[static_cast<std::size_t>(r)],
                    .columns    = columns};
        }
        return execution::Qwen4Forward<WorkspaceLayoutBuilder>(config, rank_operands, ranks, {},
                                                               first, last);
    };
    const auto forward = [&](Layouts& layout, bool prefill, std::int32_t width, std::int32_t batch,
                             std::int32_t first, std::int32_t last) {
        auto f = make(layout, prefill, width, batch, first, last);
        std::array<Tensor, 2> none{};
        std::array<Tensor, 2> out{};
        f.run(none, config.ple ? &none : nullptr, out);
    };
    // The output head over `columns` mixed hiddens: whole at one device; at tp 2 each rank's
    // vocabulary half and rank 0's staging of rank 1's half (TextContext::logits_tp2).
    const auto logits = [&](Layouts& layout, std::int32_t columns) {
        if (!tp2) {
            linear_scratch(layout[0], parameters.text.output_head, columns, columns);
            return;
        }
        for (int r = 0; r < ranks; ++r) {
            auto& own                               = layout[static_cast<std::size_t>(r)];
            const execution::LinearParameters& head = rank_parameters(r).text.output_head;
            const execution::LinearParameters& peer = rank_parameters(1 - r).text.output_head;
            auto call                               = own.scope();
            (void)execution::workspace::tp_logits(own, head.weight.n, peer.weight.n, columns,
                                                  r == 0);
            reserve_scratch(own,
                            execution::output_head_split_workspace_bytes(head, columns, columns));
        }
    };
    // Output mixer (on every rank) + output head over `columns` carried hiddens (Text or MTP
    // head: same mixer shapes).
    const auto head = [&](Layouts& layout, std::int32_t columns) {
        matrix(layout, DType::BF16, H, columns);
        auto f = make(layout, false, columns, 1, columns, columns);
        std::array<Tensor, 2> none{}, x{};
        f.head(none, x);
        logits(layout, columns);
    };
    const auto mtp = [&](Layouts& layout, bool prefill, std::int32_t width, std::int32_t batch,
                         bool append_only, std::int32_t first = 0) {
        auto f =
            make(layout, prefill, width, batch, first > 0 ? first : width * batch, width * batch);
        std::array<Tensor, 2> none{}, out{};
        f.run_mtp({&*parameters.qwen4_mtp, tp2 ? &*plan.peer_parameters->qwen4_mtp : nullptr}, none,
                  nullptr, none, out, append_only);
    };
    const auto public_tokens = dimension(parameters.model.resources().public_token_count);
    const bool mtp_enabled   = plan.speculative_backend == SpeculativeBackend::Mtp;
    // The Qwen4 MTP layer; a dense Qwen3.5 model's own MTP head and proposal are planned by
    // build_qwen3_5_head_workspace_plan.
    const bool mtp_layer = mtp_enabled && config.qwen4();
    const auto drafts    = static_cast<std::int32_t>(plan.draft_window);
    const auto verify    = drafts + 1;
    if (mtp_layer && (!parameters.qwen4_mtp || (tp2 && !plan.peer_parameters->qwen4_mtp))) {
        throw std::logic_error("Qwen4Exp MTP workspace plan has no MTP Parameters");
    }

    // With Vision a prompt chunk carries three-axis RoPE positions and its visual scatter indices
    // on every rank, and its MTP alignment the composed shifted input (qwen4_text.cpp).
    const bool vision = plan.features.vision;
    WorkspacePlan out;
    {
        Layouts text_prefill;
        matrix(text_prefill, DType::I32, 1, chunk);                 // ids
        matrix(text_prefill, DType::I32, 1, chunk);                 // positions
        matrix(text_prefill, DType::I32, vision ? 3 : 1, chunk);    // RoPE positions
        if (vision) { matrix(text_prefill, DType::I32, 1, chunk); } // visual columns
        if (config.ple) {
            matrix(text_prefill, DType::BF16, dimension(config.ple->embed_dim), chunk);
        }
        if (tp2) { matrix(text_prefill, DType::BF16, wide, chunk); } // the chunk's hidden
        forward(text_prefill, true, chunk, 1, 1, chunk);
        if (!plan.causal_scoring) {
            {
                std::array<std::optional<WorkspaceLayoutBuilder::Scope>, 2> scope;
                for (int r = 0; r < ranks; ++r) {
                    scope[static_cast<std::size_t>(r)].emplace(
                        text_prefill[static_cast<std::size_t>(r)].scope());
                }
                head(text_prefill, 1);
                reserve_scratch(text_prefill[0],
                                ops::sampling_workspace_capacity_bytes(public_tokens, 1, 1));
            }
            if (mtp_layer) {
                // MTP prompt KV: append-only columns, then the last column's layer, its proposal
                // and the autoregressive drafts (TextContext::qwen4_mtp_prefill).
                matrix(text_prefill, DType::I32, 1, chunk); // shifted ids
                if (vision) {
                    matrix(text_prefill, DType::BF16, H, chunk); // composed shifted input
                    matrix(text_prefill, DType::I32, 1, chunk);  // its visual columns
                }
                {
                    std::array<std::optional<WorkspaceLayoutBuilder::Scope>, 2> scope;
                    for (int r = 0; r < ranks; ++r) {
                        scope[static_cast<std::size_t>(r)].emplace(
                            text_prefill[static_cast<std::size_t>(r)].scope());
                    }
                    if (vision) { matrix(text_prefill, DType::I32, 3, chunk); } // RoPE columns
                    matrix(text_prefill, DType::BF16, wide, chunk);
                    mtp(text_prefill, true, chunk, 1, true, 1);
                }
                {
                    std::array<std::optional<WorkspaceLayoutBuilder::Scope>, 2> scope;
                    for (int r = 0; r < ranks; ++r) {
                        scope[static_cast<std::size_t>(r)].emplace(
                            text_prefill[static_cast<std::size_t>(r)].scope());
                    }
                    if (vision) { matrix(text_prefill, DType::I32, 3, 1); } // RoPE column
                    mtp(text_prefill, true, 1, 1, false);
                    head(text_prefill, 1);
                    matrix(text_prefill, DType::BF16, wide, 1);
                    matrix(text_prefill, DType::I32, 1, 1);
                    mtp(text_prefill, true, 1, 1, false);
                }
            }
        }
        out.text_prefill = finish(text_prefill);
        out.mtp_prefill  = mtp_enabled ? out.text_prefill : 0;
    }

    if (plan.causal_scoring) {
        // ProgramImpl::causal_score's flush of one score tile: with hyper-connections the Text
        // output mixer on rank 0 (without them the carried hidden is already the final-normed
        // one), then the head (at tp 2: project_score_tile_split, rank 1's copy of the mixed
        // hidden in its persistent score_hidden).
        const auto tile = static_cast<std::int32_t>(kCausalScoreTile);
        Layouts causal_score;
        (void)causal_score[0].alloc(DType::BF16, {vocab, tile});
        (void)causal_score[0].alloc(DType::I32, {1, tile});
        (void)causal_score[0].alloc(DType::FP32, {1, tile});
        if (config.hyper_connection) {
            (void)causal_score[0].alloc(DType::BF16, {H, tile});
            reserve_scratch(causal_score[0], ops::hyper_connection_mix_workspace_bytes(
                                                 tile, tile, false, H,
                                                 dimension(config.hyper_connection->hc_lowrank)));
        }
        if (tp2) {
            for (int r = 0; r < ranks; ++r) {
                auto& own                               = causal_score[static_cast<std::size_t>(r)];
                const execution::LinearParameters& head = rank_parameters(r).text.output_head;
                const execution::LinearParameters& peer = rank_parameters(1 - r).text.output_head;
                (void)execution::workspace::tp_logits(own, head.weight.n, peer.weight.n, tile,
                                                      r == 0);
                reserve_scratch(own, execution::output_head_split_workspace_bytes(head, 1, tile));
            }
        } else {
            linear_scratch(causal_score[0], parameters.text.output_head, 1, kCausalScoreTile);
        }
        out.causal_score = finish(causal_score);
    }
    if (!plan.causal_scoring && !mtp_layer) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            Layouts ordinary;
            if (tp2) { matrix(ordinary, DType::BF16, wide, batch); } // rank 1's hidden
            forward(ordinary, false, 1, batch, batch, batch);
            head(ordinary, batch);
            reserve_scratch(ordinary[0],
                            ops::sampling_workspace_capacity_bytes(public_tokens, batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }
    if (!plan.causal_scoring && plan.speculative_backend != SpeculativeBackend::None) {
        // Both backends verify K+1 columns per lane through the same Text forward and head
        // (TextContext::qwen4_target_verify); a DFlash2 round's drafter phases are the head plan's.
        std::size_t& round = mtp_enabled ? out.mtp_round : out.dflash_round;
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = verify * batch;
            Layouts target;
            if (tp2) { matrix(target, DType::BF16, wide, aggregate); } // rank 1's hidden
            forward(target, false, verify, batch, aggregate, aggregate);
            head(target, aggregate);
            round = std::max(round, finish(target));
            if (!mtp_layer) { continue; }
            Layouts alignment;
            mtp(alignment, false, verify, batch, false);
            Layouts ar;
            mtp(ar, false, 1, batch, false);
            Layouts proposal;
            head(proposal, batch);
            const std::size_t accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    public_tokens, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), accept});
        }
        if (mtp_layer) {
            // The prefix-reuse bridge: one MTP column and its proposal drafts on the prefill row
            // (with Vision, rank 1's copy of a visual column's merged embedding).
            Layouts bridge;
            matrix(bridge, DType::I32, 1, 3);
            if (vision) { matrix(bridge, DType::BF16, H, 1); }
            mtp(bridge, true, 1, 1, false);
            head(bridge, 1);
            out.ordinary_round = finish(bridge);
        }
    }
    finish_qwen4_workspace_plan(plan, out);
    return out;
}

// The general capacity of a Qwen4-forward plan and, with Vision, the tower's items past it.
void finish_qwen4_workspace_plan(const SequencePlanImpl& plan, WorkspacePlan& out) {
    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        // The tower's rank encodes each item past its general prefix and keeps the merged
        // embeddings there for the item's chunks; at tp 2 the other rank only receives them.
        out.vision = execution::VisionContext::plan_workspace(
            *plan.parameters->model.config().vision, *plan.vision_parameters->vision,
            vision_item_tokens(plan), out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
        if (plan.tp == execution::kTensorParallelWidth) {
            out.vision_receiver = execution::VisionContext::plan_receiver(*out.vision);
            out.receiver_capacity =
                std::max(out.general_capacity, out.vision_receiver->capacity_bytes);
        }
    }
}

// A dense Qwen3.5 model: the Text forward is the Qwen4 forward's plain residual
// (build_qwen4_workspace_plan); its MTP head and proposal and its DFlash2 drafter are the Qwen3.5
// ones (TextContext::mtp_prompt_chunk, the MTP entry points, the drafter contexts), planned with
// the call's roots, logits and sampling by build_qwen3_5_head_workspace_plan. Each phase runs the
// Text forward, then that head over the same call's roots, so its peak is bounded by the larger
// of the two figures.
WorkspacePlan build_dense_workspace_plan(const SequencePlanImpl& plan) {
    WorkspacePlan out       = build_qwen4_workspace_plan(plan);
    const WorkspacePlan own = build_qwen3_5_head_workspace_plan(plan);
    out.text_prefill        = std::max(out.text_prefill, own.text_prefill);
    out.ordinary_round      = std::max(out.ordinary_round, own.ordinary_round);
    out.mtp_prefill         = std::max(out.mtp_prefill, own.mtp_prefill);
    out.mtp_round           = std::max(out.mtp_round, own.mtp_round);
    out.causal_score        = std::max(out.causal_score, own.causal_score);
    out.dflash_context      = std::max(out.dflash_context, own.dflash_context);
    out.dflash_round        = std::max(out.dflash_round, own.dflash_round);
    finish_qwen4_workspace_plan(plan, out);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    if (plan.parameters->model.config().text.qwen4()) { return build_qwen4_workspace_plan(plan); }
    return build_dense_workspace_plan(plan);
}

// The Qwen3.5 heads of a dense model at one device (tp 2: build_qwen3_5_head_workspace_plan_tp2):
// each phase's call roots (the chunk's ids, positions, RoPE and residual, as TextContext keeps them
// around the Text forward), its logits and sampling, the MTP head's prompt alignment, forwards and
// proposals, and the DFlash2 drafter's context and rounds. The Text layers are the Qwen4 forward's,
// planned by build_qwen4_workspace_plan.
WorkspacePlan build_qwen3_5_head_workspace_plan(const SequencePlanImpl& plan) {
    if (plan.tp != 1) { return build_qwen3_5_head_workspace_plan_tp2(plan); }
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace::text_prefill_roots(layout, config, tokens, plan.features.vision ? 3 : 0,
                                            plan.features.vision ? tokens : 0);
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout,
                                    const execution::LinearParameters& p, int first, int last) {
        scratch(layout, ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                             p.policy, first, last));
    };
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, parameters.mtp->output, first, last);
        scratch(layout,
                execution::dense_ffn_workspace_bytes(parameters.mtp->ffn, first, last, true));
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, dimension(parameters.proposal->rows), columns);
            linear_scratch(layout, parameters.proposal->head, columns, columns);
        } else {
            linear_scratch(layout, parameters.mtp->output_head, columns, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace::mtp_stem(layout, config, tokens, !preembedded);
        linear_scratch(layout, parameters.mtp->input_projection, 1, tokens);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace::mtp_attention_projection(layout, config, tokens);
        scratch(layout, execution::mtp_projection_workspace_bytes(parameters.mtp->projection,
                                                                  tokens, tokens));
        (void)workspace::mtp_attention_results(layout, config, tokens);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, envelope, 1, tokens, tokens));
        (void)workspace::mtp_post_attention(layout, config, tokens);
        mtp_post_mixer(layout, tokens, tokens);
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            scratch(layout,
                    execution::mtp_kv_workspace_bytes(parameters.mtp->projection, first, last));
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
        }
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout,
                execution::mtp_query_gate_workspace_bytes(parameters.mtp->projection, 1, 1));
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        mtp_post_mixer(layout, 1, 1);
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    if (!plan.causal_scoring) {
        scratch(text_prefill,
                ops::sampling_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), 1, 1));
    }
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        matrix(causal_score, DType::BF16, dimension(config.vocab_size),
               static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        linear_scratch(causal_score, parameters.text.output_head, 1, kCausalScoreTile);
        out.causal_score = finish(causal_score);
    }

    if (!plan.causal_scoring) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder ordinary;
            matrix(ordinary, DType::BF16, dimension(config.hidden_size), batch);
            scratch(ordinary,
                    ops::sampling_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), chunk);
            (void)workspace::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            dimension(parameters.model.resources().public_token_count), drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace::mtp_attention_projection(layout, config, tokens);
                scratch(layout, execution::mtp_projection_workspace_bytes(
                                    parameters.mtp->projection, tokens, tokens));
                (void)workspace::mtp_attention_results(layout, config, tokens);
                scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                    {dimension(config.attention->head_dim),
                                     dimension(config.attention->num_attention_heads),
                                     dimension(config.attention->num_key_value_heads)},
                                    plan.kv_storage, text_envelope, batch, width, width));
                (void)workspace::mtp_post_attention(layout, config, tokens);
                mtp_post_mixer(layout, tokens, tokens);
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), drafts, drafts,
                    batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.dflash2()) {
        out.dflash_context = dflash_context_workspace(plan, chunk, 1, false);
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = verify * batch;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
            out.dflash_round =
                std::max({out.dflash_round, finish(target), dflash_accept_workspace(plan, batch),
                          dflash_context_workspace(plan, verify, batch, true),
                          dflash_proposal_workspace(plan, verify, batch)});
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        out.vision = execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *plan.vision_parameters->vision,
            vision_item_tokens(plan), out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(const execution::Parameters& parameters,
                             const execution::Parameters* peer_parameters, DeviceContext& device,
                             const EngineOptions& options) {
    if (!parameters.model.config().text.attention ||
        parameters.model.config().text.full_attention_layers == 0) {
        throw std::invalid_argument("Qwen3.5 Program requires at least one full-attention layer");
    }
    if (parameters.model.options() != models::load_options(options)) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    if (parameters.draft &&
        options.max_context > parameters.model.config().draft->max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    // YaRN (ported 2026-10-24): the config's max_position_embeddings is the model's TRUE native
    // window and stays untouched; an opt-in rope factor raises the addressable ceiling to
    // native x factor, which is what the rope table installed at startup actually serves.
    // Factor 1 (the default) leaves this identical to the pre-YaRN engine.
    const double yarn_factor = options.rope_yarn_factor > 1.0F ? options.rope_yarn_factor : 1.0;
    const std::uint64_t context_ceiling = static_cast<std::uint64_t>(
        static_cast<double>(parameters.model.config().text.max_position_embeddings) * yarn_factor);
    if (context_ceiling > ops::kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument(
            "rope_yarn_factor extends the context past the attention visible-keys ceiling"
            " (native context x 4 is the supported maximum)");
    }
    if (options.max_context == 0 || options.max_context > context_ceiling) {
        throw std::invalid_argument("max_context exceeds the configured position capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        if (!(options.speculative.min_draft_probability >= 0.0F &&
              options.speculative.min_draft_probability < 1.0F)) {
            throw std::invalid_argument("MTP minimum draft probability must be in [0,1)");
        }
        break;
    case SpeculativeBackend::DFlash2:
        if (!parameters.draft) {
            throw std::invalid_argument("DFlash2 is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 15) {
            throw std::invalid_argument("DFlash2 draft window must be in [1,15]");
        }
        break;
    }
    if (device.compute_capability() != 120) {
        throw std::invalid_argument("Qwen3.5 family runtime requires compute capability 12.0");
    }
    if (!parameters.qwen4 ||
        (options.tp == 2 && (peer_parameters == nullptr || !peer_parameters->qwen4))) {
        throw std::logic_error("Qwen3.5 Program: a rank's Parameters have no Text forward blocks");
    }
    if (parameters.model.config().text.qwen4()) {
        // Qwen4Exp Text at one device (T20) or at tp 2 (T31), and MTP speculative decoding at
        // one device (T24).
        if (options.tp == 2 && models::load_options(options).hc_split_streams) {
            throw std::invalid_argument("Qwen4ExpForCausalLM: the stream-split hyper-connection "
                                        "mixers have no execution route (replicated mixers only)");
        }
        if (options.tp == 2 && options.speculative.backend == SpeculativeBackend::Mtp &&
            parameters.model.config().text.qwen4_routed_experts() &&
            options.routed_experts.backend != RoutedExpertBackend::Offload) {
            throw std::invalid_argument(
                "Qwen4ExpForCausalLM: MTP at tp 2 runs with the offload expert backend");
        }
        // T24: MTP over the full vocabulary. DFlash has no Qwen4Exp drafter; a trimmed proposal
        // head (draft vocabulary) is a later lever.
        if (options.speculative.backend != SpeculativeBackend::None &&
            options.speculative.backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument("Qwen4ExpForCausalLM: only MTP speculation is implemented");
        }
        if (options.speculative.backend == SpeculativeBackend::Mtp) {
            if (!parameters.qwen4_mtp) {
                throw std::invalid_argument("Qwen4ExpForCausalLM: the artifact has no MTP layer");
            }
            if (options.speculative.proposal_head != ProposalHead::Full) {
                throw std::invalid_argument(
                    "Qwen4ExpForCausalLM: MTP proposes over the full vocabulary only");
            }
        }
    }
    validate_tensor_parallel_options(parameters, peer_parameters, options);
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                    = std::make_unique<SequencePlanImpl>();
    impl->parameters             = inputs.parameters;
    impl->vision_parameters      = inputs.vision_parameters;
    impl->peer_parameters        = inputs.peer_parameters;
    impl->max_vision_item_tokens = inputs.max_vision_item_tokens;
    impl->capacity               = inputs.capacity;
    impl->main_page_groups       = main_page_groups;
    impl->kv_capacity            = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency        = inputs.max_concurrency;
    impl->prefill_chunk          = inputs.prefill_chunk;
    impl->kv_lease_window_tokens = inputs.kv_lease_window_tokens;
    impl->draft_window           = inputs.draft_window;
    impl->mtp_min_p              = inputs.mtp_min_p;
    impl->mtp_window_rule        = inputs.mtp_window_rule;
    impl->speculative_backend    = inputs.speculative_backend;
    impl->proposal_head          = inputs.proposal_head;
    impl->features               = inputs.features;
    impl->use_cuda_graph         = inputs.use_cuda_graph;
    impl->causal_scoring         = inputs.causal_scoring;
    impl->device                 = inputs.device;
    impl->tp                     = inputs.tp;
    impl->tp_mailbox             = inputs.tp_mailbox;
    impl->context_cache          = inputs.context_cache;
    impl->kv_storage             = inputs.kv_storage;
    impl->qwen4_experts          = inputs.qwen4_experts;
    impl->elastic_kv             = inputs.elastic_kv;
    impl->kv_ram_active          = inputs.kv_ram_active;
    impl->draft_prefix           = inputs.draft_prefix;
    impl->draft_extras           = inputs.draft_extras;
    impl->draft_split            = inputs.draft_split;
    // T42: Qwen4Exp prompts stream in chunks wider than prefill_chunk on workspace the offload
    // backend lends from the slot pool (the layers below the streaming ring), so the Program's
    // arena and the decode slot cache stay sized for prefill_chunk. The prompt chunk is the
    // largest prompt_chunk / 2^i (> prefill_chunk) whose workspace and ring fit in the Text layers
    // a prompt may lend (Qwen4OffloadExperts::may_lend, over the backend's per-layer base slots,
    // layer_slot_profile). The sizing probe (no slots yet) plans prefill_chunk only. The ring
    // counts as the runtime lends it (Qwen4OffloadExperts::ring_layers: half a layer of experts per
    // GPU at tp 2): planning the 1-GPU ring at tp 2 cost FP4-Local (59 slots) its 16384-column
    // chunk for 8192, so twice the chunks and expert streaming per prompt (o-prefill).
    std::size_t prompt_workspace = 0;
    std::vector<std::pair<std::uint32_t, std::size_t>> prompt_steps;
    {
        // NINFER_TP_HOST_EXCHANGE=0 brings the driver-staged peer pull back (program_impl_tp2.inc).
        const char* exchange_env = std::getenv("NINFER_TP_HOST_EXCHANGE");
        const bool peer_pull     = exchange_env != nullptr && std::string_view(exchange_env) == "0";
        const auto& experts      = impl->qwen4_experts;
        const auto& text         = impl->parameters->model.config().text;
        const std::uint32_t base = impl->prefill_chunk;
        // ggml block banks (capy-flash v0.7): the zero-copy prefill needs no ring, only the
        // borrowed workspace, counted in the arena bytes of the lent layers' base slots.
        const std::vector<execution::Qwen4ExpertBank> gguf_list =
            gguf_bank_list(*impl->parameters, impl->speculative_backend);
        const bool gguf_lend = text.qwen4_routed_experts() &&
                               experts.backend == RoutedExpertBackend::Offload &&
                               experts.gguf_stream_min_tokens > 0 &&
                               execution::Qwen4OffloadExperts::gguf_banks(gguf_list);
        // Only a routed-expert model has a slot pool to lend (a dense Qwen4 has no backend).
        if (text.qwen4_routed_experts() && experts.backend == RoutedExpertBackend::Offload &&
            experts.slots_per_layer > 0 && (experts.stream_min_tokens > 0 || gguf_lend) &&
            experts.prompt_chunk > base && !impl->causal_scoring && impl->capacity > base &&
            // T64: at tp 2 the elastic expert pool is VMM memory, which a cross-device pull
            // (ops::allreduce pull_peer) cannot read without P2P; the lent workspace lives there.
            // The eager prefill all-reduce goes through the host-staged exchange (T56 C1), so the
            // lending stays on unless the peer pull is forced back (T68).
            !(impl->elastic_kv.enabled && impl->tp == 2 && peer_pull)) {
            const auto text_layers = static_cast<std::int32_t>(text.num_hidden_layers);
            // The cache's layers and their base slots as the backend lays them out (the MTP
            // layer is one more cache layer, Qwen4ExpCache::cache_layers).
            const bool mtp_layer =
                impl->speculative_backend == SpeculativeBackend::Mtp && impl->parameters->qwen4_mtp;
            const std::vector<std::int32_t> layer_base =
                execution::Qwen4OffloadExperts::layer_slot_profile(
                    text_layers + (mtp_layer ? 1 : 0), text_layers,
                    static_cast<std::int32_t>(experts.slots_per_layer),
                    static_cast<std::int32_t>(experts.elastic_columns), impl->tp);
            std::int32_t ring_layers =
                gguf_lend ? 0
                          : execution::Qwen4OffloadExperts::ring_layers(
                                static_cast<std::int32_t>(
                                    std::max<std::uint32_t>(experts.stream_ring_slots, 512)),
                                layer_base, text_layers, impl->tp);
            std::vector<std::int64_t> gguf_bytes;
            if (gguf_lend) {
                for (const auto& b : gguf_list) {
                    gguf_bytes.push_back(execution::Qwen4OffloadExperts::gguf_slot_bytes(b));
                }
                // The prefill ring on the last Text layers (as the backend lends it).
                ring_layers = execution::Qwen4OffloadExperts::gguf_layers_below(
                    layer_base, gguf_bytes, text_layers,
                    execution::Qwen4OffloadExperts::gguf_ring_bytes(gguf_bytes, text_layers,
                                                                    impl->tp));
            }
            // The workspace is the lent layers' slots of the gate/up code plane (one contiguous
            // region of the planar pool), the layers just below the ring.
            constexpr auto lend_bytes = execution::Qwen4OffloadExperts::kLendPlaneBytes;
            const auto need_for       = [&](std::uint32_t width) {
                SequencePlanImpl wide    = *impl;
                wide.prefill_chunk       = width;
                const WorkspacePlan plan = build_workspace_plan(wide);
                // The phase plans assume the chunk's carried hidden lives in prefill_hidden; on
                // lent workspace it is an arena matrix [S*H, width] BF16 too (qwen4_text.cpp
                // chunk_hidden), plus alignment slack.
                const std::size_t hidden_bytes =
                    static_cast<std::size_t>(text.residual_width()) * width * sizeof(std::uint16_t);
                return std::max(plan.text_prefill, plan.mtp_prefill) + hidden_bytes +
                       (std::size_t(64) << 20);
            };
            for (std::uint32_t chunk = experts.prompt_chunk; chunk > base; chunk /= 2) {
                const std::uint32_t width = std::min(chunk, impl->capacity);
                if (width <= base || width % kPrefillChunkAlignment != 0) { continue; }
                const std::size_t need = need_for(width);
                const std::int32_t layers =
                    gguf_lend
                        ? execution::Qwen4OffloadExperts::gguf_layers_below(
                              layer_base, gguf_bytes, text_layers - ring_layers,
                              static_cast<std::int64_t>(need))
                        : execution::Qwen4OffloadExperts::layers_below(
                              layer_base, text_layers - ring_layers,
                              static_cast<std::int64_t>((need + lend_bytes - 1) / lend_bytes));
                if (execution::Qwen4OffloadExperts::may_lend(layer_base, text_layers,
                                                             ring_layers + layers,
                                                             static_cast<std::int32_t>(impl->tp))) {
                    impl->qwen4_base_chunk = base;
                    impl->prefill_chunk    = width;
                    prompt_workspace       = need;
                    // T69: the narrower widths' needs (a chunk lends only what its width needs).
                    for (std::uint32_t w = width / 2; w > base; w /= 2) {
                        if (w % kPrefillChunkAlignment != 0) { break; }
                        prompt_steps.emplace_back(w, need_for(w));
                    }
                    std::reverse(prompt_steps.begin(), prompt_steps.end());
                    prompt_steps.emplace_back(width, need);
                    break;
                }
            }
        }
    }
    impl->persistent = persistent_layout(*impl, 0);
    if (impl->tp != 1) { impl->peer_persistent = persistent_layout(*impl, 1); }
    if (impl->qwen4_base_chunk > 0) {
        SequencePlanImpl narrow            = *impl;
        narrow.prefill_chunk               = impl->qwen4_base_chunk;
        impl->workspace                    = build_workspace_plan(narrow);
        impl->workspace.qwen4_prompt       = prompt_workspace;
        impl->workspace.qwen4_prompt_steps = std::move(prompt_steps);
    } else {
        impl->workspace = build_workspace_plan(*impl);
    }
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            // Per device; tp 2 uses its measured allowance (core/tp2/device_tuning.h).
            const std::size_t per_batch =
                impl->tp == 1 ? 12ULL * kMiB
                              : tp2::device_tuning(impl->device).tp2_ordinary_graph_allowance_bytes;
            impl->graph_allowance_bytes =
                checked_mul(per_batch, impl->max_concurrency, "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            // Per device; tp 2 uses its measured class allowance (core/tp2/device_tuning.h).
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) -> std::size_t {
                    if (impl->tp != 1) {
                        return tp2::device_tuning(impl->device).tp2_mtp_graph_class_allowance_bytes;
                    }
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto profiles = dflash_graph_profiles(impl->capacity, impl->draft_window);
            const auto per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) -> std::size_t {
                    // Per device; tp 2 uses its measured class allowance
                    // (core/tp2/device_tuning.h).
                    if (impl->tp != 1) {
                        return tp2::device_tuning(impl->device)
                            .tp2_dflash2_graph_class_allowance_bytes;
                    }
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                    return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                },
                "DFlash graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "DFlash exact-b graph allowance");
        }
    }

    // Rank 0's layout bounds rank 1's, which only omits the drafter's state; both ranks are
    // budgeted for it. With Vision the budget carries the encoding rank's workspace; the other
    // rank's budget is credited with the part it does not allocate.
    impl->device_reservation_bytes =
        checked_add(checked_add(impl->persistent.bytes + impl->persistent.elastic_bytes,
                                impl->workspace.capacity, "sequence memory plan"),
                    impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters,
                           const execution::Parameters* peer_parameters, DeviceContext& device,
                           const EngineOptions& options) {
    validate_target_options(parameters, peer_parameters, device, options);
    if (options.tp == 1 && peer_parameters != nullptr) {
        throw std::invalid_argument("single-device planning takes no rank 1 Parameters");
    }
    SequencePlanningInputs inputs{
        .parameters             = &parameters,
        .vision_parameters      = models::vision_rank(options) == 1 ? peer_parameters : &parameters,
        .peer_parameters        = peer_parameters,
        .max_vision_item_tokens = options.max_vision_tokens.value_or(kMaximumMaxVisionTokens),
        .capacity               = options.max_context,
        .max_concurrency        = options.max_concurrency,
        .prefill_chunk          = std::min(options.prefill_chunk, options.max_context),
        .kv_lease_window_tokens = options.kv_lease_window_tokens,
        .draft_window           = options.speculative.draft_tokens,
        .mtp_min_p              = options.speculative.backend == SpeculativeBackend::Mtp
                                      ? options.speculative.min_draft_probability
                                      : 0.0F,
        .mtp_window_rule        = options.speculative.window_rule,
        .speculative_backend    = options.speculative.backend,
        .kv_storage             = options.kv_cache,
        .proposal_head          = options.speculative.proposal_head,
        .features               = models::load_options(options),
        .use_cuda_graph         = options.use_cuda_graph,
        .causal_scoring         = options.purpose == EnginePurpose::CausalScoring,
        .device                 = options.device,
        .tp                     = options.tp,
        .tp_mailbox             = options.tp_mailbox,
        .context_cache          = options.context_cache,
    };
    // T43: the draft shortlist (Qwen4Exp MTP): read and split here, the rows gathered by the
    // Program. At tp 2 the prefix view is bounded by rank 0's vocabulary half.
    // T55: with no shortlist given, a Qwen4Exp MTP engine uses the sidecar file shipped next to the
    // artifact (<artifact stem>.draft_shortlist.txt) when it exists and the head supports it;
    // "none" asks for full-vocabulary proposals explicitly.
    std::string shortlist_path = options.speculative.draft_shortlist;
    if (shortlist_path.empty() && parameters.model.config().text.qwen4() &&
        options.speculative.backend == SpeculativeBackend::Mtp &&
        execution::draft_head_supported(parameters.text.output_head.weight)) {
        std::filesystem::path sidecar = options.artifact_path;
        sidecar.replace_extension(".draft_shortlist.txt");
        std::error_code ec;
        if (!options.artifact_path.empty() && std::filesystem::is_regular_file(sidecar, ec)) {
            shortlist_path = sidecar.string();
            std::fprintf(stderr, "qwen4: draft shortlist (artifact sidecar) %s\n",
                         shortlist_path.c_str());
        }
    }
    if (shortlist_path == "none") { shortlist_path.clear(); }
    if (!shortlist_path.empty()) {
        const auto& head = parameters.text.output_head.weight;
        if (!parameters.model.config().text.qwen4() ||
            options.speculative.backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument("a draft shortlist needs Qwen4Exp with MTP");
        }
        if (!execution::draft_head_supported(head)) {
            throw std::invalid_argument(
                "a draft shortlist needs a Q6 RowSplit or BF16 output head");
        }
        const auto ids = execution::read_draft_shortlist(shortlist_path.c_str(),
                                                         options.speculative.draft_shortlist_rows);
        // tp 2: rows in a multiple of 16 so each rank's half is a multiple of 8; the shortlist is
        // split over the ranks when rank 0's half (rows [0, R)) lies in the prefix view
        // (NINFER_QWEN4_DRAFT_SPLIT=0: rank 0 alone, for comparison).
        const bool tp2 = options.tp == 2;
        auto split     = execution::split_draft_shortlist(
            ids, dimension(parameters.model.resources().public_token_count), head.n, tp2 ? 16 : 8);
        const auto rows = split.prefix + static_cast<std::int32_t>(split.extras.size());
        const char* env = std::getenv("NINFER_QWEN4_DRAFT_SPLIT");
        inputs.draft_split =
            tp2 && rows / 2 <= split.prefix && !(env != nullptr && std::string(env) == "0");
        inputs.draft_prefix = split.prefix;
        inputs.draft_extras = std::move(split.extras);
    }
    if (parameters.model.config().text.qwen4()) {
        const RoutedExpertOptions& experts = options.routed_experts;
        const MissSplitOptions& split      = experts.miss_split;
        const MissSplitCosts& cost         = split.costs;
        const auto nonneg                  = [](float v) { return std::isfinite(v) && v >= 0.0F; };
        if (experts.slots_per_layer > 512 || experts.cpu_threads > 64 ||
            experts.admit_threshold < 0.0F || !(experts.half_life_tokens > 0.0F) ||
            !(experts.evict_tau_windows >= 0.0F) || !std::isfinite(experts.evict_tau_windows) ||
            experts.device_reads > 16 || split.count > 16 || split.max_reads > 16 ||
            !(split.fraction >= 0.0F && split.fraction <= 1.0F) ||
            !(cost.cpu_read_slowdown >= 1.0F) || !nonneg(cost.cpu_fixed_us) ||
            !nonneg(cost.cpu_expert_us) || !nonneg(cost.cpu_column_us) ||
            !nonneg(cost.gpu_fixed_us) || !nonneg(cost.gpu_expert_us) ||
            !nonneg(cost.gpu_column_us) || experts.prefetch_slots > 64 ||
            !(experts.prefetch_min_score >= 0.0F) || experts.prefetch_first_layer == 0 ||
            !(experts.prefetch_link_gbps > 0.0F)) {
            throw std::invalid_argument(
                "Engine routed_experts: slots_per_layer must be in [0,512], "
                "cpu_threads <= 64 (0 = auto), admit_threshold >= 0, half_life > 0, "
                "evict_tau_windows >= 0, "
                "device_reads, miss_split count/max_reads <= 16, "
                "miss_split fraction in [0,1], costs >= 0, slowdown >= 1, "
                "prefetch_slots <= 64, prefetch_min_score >= 0, "
                "prefetch_first_layer >= 1, prefetch_link_gbps > 0");
        }
        // Warm start is on by default (T25/T42) and only means something to the offload backend:
        // other backends ignore it. Prefetch must be asked for, and runs at one device only.
        if (experts.prefetch_slots > 0 &&
            (experts.backend != RoutedExpertBackend::Offload || options.tp != 1)) {
            throw std::invalid_argument("Engine routed_experts: prefetch needs the offload backend "
                                        "at one device");
        }
        inputs.qwen4_experts = experts;
        if (options.elastic_kv.enabled) {
            if (!parameters.model.config().text.qwen4_routed_experts()) {
                throw std::invalid_argument("Engine elastic_kv: trades KV against offloaded expert "
                                            "slots; a dense Qwen4Exp FFN has none");
            }
            if (experts.backend != RoutedExpertBackend::Offload ||
                options.elastic_kv.grow_tokens == 0) {
                throw std::invalid_argument("Engine elastic_kv: needs the offload expert backend "
                                            "and grow_tokens > 0");
            }
            inputs.elastic_kv = options.elastic_kv;
        }
        if (options.kv_ram_active) {
            if (!parameters.model.config().text.sparse_attention) {
                throw std::invalid_argument("Engine kv_ram_active: stages the QSA selection; a "
                                            "Qwen4Exp without sparse attention has none");
            }
            inputs.kv_ram_active = true;
        }
    } else if (options.elastic_kv.enabled) {
        throw std::invalid_argument("Engine elastic_kv: Qwen4Exp (offloaded experts) only");
    } else if (options.kv_ram_active) {
        throw std::invalid_argument("Engine kv_ram_active: Qwen4Exp only");
    }
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_5::detail::SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.5 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.5 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.5 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

std::array<std::byte*, 4> Qwen4PersistentLayout::bind_pool(DeviceSpan backing) const {
    std::array<std::byte*, 4> planes{};
    for (std::size_t q = 0; q < planes.size(); ++q) {
        planes[q] = static_cast<std::byte*>(cache_pool[q].bind(backing).data);
    }
    return planes;
}

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5 {

std::uint64_t active_kv_host_pinned_bytes(const Config& config, const EngineOptions& options) {
    const TextConfig& text = config.text;
    if (!options.kv_ram_active || !text.qwen4() || !text.sparse_attention || !text.attention) {
        return 0;
    }
    // The text pool's physical pages: the explicit kv_capacity, or the most an automatic one can
    // reach (max_concurrency x max_context). The planes take this much address space; their RAM
    // follows the pages in use, but the weights leave room for all of it in the page-locked limit.
    const std::uint32_t logical_pages = detail::page_count(options.max_context);
    const std::uint64_t pages64 =
        options.kv_capacity.mode == KvCapacityMode::Explicit
            ? std::max<std::uint64_t>(detail::page_count(options.kv_capacity.explicit_tokens),
                                      logical_pages)
            : static_cast<std::uint64_t>(std::max(options.max_concurrency, 1U)) * logical_pages;
    if (pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("active KV in RAM: the text pool's page count exceeds uint32");
    }
    // The MTP pool (with MTP): the same pages plus each lane's draft window (persistent_layout).
    const bool mtp            = options.speculative.backend == SpeculativeBackend::Mtp;
    const std::uint64_t lanes = std::max(options.max_concurrency, 1U);
    const std::uint64_t mtp_pages64 =
        pages64 + (mtp ? lanes * ((static_cast<std::uint64_t>(options.speculative.draft_tokens) +
                                   static_cast<std::uint64_t>(kPagedKVPageSize) - 2U) /
                                  static_cast<std::uint64_t>(kPagedKVPageSize))
                       : 0U);
    if (mtp_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("active KV in RAM: the MTP pool's page count exceeds uint32");
    }
    const TextConfig rank        = execution::shard_text_config(text, options.tp);
    const std::uint64_t per_rank = active_kv_host_plane_bytes(DecoderStateSpec{
        .full_attention_layers     = text.full_attention_layers,
        .mtp_layers                = 1,
        .capacity                  = options.max_context,
        .kv_heads                  = execution::dimension(rank.attention->num_key_value_heads),
        .attention_head_dim        = execution::dimension(text.attention->head_dim),
        .kv_storage                = options.kv_cache,
        .enable_mtp                = mtp,
        .kv_table_rows             = static_cast<std::int32_t>(lanes),
        .text_physical_page_groups = static_cast<std::uint32_t>(pages64),
        .mtp_physical_page_groups  = static_cast<std::uint32_t>(mtp_pages64),
        .qsa_index                 = true,
        .kv_ram_active             = true,
        .staging_selection_groups  = detail::qsa_selection_groups(text),
    });
    return per_rank * static_cast<std::uint64_t>(options.tp);
}

} // namespace ninfer::models::qwen3_5

#include "models/qwen3_5/program/tp2/startup_tp2.inc"
