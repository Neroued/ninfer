#include <ninfer/targets/qwen3_6/round_state.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr std::size_t kArenaAlign = 256;

TensorRegion add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(label);
    }
    return static_cast<std::int32_t>(value);
}

void validate_spec(const RoundStateSpec& spec) {
    if (spec.hidden <= 0) { throw std::invalid_argument("RoundState hidden must be positive"); }
    if (spec.output_rows <= 0) {
        throw std::invalid_argument("RoundState output_rows must be positive");
    }
    if (spec.stats_counters <= 0) {
        throw std::invalid_argument("RoundState stats_counters must be positive");
    }
    (void)checked_i32(static_cast<std::uint64_t>(spec.draft_window) + 1ULL,
                      "RoundState draft window exceeds int32");
}

} // namespace

RoundStateLayout begin_round_state_layout(LayoutBuilder& builder, const RoundStateSpec& spec) {
    validate_spec(spec);
    const std::int32_t columns = checked_i32(static_cast<std::uint64_t>(spec.draft_window) + 1ULL,
                                             "RoundState columns exceed int32");
    RoundStateLayout layout;
    layout.spec       = spec;
    layout.token      = add_tensor(builder, DType::I32, {1}, "step token");
    layout.pos        = add_tensor(builder, DType::I32, {1}, "step position");
    layout.rope_pos   = add_tensor(builder, DType::I32, {1}, "step rope position");
    layout.rope_delta = add_tensor(builder, DType::I32, {1}, "step rope delta");
    layout.logits = add_tensor(builder, DType::BF16, {spec.output_rows, columns}, "step logits");
    layout.verify_hidden =
        add_tensor(builder, DType::BF16, {spec.hidden, columns}, "step verify hidden");
    return layout;
}

void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout) {
    if (layout.complete) { throw std::logic_error("RoundState layout is already complete"); }
    validate_spec(layout.spec);
    const std::int32_t columns =
        checked_i32(static_cast<std::uint64_t>(layout.spec.draft_window) + 1ULL,
                    "RoundState columns exceed int32");
    const std::int32_t drafts = checked_i32(std::max<std::uint64_t>(1ULL, layout.spec.draft_window),
                                            "RoundState drafts exceed int32");
    const auto i32            = [&](std::int32_t count, const char* label) {
        return add_tensor(builder, DType::I32, {count}, label);
    };
    layout.target_tokens    = i32(columns, "step target tokens");
    layout.drafts           = i32(drafts, "step drafts");
    layout.sampled_out      = i32(columns, "step sampled output");
    layout.num_sampled      = i32(1, "step sampled count");
    layout.verify_ids       = i32(columns, "step verify ids");
    layout.shifted_ids      = i32(columns, "step shifted ids");
    layout.positions        = i32(columns, "step positions");
    layout.window_base      = i32(1, "step window base");
    layout.accepted         = i32(1, "step accepted drafts");
    layout.gdn_initial_slot = i32(1, "step GDN initial slot");
    layout.ar_pos           = i32(1, "step MTP autoregressive position");
    layout.mtp_ar_hidden =
        add_tensor(builder, DType::BF16, {layout.spec.hidden, 1}, "step MTP autoregressive hidden");
    layout.stats    = add_tensor(builder, DType::I64, {layout.spec.stats_counters}, "step stats");
    layout.complete = true;
}

RoundState::RoundState(DeviceSpan backing, const RoundStateLayout& layout) {
    if (!layout.complete) { throw std::invalid_argument("RoundState layout is incomplete"); }
    token            = layout.token.bind(backing);
    pos              = layout.pos.bind(backing);
    rope_pos         = layout.rope_pos.bind(backing);
    rope_delta       = layout.rope_delta.bind(backing);
    logits           = layout.logits.bind(backing);
    verify_hidden    = layout.verify_hidden.bind(backing);
    target_tokens    = layout.target_tokens.bind(backing);
    drafts           = layout.drafts.bind(backing);
    sampled_out      = layout.sampled_out.bind(backing);
    num_sampled      = layout.num_sampled.bind(backing);
    verify_ids       = layout.verify_ids.bind(backing);
    shifted_ids      = layout.shifted_ids.bind(backing);
    positions        = layout.positions.bind(backing);
    window_base      = layout.window_base.bind(backing);
    accepted         = layout.accepted.bind(backing);
    gdn_initial_slot = layout.gdn_initial_slot.bind(backing);
    ar_pos           = layout.ar_pos.bind(backing);
    mtp_ar_hidden    = layout.mtp_ar_hidden.bind(backing);
    stats            = layout.stats.bind(backing);
}

} // namespace ninfer::targets::qwen3_6
