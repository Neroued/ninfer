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
    if (spec.enable_mtp && spec.draft_window == 0) {
        throw std::invalid_argument("RoundState cannot enable MTP with an empty draft window");
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
    const std::int32_t stats_counters =
        checked_i32(4ULL + layout.spec.draft_window, "RoundState stats exceed int32");
    const auto i32 = [&](std::int32_t count, const char* label) {
        return add_tensor(builder, DType::I32, {count}, label);
    };
    layout.gdn_initial_slot             = i32(1, "step GDN initial slot");
    layout.speculative.target_argmax    = i32(columns, "step target argmax");
    layout.speculative.draft_tokens     = i32(drafts, "step draft tokens");
    layout.speculative.round_tokens     = i32(columns, "step speculative round tokens");
    layout.speculative.produced_count   = i32(1, "step speculative produced count");
    layout.speculative.target_input_ids = i32(columns, "step target input ids");
    layout.speculative.target_positions = i32(columns, "step target positions");
    layout.speculative.accepted_drafts  = i32(1, "step accepted drafts");
    layout.speculative.stats =
        add_tensor(builder, DType::I64, {stats_counters}, "step speculative stats");
    if (layout.spec.enable_mtp) {
        layout.mtp.emplace();
        layout.mtp->alignment_ids = i32(columns, "step MTP alignment ids");
        layout.mtp->position      = i32(1, "step MTP autoregressive position");
        layout.mtp->ar_hidden     = add_tensor(builder, DType::BF16, {layout.spec.hidden, 1},
                                               "step MTP autoregressive hidden");
    }
    layout.complete = true;
}

SpeculativeRoundState::SpeculativeRoundState(DeviceSpan backing,
                                             const SpeculativeRoundStateLayout& layout)
    : target_argmax(layout.target_argmax.bind(backing)),
      draft_tokens(layout.draft_tokens.bind(backing)),
      round_tokens(layout.round_tokens.bind(backing)),
      produced_count(layout.produced_count.bind(backing)),
      target_input_ids(layout.target_input_ids.bind(backing)),
      target_positions(layout.target_positions.bind(backing)),
      accepted_drafts(layout.accepted_drafts.bind(backing)), stats(layout.stats.bind(backing)) {}

MtpRoundState::MtpRoundState(DeviceSpan backing, const MtpRoundStateLayout& layout)
    : alignment_ids(layout.alignment_ids.bind(backing)), position(layout.position.bind(backing)),
      ar_hidden(layout.ar_hidden.bind(backing)) {}

RoundState::RoundState(DeviceSpan backing, const RoundStateLayout& layout) {
    if (!layout.complete) { throw std::invalid_argument("RoundState layout is incomplete"); }
    token            = layout.token.bind(backing);
    pos              = layout.pos.bind(backing);
    rope_pos         = layout.rope_pos.bind(backing);
    rope_delta       = layout.rope_delta.bind(backing);
    logits           = layout.logits.bind(backing);
    verify_hidden    = layout.verify_hidden.bind(backing);
    gdn_initial_slot = layout.gdn_initial_slot.bind(backing);
    speculative      = SpeculativeRoundState(backing, layout.speculative);
    if (layout.mtp) { mtp.emplace(backing, *layout.mtp); }
}

} // namespace ninfer::targets::qwen3_6
