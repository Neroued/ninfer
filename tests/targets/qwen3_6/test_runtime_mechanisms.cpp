#include "core/layout.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/mtp_alignment.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_topology() {
    static_assert(q36::kHybridAttentionInterval == 4);
    static_assert(q36::full_attention_layers(64) == 16);
    static_assert(q36::gdn_layers(64) == 48);
    for (std::int32_t layer = 0; layer < 64; ++layer) {
        expect(q36::is_full_attention_layer(layer) == ((layer + 1) % 4 == 0), "hybrid layer kind");
        if (q36::is_full_attention_layer(layer)) {
            expect(q36::full_attention_index(layer) == layer / 4, "full-attention index");
        } else {
            expect(q36::gdn_index(layer) == layer - layer / 4, "GDN index");
        }
    }
}

q36::DecoderStateSpec decoder_spec(ninfer::DType dtype, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers = 2,
        .mtp_layers            = 1,
        .capacity              = 129,
        .kv_heads              = 2,
        .attention_head_dim    = 64,
        .kv_dtype              = dtype,
        .kv_quant_group        = dtype == ninfer::DType::I8 ? ninfer::kKvQuantGroup : 0,
        .enable_mtp            = mtp,
        .gdn =
            {
                .layers         = 3,
                .conv_dim       = 10,
                .conv_width     = 3,
                .value_heads    = 4,
                .value_head_dim = 5,
                .key_head_dim   = 6,
                .snapshot_slots = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 =
        q36::plan_decoder_state(bf16_builder, decoder_spec(ninfer::DType::BF16, false));
    const std::size_t bf16_bytes = bf16_builder.finish(256);
    expect(bf16_bytes != 0, "BF16 decoder layout has storage");
    expect(bf16.text_kv.k.size() == 2 && bf16.text_kv.v.size() == 2, "Text KV layer planes");
    expect(bf16.text_kv.padded_context == 256, "Text KV capacity padding");
    expect(bf16.text_kv.k_scale.empty() && bf16.text_kv.v_scale.empty(),
           "BF16 KV has no scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.gdn.conv.size() == 3 && bf16.gdn.ssm.size() == 3, "GDN layer storage");
    expect(bf16.gdn.spec.snapshot_slots == 4, "GDN snapshot geometry");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 =
        q36::plan_decoder_state(int8_builder, decoder_spec(ninfer::DType::I8, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.k_scale.size() == 2 && int8.text_kv.v_scale.size() == 2,
           "INT8 Text KV scale planes");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->k.size() == 1, "enabled MTP has one KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->k_scale.size() == 1 && int8.mtp_kv->v_scale.size() == 1,
           "INT8 MTP KV scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");
}

void test_round_layout() {
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{
                     .hidden = 32, .output_rows = 128, .draft_window = 5, .stats_counters = 9});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    const std::size_t bytes = builder.finish(256);
    expect(round.complete && bytes != 0, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 6, "round logits shape");
    expect(round.verify_hidden.shape[0] == 32 && round.verify_hidden.shape[1] == 6,
           "round verification hidden shape");
    expect(round.drafts.shape[0] == 5 && round.sampled_out.shape[0] == 6,
           "round speculative vector shapes");
    expect(round.verify_hidden.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.target_tokens.region.offset,
           "exact prefill extension retains established round-region order");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControl control = q36::build_vision_control(prompt);
    expect(control.cu_seqlens == std::vector<std::int32_t>({0, 4, 8, 12}), "Vision segment bounds");
    expect(control.scatter_indices == std::vector<std::int32_t>({1, 3, 5}), "Vision scatter order");
    expect(control.position_ids.size() == 24 && control.position_table_indices.size() == 48 &&
               control.position_table_weights.size() == 48,
           "Vision position-control sizes");
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_begin == 0 && control.items[0].merged_count == 1 &&
               control.items[0].segment_length == 4 && control.items[0].segment_count == 1 &&
               control.items[0].scatter_begin == 0 && control.items[0].scatter_count == 1,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_begin == 1 && control.items[1].merged_count == 2 &&
               control.items[1].segment_length == 4 && control.items[1].segment_count == 2 &&
               control.items[1].scatter_begin == 1 && control.items[1].scatter_count == 2,
           "video item control offsets");
}

} // namespace

int main() {
    test_topology();
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}
