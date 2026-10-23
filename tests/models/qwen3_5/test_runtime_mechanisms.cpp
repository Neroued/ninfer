#include "models/common/qwen/vision/prefill.h"
#include "core/layout.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/common/qwen/vision/control.h"

#include "models/common/qwen/prefix_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q35 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q35::DecoderStateSpec decoder_spec(ninfer::KvCacheStorage storage, bool mtp) {
    return q35::DecoderStateSpec{
        .full_attention_layers = 2,
        .mtp_layers            = 1,
        .capacity              = 129,
        .kv_heads              = 2,
        .attention_head_dim    = 256,
        .kv_storage            = storage,
        .enable_mtp            = mtp,
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q35::DecoderStateLayout bf16 = q35::plan_decoder_state(
        bf16_builder, decoder_spec(ninfer::KvCacheStorage::BFloat16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pages.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.execution_tables.spec.logical_page_capacity == 3 &&
               bf16.text_kv.execution_tables.spec.table_rows == 1,
           "Text KV execution table covers a partial final logical page");
    expect(bf16.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::BF16 &&
               bf16.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::FP16 &&
               bf16.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::BF16 &&
               bf16.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "BF16 KV has BF16 K and FP16 V without scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.text_kv.pages.payload_bytes() == 2ULL * 2 * 256 * 64 * 2 * 2,
           "BF16 page record contains K/V for every layer and head");

    auto probe_spec          = decoder_spec(ninfer::KvCacheStorage::BFloat16, true);
    probe_spec.capacity      = 262144;
    probe_spec.kv_table_rows = 8;
    ninfer::LayoutBuilder probe_builder;
    const auto probe = q35::plan_decoder_state(probe_builder, probe_spec);
    expect(probe.text_kv.execution_tables.spec.logical_page_capacity == 4096 &&
               probe.text_kv.execution_tables.spec.table_rows == 8 &&
               probe.text_kv.pages.page_bytes == bf16.text_kv.pages.page_bytes &&
               probe.mtp_kv->execution_tables.spec.logical_page_capacity == 4096,
           "logical context and concurrency enlarge tables without changing page records");

    ninfer::LayoutBuilder int8_builder;
    const q35::DecoderStateLayout int8 = q35::plan_decoder_state(
        int8_builder, decoder_spec(ninfer::KvCacheStorage::Int8Group64, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pages.planes.size() == 8 &&
               int8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pages.planes.size() == 4 &&
               int8.mtp_kv->execution_tables.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.text_kv.pages.payload_bytes() == 2ULL * (256 + 4 * 2) * 2 * 64 * 2 &&
               int8.mtp_kv->pages.payload_bytes() == 2ULL * (256 + 4 * 2) * 64 * 2,
           "INT8 Text/MTP record payload includes group scales");

    q35::DecoderStateSpec fp8_spec = decoder_spec(ninfer::KvCacheStorage::Fp8E4M3Row256, true);
    ninfer::LayoutBuilder fp8_builder;
    const q35::DecoderStateLayout fp8 = q35::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 1,
           "FP8 Text KV has row-scaled code and scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 1,
           "FP8 MTP KV has row-scaled code and scale planes");
    expect(fp8.text_kv.pages.payload_bytes() == 2ULL * (256 + 2) * 2 * 64 * 2 &&
               fp8.mtp_kv->pages.payload_bytes() == 2ULL * (256 + 2) * 64 * 2,
           "FP8 Text/MTP record payload includes row scales");

    ninfer::LayoutBuilder nvfp4_builder;
    const q35::DecoderStateLayout nvfp4 = q35::plan_decoder_state(
        nvfp4_builder, decoder_spec(ninfer::KvCacheStorage::Nvfp4Group16, true));
    (void)nvfp4_builder.finish(256);
    expect(nvfp4.text_kv.pages.planes.size() == 8 &&
               nvfp4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               nvfp4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[2].geometry.leading_extent == 16,
           "NVFP4 Text KV has packed E2M1 code and E4M3 scale planes");
    constexpr std::size_t nvfp4_vector_bytes = 128 + 16;
    constexpr std::size_t text_vectors       = 2ULL * 64 * 2;
    constexpr std::size_t mtp_vectors        = 64ULL * 2;
    expect(nvfp4.text_kv.pages.payload_bytes() == 2ULL * nvfp4_vector_bytes * text_vectors &&
               nvfp4.mtp_kv &&
               nvfp4.mtp_kv->pages.payload_bytes() == 2ULL * nvfp4_vector_bytes * mtp_vectors,
           "NVFP4 Text/MTP page record payload bytes");

    ninfer::LayoutBuilder k8v4_builder;
    const q35::DecoderStateLayout k8v4 = q35::plan_decoder_state(
        k8v4_builder, decoder_spec(ninfer::KvCacheStorage::Fp8KeyNvfp4Value, true));
    (void)k8v4_builder.finish(256);
    expect(k8v4.text_kv.pages.planes.size() == 8 &&
               k8v4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               k8v4.text_kv.pages.planes[0].geometry.leading_extent == 256 &&
               k8v4.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               k8v4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               k8v4.text_kv.pages.planes[2].geometry.leading_extent == 1 &&
               k8v4.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[3].geometry.leading_extent == 16,
           "K8V4 Text KV preserves independent key/value code and scale geometry");
    constexpr std::size_t k8_vector_bytes = 256 + 2;
    constexpr std::size_t v4_vector_bytes = 128 + 16;
    expect(
        k8v4.text_kv.pages.payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * text_vectors &&
            k8v4.mtp_kv &&
            k8v4.mtp_kv->pages.payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * mtp_vectors,
        "K8V4 Text/MTP asymmetric page record payload bytes");
}

void test_visual_embedding_windows() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const auto first = ninfer::models::qwen::visual_embedding_slice(scatter, 8, 0, 4);
    expect(first.source_begin == 0 && first.destination_columns == std::vector<std::int32_t>({2}),
           "first input window selects only its visual columns");
    const auto final = ninfer::models::qwen::visual_embedding_slice(scatter, 8, 4, 4);
    expect(final.source_begin == 1 &&
               final.destination_columns == std::vector<std::int32_t>({0, 3}),
           "last input window retains visual columns at both boundaries");
    const auto shifted = ninfer::models::qwen::visual_embedding_slice(scatter, 8, 1, 3);
    expect(shifted.source_begin == 0 &&
               shifted.destination_columns == std::vector<std::int32_t>({1}),
           "root MTP history uses token positions after its skipped first input");
    expect(ninfer::models::qwen::visual_embedding_slice(scatter, 8, 8, 1).empty(),
           "generated input beyond the prompt has no visual embedding");
}

void test_vision_control() {
    ninfer::models::qwen::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types = {0, static_cast<std::uint8_t>(ninfer::models::frontend::Modality::Image),
                          0, static_cast<std::uint8_t>(ninfer::models::frontend::Modality::Video),
                          0, static_cast<std::uint8_t>(ninfer::models::frontend::Modality::Video),
                          0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        {.modality    = ninfer::models::frontend::Modality::Image,
                  .grid        = {.temporal = 1, .height = 2, .width = 2},
                  .patch_begin = 0,
                  .patch_count = 4,
                  .token_spans = {{.begin = 1, .count = 1}}},
        {.modality    = ninfer::models::frontend::Modality::Video,
                  .grid        = {.temporal = 2, .height = 2, .width = 2},
                  .patch_begin = 4,
                  .patch_count = 8,
                  .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const ninfer::models::qwen::VisionControlPlan plan = ninfer::models::qwen::plan_vision_control(
        prompt, {.spatial_merge_size = 2, .position_grid_side = 48});
    const ninfer::models::qwen::VisionControl control =
        ninfer::models::qwen::build_vision_control(prompt, plan, 0);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");

    const ninfer::models::qwen::VisionControl suffix =
        ninfer::models::qwen::build_vision_control(prompt, plan, 1);
    expect(suffix.prepared_item_begin == 1 && suffix.items.size() == 1 &&
               suffix.items[0].patch_begin == control.items[1].patch_begin &&
               suffix.items[0].scatter_indices == control.items[1].scatter_indices &&
               suffix.items[0].position_ids == control.items[1].position_ids,
           "Vision suffix control contents");
}

ninfer::models::qwen::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    ninfer::models::qwen::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(ninfer::models::frontend::Modality::Image),
                          static_cast<std::uint8_t>(ninfer::models::frontend::Modality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    auto item =
        ninfer::models::frontend::VisionItem{.modality = ninfer::models::frontend::Modality::Image,
                                             .grid     = {.temporal = 1, .height = 2, .width = 4},
                                             .patch_begin = 0,
                                             .patch_count = 8,
                                             .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(ninfer::models::qwen::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    ninfer::models::qwen::PreparedPromptData original = identity_prompt();
    std::vector<ninfer::TokenId> ledger               = original.token_ids;
    ninfer::models::qwen::ResidentPrefixIdentity resident;
    ninfer::models::qwen::PrefixShortlistDigests digests;
    resident.reserve(16);
    resident.assign(original);
    digests.reserve(16);
    digests.assign(original);

    expect(
        ninfer::models::qwen::prefix_matches(original, ledger, resident, original.token_ids.size()),
        "identical multimodal prefix identity");

    ninfer::models::qwen::PreparedPromptData changed_media = identity_prompt(2);
    expect(!ninfer::models::qwen::prefix_matches(changed_media, ledger, resident,
                                                 changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(ninfer::models::qwen::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!ninfer::models::qwen::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    ninfer::models::qwen::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!ninfer::models::qwen::prefix_matches(changed_position, ledger, resident,
                                                 changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    ninfer::models::qwen::PreparedPromptData changed_decomposition = identity_prompt();
    changed_decomposition.identity.rewrite_execution_frontiers     = {1};
    expect(!ninfer::models::qwen::prefix_matches(changed_decomposition, ledger, resident,
                                                 changed_decomposition.token_ids.size()),
           "different GDN execution decomposition must not reuse resident state");
    changed_decomposition.identity.rewrite_execution_frontiers = {4};
    expect(ninfer::models::qwen::prefix_matches(changed_decomposition, ledger, resident, 3),
           "execution decomposition wholly after the frontier changed prefix identity");

    ninfer::models::qwen::PreparedPromptData resident_future = identity_prompt();
    resident_future.identity.rewrite_execution_frontiers     = {1, 4};
    ninfer::models::qwen::ResidentPrefixIdentity resident_with_future;
    resident_with_future.assign(resident_future);
    ninfer::models::qwen::PreparedPromptData incoming_future = identity_prompt();
    incoming_future.identity.rewrite_execution_frontiers     = {1, 3};
    expect(ninfer::models::qwen::prefix_matches(incoming_future, ledger, resident_with_future, 1),
           "resident execution decomposition after the frontier changed prefix identity");
    expect(!ninfer::models::qwen::prefix_matches(incoming_future, ledger, resident_with_future, 3),
           "different execution decomposition inside the frontier reused resident state");

    ninfer::models::qwen::PrefixShortlistDigests future_digest;
    future_digest.assign(resident_future);
    ninfer::models::qwen::PrefixShortlistDigests incoming_digest;
    incoming_digest.assign(incoming_future);
    expect(future_digest.at(1) == incoming_digest.at(1),
           "future execution boundaries changed an earlier content shortlist");
    expect(future_digest.at(3) != incoming_digest.at(3),
           "different in-prefix execution boundaries shared a shortlist digest");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    const std::array<ninfer::TokenId, 1> generated{12};
    digests.append_generated(generated, original.rope_delta);
    append_text_token(original, 12, 4);
    ninfer::models::qwen::PrefixShortlistDigests rebuilt;
    rebuilt.assign(original);
    expect(digests.at(ledger.size()) == rebuilt.at(ledger.size()),
           "incremental generated-token shortlist diverged from a full rebuild");
    expect(ninfer::models::qwen::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    ninfer::models::qwen::PreparedPromptData accepted_rebuild = identity_prompt();
    const std::array<ninfer::TokenId, 3> proposed{20, 21, 22};
    const std::span<const ninfer::TokenId> accepted(proposed.data(), 2);
    std::vector<ninfer::TokenId> accepted_ledger = accepted_rebuild.token_ids;
    accepted_ledger.insert(accepted_ledger.end(), accepted.begin(), accepted.end());
    ninfer::models::qwen::ResidentPrefixIdentity accepted_resident;
    ninfer::models::qwen::PrefixShortlistDigests accepted_digests;
    accepted_resident.assign(accepted_rebuild);
    accepted_digests.assign(accepted_rebuild);
    accepted_resident.append_generated(accepted.size(), accepted_rebuild.rope_delta, 1);
    accepted_digests.append_generated(accepted, accepted_rebuild.rope_delta, 1);
    append_text_token(accepted_rebuild, accepted[0], 4);
    append_text_token(accepted_rebuild, accepted[1], 5);
    accepted_rebuild.identity.rewrite_execution_frontiers = {5};
    ninfer::models::qwen::PrefixShortlistDigests rebuilt_accepted_digests;
    rebuilt_accepted_digests.assign(accepted_rebuild);
    expect(ninfer::models::qwen::prefix_matches(accepted_rebuild, accepted_ledger,
                                                accepted_resident, accepted_ledger.size()) &&
               accepted_digests.at(accepted_ledger.size()) ==
                   rebuilt_accepted_digests.at(accepted_ledger.size()),
           "accepted generated prefix and rebuilt history formed different cache identities");

    const ninfer::models::qwen::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    digests.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    ninfer::models::qwen::PrefixShortlistDigests prompt_digest;
    prompt_digest.assign(prompt_only);
    expect(digests.at(ledger.size()) == prompt_digest.at(ledger.size()),
           "truncated shortlist did not restore the original frontier digest");
    expect(ninfer::models::qwen::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_regular_prefix_identity() {
    const auto prompt = [](std::size_t count, std::int32_t origin = 0) {
        ninfer::models::qwen::PreparedPromptData value;
        value.token_ids.assign(count, 12);
        value.token_types.assign(count, 0);
        value.positions.resize(3 * count);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < count; ++i) {
                value.positions[axis * count + i] = origin + static_cast<std::int32_t>(i);
            }
        }
        return value;
    };
    auto original = prompt(128, 7);
    ninfer::models::qwen::ResidentPrefixIdentity left, right;
    left.assign(original);
    right.assign(original);
    expect(left.equals(right), "independently prepared regular position identities differ");
    const auto check_against_input = [&](const auto& value, const auto& identity) {
        // matches reads the stored arrays against the public input, without the prefix shortcut.
        for (std::size_t count = 0; count <= value.token_ids.size(); ++count) {
            expect(left.prefix_equals(identity, count) == left.matches(value, count),
                   "resident identity comparison disagrees with the prepared input");
        }
    };
    for (std::size_t axis = 0; axis < 3; ++axis) {
        auto changed = original;
        changed.positions[axis * 128 + 64] += 1;
        right.assign(changed);
        check_against_input(changed, right);
    }
    auto changed            = original;
    changed.token_types[64] = 1;
    right.assign(changed);
    check_against_input(changed, right);
    changed = prompt(128, 8);
    right.assign(changed);
    check_against_input(changed, right);
    changed                                      = original;
    changed.identity.rewrite_execution_frontiers = {64};
    right.assign(changed);
    check_against_input(changed, right);

    left.append_generated(2, 7);
    right.assign(prompt(130, 7));
    expect(left.equals(right), "regular generated positions differ from a full rebuild");
    left.append_generated(1, 8);
    right.assign(prompt(131, 7));
    expect(left.prefix_equals(right, 130) && !left.equals(right),
           "a changed generated position was ignored");
    left.truncate(128);
    left.append_generated(3, 7);
    expect(left.equals(right), "truncation retained discarded position differences");

    left.truncate(0);
    left.append_generated(3, -4);
    right.assign(prompt(3, -4));
    expect(left.equals(right), "empty truncated identity retained an old position origin");
    ninfer::models::qwen::ResidentPrefixIdentity saved = left;
    right.assign(prompt(3, 11));
    left.swap(right);
    expect(right.equals(saved) && !left.equals(saved), "swap lost the exact position identity");
    left.clear();
    left.append_generated(3, -4);
    expect(left.equals(saved), "cleared identity retained its prior position metadata");
    auto moved = std::move(left);
    left.append_generated(1, 0);
    left.append_generated(1, 1);
    right.assign(prompt(2));
    expect(moved.equals(saved) && left.prefix_equals(right, 1) && !left.equals(right),
           "reusing a moved-from identity ignored a new position difference");

    auto vision       = identity_prompt();
    auto other_vision = identity_prompt(2);
    left.assign(vision);
    right.assign(other_vision);
    check_against_input(other_vision, right);
}

} // namespace

int main() {
    test_decoder_layout();
    test_visual_embedding_windows();
    test_vision_control();
    test_prefix_identity();
    test_regular_prefix_identity();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.5 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.5 runtime mechanism checks passed\n";
    return 0;
}
