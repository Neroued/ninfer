#include "models/common/vision/window.h"
#include "core/layout.h"
#include "models/qwen3_5/program/frames.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/common/qwen/vision/control.h"

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

    q35::DecoderStateSpec fp8_spec =
        decoder_spec(ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, true);
    ninfer::LayoutBuilder fp8_builder;
    const q35::DecoderStateLayout fp8 = q35::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::U8 &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 2,
           "FP8 Text KV has G128 code and UE8M0 scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 2,
           "FP8 MTP KV has G128 code and UE8M0 scale planes");
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
               k8v4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[2].geometry.leading_extent == 2 &&
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

void test_frame_layout() {
    constexpr std::uint32_t capacity = 128;
    constexpr std::array rows{1U, 9U, 10U, 64U, capacity};
    constexpr std::array backends{ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp,
                                  ninfer::SpeculativeBackend::DFlash2};
    const auto check_regions = [](std::vector<ninfer::LayoutRegion> regions, std::size_t bytes,
                                  std::size_t maximum) {
        expect(bytes <= maximum, "compact controls exceed their maximum backing");
        std::sort(regions.begin(), regions.end(),
                  [](const auto& a, const auto& b) { return a.offset < b.offset; });
        std::size_t end = 0;
        for (const auto& region : regions) {
            expect(region.bytes && region.offset >= end && region.offset % region.alignment == 0 &&
                       region.offset + region.bytes <= bytes,
                   "control fields overlap or exceed their aligned slab");
            end = region.offset + region.bytes;
        }
    };
    for (const auto backend : backends) {
        const auto drafts = backend == ninfer::SpeculativeBackend::Mtp       ? 3U
                            : backend == ninfer::SpeculativeBackend::DFlash2 ? 15U
                                                                             : 0U;
        const q35::FrameSpec spec{.hidden         = 16,
                                  .output_rows    = 32,
                                  .public_tokens  = 32,
                                  .prefill_chunk  = 128,
                                  .batch_capacity = capacity,
                                  .draft_window   = drafts,
                                  .backend        = backend};
        ninfer::LayoutBuilder builder;
        const auto frame = q35::plan_frames(builder, spec);
        std::array<std::size_t, 3> previous{};
        for (const auto count : rows) {
            const auto control = q35::plan_frame_controls(spec, count);
            const auto& in     = control.ingress;
            const auto& commit = control.commit;
            const auto& out    = control.egress;
            const auto width   = drafts + 1;
            std::vector<ninfer::LayoutRegion> input_regions, commit_regions, output_regions;
            const auto tensor = [&](auto& regions, const ninfer::TensorRegion& value,
                                    ninfer::DType dtype, std::int32_t x, std::int32_t y = 1) {
                expect(value.dtype == dtype &&
                           value.shape == std::array<std::int32_t, 4>{x, y, 1, 1} &&
                           value.region.bytes == std::size_t(x) * y * ninfer::dtype_size(dtype),
                       "control field does not match its physical row geometry");
                regions.push_back(value.region);
            };
            for (const auto* field :
                 {&in.live_rows, &in.anchors, &in.frontiers, &in.backend_frontiers, &in.extents,
                  &in.valid_columns, &in.proposal_valid_columns, &in.proposal_live_rows,
                  &in.main_table_rows, &in.backend_table_rows, &in.residency_slots, &in.rope_deltas,
                  &in.sampling_positions, &in.history_skip})
                tensor(input_regions, *field, ninfer::DType::I32, count);
            for (const auto* field : {&in.state_source_offsets, &in.state_destination_offsets})
                tensor(input_regions, *field, ninfer::DType::I64, count);
            tensor(input_regions, in.previous_rope, ninfer::DType::I32, count, 3);
            tensor(input_regions, in.decode_positions, ninfer::DType::I32, count * width);
            tensor(input_regions, in.decode_rope, ninfer::DType::I32, count * width, 3);
            expect(in.sampling.bytes == count * sizeof(ninfer::ops::SamplingConfig),
                   "sampling controls retain unused resident rows");
            input_regions.push_back(in.sampling);
            if (backend == ninfer::SpeculativeBackend::Mtp) {
                tensor(input_regions, in.step_positions, ninfer::DType::I32, count, drafts);
                tensor(input_regions, in.step_valid, ninfer::DType::I32, count, drafts);
                tensor(input_regions, in.step_rope, ninfer::DType::I32, 3 * count, drafts);
            }
            tensor(commit_regions, commit.state_destination_offsets, ninfer::DType::I64, count);
            for (const auto* field :
                 {&commit.last_columns, &commit.accepted_columns, &commit.count_columns,
                  &commit.draft_append_starts, &commit.draft_append_ends})
                tensor(commit_regions, *field, ninfer::DType::I32, count);
            tensor(output_regions, out.tokens, ninfer::DType::I32, width, count);
            tensor(output_regions, out.counts, ninfer::DType::I32, count);
            check_regions(std::move(input_regions), in.bytes, frame.ingress.bytes);
            check_regions(std::move(commit_regions), commit.bytes, frame.commit.bytes);
            check_regions(std::move(output_regions), out.bytes, frame.egress.bytes);
            const std::array sizes{in.bytes, commit.bytes, out.bytes};
            const std::array maximum{frame.host.ingress.bytes, frame.host.commit.bytes,
                                     frame.host.egress.bytes};
            for (std::size_t i = 0; i < sizes.size(); ++i) {
                expect(sizes[i] >= previous[i] && sizes[i] <= maximum[i],
                       "control transfer size is not monotone or exceeds Host backing");
                if (count == 1)
                    expect(sizes[i] < maximum[i], "one-row transfer includes resident capacity");
            }
            previous = sizes;
        }
        if (frame.mtp) {
            ninfer::HostBuffer backing(builder.finish(256));
            ninfer::HostBuffer ingress(frame.ingress.bytes);
            q35::MtpFrame mtp(backing.span(), *frame.mtp);
            for (const auto count : rows) {
                const auto controls     = q35::plan_frame_controls(spec, count);
                mtp.step_positions      = controls.ingress.step_positions.bind(ingress.span());
                mtp.step_valid_columns  = controls.ingress.step_valid.bind(ingress.span());
                mtp.step_rope_positions = controls.ingress.step_rope.bind(ingress.span());
                const auto step         = mtp.step_view(1, count);
                const auto proposals    = mtp.proposals(count);
                expect(proposals.ne[0] == static_cast<int>(drafts) &&
                           proposals.ne[1] == static_cast<int>(count) &&
                           proposals.nb[0] == capacity * sizeof(ninfer::TokenId) &&
                           proposals.nb[1] == sizeof(ninfer::TokenId),
                       "MTP proposal view lost its allocated draft-step stride");
                expect(static_cast<const std::byte*>(step.ids.data) -
                                   static_cast<const std::byte*>(mtp.drafts.data) ==
                               capacity * sizeof(ninfer::TokenId) &&
                           static_cast<const std::byte*>(step.positions.data) -
                                   static_cast<const std::byte*>(mtp.step_positions.data) ==
                               count * sizeof(std::int32_t) &&
                           step.rope_positions.nb[1] == count * sizeof(std::int32_t),
                       "MTP step confuses resident proposal and physical control strides");
            }
        }
    }
}

void test_visual_embedding_windows() {
    ninfer::models::vision::WindowPlan plan;
    plan.outputs     = {{0, 0, {2, 4, 7}}};
    const auto first = ninfer::models::vision::embedding_slices(plan, 0, 4);
    expect(first.size() == 1 && first[0].source_begin == 0 &&
               first[0].destination_columns == std::vector<std::int32_t>{2},
           "first window selects only its visual columns");
    const auto final = ninfer::models::vision::embedding_slices(plan, 4, 4);
    expect(final.size() == 1 && final[0].source_begin == 1 &&
               final[0].destination_columns == std::vector<std::int32_t>({0, 3}),
           "last window retains both visual boundaries");
    const auto shifted = ninfer::models::vision::embedding_slices(plan, 1, 3);
    expect(shifted.size() == 1 && shifted[0].source_begin == 0 &&
               shifted[0].destination_columns == std::vector<std::int32_t>{1},
           "MTP shifted history selects matching visual positions");
    expect(ninfer::models::vision::embedding_slices(plan, 8, 1).empty(),
           "generated tokens have no visual embedding");
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

    const ninfer::models::qwen::VisionConfig config{.patch_size          = 1,
                                                    .temporal_patch_size = 1,
                                                    .spatial_merge_size  = 2,
                                                    .position_grid_side  = 48};
    for (const auto& item : prompt.vision_items) {
        auto payload = std::make_shared<ninfer::models::frontend::PreparedMediaPayload>();
        payload->features_per_patch = config.patch_width();
        payload->patch_elements     = item.patch_count * config.patch_width();
        payload->patches            = std::make_unique<std::uint16_t[]>(payload->patch_elements);
        prompt.media_payloads.push_back(std::move(payload));
    }
    const auto input = ninfer::models::qwen::prepare_vision_input(prompt, config, 2);
    const auto plan  = ninfer::models::vision::plan_window(input->units, 2, 7, 2, 1);
    const auto batch = ninfer::models::qwen::encoder_batch(*input, *plan, 0);
    expect(input->units.size() == 3 && plan->units.size() == 2 && batch.features == 2,
           "video temporal grids form independently encodable units");
    expect(batch.cu_seqlens == std::vector<std::int32_t>({0, 4, 8}) &&
               batch.positions ==
                   std::vector<std::int32_t>({0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1}),
           "selected grids retain spatial rotary coordinates");
    expect(batch.position_indices[0] == 0 && batch.position_indices[4] == 47 &&
               batch.position_indices[8] == 2256 && batch.position_indices[12] == 2303 &&
               batch.position_weights[0] == 1 && batch.position_weights[4] == 1,
           "position interpolation maps grid corners to the learned table corners");
    expect(batch.patches[0].data() == prompt.media_payloads[1]->span().data() &&
               batch.patches[1].data() == prompt.media_payloads[1]->span().data() + 12,
           "window borrows only its temporal patch ranges");
}


} // namespace

int main() {
    test_decoder_layout();
    test_frame_layout();
    test_visual_embedding_windows();
    test_vision_control();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.5 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.5 runtime mechanism checks passed\n";
    return 0;
}
