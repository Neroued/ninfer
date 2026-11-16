#include "models/qwen4_exp/state/state_image.h"
#include "models/qwen4_exp/state/kv_cache.h"
#include "models/qwen4_exp/program/buffers.h"
#include "models/qwen4_exp/work.h"
#include "core/buffer.h"
#include "core/transfer.h"
#include "artifact/fixture.h"

#include <cstring>
#include <iostream>

namespace {
using namespace ninfer;
using test::artifact_fixture::require;
namespace q4 = models::qwen4_exp;

void binding_contract() {
    q4::WorkContract contract{};
    contract.shape.max_concurrency = 17;
    contract.shape.max_context     = 128;
    contract.vocabulary = contract.public_tokens = 32;
    q4::ContextWork context;
    for (std::uint32_t lane = 0; lane < 17; ++lane) context.releases.push_back({{lane}, 1});
    context.bindings.push_back({.member = {{1, 1}, {{3}, 2}}});
    contract.validate({{}, context}, runtime::FrameKind::Context);
    context.bindings.push_back({.member = {{2, 1}, {{4}, 2}}});
    bool rejected = false;
    try {
        contract.validate({{}, context}, runtime::FrameKind::Context);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "one Context work accepted multiple bindings into its single counts staging");
}

q4::Config config(bool mtp) {
    q4::Config c{};
    c.text.hidden     = 2560;
    c.text.hc         = {4, 320};
    c.text.gdn        = {16, 48, 128, 128, 4};
    c.text.qsa        = {24, 2, 256, 4, 128, 4, 512, 1e7F, 1, 64, {11, 11, 10}, true};
    c.text.ple        = {1, 128, 2500012, 248044, 2560, 3, 8, 160, 4, 3};
    c.text.gdn_layers = 36;
    c.text.qsa_layers = 12;
    c.mtp             = mtp;
    return c;
}

void controls() {
    for (const bool mtp : {false, true}) {
        EngineOptions options;
        options.max_concurrency = 128;
        options.speculative     = {mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
                               mtp ? 3U : 0U};
        const auto c = config(mtp);
        LayoutBuilder builder;
        const auto backing =
            q4::plan_buffers(builder, c, options, 1024, 128, c.text.gdn, 128, 2, true);
        HostBuffer host(backing.host_region.bytes);
        auto view = backing;
        for (const std::uint32_t rows : {1U, 9U, 10U, 64U, 128U, 1U}) {
            q4::select_buffer_rows(view, backing, rows);
            const auto& b   = view.batch;
            const auto bind = [&](const TensorRegion& r) { (void)r.bind(host.span()); };
            for (const auto* r : std::initializer_list<const TensorRegion*>{
                     &view.anchors, &view.decode.positions, &view.decode.rope_positions,
                     &b.live_rows, &b.valid_columns, &b.source_offsets, &b.destination_offsets,
                     &b.main_tables, &b.base_positions, &b.record_offsets, &b.raw_tail,
                     &b.raw_tail_offsets, &b.count_slots, &b.sampling_positions,
                     &view.commit_destination_offsets, &b.accepted_columns, &b.count_columns})
                bind(*r);
            bind(*view.sampling);
            require(b.live_rows.shape[0] == int(rows) && b.source_offsets.shape[0] == int(rows) &&
                        view.ingress.offset + view.ingress.bytes <= view.commit_ingress.offset &&
                        view.commit_ingress.offset + view.commit_ingress.bytes <=
                            backing.commit_ingress.offset + backing.commit_ingress.bytes,
                    "compact controls overlap or exceed their backing");
            require(view.sampled_ids->shape[1] == int(rows) &&
                        view.sampled_ids->region.offset + view.sampled_ids->region.bytes <=
                            view.output_counts->region.offset &&
                        view.output_counts->region.offset + view.output_counts->region.bytes <=
                            backing.output_counts->region.offset +
                                backing.output_counts->region.bytes,
                    "compact results exceed their backing");
            if (rows == 1)
                require(view.ingress.bytes < backing.ingress.bytes / 4,
                        "single-row controls still transfer resident capacity");
            if (!mtp) continue;
            for (const auto& step : view.mtp->draft_steps) {
                bind(step.live_rows);
                bind(step.source_offsets);
                bind(step.record_offsets);
            }
            const auto history = q4::history_control_view(backing.mtp->history, 3, rows);
            bind(history.ids);
            bind(history.rope_positions);
            bind(history.source_offsets);
            bind(history.record_offsets);
            require(history.ids.shape[0] == int(3 * rows) &&
                        history.ingress.offset + history.ingress.bytes <=
                            backing.mtp->history.ingress.offset +
                                backing.mtp->history.ingress.bytes,
                    "compact history exceeds its backing");
        }
    }
}

void image(DeviceContext& device, bool mtp) {
    const auto layout         = q4::plan_state_image(config(mtp), config(mtp).text.gdn);
    const std::int64_t source = 256, destination = layout.images.image_bytes + 768;
    DeviceBuffer backing(device.device(), destination + layout.images.image_bytes);
    q4::StateViews state(backing.span(), layout);
    PinnedHostBuffer expected(state.storage().image_bytes()),
        returned(state.storage().image_bytes());
    auto* bytes = static_cast<unsigned char*>(expected.data());
    for (std::size_t i = 0; i < expected.size(); ++i) bytes[i] = (i * 29 + i / 257) & 255;
    state.storage().copy_from_host(expected.span(), source, device.execution_view());
    state.storage().copy(source, destination, device.execution_view());
    state.storage().copy_to_host(destination, returned.span(), device.execution_view());
    device.synchronize();
    require(std::memcmp(expected.data(), returned.data(), expected.size()) == 0,
            "complete image transfer changed words");
    const auto conv   = state.gdn().conv_tensor(35, destination);
    const auto sample = state.sample_hidden(destination);
    require(conv.data != sample.data &&
                static_cast<std::byte*>(sample.data) -
                        static_cast<std::byte*>(state.sample_hidden(source).data) ==
                    destination - source,
            "native views lost image stride");
    if (mtp) require(state.conditioning(destination).ne[0] == 10240, "MTP conditioning width");
    state.storage().clear(source, device.execution_view());
    state.storage().clear(destination, device.execution_view());
    device.synchronize();

    const std::array<std::int32_t, 5> words{0, 1, 6, 1, 1};
    DeviceBuffer meta(device.device(), sizeof(words));
    meta.copy_from_host(words.data(), sizeof(words));
    const auto scalar = [&](int i) {
        return Tensor(meta.span().subspan(i * 4, 4), DType::I32, {1});
    };
    const std::array<std::int64_t, 2> offsets{source, destination};
    DeviceBuffer selected(device.device(), sizeof(offsets));
    selected.copy_from_host(offsets.data(), sizeof(offsets));
    const auto selected_row = [&](int i) {
        return Tensor(selected.span().subspan(i * 8, 8), DType::I64, {1});
    };
    auto before = state.main_tail(11, selected_row(0)),
         after  = state.main_tail(11, selected_row(1));
    std::vector<std::uint16_t> old_keys(128 * 3), new_keys(128);
    for (int i = 0; i < 128; ++i) {
        old_keys[i]       = 0x3f00 + i;
        old_keys[128 + i] = 0x4000 + i;
        new_keys[i]       = 0x4100 + i;
    }
    const std::array<std::int32_t, 3> coordinates{4, 8, 12}, new_coordinates{6, 10, 14};
    CUDA_CHECK(cudaMemcpy(before.raw_keys.tensor(source).data, old_keys.data(), old_keys.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(before.first_rope.tensor(source).data, coordinates.data(),
                          sizeof(coordinates), cudaMemcpyHostToDevice));
    DeviceBuffer raw(device.device(), new_keys.size() * 2),
        rope(device.device(), sizeof(new_coordinates));
    raw.copy_from_host(new_keys.data(), raw.size());
    rope.copy_from_host(new_coordinates.data(), rope.size());
    ops::qsa_tail_extract(before,
                          {Tensor(raw.span(), DType::BF16, {128, 1, 1}),
                           Tensor(rope.span(), DType::I32, {3, 1, 1}), scalar(2), scalar(4)},
                          scalar(3), after, {}, device.execution_view());

    auto ple_before = state.ple(selected_row(0)), ple_after = state.ple(selected_row(1));
    std::vector<std::uint16_t> history(10240 * 9), record(10240, 0x4321);
    for (int j = 0; j < 9; ++j) std::fill_n(history.begin() + j * 10240, 10240, 0x3f00 + j);
    CUDA_CHECK(cudaMemcpy(ple_before.history.tensor(source).data, history.data(),
                          history.size() * 2, cudaMemcpyHostToDevice));
    DeviceBuffer n(device.device(), record.size() * 2);
    n.copy_from_host(record.data(), n.size());
    ops::ple_state_extract(ple_before, {Tensor(n.span(), DType::BF16, {10240, 1, 1}), scalar(4)},
                           scalar(3), ple_after, {}, device.execution_view());
    device.synchronize();
    std::vector<std::uint16_t> result(old_keys.size());
    CUDA_CHECK(cudaMemcpy(result.data(), after.raw_keys.tensor(destination).data, result.size() * 2,
                          cudaMemcpyDeviceToHost));
    require(std::equal(old_keys.begin(), old_keys.begin() + 256, result.begin()) &&
                std::equal(new_keys.begin(), new_keys.end(), result.begin() + 256),
            "strided QSA accepted prefix");
    std::array<std::int32_t, 3> actual{};
    CUDA_CHECK(cudaMemcpy(actual.data(), after.first_rope.tensor(destination).data, sizeof(actual),
                          cudaMemcpyDeviceToHost));
    require(actual == coordinates, "QSA tail used current rather than block-first coordinates");
    std::vector<std::uint16_t> ple_result(history.size());
    CUDA_CHECK(cudaMemcpy(ple_result.data(), ple_after.history.tensor(destination).data,
                          ple_result.size() * 2, cudaMemcpyDeviceToHost));
    require(std::equal(history.begin() + 10240, history.end(), ple_result.begin()) &&
                std::equal(record.begin(), record.end(), ple_result.end() - 10240),
            "strided PLE accepted prefix");
}

void cache(DeviceContext& device, KvCacheStorage storage, std::uint32_t layers,
           std::size_t per_layer) {
    LayoutBuilder b;
    const auto layout = q4::plan_qsa_cache(b, config(false).text.qsa, layers, 130, 2, storage);
    DeviceBuffer tables(device.device(), b.finish());
    const std::int64_t source = 256, destination = layout.pages.page_bytes + 768;
    DeviceBuffer backing(device.device(), destination + layout.pages.page_bytes);
    q4::QsaCache cache(backing.span(), tables.span(), layout);
    require(layout.pages.payload_bytes() == per_layer * layers, "QSA record payload");
    const auto view = cache.layer(layers - 1);
    require(view.main.k_pages.records.page_offsets.data ==
                    view.index.keys.records.page_offsets.data &&
                view.index.keys.shape()[1] == 16 && view.main.k_pages.shape()[1] == 64,
            "QSA page record view");
    for (std::size_t i = 0; i < layout.pages.planes.size(); ++i) {
        const auto plane = layout.pages.planes[i].tensor.bind(backing.span().subspan(source));
        CUDA_CHECK(cudaMemsetAsync(plane.data, (i + 1) & 255, plane.bytes(), device.stream()));
    }
    const auto bytes = layout.pages.page_bytes;
    copy_bytes_async(backing.span().subspan(destination, bytes),
                     backing.span().subspan(source, bytes), device.execution_view());
    PinnedHostBuffer original(bytes), restored(bytes);
    copy_bytes_async(original.span(), backing.span().subspan(destination, bytes),
                     device.execution_view());
    CUDA_CHECK(cudaMemsetAsync(backing.span().subspan(destination, bytes).data, 0, bytes,
                               device.stream()));
    copy_bytes_async(backing.span().subspan(destination, bytes), original.span(),
                     device.execution_view());
    copy_bytes_async(restored.span(), backing.span().subspan(destination, bytes),
                     device.execution_view());
    device.synchronize();
    for (std::size_t i = 0; i < layout.pages.planes.size(); ++i) {
        const auto& plane = layout.pages.planes[i].tensor.region;
        const auto* begin = static_cast<const std::byte*>(restored.data()) + plane.offset;
        require(std::all_of(begin, begin + plane.bytes,
                            [&](std::byte byte) { return byte == std::byte((i + 1) & 255); }),
                "QSA record copy/Host roundtrip dropped a plane");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        controls();
        binding_contract();
        if (argc == 2 && std::string_view(argv[1]) == "--host") {
            std::cout << "Qwen4Exp control layouts passed\n";
            return 0;
        }
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        DeviceContext device;
        image(device, false);
        image(device, true);
        const std::array formats{KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                 KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                                 KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value};
        const std::array<std::size_t, 5> bytes{135168, 71680, 70144, 40960, 55552};
        for (std::size_t i = 0; i < formats.size(); ++i) {
            cache(device, formats[i], 12, bytes[i]);
            cache(device, formats[i], 1, bytes[i]);
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
