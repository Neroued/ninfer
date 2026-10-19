#include "models/qwen4_exp/state/state_image.h"
#include "models/qwen4_exp/state/kv_cache.h"
#include "core/buffer.h"
#include "artifact/fixture.h"

#include <cstring>
#include <iostream>

namespace {
using namespace ninfer;
using test::artifact_fixture::require;
namespace q4 = models::qwen4_exp;

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

void image(DeviceContext& device, bool mtp) {
    LayoutBuilder b;
    const auto layout = q4::plan_state_image(b, config(mtp), config(mtp).text.gdn, 2);
    DeviceBuffer backing(device.device(), b.finish());
    q4::StateViews state(backing.span(), layout);
    PinnedHostBuffer expected(state.storage().image_bytes()),
        returned(state.storage().image_bytes());
    auto* bytes = static_cast<unsigned char*>(expected.data());
    for (std::size_t i = 0; i < expected.size(); ++i) bytes[i] = (i * 29 + i / 257) & 255;
    state.storage().copy_from_host(expected.span(), 0, device.execution_view());
    state.storage().copy(0, 1, device.execution_view());
    state.storage().copy_to_host(1, returned.span(), device.execution_view());
    device.synchronize();
    require(std::memcmp(expected.data(), returned.data(), expected.size()) == 0,
            "complete image transfer changed words");
    const auto conv   = state.gdn().conv_slot(35, 1);
    const auto sample = state.sample_hidden().slice(1, 1, 1);
    require(conv.data != sample.data &&
                state.sample_hidden().nb[1] == std::int64_t(layout.images.image_stride),
            "native views lost image stride");
    if (mtp) require(state.conditioning().ne[0] == 10240, "MTP conditioning width");
    state.storage().clear_all(device.execution_view());
    device.synchronize();

    const std::array<std::int32_t, 5> words{0, 1, 6, 1, 1};
    DeviceBuffer meta(device.device(), sizeof(words));
    meta.copy_from_host(words.data(), sizeof(words));
    const auto scalar = [&](int i) {
        return Tensor(meta.span().subspan(i * 4, 4), DType::I32, {1});
    };
    auto before = state.main_tail(11, scalar(0)), after = state.main_tail(11, scalar(1));
    std::vector<std::uint16_t> old_keys(128 * 3), new_keys(128);
    for (int i = 0; i < 128; ++i) {
        old_keys[i]       = 0x3f00 + i;
        old_keys[128 + i] = 0x4000 + i;
        new_keys[i]       = 0x4100 + i;
    }
    const std::array<std::int32_t, 3> coordinates{4, 8, 12}, new_coordinates{6, 10, 14};
    CUDA_CHECK(cudaMemcpy(before.raw_keys.data, old_keys.data(), old_keys.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(before.first_rope.data, coordinates.data(), sizeof(coordinates),
                          cudaMemcpyHostToDevice));
    DeviceBuffer raw(device.device(), new_keys.size() * 2),
        rope(device.device(), sizeof(new_coordinates));
    raw.copy_from_host(new_keys.data(), raw.size());
    rope.copy_from_host(new_coordinates.data(), rope.size());
    ops::qsa_tail_extract(before,
                          {Tensor(raw.span(), DType::BF16, {128, 1, 1}),
                           Tensor(rope.span(), DType::I32, {3, 1, 1}), scalar(2), scalar(4)},
                          scalar(3), after, device.execution_view());

    auto ple_before = state.ple(scalar(0)), ple_after = state.ple(scalar(1));
    std::vector<std::uint16_t> history(10240 * 9), record(10240, 0x4321);
    for (int j = 0; j < 9; ++j) std::fill_n(history.begin() + j * 10240, 10240, 0x3f00 + j);
    CUDA_CHECK(cudaMemcpy(ple_before.history.data, history.data(), history.size() * 2,
                          cudaMemcpyHostToDevice));
    DeviceBuffer n(device.device(), record.size() * 2);
    n.copy_from_host(record.data(), n.size());
    ops::ple_state_extract(ple_before, {Tensor(n.span(), DType::BF16, {10240, 1, 1}), scalar(4)},
                           scalar(3), ple_after, device.execution_view());
    device.synchronize();
    std::vector<std::uint16_t> result(old_keys.size());
    CUDA_CHECK(cudaMemcpy(result.data(), after.raw_keys.slice(2, 1, 1).data, result.size() * 2,
                          cudaMemcpyDeviceToHost));
    require(std::equal(old_keys.begin(), old_keys.begin() + 256, result.begin()) &&
                std::equal(new_keys.begin(), new_keys.end(), result.begin() + 256),
            "strided QSA accepted prefix");
    std::array<std::int32_t, 3> actual{};
    CUDA_CHECK(cudaMemcpy(actual.data(), after.first_rope.slice(1, 1, 1).data, sizeof(actual),
                          cudaMemcpyDeviceToHost));
    require(actual == coordinates, "QSA tail used current rather than block-first coordinates");
    std::vector<std::uint16_t> ple_result(history.size());
    CUDA_CHECK(cudaMemcpy(ple_result.data(), ple_after.history.slice(2, 1, 1).data,
                          ple_result.size() * 2, cudaMemcpyDeviceToHost));
    require(std::equal(history.begin() + 10240, history.end(), ple_result.begin()) &&
                std::equal(record.begin(), record.end(), ple_result.end() - 10240),
            "strided PLE accepted prefix");
}

void cache(DeviceContext& device, KvCacheStorage storage, std::uint32_t layers,
           std::size_t per_layer) {
    LayoutBuilder b;
    const auto layout = q4::plan_qsa_cache(b, config(false).text.qsa, layers, 130, 2, 2, storage);
    DeviceBuffer backing(device.device(), b.finish());
    q4::QsaCache cache(backing.span(), layout);
    auto& pool = cache.pages();
    require(pool.device_copy_work(1).payload_bytes == per_layer * layers, "QSA pool payload");
    const auto view = cache.layer(layers - 1);
    require(view.main.block_tables.data == view.index.block_tables.data &&
                view.index.keys.ne[1] == 16 && view.main.k_pages.ne[1] == 64,
            "QSA page group view");
    for (std::size_t i = 0; i < pool.plane_count(); ++i) {
        const auto& plane = pool.plane(i);
        CUDA_CHECK(cudaMemsetAsync(plane.data, (i + 1) & 255, plane.bytes(), device.stream()));
    }
    const std::array<std::int32_t, 1> src{0}, dst{1};
    pool.zero_pages(dst, device.execution_view());
    pool.copy_page(src[0], dst[0], device.execution_view());
    const auto host_layout = plan_host_kv_page_layout(pool.geometry());
    PinnedHostBuffer original(host_layout.page_stride), restored(host_layout.page_stride);
    pool.copy_to_host(dst, original.span(), host_layout, device.execution_view());
    pool.zero_pages(dst, device.execution_view());
    pool.copy_from_host(original.span(), host_layout, dst, device.execution_view());
    pool.copy_to_host(dst, restored.span(), host_layout, device.execution_view());
    device.synchronize();
    for (std::size_t i = 0; i < host_layout.planes.size(); ++i) {
        const auto& plane = host_layout.planes[i];
        const auto* begin = static_cast<const std::byte*>(restored.data()) + plane.offset;
        require(std::all_of(begin, begin + plane.page_payload_bytes,
                            [&](std::byte byte) { return byte == std::byte((i + 1) & 255); }),
                "QSA copy/Host roundtrip dropped a plane");
    }
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        DeviceContext device;
        image(device, false);
        image(device, true);
        const std::array formats{KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                 KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                 KvCacheStorage::Fp8KeyNvfp4Value};
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
