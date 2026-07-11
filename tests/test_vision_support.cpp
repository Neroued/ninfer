#include "model/vision_ops.h"
#include "kernels/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace qus;
using namespace qus::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    model::ProcessedInput input;
    input.stats.raw_patches   = 32;
    input.stats.vision_tokens = 8;
    model::VisionItem image;
    image.modality    = model::Modality::Image;
    image.grid        = {1, 4, 4};
    image.patch_begin = 0;
    image.patch_count = 16;
    image.token_spans = {{5, 4}};
    model::VisionItem video;
    video.modality                             = model::Modality::Video;
    video.grid                                 = {2, 2, 4};
    video.patch_begin                          = 16;
    video.patch_count                          = 16;
    video.token_spans                          = {{20, 2}, {30, 2}};
    input.vision_items                         = {image, video};
    const model::detail::VisionControl control = model::detail::build_vision_control(input);
    int failures                               = 0;
    if (control.cu_seqlens != std::vector<std::int32_t>({0, 16, 24, 32})) {
        std::cerr << "vision control cu_seqlens mismatch\n";
        ++failures;
    }
    if (control.scatter_indices != std::vector<std::int32_t>({5, 6, 7, 8, 20, 21, 30, 31})) {
        std::cerr << "vision control scatter indices mismatch\n";
        ++failures;
    }
    const std::vector<std::int32_t> expected_y{0, 0, 1, 1, 0, 0, 1, 1};
    const std::vector<std::int32_t> expected_x{0, 1, 0, 1, 2, 3, 2, 3};
    for (std::size_t i = 0; i < expected_y.size(); ++i) {
        if (control.position_ids[i] != expected_y[i] ||
            control.position_ids[32 + i] != expected_x[i]) {
            std::cerr << "vision control merge order mismatch at " << i << '\n';
            ++failures;
            break;
        }
    }

    std::vector<float> source{0.0f, 0.1f, -0.1f, 1.0f, -1.0f, 3.1415927f, -7.25f};
    DBuf dsrc = to_device_f32(source);
    DBuf ddst(source.size() * 2);
    Tensor tsrc(dsrc.p, DType::FP32, {static_cast<int>(source.size())});
    Tensor tdst(ddst.p, DType::BF16, {static_cast<int>(source.size())});
    model::detail::vision_f32_to_bf16(tsrc, tdst, nullptr);
    cudaDeviceSynchronize();
    std::vector<double> reference(source.begin(), source.end());
    failures += verify("vision f32 to bf16", from_device_bf16(ddst, source.size()), reference,
                       Tolerance::bf16_elementwise());
    std::cout << (failures ? "FAIL" : "OK") << " vision support correctness\n";
    return failures ? 1 : 0;
}
