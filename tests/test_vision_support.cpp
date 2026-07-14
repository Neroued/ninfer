#include "model/vision_ops.h"
#include "model/position.h"
#include "kernels/op_tester.h"
#include "ninfer/core/device.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    model::ProcessedInput input;
    input.stats.raw_patches   = 32;
    input.stats.vision_tokens = 8;
    input.input_ids.resize(32);
    input.token_types.resize(32);
    model::VisionItem image;
    image.modality    = model::Modality::Image;
    image.grid        = {1, 4, 4};
    image.patch_begin = 0;
    image.patch_count = 16;
    image.token_spans = {{5, 4}};
    model::VisionItem video;
    video.modality     = model::Modality::Video;
    video.grid         = {2, 2, 4};
    video.patch_begin  = 16;
    video.patch_count  = 16;
    video.token_spans  = {{20, 2}, {30, 2}};
    input.vision_items = {image, video};
    std::fill(input.token_types.begin() + 5, input.token_types.begin() + 9,
              static_cast<std::uint8_t>(model::Modality::Image));
    std::fill(input.token_types.begin() + 20, input.token_types.begin() + 22,
              static_cast<std::uint8_t>(model::Modality::Video));
    std::fill(input.token_types.begin() + 30, input.token_types.begin() + 32,
              static_cast<std::uint8_t>(model::Modality::Video));
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

    const std::vector<std::int32_t> axes{
        3, 4, 5, 6, 101, 102, 103, 104, 201, 202, 203, 204,
    };
    DBuf daxes(axes.size() * sizeof(std::int32_t));
    Tensor taxes(daxes.p, DType::I32, {4, 3});
    model::detail::copy_i32(axes.data(), taxes, nullptr);
    cudaDeviceSynchronize();
    std::vector<std::int32_t> axes_out(axes.size());
    cudaMemcpy(axes_out.data(), daxes.p, axes.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
    if (axes_out != axes) {
        std::cerr << "three-axis I32 upload mismatch\n";
        ++failures;
    }

    const std::vector<std::int32_t> base_positions{2, 5, 11, 19};
    DBuf dbase = to_device_i32(std::vector<int>(base_positions.begin(), base_positions.end()));
    DBuf ddelta(sizeof(std::int32_t));
    DBuf doffset(base_positions.size() * sizeof(std::int32_t));
    Tensor tbase(dbase.p, DType::I32, {static_cast<int>(base_positions.size())});
    Tensor tdelta(ddelta.p, DType::I32, {1});
    Tensor toffset(doffset.p, DType::I32, {static_cast<int>(base_positions.size())});
    cudaStream_t stream  = nullptr;
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    model::detail::offset_positions(tbase, tdelta, toffset, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    for (const std::int32_t delta : {0, 7, -4}) {
        CUDA_CHECK(
            cudaMemcpyAsync(ddelta.p, &delta, sizeof(delta), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::int32_t> got(base_positions.size());
        CUDA_CHECK(cudaMemcpy(got.data(), doffset.p, got.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (got[i] != base_positions[i] + delta) {
                std::cerr << "captured device-delta offset mismatch\n";
                ++failures;
                break;
            }
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << (failures ? "FAIL" : "OK") << " vision support correctness\n";
    return failures ? 1 : 0;
}
