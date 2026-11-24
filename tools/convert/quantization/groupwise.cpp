#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void launch_groupwise_mse(const float*, signed char*, unsigned short*, int64_t, int, cudaStream_t);

std::vector<torch::Tensor> quantize(torch::Tensor values, int bits) {
    TORCH_CHECK(values.is_cuda() && values.is_contiguous() &&
                values.scalar_type() == torch::kFloat32);
    TORCH_CHECK(bits == 4 || bits == 5 || bits == 6 || bits == 8);
    const int group_size = bits == 8 ? 32 : 64;
    TORCH_CHECK(values.dim() == 2 && values.size(1) == group_size);
    const int64_t groups = values.size(0);
    c10::cuda::CUDAGuard guard(values.device());
    auto codes  = torch::empty({groups, group_size}, values.options().dtype(torch::kInt8));
    auto scales = torch::empty({groups}, values.options().dtype(torch::kInt16));
    if (groups > 0) {
        launch_groupwise_mse(values.data_ptr<float>(), codes.data_ptr<signed char>(),
                             reinterpret_cast<unsigned short*>(scales.data_ptr<int16_t>()), groups,
                             bits, at::cuda::getCurrentCUDAStream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {codes, scales.view(torch::kFloat16)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quantize", &quantize); }
