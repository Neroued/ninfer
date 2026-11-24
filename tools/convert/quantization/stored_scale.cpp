#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void launch_stored_scale(const float*, unsigned char*, unsigned char*, int64_t, double, bool,
                         cudaStream_t);

std::vector<torch::Tensor> quantize(torch::Tensor values, double divisor, bool nf4) {
    TORCH_CHECK(values.is_cuda() && values.is_contiguous() &&
                values.scalar_type() == torch::kFloat32);
    TORCH_CHECK(values.numel() > 0 && values.numel() % 16 == 0);
    c10::cuda::CUDAGuard guard(values.device());
    auto codes  = torch::empty({values.numel() / 2}, values.options().dtype(torch::kUInt8));
    auto scales = torch::empty({values.numel() / 16}, values.options().dtype(torch::kUInt8));
    launch_stored_scale(values.data_ptr<float>(), codes.data_ptr<unsigned char>(),
                        scales.data_ptr<unsigned char>(), values.numel(), divisor, nf4,
                        at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {codes, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quantize", &quantize); }
