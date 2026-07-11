#include "qus/kernels/attn_input_proj.h"
#include "qus/kernels/gdn_input_proj.h"

#include "qus/kernels/linear.h"

#include "kernels/linear/reference/linear_generic.h"
#include "qus/core/device.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("linear grouped input: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t n, std::int32_t k,
                      const char* label) {
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.q5090_scale_dtype != ScaleDType::FP16 || weight.group_size != 64 ||
        weight.group != 64 || weight.n != n || weight.k != k || weight.padded_shape[0] != n ||
        weight.padded_shape[1] != k || weight.qdata == nullptr || weight.scales == nullptr ||
        (qtype == QType::Q5G64_F16S && weight.qhigh == nullptr)) {
        throw std::invalid_argument(std::string("linear grouped input: invalid ") + label);
    }
}

} // namespace

void attn_input_proj(const Tensor& x, const Weight& q_weight, const Weight& gate_weight,
                     const Weight& k_weight, const Weight& v_weight, Tensor& q, Tensor& gate,
                     Tensor& k, Tensor& v, WorkspaceArena& ws, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t t           = x.ne[1];
    require_matrix(x, kHidden, t, "attention x");
    require_matrix(q, kQRows, t, "attention q");
    require_matrix(gate, kQRows, t, "attention gate");
    require_matrix(k, kKvRows, t, "attention k");
    require_matrix(v, kKvRows, t, "attention v");
    require_rowsplit(q_weight, QType::Q4G64_F16S, kQRows, kHidden, "attention q weight");
    require_rowsplit(gate_weight, QType::Q5G64_F16S, kQRows, kHidden, "attention gate weight");
    require_rowsplit(k_weight, QType::Q4G64_F16S, kKvRows, kHidden, "attention k weight");
    require_rowsplit(v_weight, QType::Q5G64_F16S, kKvRows, kHidden, "attention v weight");

    if (t <= 16) {
        linear(x, q_weight, q, ws, stream);
        linear(x, gate_weight, gate, ws, stream);
        linear(x, k_weight, k, ws, stream);
        linear(x, v_weight, v, ws, stream);
        return;
    }
    detail::linear_rowsplit_attn_input_grouped_mma_launch(x, q_weight, gate_weight, k_weight,
                                                          v_weight, q, gate, k, v, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    WorkspaceArena& ws, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kVRows  = 6144;
    constexpr std::int32_t kRows   = kQkRows + kVRows;
    const std::int32_t t           = x.ne[1];
    require_matrix(x, kHidden, t, "GDN x");
    require_matrix(qkv, kRows, t, "GDN qkv");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQkRows, kHidden, "GDN qk weight");
    require_rowsplit(v_weight, QType::Q5G64_F16S, kVRows, kHidden, "GDN v weight");

    if (t <= 16) {
        auto scratch_scope = ws.scope();
        Tensor qk          = ws.alloc(DType::BF16, {kQkRows, t});
        Tensor vv          = ws.alloc(DType::BF16, {kVRows, t});
        linear(x, qk_weight, qk, ws, stream);
        linear(x, v_weight, vv, ws, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(qkv.data, static_cast<std::size_t>(kRows) * 2, qk.data,
                                     static_cast<std::size_t>(kQkRows) * 2,
                                     static_cast<std::size_t>(kQkRows) * 2, t,
                                     cudaMemcpyDeviceToDevice, stream));
        auto* v_dst = static_cast<unsigned char*>(qkv.data) + static_cast<std::size_t>(kQkRows) * 2;
        CUDA_CHECK(cudaMemcpy2DAsync(v_dst, static_cast<std::size_t>(kRows) * 2, vv.data,
                                     static_cast<std::size_t>(kVRows) * 2,
                                     static_cast<std::size_t>(kVRows) * 2, t,
                                     cudaMemcpyDeviceToDevice, stream));
        return;
    }
    detail::linear_rowsplit_gdn_input_grouped_mma_launch(x, qk_weight, v_weight, qkv, stream);
}

} // namespace qus::kernels
