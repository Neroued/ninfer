#pragma once

#include "qus/core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace qus {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QType : std::uint16_t {
    Q4G64_F16S  = 0,
    Q5G64_F16S  = 1,
    Q6G64_F16S  = 2,
    W8G128_F16S = 3,
    BF16_CTRL   = 4,
    FP32_CTRL   = 5,
    W8G32_F16S  = 6,
    I32_CTRL    = 7,
};

enum class QuantLayout : std::uint16_t {
    RowSplit  = 0,
    Contiguous = 1,
};

enum class ModuleKind : std::uint16_t {
    TextCore      = 0,
    MtpDraft      = 1,
    VisionEncoder = 2,
};

enum class ScaleDType : std::uint16_t {
    None = 0,
    FP16 = 1,
};

enum class LoadPolicy : std::uint32_t {
    Resident         = 0,
    LazyGpu          = 1,
    CpuPinnedThenGpu = 2,
};

enum class SourceKind : std::uint32_t {
    Other             = 0,
    Embed             = 1,
    LmHead            = 2,
    FinalNorm         = 3,
    InputLayernorm    = 4,
    PostAttnLayernorm = 5,
    LmHeadDraft       = 6,
    LmHeadDraftIdmap  = 7,
    GdnALog           = 10,
    GdnDtBias         = 11,
    GdnConv1d         = 12,
    GdnInProjA        = 13,
    GdnInProjB        = 14,
    GdnInProjQ        = 15,
    GdnInProjK        = 16,
    GdnInProjV        = 17,
    GdnInProjZ        = 18,
    GdnNorm           = 19,
    GdnOutProj        = 20,
    AttnQ             = 30,
    AttnGate          = 31,
    AttnK             = 32,
    AttnV             = 33,
    AttnQNorm         = 34,
    AttnKNorm         = 35,
    AttnO             = 36,
    MlpGate           = 40,
    MlpUp             = 41,
    MlpDown           = 42,
    MtpFc             = 50,
    MtpPreFcNormEmb   = 51,
    MtpPreFcNormHid   = 52,
    MtpNorm           = 53,
    VisPatchEmbed     = 60,
    VisPatchEmbedBias = 61,
    VisPosEmbed       = 62,
    VisBlockQkv       = 63,
    VisBlockQkvBias   = 64,
    VisBlockProj      = 65,
    VisBlockProjBias  = 66,
    VisBlockFc1       = 67,
    VisBlockFc1Bias   = 68,
    VisBlockFc2       = 69,
    VisBlockFc2Bias   = 70,
    VisBlockNorm1W    = 71,
    VisBlockNorm1B    = 72,
    VisBlockNorm2W    = 73,
    VisBlockNorm2B    = 74,
    VisMergerFc1      = 75,
    VisMergerFc1Bias  = 76,
    VisMergerFc2      = 77,
    VisMergerFc2Bias  = 78,
    VisMergerNormW    = 79,
    VisMergerNormB    = 80,
};

struct Weight {
    const void* payload          = nullptr;
    std::uint64_t payload_bytes  = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                  = QType::Q4G64_F16S;
    ModuleKind module            = ModuleKind::TextCore;
    ScaleDType q5090_scale_dtype = ScaleDType::FP16;
    std::uint32_t group_size     = 0;
    std::uint32_t source_layer   = 0xFFFFFFFFU;
    std::uint32_t source_kind    = 0;
    std::int32_t shape[4]        = {1, 1, 1, 1};
    std::int32_t padded_shape[4] = {1, 1, 1, 1};
    std::uint32_t ndim           = 0;

    const void* qdata        = nullptr;
    const void* qhigh        = nullptr;
    const void* scales       = nullptr;
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    std::int32_t group       = 0;
    QuantLayout layout       = QuantLayout::RowSplit;
    DType scale_dtype        = DType::FP32;
    std::int32_t scale_ne[4] = {1, 1, 1, 1};
    std::int64_t scale_nb[4] = {0, 0, 0, 0};
};

} // namespace qus
