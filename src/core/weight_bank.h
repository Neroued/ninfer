#pragma once

#include "core/memory.h"
#include "core/weight_view.h"
#include "core/layout.h"

namespace ninfer {

inline constexpr std::uint64_t kExpertAbsentOffset = UINT64_MAX;

struct ExpertMatrixEntryLayout {
    LayoutRegion codes, high, scales;
    friend bool operator==(const ExpertMatrixEntryLayout&,
                           const ExpertMatrixEntryLayout&) = default;
};

struct ExpertBankId {
    std::uint32_t index = UINT32_MAX;

    friend bool operator==(ExpertBankId, ExpertBankId) = default;
};

struct ExpertKey {
    ExpertBankId bank;
    std::uint32_t expert = 0;

    friend bool operator==(ExpertKey, ExpertKey) = default;
};

// Rows are measured in the flattened parent, preserving its last (K) axis.
struct WeightBankRegion {
    std::uint64_t row_begin  = 0;
    std::uint32_t count      = 0;
    std::uint64_t rows       = 0;
    std::uint64_t row_stride = 0;

    friend bool operator==(const WeightBankRegion&, const WeightBankRegion&) = default;
};

struct WeightBankPlane {
    std::uint64_t offset        = 0;
    std::uint64_t expert_bytes  = 0;
    std::uint64_t expert_stride = 0;

    friend bool operator==(const WeightBankPlane&, const WeightBankPlane&) = default;
};

struct Csf32BankLayout {
    WeightBankPlane bases, nibbles, offsets, page_bases;
    std::uint64_t exceptions_offset = 0, exception_count = 0;
    friend bool operator==(const Csf32BankLayout&, const Csf32BankLayout&) = default;
};

// Pointer-free description. Plane offsets refer to the original encoded parent.
struct WeightBankLayout {
    WeightBankRegion region;
    QType format          = QType::BF16;
    QuantLayout layout    = QuantLayout::Contiguous;
    std::uint64_t columns = 0;
    WeightBankPlane codes, high, scales;
    std::uint64_t matrix_scale_begin  = 0;
    std::uint64_t matrix_scale_stride = 0;
    bool has_matrix_scales            = false;
    Csf32BankLayout csf;

    friend bool operator==(const WeightBankLayout&, const WeightBankLayout&) = default;
};

// Borrows immutable backing and Host matrix-scale metadata from the model.
struct WeightBankView {
    const WeightParent* parent = nullptr;
    WeightBankLayout layout;

    [[nodiscard]] ConstMemorySpan plane(const WeightBankPlane& plane, std::uint32_t expert) const;
    // Returns the identity when the representation has no matrix scalar.
    [[nodiscard]] float matrix_scale(std::uint32_t expert) const;
};

struct ExpertBankSource {
    WeightBankView gate_up;
    WeightBankView down;
    // Optional for CPU-only math; GPU replicas require registered views of both parents.
    CudaHostView gate_up_upload;
    CudaHostView down_upload;
};

struct Csf32BankDeviceView {
    Csf32ScaleView first;
    // Byte strides between physical matrices; the exceptions arena is shared.
    std::uint64_t bases_stride = 0, nibbles_stride = 0;
    std::uint64_t offsets_stride = 0, page_bases_stride = 0;
};

// Affine planes use base + physical_expert * stride; expert_indices optionally maps
// logical IDs to physical slots. Indexed storage uses offsets[logical_expert] instead.
// Matrix scalars follow that physical slot (or indexed logical ID); Use calibration
// retains its independent logical expert domain.
struct ExpertMatrixDeviceView {
    const std::byte* codes    = nullptr;
    const std::byte* high     = nullptr;
    const std::byte* scales   = nullptr;
    std::uint64_t code_stride = 0, high_stride = 0, scale_stride = 0;
    const std::uint64_t* offsets      = nullptr;
    const float* matrix_scales        = nullptr;
    std::uint64_t matrix_scale_stride = 0;
    // Indexed addresses refer into this backing; the owner retains selected entries through
    // execution.
    ConstMemorySpan indexed_storage;
    QType format                        = QType::BF16;
    QuantLayout layout                  = QuantLayout::Contiguous;
    const std::uint32_t* expert_indices = nullptr;
    // Affine physical slots. Indexed-storage views use their offset domain instead.
    std::uint32_t physical_count = 0;
    Csf32BankDeviceView csf;
};

struct ExpertBankDeviceView {
    ExpertMatrixDeviceView gate_up, down;
    std::uint32_t count = 0;
    Device device;
};

[[nodiscard]] ExpertMatrixEntryLayout plan_expert_matrix(LayoutBuilder&, const WeightBankLayout&);
[[nodiscard]] ExpertMatrixDeviceView indexed_weight_matrix(const WeightParent&,
                                                           const std::uint64_t* offsets,
                                                           const float* matrix_scales,
                                                           ConstMemorySpan storage);

[[nodiscard]] WeightBankLayout describe_weight_bank(const WeightGeometry& geometry,
                                                    const WeightBankRegion& region);
[[nodiscard]] WeightBankView bind_weight_bank(const WeightParent& parent,
                                              const WeightBankLayout& layout);
[[nodiscard]] Weight matrix_weight(const WeightBankView& bank, std::uint32_t expert,
                                   float input_scale = 0.0F);
// Borrows resident CUDA planes and matrix scalars; performs no allocation.
[[nodiscard]] ExpertMatrixDeviceView device_weight_bank(const WeightBankView& bank);
[[nodiscard]] ExpertBankDeviceView device_expert_bank(const WeightBankView& gate_up,
                                                      const WeightBankView& down);

} // namespace ninfer
