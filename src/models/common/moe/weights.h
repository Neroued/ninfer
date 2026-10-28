#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/weight_bank.h"
#include "ninfer/ops/moe.h"

namespace ninfer::models::moe {

struct ExpertBinding {
    const artifact::ParameterReference& gate;
    const artifact::ParameterReference& up;
    const artifact::ParameterReference& down;
};

struct MatrixSource {
    artifact::ObjectHandle object;
    WeightBankLayout layout;
};

struct BankDescription {
    MatrixSource gate_up, down;
};

// The materialized matrix geometry and global routing domain are independent: an indexed
// bank stores selected full experts, while an affine bank stores every expert's local slice.
struct ExpertDescription {
    WeightGeometry gate_up, down;
    std::uint32_t count          = 0;
    ops::ExpertSubsetKind subset = ops::ExpertSubsetKind::All;
};

[[nodiscard]] ops::RoutedSwiGLUProfile describe_experts(const ExpertDescription&,
                                                        std::int32_t top_k,
                                                        ops::LinearPolicy gate_up,
                                                        ops::LinearPolicy down);

struct ResidentExpertEntry {
    std::uint32_t expert;
    const WeightParent* gate_up;
    const WeightParent* down;
};

// Owns immutable global-ID metadata; payloads are borrowed from final local weight storage.
class ResidentExpertBank {
public:
    ResidentExpertBank(std::span<const ResidentExpertEntry>, std::uint32_t global_count, Device);
    ResidentExpertBank(const WeightParent& gate_up, const WeightParent& down);

    [[nodiscard]] const ExpertBankDeviceView& view() const { return view_; }

    [[nodiscard]] ops::ExpertSubset subset() const { return {subset_, mask_}; }

    [[nodiscard]] std::size_t bytes() const { return metadata_.size(); }
private:
    DeviceBuffer metadata_;
    ExpertBankDeviceView view_;
    Tensor mask_;
    ops::ExpertSubsetKind subset_ = ops::ExpertSubsetKind::All;
};

// Logical expert order must form affine gate/up and down banks in the stored parents.
[[nodiscard]] BankDescription describe_bank(const artifact::Reader&,
                                            std::span<const ExpertBinding>);
// Three rank-3 logical [E,I,H], [E,I,H], [E,H,I] bindings over affine stored banks.
[[nodiscard]] BankDescription describe_bank(const artifact::Reader&, const ExpertBinding&);
[[nodiscard]] std::vector<ExpertBankSource> bind_banks(std::span<const BankDescription>,
                                                       const artifact::MaterializedArtifact&,
                                                       std::span<const Device>);

} // namespace ninfer::models::moe
