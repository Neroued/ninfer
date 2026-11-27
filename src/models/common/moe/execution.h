#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::models::moe {

struct Parameters {
    ops::MoeProfile profile;
    Weight control;
    ops::SwiGLUWeights shared;
    ExpertBankId bank;
    ops::ExpertActivationCalibration activation_calibration;
    ExpertBankDeviceView resident;
    ops::ExpertSubset subset;
};

struct LayerSource {
    Tensor hidden, output;
    ops::MoeTokenValidity validity;
    ops::MoeEpilogue mode = ops::MoeEpilogue::Write;
};

class ExpertExchange;
class DeviceExpertPlacement;

// A matched layer stage borrows a single frame and immutable placement until consumption.
struct ExpertExchangeWork {
    ExpertExchange& frame;
    const DeviceExpertPlacement& placement;
    int physical_columns;
    Tensor* observation = nullptr;
};

void resident_layer(const Parameters&, const LayerSource*, const ExpertExchangeWork*, MemoryArena&,
                    CudaExecutionView);

[[nodiscard]] std::size_t
workspace_bytes(const ops::MoeProfile& profile, ops::MoeEpilogue epilogue, std::int32_t first,
                std::int32_t last, ops::ExpertSubsetKind subset = ops::ExpertSubsetKind::All);
// BF16 [H,T] output. Write overwrites it; AddResidual adds to its existing value.
// The complete GPU path preserves the fused Op and supports CUDA Graph execution.
void execute(const Tensor& hidden, const Parameters& parameters,
             const ExpertBankDeviceView& experts, ops::MoeEpilogue epilogue, Tensor& output,
             ops::MoeTokenValidity validity, MemoryArena& workspace, CudaExecutionView execution,
             const ops::MoeExecutionHints& hints = {});

} // namespace ninfer::models::moe
