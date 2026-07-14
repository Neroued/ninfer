#include "targets/qwen3_6_27b_rtx5090/impl/diagnostic/activation_dump_access.h"

#include "targets/qwen3_6_27b_rtx5090/impl/program/program.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

void ActivationDumpAccess::attach(Program& program, void* context, schedule::TextTapCallback text,
                                  schedule::VisionTapCallback vision) {
    if (context == nullptr || (text == nullptr && vision == nullptr)) {
        throw std::invalid_argument("activation dump attachment requires a context and callback");
    }
    program.impl_->diagnostic_context    = context;
    program.impl_->diagnostic_text_tap   = text;
    program.impl_->diagnostic_vision_tap = vision;
}

void ActivationDumpAccess::detach(Program& program) noexcept {
    program.impl_->diagnostic_context    = nullptr;
    program.impl_->diagnostic_text_tap   = nullptr;
    program.impl_->diagnostic_vision_tap = nullptr;
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
