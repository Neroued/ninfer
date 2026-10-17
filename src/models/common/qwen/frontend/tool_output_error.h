#pragma once

#include "ninfer/types.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen::frontend {

enum class ToolOutputErrorKind { InvalidModelOutput, ContractViolation };

struct ToolOutputError {
    ToolOutputErrorKind kind;
    std::string message;

    std::exception_ptr exception() const {
        if (kind == ToolOutputErrorKind::ContractViolation)
            return std::make_exception_ptr(
                std::logic_error("tool constraint/decoder disagreement: " + message));
        return std::make_exception_ptr(RequestError(RequestErrorKind::InvalidToolOutput, message,
                                                    {}, RequestErrorSource::Tools));
    }
};

} // namespace ninfer::models::qwen::frontend
