#pragma once

#include "ninfer/types.h"
#include "runtime/engine/product_cookie.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27BRtx5090 = qwen3_6_27b_rtx5090::Package;

struct LoadedQwen3_6_27BRtx5090 {
    std::unique_ptr<Qwen3_6_27BRtx5090::LoadedModel> model;
    Qwen3_6_27BRtx5090::Frontend frontend;

    explicit LoadedQwen3_6_27BRtx5090(
        std::unique_ptr<Qwen3_6_27BRtx5090::LoadedModel> stable_model);
    ~LoadedQwen3_6_27BRtx5090();

    LoadedQwen3_6_27BRtx5090(const LoadedQwen3_6_27BRtx5090&)            = delete;
    LoadedQwen3_6_27BRtx5090& operator=(const LoadedQwen3_6_27BRtx5090&) = delete;
};

struct Qwen3_6_27BRtx5090Instance {
    using Package = Qwen3_6_27BRtx5090;

    runtime::ProductCookie cookie;
    std::unique_ptr<LoadedQwen3_6_27BRtx5090> loaded;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27BRtx5090::Program> program;

    Qwen3_6_27BRtx5090Instance(runtime::ProductCookie product_cookie,
                               std::unique_ptr<LoadedQwen3_6_27BRtx5090> stable_loaded,
                               Qwen3_6_27BRtx5090::SequencePlan sequence_plan,
                               DeviceContext& device);
    ~Qwen3_6_27BRtx5090Instance();

    Qwen3_6_27BRtx5090Instance(const Qwen3_6_27BRtx5090Instance&)            = delete;
    Qwen3_6_27BRtx5090Instance& operator=(const Qwen3_6_27BRtx5090Instance&) = delete;
};

using ActiveTarget = std::variant<std::unique_ptr<Qwen3_6_27BRtx5090Instance>>;
using PreparedValue = std::variant<Qwen3_6_27BRtx5090::PreparedPrompt>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
