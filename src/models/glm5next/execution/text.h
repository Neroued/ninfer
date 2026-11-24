#pragma once
#include "models/common/parallel_execution.h"
#include "models/common/vision/execution.h"
#include "models/glm5next/execution/batch.h"
#include "models/glm5next/execution/mla.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/execution/workspace.h"
#include "models/glm5next/state/context.h"

namespace ninfer::models::glm5next::execution {
struct TextExecution {
    MemoryArena& workspace;
    const StateViews& states;
    const CacheViews& cache;
    Tensor raw_keys, raw_gates;
    ParallelExecution* parallel = nullptr;
    Tensor features;
};

class TextRun {
public:
    TextRun(const Parameters&, TextExecution, BatchView, StateUpdate, CudaExecutionView);
    void stem(std::optional<models::vision::EmbeddingView> = {});
    void local_mixer(std::uint32_t layer, const MlaSubmit& = {});
    [[nodiscard]] MlaRun local_mla(std::uint32_t layer);
    void finish_mixer(std::uint32_t layer);
    void ffn(std::uint32_t layer);
    void finish_layer(std::uint32_t layer);
    void finish(Tensor& hidden);

    [[nodiscard]] const TokenPartition& columns() const noexcept { return columns_; }
private:
    void next_pre(const ops::MhcPreWeights&, const Tensor& feature = {},
                  std::optional<std::uint32_t> next_mixer = {});
    void gather_input(std::optional<std::uint32_t> next_mixer);
    void broadcast_mla_columns(Tensor&, int begin, int end, std::uint32_t root);
    void exchange_packed_mla(const PackedPrefillUpdate&, Tensor& exchange, bool input);
    void reduce_output();
    const Parameters& parameters_;
    TextExecution execution_;
    BatchView batch_;
    StateUpdate update_;
    CudaExecutionView cuda_;
    TokenPartition columns_;
    MemoryArena::Scope scope_;
    workspace::Roots roots_;
    std::uint32_t coefficients_ = 0;
    workspace::AttentionBuffers attention_;
};
} // namespace ninfer::models::glm5next::execution
