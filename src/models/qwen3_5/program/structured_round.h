#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include "text/structured_output.h"
#include <array>
#include <exception>
#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5 {
// Program-owned addresses survive CUDA Graph capture/replay. Grammar state belongs to the
// request, never to a reusable KV checkpoint. Only Engine's output commit advances it.
class StructuredRound {
public:
    StructuredRound(Tensor masks, std::uint32_t vocab, std::uint32_t width, std::uint32_t lanes);
    const std::uint32_t* device_mask(std::uint32_t lane) const;

    int stride() const { return words_; }

    void fill(std::uint32_t lane, const text::GrammarState& grammar,
              std::span<const TokenId> drafts, cudaStream_t stream);
    void begin_dflash();
    void set_dflash_row(std::uint32_t row, std::uint32_t lane, std::uint32_t extent,
                        std::shared_ptr<text::GrammarState> grammar);
    void enqueue_dflash(const Tensor& drafts, cudaStream_t stream);
    void check() const;
private:
    static void CUDART_CB callback(void* self) noexcept;
    Tensor masks_;
    int words_;
    std::uint32_t width_, lanes_;
    PinnedHostBuffer host_masks_, host_drafts_;

    struct Row {
        std::uint32_t lane = 0, extent = 0;
        std::shared_ptr<text::GrammarState> grammar;
    };

    std::array<Row, 8> rows_{};
    std::exception_ptr error_;
};
} // namespace ninfer::models::qwen3_5
