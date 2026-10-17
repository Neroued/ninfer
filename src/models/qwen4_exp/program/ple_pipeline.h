#pragma once

#include "models/qwen4_exp/program/resources.h"

namespace ninfer::models::qwen4_exp {

class PlePipeline {
public:
    PlePipeline(Resources&, const DeviceContext&);
    ~PlePipeline();
    [[nodiscard]] ple::Ticket require(std::uint32_t tokens, std::span<const ple::InputRow>);
    [[nodiscard]] ple::Ticket begin(std::uint32_t tokens, std::span<const ple::InputRow>);
    void append_device(ple::Ticket, const Tensor& ids, std::span<const std::uint32_t> extents,
                       std::uint32_t step);
    // Capture only the CUDA handoff. Host bookkeeping follows the submitted chain replay.
    void capture_append(std::uint32_t slot, const Tensor& ids, std::uint32_t step,
                        CudaExecutionView);
    void append_submitted(ple::Ticket, std::span<const std::uint32_t> extents, std::uint32_t step);
    void seal(ple::Ticket, std::span<const std::uint32_t> valid_columns);
    [[nodiscard]] std::span<const std::int32_t> input(ple::Ticket, std::uint32_t row);
    void prefetch(std::uint32_t tokens, std::span<const ple::InputRow>);
    [[nodiscard]] ple::DeviceEmbedding consume(ple::Ticket);
    void consumed(ple::Ticket);
    void progress();
    void wait_compute();
    void cancel(std::uint64_t sequence);
    void drain();
private:
    struct Window {
        ple::Ticket ticket;
        std::uint32_t tokens;
        std::vector<ple::SequenceIdentity> rows;
        std::vector<std::int32_t> ids;
        std::array<std::uint32_t, kMaximumConcurrency> known{}, valid{};
        bool seal_requested = false, sealed = false;
        std::uint32_t append_count = 0, next_append = 0;
        bool uploaded = false, acquired = false, consumed = false, cancelled = false;
        std::exception_ptr error;
    };

    struct DeviceAppend {
        CudaCompletionEvent produced, copied;
        ple::Ticket ticket{};
        std::array<bool, kMaximumConcurrency> active{};
        std::uint32_t batch = 0;
        bool pending        = false;

        explicit DeviceAppend(const DeviceContext& device) : produced(device), copied(device) {}
    };

    [[nodiscard]] bool matches(const Window&, std::uint32_t, std::span<const ple::InputRow>) const;
    [[nodiscard]] std::optional<ple::Ticket> prepare(std::uint32_t, std::span<const ple::InputRow>,
                                                     bool);
    Window& window(ple::Ticket);
    [[nodiscard]] MemorySpan append_destination(std::uint32_t slot, std::uint32_t step,
                                                std::uint32_t batch);
    void abandon_consumer(Window&);
    Resources& resources_;
    CudaExecutionView cuda_;
    CudaCompletionEvent completed_;
    std::array<std::optional<Window>, 2> windows_;
    std::vector<DeviceAppend> appends_;
};

} // namespace ninfer::models::qwen4_exp
