#pragma once

#include "models/qwen4_exp/ple/upload.h"
#include "runtime/executor/contract.h"

#include <optional>

namespace ninfer::models::qwen4_exp {

class Resources;

class PlePipeline {
public:
    PlePipeline(Resources&, const DeviceContext&);
    ~PlePipeline();
    // A missing ticket means the required slot is still retiring. Successful calls claim
    // matching prefetch work for this execution and copy all input identities and IDs.
    [[nodiscard]] std::optional<ple::Ticket> try_require(runtime::ExecutionTicket,
                                                         std::uint32_t tokens,
                                                         std::uint32_t physical_rows,
                                                         std::span<const ple::InputRow>);
    [[nodiscard]] std::optional<ple::Ticket> try_begin(runtime::ExecutionTicket,
                                                       std::uint32_t tokens,
                                                       std::uint32_t physical_rows,
                                                       std::span<const ple::InputRow>);
    void append_device(ple::Ticket, const Tensor& ids, std::span<const std::uint32_t> extents,
                       std::uint32_t step);
    // Capture only the CUDA handoff. Host bookkeeping follows the submitted chain replay.
    void capture_append(std::uint32_t slot, const Tensor& ids, std::uint32_t step,
                        CudaExecutionView);
    void append_submitted(ple::Ticket, std::span<const std::uint32_t> extents, std::uint32_t step);
    void seal(ple::Ticket, std::span<const std::uint32_t> valid_columns);
    [[nodiscard]] bool input_ready(ple::Ticket);
    // IDs are available once appends have retired, before embedding preparation completes.
    // The borrowed input remains valid until progress() retires this window.
    [[nodiscard]] std::span<const std::int32_t> input(ple::Ticket, std::uint32_t row);
    void prefetch(std::uint32_t tokens, std::span<const ple::InputRow>);
    // Returns after upload submission; the consumer waits on DeviceEmbedding::ready and
    // records DeviceEmbedding::consumed after its last reader, including graph replay.
    [[nodiscard]] std::optional<ple::DeviceEmbedding> try_consume(ple::Ticket);
    void consumer_submitted(ple::Ticket);
    void progress();
    void cancel(runtime::ExecutionTicket);
    // Drops unclaimed prefetch only; an owned batch is cancelled through its execution ticket.
    void cancel_sequence(std::uint64_t sequence);
    [[nodiscard]] bool retired(runtime::ExecutionTicket) const;
    [[nodiscard]] bool idle() const;
    // Terminal shutdown; stops the Host producer and retires CUDA consumers. Repeated calls
    // are harmless. Ordinary execution advances through progress() and readiness queries.
    void drain();
private:
    struct Window {
        ple::Ticket ticket;
        std::optional<runtime::ExecutionTicket> owner;
        std::uint32_t tokens = 0, physical_rows = 0;
        bool allocated = false;
        std::vector<ple::SequenceIdentity> rows;
        std::vector<std::int32_t> ids;
        std::vector<std::uint32_t> known, valid;
        void reset();
        bool seal_requested = false, sealed = false;
        std::uint32_t append_count = 0, next_append = 0;
        bool uploaded = false, acquired = false, consumed = false, cancelled = false;
        std::exception_ptr error;
    };

    struct DeviceAppend {
        CudaCompletionEvent produced, copied;
        ple::Ticket ticket{};
        std::vector<std::uint8_t> active;
        std::uint32_t batch = 0;
        bool pending        = false;

        DeviceAppend(const DeviceContext& device, std::uint32_t capacity)
            : produced(device), copied(device), active(capacity) {}
    };

    [[nodiscard]] bool matches(const Window&, std::uint32_t, std::uint32_t,
                               std::span<const ple::InputRow>) const;
    [[nodiscard]] std::optional<ple::Ticket> prepare(std::uint32_t, std::uint32_t,
                                                     std::span<const ple::InputRow>,
                                                     std::optional<runtime::ExecutionTicket> owner);
    Window& window(ple::Ticket);
    [[nodiscard]] MemorySpan append_destination(std::uint32_t slot, std::uint32_t step,
                                                std::uint32_t batch);
    void abandon_consumer(Window&);
    void cancel(Window&);
    Resources& resources_;
    CudaExecutionView cuda_;
    std::array<Window, 2> windows_;
    std::vector<ple::Append> additions_;
    std::vector<DeviceAppend> appends_;
    bool drained_ = false;
};

} // namespace ninfer::models::qwen4_exp
