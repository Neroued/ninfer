#pragma once

#include "runtime/executor/execution_pool.h"

namespace ninfer::runtime {

template <class Types>
class Executor {
public:
    virtual ~Executor()                                                              = default;
    virtual void bind_to_current_thread()                                            = 0;
    [[nodiscard]] virtual SessionEpoch epoch() const noexcept                        = 0;
    [[nodiscard]] virtual ExecutionCapacity capacity() const noexcept                = 0;
    [[nodiscard]] virtual std::optional<ExecutionReservation> try_reserve(FrameKind) = 0;
    [[nodiscard]] virtual ExecutionTicket submit(ExecutionReservation, typename Types::Work) = 0;
    virtual void supply(ExecutionTicket, DecisionInput)                                      = 0;
    [[nodiscard]] virtual CommitReceipt commit(ExecutionTicket, typename Types::Commit)      = 0;
    virtual void discard(ExecutionTicket)                                                    = 0;
    [[nodiscard]] virtual std::optional<ExecutionEvent<Types>> poll()                        = 0;
    virtual void flush()                                                                     = 0;
    // Closes submission and completes all physical readers before control owners are released.
    virtual void drain()                                         = 0;
    virtual void fail(std::string message)                       = 0;
    [[nodiscard]] virtual bool failed() const noexcept           = 0;
    [[nodiscard]] virtual bool has_pending_work() const noexcept = 0;
    [[nodiscard]] virtual MemorySummary memory_summary()         = 0;
    virtual void reset_memory_peaks() noexcept                   = 0;
};

} // namespace ninfer::runtime
