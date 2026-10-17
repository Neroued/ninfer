#include "models/qwen4_exp/ple/upload.h"

#include "models/qwen4_exp/ple/parameters.h"
#include "core/buffer.h"
#include "core/transfer.h"

#include <exception>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::ple {

struct UploadSlots::Impl {
    struct Slot {
        PinnedHostBuffer host;
        DeviceBuffer device;
        CudaCompletionEvent uploaded, consumed;
        std::uint64_t generation = 0;
        std::uint32_t tokens = 0, batch = 0;
        cudaStream_t transfer = nullptr, consumer = nullptr;
        bool active = false, consumption_recorded = false;
        std::exception_ptr failure;

        Slot(const DeviceContext& context, std::size_t bytes)
            : host(bytes), device(context.device(), bytes), uploaded(context), consumed(context) {}

        void retire() noexcept {
            active = consumption_recorded = false;
            transfer = consumer = nullptr;
            failure             = {};
        }

        void synchronize() const {
            if (transfer) CUDA_CHECK(cudaStreamSynchronize(transfer));
            if (consumer && consumer != transfer) CUDA_CHECK(cudaStreamSynchronize(consumer));
        }
    };

    Device device;
    std::uint32_t max_positions;
    std::vector<Slot> slots;
    std::vector<MemorySpan> outputs;

    Impl(const DeviceContext& context, std::uint32_t positions, std::uint32_t count)
        : device(context.device()), max_positions(positions) {
        (void)UploadSlots::memory(positions, count);
        CudaDeviceScope scope(device);
        const auto bytes = std::size_t(positions) * kEmbeddingBytes;
        slots.reserve(count);
        outputs.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            slots.emplace_back(context, bytes);
            outputs.push_back(slots.back().host.span());
        }
    }

    ~Impl() {
        try {
            CudaDeviceScope scope(device);
            for (const auto& slot : slots) {
                if (slot.active) slot.synchronize();
            }
        } catch (...) { std::terminate(); }
    }

    Slot& current(Ticket ticket) {
        auto& slot = slots.at(ticket.slot);
        if (!slot.active || slot.generation != ticket.generation) {
            throw std::logic_error("PLE ticket has no current upload");
        }
        if (slot.failure) std::rethrow_exception(slot.failure);
        return slot;
    }

    void validate(CudaExecutionView execution) const {
        if (execution.device() != device || !execution.stream()) {
            throw std::invalid_argument(
                "PLE transfer/consumer belongs to another execution device");
        }
    }

    std::size_t bytes(std::uint32_t tokens, std::uint32_t batch) const {
        if (!tokens || !batch || batch > 8 || std::uint64_t(tokens) * batch > max_positions) {
            throw std::invalid_argument("invalid PLE upload shape");
        }
        return std::size_t(tokens) * batch * kEmbeddingBytes;
    }

    [[noreturn]] void submission_failed(Slot& slot) {
        const auto error = std::current_exception();
        // A failed event record can follow a successful copy/consumer submission. Drain the
        // recorded streams before making storage reusable; logical cancellation is the caller's.
        slot.failure = error;
        slot.synchronize();
        slot.retire();
        std::rethrow_exception(error);
    }
};

UploadSlots::UploadSlots(const DeviceContext& context, std::uint32_t max_positions,
                         std::uint32_t slots)
    : impl_(std::make_unique<Impl>(context, max_positions, slots)) {}

UploadMemory UploadSlots::memory(std::uint32_t positions, std::uint32_t slots) {
    if (!positions || positions > INT32_MAX || !slots ||
        std::size_t(slots) > SIZE_MAX / kEmbeddingBytes / positions)
        throw std::invalid_argument("invalid PLE upload capacity");
    const auto bytes = std::size_t(positions) * slots * kEmbeddingBytes;
    return {sizeof(Impl) + std::size_t(slots) * (sizeof(Impl::Slot) + sizeof(MemorySpan)), bytes,
            bytes};
}

UploadSlots::~UploadSlots() = default;

std::span<const MemorySpan> UploadSlots::host_slots() const noexcept { return impl_->outputs; }

void UploadSlots::upload(const HostEmbedding& input, CudaExecutionView transfer) {
    auto& p = *impl_;
    p.validate(transfer);
    const auto bytes = p.bytes(input.window.tokens, input.window.batch);
    auto& slot       = p.slots.at(input.ticket.slot);
    if (slot.active || !input.ticket.generation || input.ticket.generation <= slot.generation) {
        throw std::logic_error("PLE upload slot is busy or ticket is stale");
    }
    if (input.values.data != slot.host.data() || input.values.bytes != bytes ||
        input.values.device != Device::cpu() || !input.values.host_pinned) {
        throw std::invalid_argument("PLE upload does not reference its pinned slot");
    }
    slot.generation = input.ticket.generation;
    slot.tokens     = input.window.tokens;
    slot.batch      = input.window.batch;
    slot.active     = true;
    slot.transfer   = transfer.stream();
    try {
        copy_bytes_async(slot.device.span().subspan(0, bytes), input.values, transfer);
        slot.uploaded.record(transfer.stream());
    } catch (...) { p.submission_failed(slot); }
}

DeviceEmbedding UploadSlots::acquire(Ticket ticket, CudaExecutionView consumer) {
    auto& p = *impl_;
    p.validate(consumer);
    auto& slot = p.current(ticket);
    if (slot.consumption_recorded || (slot.consumer && slot.consumer != consumer.stream())) {
        throw std::logic_error("invalid PLE consumer ordering");
    }
    slot.consumer = consumer.stream();
    return binding(ticket.slot, slot.tokens, slot.batch);
}

void UploadSlots::consumer_submitted(Ticket ticket, CudaExecutionView consumer) {
    auto& p = *impl_;
    p.validate(consumer);
    auto& slot = p.current(ticket);
    if (slot.consumer != consumer.stream() || slot.consumption_recorded) {
        throw std::logic_error("PLE consumer completion has no matching acquisition");
    }
    slot.consumption_recorded = true;
}

bool UploadSlots::upload_complete(Ticket ticket) const {
    return impl_->current(ticket).uploaded.ready();
}

bool UploadSlots::try_release(Ticket ticket) {
    auto& slot = impl_->slots.at(ticket.slot);
    if (slot.failure) std::rethrow_exception(slot.failure);
    if (!ticket.generation || ticket.generation < slot.generation ||
        (slot.active && ticket.generation != slot.generation)) {
        throw std::logic_error("PLE upload release has a stale or mismatched ticket");
    }
    if (!slot.active) {
        slot.generation = ticket.generation;
        return true;
    }
    if (!slot.uploaded.ready() ||
        (slot.consumer && (!slot.consumption_recorded || !slot.consumed.ready()))) {
        return false;
    }
    slot.retire();
    return true;
}

Tensor UploadSlots::device_slot(std::uint32_t slot, std::uint32_t tokens,
                                std::uint32_t batch) const {
    auto& p          = *impl_;
    const auto bytes = p.bytes(tokens, batch);
    return Tensor(
        p.slots.at(slot).device.span().subspan(0, bytes), DType::BF16,
        {kEmbeddingElements, static_cast<std::int32_t>(tokens), static_cast<std::int32_t>(batch)});
}

DeviceEmbedding UploadSlots::binding(std::uint32_t slot, std::uint32_t tokens,
                                     std::uint32_t batch) const {
    auto& storage = impl_->slots.at(slot);
    return {device_slot(slot, tokens, batch), storage.uploaded, storage.consumed};
}

UploadMemory UploadSlots::memory() const noexcept {
    const auto& p = *impl_;
    return memory(p.max_positions, static_cast<std::uint32_t>(p.slots.size()));
}

} // namespace ninfer::models::qwen4_exp::ple
