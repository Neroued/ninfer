#include "distributed/pipeline_link.h"
#include "core/transfer.h"

#include <limits>
#include <stdexcept>

namespace ninfer::distributed {
namespace {
template <class Span>
std::size_t payload_bytes(std::span<const Span> fields, Device device) {
    std::size_t bytes = 0;
    for (const auto& field : fields) {
        if (field.device != device || !field.data || !field.bytes ||
            field.bytes > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::invalid_argument("invalid pipeline boundary field");
        bytes += field.bytes;
    }
    return bytes;
}
} // namespace

PpStageLink::PpStageLink(const Group& group, DeviceContext& device, MemorySpan backing)
    : slots_{Slot(device), Slot(device)}, group_(group), stream_(device),
      transfer_(device.execution_view(stream_.stream())), backing_(backing) {
    if (group.size() != 2 || group.device() != device.device() ||
        backing.device != group.device() || !backing.data || !backing.bytes ||
        backing.bytes % slots)
        throw std::invalid_argument("pipeline link requires two endpoint buffers on its device");
}

GroupRequirements PpStageLink::requirements(const GroupDescription& group, RankId rank,
                                            std::size_t bytes) {
    if (group.usage != GroupUsage::PeerExchange || group.members.size() != 2 || !bytes)
        throw std::invalid_argument("invalid directed pipeline edge");
    const auto own = group.index(rank);
    return {group.id, {}, {{1U - own, own == 0 ? bytes : 0, own == 1 ? bytes : 0}}};
}

const CudaCompletionEvent& PpStageLink::completion(const Slot& slot) const {
    return group_.rank() == 0 ? slot.transferred : slot.copied;
}

PpStageLink::Slot* PpStageLink::reserve(std::uint64_t sequence, std::size_t bytes) {
    if (sequence <= last_) throw std::logic_error("stale pipeline transfer sequence");
    if (!bytes || bytes > backing_.bytes / slots)
        throw std::out_of_range("pipeline payload exceeds the prepared slot");
    auto& slot = slots_[next_];
    return !slot.sequence || completion(slot).ready() ? &slot : nullptr;
}

PpStageLink::Transfer PpStageLink::submitted(std::uint64_t sequence) {
    const Transfer receipt(*this, next_, sequence);
    slots_[next_].sequence = last_ = sequence;
    next_                          = (next_ + 1) % slots;
    return receipt;
}

std::optional<PpStageLink::Transfer> PpStageLink::try_send(std::uint64_t sequence,
                                                           std::span<const ConstMemorySpan> fields,
                                                           CudaExecutionView producer) {
    if (group_.rank() != 0 || producer.device() != group_.device())
        throw std::logic_error("invalid pipeline producer endpoint");
    const auto bytes = payload_bytes(fields, producer.device());
    auto* slot       = reserve(sequence, bytes);
    if (!slot) return {};
    auto packet        = backing_.subspan(next_ * (backing_.bytes / slots), bytes);
    std::size_t offset = 0;
    for (auto field : fields) {
        copy_bytes_async(packet.subspan(offset, field.bytes), field, producer);
        offset += field.bytes;
    }
    slot->copied.record(producer.stream());
    slot->copied.wait(transfer_.stream());
    const PeerTransfer send{1, packet, {}};
    group_.exchange({&send, 1}, transfer_);
    slot->transferred.record(transfer_.stream());
    return submitted(sequence);
}

std::optional<PpStageLink::Transfer> PpStageLink::try_receive(std::uint64_t sequence,
                                                              std::span<const MemorySpan> fields,
                                                              CudaExecutionView consumer) {
    if (group_.rank() != 1 || consumer.device() != group_.device())
        throw std::logic_error("invalid pipeline consumer endpoint");
    const auto bytes = payload_bytes(fields, consumer.device());
    auto* slot       = reserve(sequence, bytes);
    if (!slot) return {};
    auto packet = backing_.subspan(next_ * (backing_.bytes / slots), bytes);
    const PeerTransfer receive{0, {}, packet};
    group_.exchange({&receive, 1}, transfer_);
    slot->transferred.record(transfer_.stream());
    slot->transferred.wait(consumer.stream());
    std::size_t offset = 0;
    for (auto field : fields) {
        copy_bytes_async(field, packet.subspan(offset, field.bytes), consumer);
        offset += field.bytes;
    }
    slot->copied.record(consumer.stream());
    return submitted(sequence);
}

bool PpStageLink::retired(Transfer receipt) const {
    if (receipt.link_ != this) throw std::logic_error("pipeline transfer belongs to another link");
    const auto& slot = slots_[receipt.slot_];
    return slot.sequence > receipt.sequence_ || completion(slot).ready();
}

void PpStageLink::drain() const {
    for (const auto& slot : slots_)
        if (slot.sequence) completion(slot).synchronize();
}

} // namespace ninfer::distributed
