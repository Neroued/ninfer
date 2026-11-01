#include "models/common/vision/input.h"

#include <cstring>

namespace ninfer::models::vision {

void InputReceiver::bind(MemorySpan storage) {
    if (id_ || (storage.bytes && (storage.device != Device::cpu() || !storage.host_pinned)))
        throw std::invalid_argument("Vision input receive slot must be idle pinned Host storage");
    storage_ = storage;
}

void InputReceiver::begin(std::uint64_t id, std::span<const std::size_t> parts) {
    if (!id || id_) throw std::logic_error("Vision input receive slot is occupied");
    std::size_t offset = 0;
    std::vector<Part> layout;
    layout.reserve(parts.size());
    for (auto bytes : parts) {
        if (!bytes || bytes > storage_.bytes - offset)
            throw std::invalid_argument("Vision input exceeds the prepared staging capacity");
        layout.push_back({offset, bytes, 0});
        offset += bytes;
    }
    parts_ = std::move(layout);
    id_    = id;
}

void InputReceiver::append(std::uint64_t id, std::size_t part, std::size_t offset,
                           std::span<const std::byte> bytes) {
    if (id != id_) throw std::logic_error("Vision input chunk belongs to another window");
    auto& p = parts_.at(part);
    if (offset != p.received || bytes.size() > p.bytes - p.received)
        throw std::invalid_argument("Vision input chunk differs from its reserved part");
    std::memcpy(static_cast<std::byte*>(storage_.data) + p.offset + offset, bytes.data(),
                bytes.size());
    p.received += bytes.size();
}

std::vector<ConstMemorySpan> InputReceiver::finish(std::uint64_t id) const {
    if (id != id_) throw std::logic_error("Vision input completion belongs to another window");
    std::vector<ConstMemorySpan> parts;
    for (const auto& part : parts_) {
        if (part.received != part.bytes)
            throw std::logic_error("Vision input was sealed before its payload arrived");
        parts.push_back(storage_.subspan(part.offset, part.bytes));
    }
    return parts;
}

void InputReceiver::release(std::uint64_t id) {
    if (id != id_) throw std::logic_error("Vision input release belongs to another window");
    id_ = 0;
    parts_.clear();
}

void encode_window(distributed::wire::Encoder& out, const WindowUse& use) {
    out.u64(use.id);
    out.u64(use.owner.sequence.value);
    out.u64(use.owner.sequence.generation);
    out.u32(use.owner.residency.lane.value);
    out.u64(use.owner.residency.generation);
    const auto& plan = *use.plan;
    out.u32(plan.consume_end);
    out.u32(plan.features);
    out.u64(plan.units.size());
    for (const auto& u : plan.units) {
        out.u32(u.item);
        out.u32(u.index);
        out.u64(u.patch_begin);
        out.u32(u.patches);
        out.u32(u.token_begin);
        out.u32(u.features);
        out.u64(u.cost);
    }
    out.u64(plan.placements.size());
    for (const auto& p : plan.placements) {
        out.u32(p.unit);
        out.u32(p.rank);
        out.u32(p.feature_begin);
    }
    out.u64(plan.outputs.size());
    for (const auto& b : plan.outputs) {
        out.u32(b.rank);
        out.u32(b.feature_begin);
        out.u64(b.token_positions.size());
        for (auto p : b.token_positions) out.i32(p);
    }
}

WindowUse decode_window(distributed::wire::Decoder& in) {
    WindowUse use;
    use.id              = in.u64();
    use.owner.sequence  = {in.u64(), in.u64()};
    use.owner.residency = {{in.u32()}, in.u64()};
    auto plan           = std::make_shared<WindowPlan>();
    plan->consume_end   = in.u32();
    plan->features      = in.u32();
    plan->units.resize(in.count(36));
    for (auto& u : plan->units) {
        u.item        = in.u32();
        u.index       = in.u32();
        u.patch_begin = in.u64();
        u.patches     = in.u32();
        u.token_begin = in.u32();
        u.features    = in.u32();
        u.cost        = in.u64();
    }
    plan->placements.resize(in.count(12));
    for (auto& p : plan->placements) p = {in.u32(), in.u32(), in.u32()};
    plan->outputs.resize(in.count(16));
    for (auto& b : plan->outputs) {
        b.rank          = in.u32();
        b.feature_begin = in.u32();
        b.token_positions.resize(in.count(4));
        for (auto& p : b.token_positions) p = in.i32();
    }
    use.plan    = std::move(plan);
    use.produce = true;
    return use;
}

} // namespace ninfer::models::vision
