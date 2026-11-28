#include "models/common/qwen/vision/wire.h"

#include <bit>
#include <cstring>

namespace ninfer::models::qwen {
namespace {
namespace wire = distributed::wire;

// Streaming uses the same little-endian scalar representation as the wire encoder.
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(float) == sizeof(std::uint32_t));

struct StreamOwner {
    std::shared_ptr<const VisionWindowInput> input;
    wire::Bytes descriptor;
};

bool same_member(const runtime::ExecutionMember& a, const runtime::ExecutionMember& b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}

auto coordinates(VisionEncoderBatch& batch) {
    return std::array{std::as_writable_bytes(std::span(batch.positions)),
                      std::as_writable_bytes(std::span(batch.position_indices)),
                      std::as_writable_bytes(std::span(batch.position_weights))};
}

void append(std::span<std::byte> target, std::size_t& received, std::size_t offset,
            std::span<const std::byte> bytes) {
    if (offset != received || bytes.size() > target.size() - received)
        throw std::invalid_argument("Vision input chunk differs from its reserved part");
    if (!bytes.empty()) std::memcpy(target.data() + offset, bytes.data(), bytes.size());
    received += bytes.size();
}
} // namespace

void VisionInputBindings::install(VisionWork work) {
    if (!work.window.id || !work.window.plan || !work.input ||
        !work.window.owner.sequence.generation || !work.window.owner.residency.generation ||
        (current_ && work.window.id <= current_->window.id))
        throw std::invalid_argument("Vision input window is empty or stale");
    current_ = std::move(work);
}

void VisionInputBindings::release_payload(std::uint64_t id) {
    if (!current_ || current_->window.id != id || !current_->input)
        throw std::logic_error("Vision input payload has no matching production");
    current_->input.reset();
}

VisionWork VisionInputBindings::get(std::uint64_t id, bool produce,
                                    const runtime::ExecutionMember& owner) const {
    if (!current_ || current_->window.id != id || (produce && !current_->input) ||
        !same_member(current_->window.owner, owner))
        throw std::logic_error("Vision work has no installed window input for its owner");
    auto work           = *current_;
    work.window.produce = produce;
    if (!produce) work.input.reset();
    return work;
}

std::vector<runtime::InputResource>
stream_vision_inputs(const VisionWork& work, std::span<const distributed::RankId> ranks) {
    if (!work.input || work.input->batches.size() != ranks.size())
        throw std::logic_error("Vision production does not cover the execution ranks");
    std::vector<runtime::InputResource> result;
    result.reserve(ranks.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        const auto& batch = work.input->batches[rank];
        wire::Encoder descriptor;
        vision::encode_window(descriptor, work.window);
        descriptor.u32(batch.features);
        descriptor.u64(batch.cu_seqlens.size());
        for (auto value : batch.cu_seqlens) descriptor.i32(value);
        descriptor.u64(batch.patches.size());
        for (auto part : batch.patches) descriptor.u64(part.size_bytes());
        auto owner =
            std::make_shared<StreamOwner>(StreamOwner{work.input, std::move(descriptor).take()});
        wire::Encoder begin;
        begin.u64(owner->descriptor.size());
        runtime::InputResource resource{ranks[rank],
                                        work.window.id,
                                        std::move(begin).take(),
                                        {owner->descriptor,
                                         std::as_bytes(std::span(batch.positions)),
                                         std::as_bytes(std::span(batch.position_indices)),
                                         std::as_bytes(std::span(batch.position_weights))},
                                        owner};
        for (auto part : batch.patches) resource.parts.push_back(std::as_bytes(part));
        result.push_back(std::move(resource));
    }
    return result;
}

void VisionInputReceiver::decode_descriptor(Pending& pending, std::uint32_t max_features,
                                            std::uint32_t rank, std::uint32_t ranks) {
    wire::Decoder in(pending.descriptor);
    auto window      = vision::decode_window(in);
    const auto& plan = *window.plan;
    if (window.id != pending.id || !plan.features || plan.features > max_features ||
        !window.owner.sequence.generation || !window.owner.residency.generation)
        throw std::invalid_argument("Vision input differs from its prepared window capacity");

    auto input = std::make_shared<VisionWindowInput>();
    input->batches.resize(ranks);
    auto& batch    = input->batches.at(rank);
    batch.features = in.u32();
    batch.cu_seqlens.resize(in.count(4));
    for (auto& value : batch.cu_seqlens) value = in.i32();
    std::vector<std::size_t> sizes(in.count(8));
    for (auto& size : sizes) {
        size = in.u64();
        if (!size || size % sizeof(std::uint16_t))
            throw std::invalid_argument("Vision patch payload has an incomplete BF16 element");
    }
    in.finish();

    if (batch.cu_seqlens.size() != sizes.size() + 1 || batch.cu_seqlens.front() != 0)
        throw std::invalid_argument("Vision patch payload differs from its encoder segments");
    std::size_t segment    = 0;
    std::uint64_t features = 0;
    for (const auto& placement : plan.placements) {
        if (placement.rank != rank) continue;
        const auto& unit = plan.units.at(placement.unit);
        if (segment >= sizes.size() || !unit.patches ||
            std::int64_t(batch.cu_seqlens[segment + 1]) - batch.cu_seqlens[segment] !=
                unit.patches ||
            sizes[segment] / sizeof(std::uint16_t) < unit.patches)
            throw std::invalid_argument("Vision patch payload differs from its window placement");
        ++segment;
        features += unit.features;
    }
    if (segment != sizes.size() || features != batch.features || features > plan.features)
        throw std::invalid_argument("Vision feature count differs from its window placement");
    patches_.begin(pending.id, sizes);
    const auto count = static_cast<std::size_t>(batch.cu_seqlens.back());
    batch.positions.resize(2 * count);
    batch.position_indices.resize(4 * count);
    batch.position_weights.resize(4 * count);
    pending.window     = std::move(window);
    pending.input      = std::move(input);
    pending.descriptor = {};
}

std::optional<std::uint64_t> VisionInputReceiver::receive(wire::Kind kind,
                                                          std::span<const std::byte> bytes,
                                                          std::uint32_t max_features,
                                                          std::uint32_t rank, std::uint32_t ranks,
                                                          VisionInputBindings& bindings) {
    wire::Decoder in(bytes);
    const auto id = in.u64();
    if (kind == wire::Kind::InputRelease) {
        in.finish();
        bindings.release_payload(id);
        patches_.release(id);
        return {};
    }
    if (kind == wire::Kind::InputBegin) {
        if (!id || !max_features || pending_ || patches_.occupied())
            throw std::logic_error("Vision input receive slot is unavailable");
        wire::Decoder metadata(in.bytes());
        in.finish();
        const auto size = metadata.u64();
        metadata.finish();
        // Each unit consumes at least one feature. Charge units, placements, output
        // headers/scatter, segment ends, and patch part lengths to that domain.
        if (!size || size > 128 + 96ULL * max_features)
            throw std::invalid_argument("Vision window descriptor exceeds its prepared capacity");
        pending_.emplace(Pending{.id = id, .descriptor = wire::Bytes(size)});
        return {};
    }
    if (!pending_ || pending_->id != id)
        throw std::logic_error("Vision input has no active receiving window");
    auto& pending = *pending_;
    if (kind == wire::Kind::InputChunk) {
        const auto part   = in.u64();
        const auto offset = in.u64();
        const auto chunk  = in.bytes();
        in.finish();
        if (!pending.input) {
            if (part != 0) throw std::invalid_argument("Vision window descriptor has not arrived");
            append(pending.descriptor, pending.descriptor_received, offset, chunk);
            if (pending.descriptor_received == pending.descriptor.size())
                decode_descriptor(pending, max_features, rank, ranks);
        } else if (part >= 1 && part <= pending.received.size()) {
            auto metadata = coordinates(pending.input->batches.at(rank));
            append(metadata[part - 1], pending.received[part - 1], offset, chunk);
        } else if (part > pending.received.size()) {
            patches_.append(id, part - 1 - pending.received.size(), offset, chunk);
        } else {
            throw std::invalid_argument("Vision window descriptor already arrived");
        }
        return {};
    }
    if (kind != wire::Kind::InputEnd) throw std::invalid_argument("unknown Vision input action");
    in.finish();
    if (!pending.input)
        throw std::logic_error("Vision input completed before the window descriptor arrived");
    auto& batch         = pending.input->batches.at(rank);
    const auto metadata = coordinates(batch);
    for (std::size_t i = 0; i < metadata.size(); ++i)
        if (pending.received[i] != metadata[i].size())
            throw std::logic_error("Vision input completed before its coordinates arrived");
    for (auto part : patches_.finish(id))
        batch.patches.emplace_back(static_cast<const std::uint16_t*>(part.data),
                                   part.bytes / sizeof(std::uint16_t));
    bindings.install({std::move(pending.window), std::move(pending.input)});
    pending_.reset();
    return id;
}

} // namespace ninfer::models::qwen
