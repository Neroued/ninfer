#include "models/glm5next/parallel/codec.h"

namespace ninfer::models::glm5next {
namespace {
namespace wire = distributed::wire;

void integers(wire::Encoder& out, std::span<const std::int32_t> values) {
    out.u64(values.size());
    for (auto value : values) out.i32(value);
}

std::vector<std::int32_t> integers(wire::Decoder& in) {
    std::vector<std::int32_t> values(in.count(4));
    for (auto& value : values) value = in.i32();
    return values;
}
} // namespace

std::vector<runtime::InputResource>
ExecutionCodec::prepare_inputs(const Work& work, const distributed::ExecutionTopology& topology) {
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    if (!forward || !forward->visual || !forward->visual->window.produce) return {};
    const auto& visual = *forward->visual;
    if (!visual.input || visual.input->batches.size() != topology.ranks.size())
        throw std::logic_error("Vision production does not cover the execution ranks");
    inputs_.install(visual);
    std::vector<runtime::InputResource> result;
    for (const auto& rank : topology.ranks) {
        const auto& batch = visual.input->batches.at(rank.rank);
        runtime::InputResource resource{rank.rank, visual.window.id, {}, {}, visual.input};
        wire::Encoder out;
        models::vision::encode_window(out, visual.window);
        out.u32(batch.features);
        integers(out, batch.positions);
        integers(out, batch.cu_seqlens);
        out.u64(batch.patches.size());
        for (auto part : batch.patches) {
            out.u64(part.size_bytes());
            resource.parts.push_back(std::as_bytes(part));
        }
        resource.metadata = std::move(out).take();
        result.push_back(std::move(resource));
    }
    return result;
}

std::vector<runtime::InputRelease> ExecutionCodec::retire_inputs(const Work& work) {
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    if (!forward || !forward->visual || !forward->visual->window.produce) return {};
    const auto id = forward->visual->window.id;
    inputs_.release_payload(id);
    return {{id}};
}

std::optional<std::uint64_t> ExecutionCodec::receive_input(wire::Kind kind,
                                                           std::span<const std::byte> bytes) {
    wire::Decoder in(bytes);
    const auto id = in.u64();
    if (kind == wire::Kind::InputRelease) {
        in.finish();
        inputs_.release_payload(id);
        receiver_.release(id);
        return {};
    }
    if (kind == wire::Kind::InputBegin) {
        if (!contract_.vision_max_features || receiving_)
            throw std::logic_error("Vision input receive slot is unavailable");
        wire::Decoder metadata(in.bytes());
        in.finish();
        auto use = models::vision::decode_window(metadata);
        if (use.id != id || use.plan->features > contract_.vision_max_features)
            throw std::invalid_argument("Vision input differs from its prepared window capacity");
        auto input = std::make_shared<vision::WindowInput>();
        input->batches.resize(ranks_);
        auto& batch      = input->batches.at(rank_);
        batch.features   = metadata.u32();
        batch.positions  = integers(metadata);
        batch.cu_seqlens = integers(metadata);
        std::vector<std::size_t> sizes(metadata.count(8));
        for (auto& size : sizes) {
            size = metadata.u64();
            if (size % sizeof(std::uint16_t))
                throw std::invalid_argument("Vision patch payload has an incomplete BF16 element");
        }
        metadata.finish();
        receiver_.begin(id, sizes);
        receiving_ = ReceivingInput{std::move(use), std::move(input)};
        return {};
    }
    if (!receiving_ || receiving_->window.id != id)
        throw std::logic_error("Vision input chunk has no active receiving window");
    if (kind == wire::Kind::InputChunk) {
        const auto part = in.u64(), offset = in.u64();
        const auto chunk = in.bytes();
        in.finish();
        receiver_.append(id, part, offset, chunk);
        return {};
    }
    if (kind != wire::Kind::InputEnd) throw std::invalid_argument("unknown Vision input action");
    in.finish();
    auto& input = *receiving_->input;
    for (auto part : receiver_.finish(id))
        input.batches.at(rank_).patches.emplace_back(static_cast<const std::uint16_t*>(part.data),
                                                     part.bytes / sizeof(std::uint16_t));
    inputs_.install({std::move(receiving_->window), std::move(receiving_->input)});
    receiving_.reset();
    return id;
}

} // namespace ninfer::models::glm5next
