#include "models/qwen4_exp/parallel/codec.h"
#include "models/qwen4_exp/parallel/vision.h"

#include <algorithm>
#include <cstring>

namespace ninfer::models::qwen4_exp {
namespace {
namespace wire = distributed::wire;

void integers(wire::Encoder& out, const std::vector<std::int32_t>& values) {
    out.u64(values.size());
    for (auto value : values) out.i32(value);
}

std::vector<std::int32_t> integers(wire::Decoder& in) {
    std::vector<std::int32_t> values(in.count(4));
    for (auto& value : values) value = in.i32();
    return values;
}

void grid(wire::Encoder& out, const qwen::VisionGrid& g) {
    out.i32(g.temporal);
    out.i32(g.height);
    out.i32(g.width);
}

qwen::VisionGrid grid(wire::Decoder& in) { return {in.i32(), in.i32(), in.i32()}; }

runtime::InputResource encode_input(std::uint64_t id,
                                    const std::shared_ptr<const qwen::VisionExecutionInput>& input,
                                    std::uint32_t rank, std::uint32_t ranks) {
    runtime::InputResource resource{rank, id, {}, {}, input};
    wire::Encoder out;
    out.u32(input->prompt_tokens);
    out.u64(input->plan.max_merged_count);
    out.u32(input->plan.control->prepared_item_begin);
    out.u64(input->plan.control->items.size());
    for (const auto& c : input->plan.control->items) {
        out.u8(static_cast<std::uint8_t>(c.modality));
        grid(out, c.grid);
        out.u64(c.patch_begin);
        out.u64(c.patch_count);
        out.u64(c.merged_count);
        out.i32(c.segment_length);
        out.i32(c.segment_count);
        integers(out, c.position_ids);
        integers(out, c.scatter_indices);
        integers(out, c.position_table_indices);
        out.u64(c.position_table_weights.size());
        for (auto weight : c.position_table_weights) out.f32(weight);
    }
    out.u64(input->plan.uses.size());
    for (const auto& use : input->plan.uses) {
        out.u32(use.begin);
        out.u32(use.end);
        out.u32(use.prepared_item_index);
        out.u32(use.control_index);
    }
    out.u64(input->vision_items.size());
    for (const auto& item : input->vision_items) {
        out.u8(static_cast<std::uint8_t>(item.modality));
        grid(out, item.grid);
        out.u64(item.patch_begin);
        out.u64(item.patch_count);
    }
    const auto owners = vision_owners(*input, ranks);
    out.u64(input->media_payloads.size());
    for (std::size_t i = 0; i < input->media_payloads.size(); ++i) {
        const auto& payload = input->media_payloads[i];
        const auto values =
            owners[i] == rank && payload ? payload->span() : std::span<const std::uint16_t>{};
        out.u64(values.size());
        resource.parts.push_back(std::as_bytes(values));
    }
    resource.metadata = std::move(out).take();
    return resource;
}
} // namespace

std::vector<runtime::InputResource>
ExecutionCodec::prepare_inputs(const Work& work, const distributed::ExecutionTopology& topology) {
    std::vector<runtime::InputResource> resources;
    const auto* context = std::get_if<ContextWork>(&work.operation);
    if (!context) return resources;
    for (const auto& binding : context->bindings) {
        if (!binding.vision) continue;
        auto found =
            std::find_if(input_owners_.begin(), input_owners_.end(),
                         [&](const auto& owner) { return owner.second.input == binding.vision; });
        if (found == input_owners_.end()) {
            const auto id = ++next_input_;
            inputs_.install(id, binding.vision);
            found = input_owners_.emplace(id, InputOwner{binding.vision}).first;
            for (const auto& rank : topology.ranks)
                resources.push_back(
                    encode_input(id, binding.vision, rank.rank, topology.ranks.size()));
        }
        ++found->second.references;
        const auto& r = binding.member.residency;
        if (!residency_inputs_.emplace(std::pair{r.lane.value, r.generation}, found->first).second)
            throw std::logic_error("Vision input residency is already installed");
    }
    return resources;
}

std::vector<runtime::InputRelease> ExecutionCodec::retire_inputs(const Work& work) {
    std::vector<runtime::InputRelease> releases;
    const auto* context = std::get_if<ContextWork>(&work.operation);
    if (!context) return releases;
    for (const auto& residency : context->releases) {
        const auto found = residency_inputs_.find({residency.lane.value, residency.generation});
        if (found == residency_inputs_.end()) continue;
        const auto id = found->second;
        if (--input_owners_.at(id).references == 0) {
            inputs_.release(id);
            input_owners_.erase(id);
            releases.push_back({id});
        }
        residency_inputs_.erase(found);
    }
    return releases;
}

std::optional<std::uint64_t> ExecutionCodec::receive_input(wire::Kind kind,
                                                           std::span<const std::byte> bytes) {
    wire::Decoder in(bytes);
    const auto id = in.u64();
    if (kind == wire::Kind::InputRelease) {
        inputs_.release(id);
        in.finish();
        return {};
    }
    if (kind == wire::Kind::InputBegin) {
        if (!contract_.vision) throw std::invalid_argument("Qwen4Exp Vision was not prepared");
        wire::Decoder data(in.bytes());
        in.finish();
        ReceivingInput r;
        r.input                      = std::make_shared<qwen::VisionExecutionInput>();
        auto& v                      = *r.input;
        v.prompt_tokens              = data.u32();
        v.plan.max_merged_count      = data.u64();
        auto control                 = std::make_shared<qwen::VisionControl>();
        control->prepared_item_begin = data.u32();
        control->items.resize(data.count(65));
        for (auto& c : control->items) {
            c.modality               = static_cast<qwen::Modality>(data.u8());
            c.grid                   = grid(data);
            c.patch_begin            = data.u64();
            c.patch_count            = data.u64();
            c.merged_count           = data.u64();
            c.segment_length         = data.i32();
            c.segment_count          = data.i32();
            c.position_ids           = integers(data);
            c.scatter_indices        = integers(data);
            c.position_table_indices = integers(data);
            c.position_table_weights.resize(data.count(4));
            for (auto& weight : c.position_table_weights) weight = data.f32();
        }
        v.plan.control = std::move(control);
        v.plan.uses.resize(data.count(16));
        for (auto& use : v.plan.uses) use = {data.u32(), data.u32(), data.u32(), data.u32()};
        v.vision_items.resize(data.count(29));
        for (auto& item : v.vision_items) {
            item.modality    = static_cast<qwen::Modality>(data.u8());
            item.grid        = grid(data);
            item.patch_begin = data.u64();
            item.patch_count = data.u64();
        }
        v.media_payloads.resize(data.count(8));
        r.parts.resize(v.media_payloads.size());
        r.received.resize(r.parts.size());
        for (std::size_t i = 0; i < r.parts.size(); ++i) {
            const auto elements = data.u64();
            if (!elements) continue;
            auto payload            = std::make_shared<qwen::PreparedMediaPayload>();
            payload->patch_elements = elements;
            payload->patches        = std::make_unique<std::uint16_t[]>(elements);
            v.media_payloads[i]     = payload;
            r.parts[i]              = std::move(payload);
        }
        data.finish();
        if (!receiving_.emplace(id, std::move(r)).second)
            throw std::logic_error("duplicate incoming Vision input");
        return {};
    }
    auto& r = receiving_.at(id);
    if (kind == wire::Kind::InputChunk) {
        const auto part = in.u64(), offset = in.u64();
        const auto chunk = in.bytes();
        in.finish();
        auto& payload = r.parts.at(part);
        if (!payload || offset != r.received[part] ||
            chunk.size() > payload->span().size_bytes() - offset)
            throw std::logic_error("Vision input chunk differs from its reserved part");
        std::memcpy(reinterpret_cast<std::byte*>(payload->patches.get()) + offset, chunk.data(),
                    chunk.size());
        r.received[part] += chunk.size();
        return {};
    }
    in.finish();
    for (std::size_t i = 0; i < r.parts.size(); ++i)
        if (r.parts[i] && r.received[i] != r.parts[i]->span().size_bytes())
            throw std::logic_error("Vision input was sealed before its payload arrived");
    inputs_.install(id, std::move(r.input));
    receiving_.erase(id);
    return id;
}

} // namespace ninfer::models::qwen4_exp
