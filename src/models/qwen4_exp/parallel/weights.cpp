#include "models/qwen4_exp/parallel/weights.h"

#include "artifact/reader.h"

namespace ninfer::models::qwen4_exp {

WeightSelection select_weights(const artifact::Reader& reader, const ModelDefinition& definition,
                               const RankPlan& rank, Device device) {
    WeightSelection out;
    out.bindings.resize(definition.weight_data().size());
    std::vector<bool> selected(out.bindings.size());
    const auto add = [&](artifact::EncodedTensor tensor, std::uint64_t domain = 0) {
        const auto index = out.requests.size();
        out.requests.push_back({std::move(tensor), device, artifact::HostAccess::CpuOnly, domain});
        return index;
    };
    for (const auto& a : rank.assemblies) {
        std::vector<artifact::MatrixSlice> slices;
        for (const auto& s : a.slices)
            append_parameter_slice(slices, reader, definition.parameter(s.parameter), s);
        const auto request = add(artifact::slice_tensor(reader, a.shape, slices));
        for (const auto& binding : a.bindings) {
            if (selected.at(binding.parameter.index))
                throw std::logic_error("parallel plan binds a parameter twice");
            selected[binding.parameter.index]     = true;
            out.bindings[binding.parameter.index] = {binding.shape,
                                                     {{request, binding.begin, binding.end}}};
        }
    }
    if (rank.size > 1) {
        const auto mark = [&](const MoeWeights& w) {
            for (const auto id : {w.gate, w.up, w.down}) selected[id.index] = true;
        };
        for (const auto& b : definition.weights().text.layers) mark(b.moe);
        if (definition.weights().mtp) mark(definition.weights().mtp->layer.moe);
        out.experts.resize(definition.expert_banks().size());
        for (std::size_t bank = 0; bank < out.experts.size(); ++bank) {
            const auto& b     = definition.expert_banks()[bank];
            const auto matrix = [&](const moe::MatrixSource& m, std::uint32_t e) {
                const artifact::MatrixSlice s{
                    m.object, m.layout.region.row_begin + e * m.layout.region.row_stride, 0,
                    m.layout.region.rows, m.layout.columns};
                const artifact::Shape shape{m.layout.region.rows, m.layout.columns};
                return add(artifact::slice_tensor(reader, shape, std::span(&s, 1)), bank + 1);
            };
            for (const auto expert : rank.experts.at(bank)) {
                const auto gu   = matrix(b.gate_up, expert);
                const auto down = matrix(b.down, expert);
                out.experts[bank].entries.push_back({expert, gu, down});
            }
        }
        for (std::uint32_t i = rank.ple_shards.begin;
             i < rank.ple_shards.begin + rank.ple_shards.size; ++i)
            out.ple.push_back(add(
                artifact::whole_tensor(reader, definition.ple_parameters().shards.at(i).object)));
    }
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (selected[i]) continue;
        const auto& reference = definition.parameter({i});
        auto& binding         = out.bindings[i];
        binding.shape         = reference.shape;
        for (const auto& part : reference.binding.parts)
            binding.parts.push_back(
                {add(artifact::whole_tensor(reader, part.object)), part.begin, part.end});
    }
    return out;
}

std::vector<BoundWeight> bind_selected_weights(const ModelDefinition& definition,
                                               const WeightSelection& selection,
                                               std::span<const artifact::StorageId> requests,
                                               std::span<const WeightParent> parents) {
    std::vector<BoundWeight> out;
    out.reserve(selection.bindings.size());
    for (std::size_t i = 0; i < selection.bindings.size(); ++i) {
        auto weight       = definition.weight_data()[i];
        const auto& local = selection.bindings[i];
        if (!local.parts.empty()) {
            weight.view = {local.shape, {}};
            for (const auto& part : local.parts)
                weight.view.parts.push_back(
                    {&parents[requests[part.request].index], part.begin, part.end});
        }
        out.push_back(std::move(weight));
    }
    return out;
}
} // namespace ninfer::models::qwen4_exp
