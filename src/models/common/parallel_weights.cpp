#include "models/common/parallel_weights.h"

#include "artifact/reader.h"

namespace ninfer::models {

SelectedWeights select_weights(const artifact::Reader& reader,
                               std::span<const artifact::ParameterReference> references,
                               std::span<const ParameterAssembly> assemblies,
                               std::span<const ExpertWeightPartition> experts, Device device) {
    SelectedWeights out;
    out.bindings.resize(references.size());
    std::vector<bool> selected(references.size());
    const auto add = [&](artifact::EncodedTensor tensor, std::uint64_t domain = 0) {
        const auto index = out.requests.size();
        out.requests.push_back({std::move(tensor), device, artifact::HostAccess::CpuOnly, domain});
        return index;
    };
    const auto claim = [&](WeightId parameter) {
        if (selected.at(parameter.index))
            throw std::logic_error("parallel plan binds a parameter twice");
        selected[parameter.index] = true;
    };
    for (const auto& assembly : assemblies) {
        std::vector<artifact::MatrixSlice> slices;
        for (const auto& slice : assembly.slices) {
            if (slice.parameter.index >= references.size())
                throw std::logic_error("parallel slice references an unknown parameter");
            append_parameter_slice(slices, reader, references[slice.parameter.index], slice);
        }
        const auto request = add(artifact::slice_tensor(reader, assembly.shape, slices));
        for (const auto& binding : assembly.bindings) {
            claim(binding.parameter);
            out.bindings[binding.parameter.index] = {binding.shape,
                                                     {{request, binding.begin, binding.end}}};
        }
    }
    out.experts.resize(experts.size());
    for (std::size_t bank = 0; bank < experts.size(); ++bank) {
        const auto& partition = experts[bank];
        for (const auto parameter : partition.parameters) claim(parameter);
        const auto matrix = [&](const moe::MatrixSource& source, std::uint32_t expert) {
            const artifact::MatrixSlice slice{source.object,
                                              source.layout.region.row_begin +
                                                  expert * source.layout.region.row_stride,
                                              0, source.layout.region.rows, source.layout.columns};
            const artifact::Shape shape{source.layout.region.rows, source.layout.columns};
            return add(artifact::slice_tensor(reader, shape, std::span(&slice, 1)), bank + 1);
        };
        for (const auto expert : partition.experts) {
            const auto gate_up = matrix(partition.bank.gate_up, expert);
            const auto down    = matrix(partition.bank.down, expert);
            out.experts[bank].entries.push_back({expert, gate_up, down});
        }
    }
    for (std::size_t i = 0; i < references.size(); ++i) {
        if (selected[i]) continue;
        const auto& reference = references[i];
        auto& binding         = out.bindings[i];
        binding.shape         = reference.shape;
        for (const auto& part : reference.binding.parts)
            binding.parts.push_back(
                {add(artifact::whole_tensor(reader, part.object)), part.begin, part.end});
    }
    return out;
}

std::vector<BoundWeight> bind_selected_weights(std::span<const BoundWeight> descriptions,
                                               const SelectedWeights& selection,
                                               std::span<const artifact::StorageId> requests,
                                               std::span<const WeightParent> parents) {
    if (descriptions.size() != selection.bindings.size())
        throw std::logic_error("selected weight count differs from bound descriptions");
    std::vector<BoundWeight> out;
    out.reserve(descriptions.size());
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
        auto weight       = descriptions[i];
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

} // namespace ninfer::models
