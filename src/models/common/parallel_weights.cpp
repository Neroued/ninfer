#include "models/common/parallel_weights.h"

#include "artifact/reader.h"

#include <algorithm>

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
        const auto& source    = experts[bank].bank;
        const auto& partition = experts[bank].partition;
        for (const auto parameter : experts[bank].parameters) claim(parameter);
        const auto count        = source.gate_up.layout.region.count;
        const auto hidden       = source.gate_up.layout.columns;
        const auto intermediate = source.down.layout.columns;
        const auto range        = partition.intermediate;
        if (partition.ids.empty() || !range.size || range.begin > intermediate ||
            range.size > intermediate - range.begin ||
            source.gate_up.layout.region.rows != 2 * intermediate ||
            source.down.layout.region.rows != hidden || source.down.layout.region.count != count)
            throw artifact::ArtifactError("expert partition differs from its source bank");
        std::vector<bool> selected_ids(count);
        for (const auto id : partition.ids) {
            if (id >= count || selected_ids[id])
                throw artifact::ArtifactError("expert partition has invalid or duplicate IDs");
            selected_ids[id] = true;
        }
        auto& selection = out.experts[bank];
        if (partition.ids.size() == count) {
            if (!std::is_sorted(partition.ids.begin(), partition.ids.end()))
                throw artifact::ArtifactError("affine expert partition requires global ID order");
            std::vector<artifact::MatrixSlice> gu, down;
            for (const auto id : partition.ids) {
                const auto& g  = source.gate_up;
                const auto& d  = source.down;
                const auto row = g.layout.region.row_begin + id * g.layout.region.row_stride;
                gu.push_back({g.object, row + range.begin, 0, range.size, hidden,
                              std::uint64_t(id) * 2 * range.size, 0});
                gu.push_back({g.object, row + intermediate + range.begin, 0, range.size, hidden,
                              (std::uint64_t(id) * 2 + 1) * range.size, 0});
                down.push_back({d.object,
                                d.layout.region.row_begin + id * d.layout.region.row_stride,
                                range.begin, hidden, range.size, std::uint64_t(id) * hidden, 0});
            }
            const artifact::Shape gu_shape{count, 2ULL * range.size, hidden};
            const artifact::Shape down_shape{count, hidden, range.size};
            selection.affine = {add(artifact::slice_tensor(reader, gu_shape, gu), bank + 1),
                                add(artifact::slice_tensor(reader, down_shape, down), bank + 1)};
        } else {
            if (range.begin != 0 || range.size != intermediate)
                throw artifact::ArtifactError("indexed expert partition requires whole experts");
            const auto matrix = [&](const moe::MatrixSource& source, std::uint32_t expert) {
                const artifact::MatrixSlice slice{
                    source.object,
                    source.layout.region.row_begin + expert * source.layout.region.row_stride, 0,
                    source.layout.region.rows, source.layout.columns};
                const artifact::Shape shape{source.layout.region.rows, source.layout.columns};
                return add(artifact::slice_tensor(reader, shape, std::span(&slice, 1)), bank + 1);
            };
            for (const auto expert : partition.ids) {
                const auto gate_up = matrix(source.gate_up, expert);
                const auto down    = matrix(source.down, expert);
                selection.entries.push_back({expert, gate_up, down});
            }
        }
        const auto gu =
            selection.affine ? selection.affine->gate_up : selection.entries.front().gate_up;
        const auto dn = selection.affine ? selection.affine->down : selection.entries.front().down;
        out.expert_descriptions.push_back(
            {out.requests[gu].tensor.geometry, out.requests[dn].tensor.geometry, count,
             selection.affine ? ops::ExpertSubsetKind::All : ops::ExpertSubsetKind::MaskOnes});
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

std::vector<moe::ResidentExpertBank>
bind_selected_experts(const SelectedWeights& selection,
                      std::span<const artifact::StorageId> requests,
                      std::span<const WeightParent> parents, Device device) {
    const auto parent = [&](std::size_t request) -> const WeightParent& {
        return parents[requests[request].index];
    };
    std::vector<moe::ResidentExpertBank> out;
    out.reserve(selection.experts.size());
    for (std::size_t i = 0; i < selection.experts.size(); ++i) {
        const auto& bank = selection.experts[i];
        if (bank.affine) {
            out.emplace_back(parent(bank.affine->gate_up), parent(bank.affine->down));
        } else {
            std::vector<moe::ResidentExpertEntry> entries;
            for (const auto& e : bank.entries)
                entries.push_back({e.expert, &parent(e.gate_up), &parent(e.down)});
            out.emplace_back(entries, selection.expert_descriptions[i].count, device);
        }
    }
    return out;
}

} // namespace ninfer::models
