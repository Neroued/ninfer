#include "models/common/parallel_weights.h"

#include "artifact/reader.h"

#include <algorithm>

namespace ninfer::models {
namespace {

void append_expert_slice(std::vector<artifact::MatrixSlice>& slices, artifact::MatrixSlice next) {
    if (!slices.empty()) {
        auto& prev = slices.back();
        if (prev.object == next.object && prev.column == 0 && next.column == 0 &&
            prev.columns == next.columns && prev.destination_column == 0 &&
            next.destination_column == 0 && prev.row + prev.rows == next.row &&
            prev.destination_row + prev.rows == next.destination_row) {
            prev.rows += next.rows;
            return;
        }
    }
    slices.push_back(next);
}

} // namespace

std::uint64_t rank_weight_digest(const artifact::ArtifactId& artifact, LoadOptions options,
                                 std::span<const std::uint64_t> rank_geometry,
                                 std::span<const WeightId> whole, std::span<const WeightId> omitted,
                                 std::span<const ParameterAssembly> assemblies,
                                 std::span<const ExpertPartition> experts) {
    std::uint64_t identity = 14695981039346656037ULL;
    const auto mix         = [&](std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) {
            identity ^= (value >> (i * 8)) & 255U;
            identity *= 1099511628211ULL;
        }
    };
    for (const auto byte : artifact) mix(std::to_integer<unsigned>(byte));
    mix(options.vision);
    mix(static_cast<std::uint64_t>(options.purpose));
    mix(static_cast<std::uint64_t>(options.speculative));
    mix(static_cast<std::uint64_t>(options.proposal_head));
    for (const auto value : rank_geometry) mix(value);
    mix(whole.size());
    for (const auto id : whole) mix(id.index);
    mix(omitted.size());
    for (const auto id : omitted) mix(id.index);
    mix(assemblies.size());
    for (const auto& a : assemblies) {
        mix(a.shape.size());
        for (const auto d : a.shape) mix(d);
        mix(a.slices.size());
        for (const auto& s : a.slices) {
            mix(s.parameter.index);
            mix(s.row);
            mix(s.column);
            mix(s.rows);
            mix(s.columns);
            mix(s.destination_row);
            mix(s.destination_column);
        }
        mix(a.bindings.size());
        for (const auto& b : a.bindings) {
            mix(b.parameter.index);
            mix(b.shape.size());
            for (const auto d : b.shape) mix(d);
            mix(b.begin);
            mix(b.end);
        }
    }
    mix(experts.size());
    for (const auto& bank : experts) {
        mix(bank.intermediate.begin);
        mix(bank.intermediate.size);
        mix(bank.ids.size());
        for (const auto expert : bank.ids) mix(expert);
    }
    return identity ? identity : 1ULL;
}

SelectedWeights select_weights(const artifact::Reader& reader,
                               std::span<const artifact::ParameterReference> references,
                               std::span<const BoundWeight> descriptions,
                               std::span<const WeightId> whole,
                               std::span<const ParameterAssembly> assemblies,
                               std::span<const ExpertWeightPartition> experts,
                               std::span<const WeightId> omitted, Device device) {
    if (descriptions.size() != references.size())
        throw std::logic_error("weight calibration descriptions differ from parameters");
    SelectedWeights out;
    out.bindings.resize(references.size());
    out.omitted.assign(references.size(), false);
    enum class Role { Unclassified, Weight, Expert, Omitted };
    std::vector<Role> classified(references.size(), Role::Unclassified);
    const auto add = [&](artifact::EncodedTensor tensor, std::uint64_t domain = 0) {
        const auto index = out.requests.size();
        out.requests.push_back({std::move(tensor), device, artifact::HostAccess::CpuOnly, domain});
        return index;
    };
    const auto claim = [&](WeightId parameter, Role role) {
        if (parameter.index >= references.size())
            throw std::logic_error("parallel plan classifies an unknown parameter");
        if (classified[parameter.index] != Role::Unclassified)
            throw std::logic_error("parallel plan classifies parameter '" +
                                   references[parameter.index].name + "' twice");
        classified[parameter.index] = role;
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
            claim(binding.parameter, Role::Weight);
            const auto elements = out.requests[request].tensor.geometry.elements;
            if (binding.begin > binding.end || binding.end > elements ||
                weight_element_count(binding.shape) != binding.end - binding.begin)
                throw artifact::ArtifactError("parallel assembly binding exceeds its encodings");
            out.bindings[binding.parameter.index] = {binding.shape,
                                                     {{request, binding.begin, binding.end}}};
        }
    }
    out.experts.resize(experts.size());
    for (std::size_t bank = 0; bank < experts.size(); ++bank) {
        const auto& source    = experts[bank].bank;
        const auto& partition = experts[bank].partition;
        for (const auto parameter : experts[bank].parameters) claim(parameter, Role::Expert);
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
        auto& selection       = out.experts[bank];
        const bool complete   = partition.ids.size() == count;
        const bool compressed = source.gate_up.layout.layout == QuantLayout::Csf32 ||
                                source.down.layout.layout == QuantLayout::Csf32;
        if (complete || compressed) {
            if (complete && !std::is_sorted(partition.ids.begin(), partition.ids.end()))
                throw artifact::ArtifactError("affine expert partition requires global ID order");
            std::vector<artifact::MatrixSlice> gu, down;
            for (std::size_t slot = 0; slot < partition.ids.size(); ++slot) {
                const auto id  = partition.ids[slot];
                const auto& g  = source.gate_up;
                const auto& d  = source.down;
                const auto row = g.layout.region.row_begin + id * g.layout.region.row_stride;
                append_expert_slice(gu, {g.object, row + range.begin, 0, range.size, hidden,
                                         slot * 2 * range.size});
                append_expert_slice(gu, {g.object, row + intermediate + range.begin, 0, range.size,
                                         hidden, (slot * 2 + 1) * range.size});
                append_expert_slice(
                    down, {d.object, d.layout.region.row_begin + id * d.layout.region.row_stride,
                           range.begin, hidden, range.size, slot * hidden});
            }
            const artifact::Shape gu_shape{partition.ids.size(), 2ULL * range.size, hidden};
            const artifact::Shape down_shape{partition.ids.size(), hidden, range.size};
            const auto materialize = [&](const moe::MatrixSource& matrix, const auto& shape,
                                         const auto& slices) {
                const auto& geometry = reader.geometry(matrix.object);
                if (complete && range.begin == 0 && range.size == intermediate &&
                    matrix.layout.region.row_begin == 0 && geometry.shape == shape)
                    return add(artifact::whole_tensor(reader, matrix.object));
                return add(artifact::slice_tensor(reader, shape, slices), bank + 1);
            };
            selection.packed = {materialize(source.gate_up, gu_shape, gu),
                                materialize(source.down, down_shape, down),
                                complete ? std::vector<std::uint32_t>{} : partition.ids};
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
            selection.packed ? selection.packed->gate_up : selection.entries.front().gate_up;
        const auto dn = selection.packed ? selection.packed->down : selection.entries.front().down;
        out.expert_descriptions.push_back(
            {out.requests[gu].tensor.geometry, out.requests[dn].tensor.geometry, count,
             complete ? ops::ExpertSubsetKind::All : ops::ExpertSubsetKind::MaskOnes});
    }
    for (const auto parameter : whole) {
        claim(parameter, Role::Weight);
        const auto& reference = references[parameter.index];
        auto& binding         = out.bindings[parameter.index];
        binding.shape         = reference.shape;
        for (const auto& part : reference.binding.parts)
            binding.parts.push_back(
                {add(artifact::whole_tensor(reader, part.object)), part.begin, part.end});
    }
    for (const auto parameter : omitted) {
        claim(parameter, Role::Omitted);
        out.omitted[parameter.index] = true;
    }
    for (std::size_t i = 0; i < references.size(); ++i) {
        if (classified[i] == Role::Unclassified)
            throw std::logic_error("parallel plan omits classification for parameter '" +
                                   references[i].name + "'");
        if (classified[i] == Role::Weight && out.bindings[i].parts.empty())
            throw artifact::ArtifactError("parallel weight has no local binding for '" +
                                          references[i].name + "'");
    }
    for (std::size_t parameter = 0; parameter < descriptions.size(); ++parameter) {
        if (out.omitted[parameter]) continue;
        const auto& weight = descriptions[parameter];
        for (std::size_t use = 0; use < weight.uses.size(); ++use) {
            const auto& calibration = weight.uses[use];
            if (!calibration.calibration_source || !calibration.calibration ||
                calibration.calibration->values.size() <= 1)
                continue;
            const auto& binding = *calibration.calibration_source;
            const artifact::ParameterReference source{
                weight.name + "/activation_calibration", {binding.elements}, binding};
            std::vector<artifact::MatrixSlice> slices;
            append_parameter_slice(slices, reader, source, {{}, 0, 0, 1, binding.elements, 0, 0});
            out.calibrations.push_back(
                {parameter, use, add(artifact::slice_tensor(reader, source.shape, slices))});
        }
    }
    return out;
}

std::vector<BoundWeight> bind_selected_weights(std::span<const BoundWeight> descriptions,
                                               const SelectedWeights& selection,
                                               std::span<const artifact::StorageId> requests,
                                               std::span<const WeightParent> parents) {
    if (descriptions.size() != selection.bindings.size() ||
        descriptions.size() != selection.omitted.size())
        throw std::logic_error("selected weight count differs from bound descriptions");
    std::vector<BoundWeight> out;
    out.reserve(descriptions.size());
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
        auto weight = descriptions[i];
        if (selection.omitted[i]) {
            weight.view = {};
            for (auto& use : weight.uses) use.calibration_device = {};
        } else {
            const auto& local = selection.bindings[i];
            if (local.parts.empty()) {
                // Expert weights are provided by their bank. Keep the parameter's Use metadata.
                weight.view = {};
            } else {
                weight.view = {local.shape, {}};
                for (const auto& part : local.parts)
                    weight.view.parts.push_back(
                        {&parents[requests[part.request].index], part.begin, part.end});
            }
        }
        out.push_back(std::move(weight));
    }
    for (const auto& calibration : selection.calibrations) {
        const auto& parent = parents[requests[calibration.request].index];
        const auto count =
            out[calibration.parameter].uses[calibration.use].calibration->values.size();
        if (parent.geometry.format != QType::FP32 ||
            parent.geometry.layout != QuantLayout::Contiguous || parent.geometry.elements != count)
            throw artifact::ArtifactError("activation calibration materialization differs");
        if (parent.data)
            out[calibration.parameter].uses[calibration.use].calibration_device = {
                parent.data, count * sizeof(float), parent.device};
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
        if (bank.packed) {
            if (bank.packed->logical_ids.empty())
                out.emplace_back(parent(bank.packed->gate_up), parent(bank.packed->down));
            else
                out.emplace_back(parent(bank.packed->gate_up), parent(bank.packed->down),
                                 bank.packed->logical_ids, selection.expert_descriptions[i].count,
                                 device);
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
