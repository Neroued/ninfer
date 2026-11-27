#include "models/common/parallel.h"
#include "artifact/reader.h"

#include <stdexcept>
#include <algorithm>
#include <optional>

namespace ninfer::models {
std::vector<IndexRange> partition_pipeline_layers(std::span<const PipelineLayerDemand> layers,
                                                  const ParallelOptions& parallel,
                                                  std::span<const std::uint64_t> budgets) {
    const auto stages = parallel.pp;
    if (!stages || stages > layers.size() || layers.size() >= UINT32_MAX ||
        (!budgets.empty() && budgets.size() != stages))
        throw std::invalid_argument("pipeline requires a nonempty layer interval per stage");
    std::vector<std::uint64_t> work{0}, bytes{0};
    for (auto layer : layers) {
        if (!layer.work || UINT64_MAX - work.back() < layer.work ||
            UINT64_MAX - bytes.back() < layer.resident_bytes)
            throw std::invalid_argument("invalid pipeline layer demand");
        work.push_back(work.back() + layer.work);
        bytes.push_back(bytes.back() + layer.resident_bytes);
    }
    const auto fits = [&](std::uint32_t stage, std::size_t begin, std::size_t end) {
        return budgets.empty() || bytes[end] - bytes[begin] <= budgets[stage];
    };
    std::vector<IndexRange> result(stages);
    if (!parallel.stage_layers.empty()) {
        if (parallel.stage_layers.size() != stages)
            throw std::invalid_argument("stage_layers differs from pipeline stage count");
        std::uint32_t begin = 0;
        for (std::uint32_t s = 0; s < stages; ++s) {
            const auto count = parallel.stage_layers[s];
            if (!count || count > layers.size() - begin || !fits(s, begin, begin + count))
                throw std::invalid_argument("explicit pipeline interval exceeds layers or memory");
            result[s] = {begin, count};
            begin += count;
        }
        if (begin != layers.size())
            throw std::invalid_argument("stage_layers does not cover the model");
        return result;
    }
    const auto n = static_cast<std::uint32_t>(layers.size());
    std::vector<std::vector<std::uint32_t>> cuts(stages,
                                                 std::vector<std::uint32_t>(layers.size() + 1));
    // A later stage can hide a difference between prefix costs. Minimize peak work first,
    // then minimize peak bytes among partitions attaining that same work bound.
    const auto solve = [&](std::optional<std::uint64_t> work_limit) {
        std::vector<std::optional<std::uint64_t>> previous(layers.size() + 1),
            current(layers.size() + 1);
        previous[0] = 0;
        for (std::uint32_t s = 0; s < stages; ++s) {
            std::ranges::fill(current, std::nullopt);
            for (std::uint32_t end = s + 1; end <= n - (stages - s - 1); ++end) {
                for (std::uint32_t begin = s; begin < end; ++begin) {
                    const auto demand = work[end] - work[begin];
                    if (!previous[begin] || !fits(s, begin, end) ||
                        (work_limit && demand > *work_limit))
                        continue;
                    const auto cost =
                        std::max(*previous[begin], work_limit ? bytes[end] - bytes[begin] : demand);
                    if (!current[end] || cost < *current[end]) {
                        current[end] = cost;
                        cuts[s][end] = begin;
                    }
                }
            }
            previous.swap(current);
        }
        if (!previous[n])
            throw std::invalid_argument("pipeline layer weights exceed stage capacities");
        return *previous[n];
    };
    const auto peak_work = solve({});
    (void)solve(peak_work);
    for (std::uint32_t s = stages, end = n; s > 0; --s) {
        const auto begin = cuts[s - 1][end];
        result[s - 1]    = {begin, end - begin};
        end              = begin;
    }
    return result;
}

void add_expert_roles(distributed::ExecutionTopology& topology) {
    const RankPlacement first(topology.ranks.front());
    const auto lanes = first.expert_slice().size;
    const auto replicas_per_group =
        first.ep.size > first.tp.size ? first.ep.size / first.tp.size : 1U;
    for (std::uint32_t stage = 0; stage < first.pp.size; ++stage) {
        std::vector<distributed::RankId> domain;
        for (const auto& replica : topology.replicas) {
            const auto members = std::span(replica.stage_ranks.at(stage));
            domain.insert(domain.end(), members.begin(), members.end());
            if ((replica.replica_id + 1) % replicas_per_group == 0) {
                topology.add_role("expert", domain);
                domain.clear();
            }
            for (std::size_t i = 0; i < members.size(); i += lanes)
                topology.add_role("expert_tensor", members.subspan(i, lanes));
        }
    }
}

IndexRange partition_range(std::uint32_t total, std::uint32_t parts, std::uint32_t rank) {
    if (!total || !parts || rank >= parts || total % parts)
        throw std::invalid_argument("parallel partition requires equal, nonempty ranges");
    return {rank * (total / parts), total / parts};
}

IndexRange partition_vocabulary_range(std::uint32_t vocabulary, std::uint32_t parts,
                                      std::uint32_t rank) {
    if (!vocabulary || !parts || rank >= parts || vocabulary < parts)
        throw std::invalid_argument("vocabulary partition requires nonempty shards per rank");
    if (parts == 1) return {0, vocabulary};
    constexpr std::uint32_t kAlignment = 128;
    if (std::uint64_t(vocabulary) >= std::uint64_t(kAlignment) * parts) {
        const std::uint32_t blocks = vocabulary / kAlignment;
        const std::uint32_t base   = blocks / parts;
        const std::uint32_t extra  = blocks % parts;
        const std::uint32_t block_begin =
            rank < extra ? rank * (base + 1) : extra * (base + 1) + (rank - extra) * base;
        const std::uint32_t block_count = rank < extra ? (base + 1) : base;
        const std::uint32_t begin       = block_begin * kAlignment;
        const std::uint32_t size =
            (rank + 1 == parts) ? (vocabulary - begin) : (block_count * kAlignment);
        return {begin, size};
    }
    const std::uint32_t base  = vocabulary / parts;
    const std::uint32_t extra = vocabulary % parts;
    const std::uint32_t begin =
        rank < extra ? rank * (base + 1) : extra * (base + 1) + (rank - extra) * base;
    const std::uint32_t size = rank < extra ? (base + 1) : base;
    return {begin, size};
}

void validate_vocabulary_partition(const BoundWeight& head, IndexRange range) {
    for (const auto& part : head.view.parts) {
        const auto& g    = part.parent->geometry;
        const auto block = g.layout == QuantLayout::BlockScaleK16M128x4 ||
                                   g.layout == QuantLayout::BatchedBlockScaleK16M128x4
                               ? 128
                           : g.layout == QuantLayout::Csf32 ? 32
                                                            : g.scale_block_rows;
        if (range.begin % block || range.size % block)
            throw artifact::ArtifactError("vocabulary shard splits a quantized row block");
    }
}

void append_parameter_slice(std::vector<artifact::MatrixSlice>& out, const artifact::Reader& reader,
                            const artifact::ParameterReference& reference,
                            const ParameterSlice& s) {
    const auto k = reference.shape.back();
    const auto n = weight_element_count(reference.shape) / k;
    if (!s.rows || !s.columns || s.row > n || s.rows > n - s.row || s.column > k ||
        s.columns > k - s.column)
        throw artifact::ArtifactError("parameter selection exceeds logical shape");
    std::uint64_t origin = 0, covered = 0;
    for (const auto& p : reference.binding.parts) {
        const auto end = origin + p.end - p.begin;
        const auto sk  = reader.geometry(p.object).shape.back();
        if (origin % k == 0 && end % k == 0 && p.begin % sk == 0 && sk == k) {
            const auto first = std::max(s.row, origin / k);
            const auto last  = std::min(s.row + s.rows, end / k);
            if (first < last) {
                out.push_back({p.object, p.begin / k + first - origin / k, s.column, last - first,
                               s.columns, s.destination_row + first - s.row, s.destination_column});
                covered += (last - first) * s.columns;
            }
        } else {
            const auto first_row = std::max(s.row, origin / k);
            const auto last_row  = std::min(s.row + s.rows, (end + k - 1) / k);
            for (auto row = first_row; row < last_row; ++row) {
                auto first      = std::max(row * k + s.column, origin);
                const auto last = std::min(row * k + s.column + s.columns, end);
                while (first < last) {
                    const auto source = p.begin + first - origin;
                    const auto take   = std::min(last - first, sk - source % sk);
                    out.push_back({p.object, source / sk, source % sk, 1, take,
                                   s.destination_row + row - s.row,
                                   s.destination_column + first - row * k - s.column});
                    covered += take;
                    first += take;
                }
            }
        }
        origin = end;
    }
    if (covered != s.rows * s.columns)
        throw artifact::ArtifactError("parameter selection is not covered by Binding");
}
} // namespace ninfer::models
