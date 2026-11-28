#include "models/common/parallel.h"
#include "artifact/reader.h"

#include <stdexcept>
#include <algorithm>

namespace ninfer::models {
// A shared physical pool can contain ranks from several stages. The caller checks whole
// prefixes, so a later choice cannot borrow bytes already charged to an earlier stage.
std::vector<IndexRange> partition_pipeline_layers(std::span<const PipelineLayerDemand> layers,
                                                  const ParallelOptions& parallel,
                                                  const PipelineCapacity& fits) {
    const auto stages = parallel.pp;
    if (!stages || stages > layers.size() || layers.size() >= UINT32_MAX)
        throw std::invalid_argument("pipeline requires a nonempty layer interval per stage");
    const auto n = static_cast<std::uint32_t>(layers.size());
    std::vector<std::uint64_t> work{0}, bytes{0};
    for (const auto& layer : layers) {
        if (!layer.work || UINT64_MAX - work.back() < layer.work ||
            UINT64_MAX - bytes.back() < layer.resident_bytes)
            throw std::invalid_argument("invalid pipeline layer demand");
        work.push_back(work.back() + layer.work);
        bytes.push_back(bytes.back() + layer.resident_bytes);
    }
    std::vector<IndexRange> result(stages);
    if (!parallel.stage_layers.empty()) {
        if (parallel.stage_layers.size() != stages)
            throw std::invalid_argument("stage_layers differs from pipeline stage count");
        std::uint32_t begin = 0;
        for (std::uint32_t s = 0; s < stages; ++s) {
            const auto count = parallel.stage_layers[s];
            if (!count || count > n - begin)
                throw std::invalid_argument("invalid explicit stage interval");
            result[s] = {begin, count};
            begin += count;
        }
        if (begin != n) throw std::invalid_argument("stage_layers does not cover the model");
        if (fits && !fits(result))
            throw std::invalid_argument("explicit pipeline exceeds physical weight capacity");
        return result;
    }

    // Both objective values are sums over actual intervals. Search these discrete bounds,
    // first minimizing the bottleneck work, then the peak encoded layer size at that work.
    const auto choices = [&](const auto& prefix) {
        std::vector<std::uint64_t> values;
        for (std::uint32_t begin = 0; begin < n; ++begin)
            for (std::uint32_t end = begin + 1; end <= n; ++end)
                values.push_back(prefix[end] - prefix[begin]);
        std::ranges::sort(values);
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    };
    const auto possible = [&](std::uint64_t limit, std::uint64_t byte_limit) {
        std::vector<std::vector<bool>> failed;
        if (!fits) failed.assign(stages, std::vector<bool>(n));
        const auto visit = [&](const auto& self, unsigned stage, unsigned begin) -> bool {
            if (!fits && failed[stage][begin]) return false;
            const auto following = stages - stage - 1;
            for (unsigned end = following ? begin + 1 : n; end <= n - following; ++end) {
                if (work[end] - work[begin] > limit || bytes[end] - bytes[begin] > byte_limit)
                    break;
                const auto remaining = work[n] - work[end];
                if (following && remaining / following + (remaining % following != 0) > limit)
                    continue;
                result[stage] = {begin, end - begin};
                if (fits && !fits(std::span(result).first(stage + 1))) continue;
                if (!following || self(self, stage + 1, end)) return true;
            }
            if (!fits) failed[stage][begin] = true;
            return false;
        };
        return visit(visit, 0, 0);
    };
    const auto work_limits = choices(work);
    const auto chosen_work = std::ranges::partition_point(
        work_limits, [&](auto value) { return !possible(value, UINT64_MAX); });
    if (chosen_work == work_limits.end())
        throw std::invalid_argument(
            "pipeline target layers cannot fit the available physical memory pools");
    const auto byte_limits  = choices(bytes);
    const auto chosen_bytes = std::ranges::partition_point(
        byte_limits, [&](auto value) { return !possible(*chosen_work, value); });
    if (chosen_bytes == byte_limits.end() || !possible(*chosen_work, *chosen_bytes))
        throw std::logic_error("pipeline capacity changed during layer planning");
    return result;
}

std::vector<std::uint32_t> expert_replica_members(const distributed::ExecutionTopology& topology,
                                                  const distributed::GroupDescription& group) {
    std::vector<std::uint32_t> replicas;
    for (auto rank : group.members) {
        const auto id = topology.replica_for_rank(rank).replica_id;
        if (std::ranges::find(replicas, id) == replicas.end()) replicas.push_back(id);
    }
    return replicas;
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
