#include "models/common/parallel.h"
#include "artifact/reader.h"

#include <stdexcept>

namespace ninfer::models {
IndexRange partition_range(std::uint32_t total, std::uint32_t parts, std::uint32_t rank) {
    if (!total || !parts || rank >= parts || total % parts)
        throw std::invalid_argument("parallel partition requires equal, nonempty ranges");
    return {rank * (total / parts), total / parts};
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
