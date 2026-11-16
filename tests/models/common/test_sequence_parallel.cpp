#include "models/common/parallel_execution.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void ownership(int columns, unsigned ranks) {
    const TokenPartition first(columns, ranks, 0);
    constexpr int width = 3;
    std::vector<std::int32_t> storage(width * first.padded, -1);
    Tensor full(storage.data(), Device::cpu(), DType::I32, {width, first.padded});
    int covered = 0;
    for (unsigned rank = 0; rank < ranks; ++rank) {
        const TokenPartition p(columns, ranks, rank);
        require(p.begin == covered || p.count == 0, "partition omitted or repeated real tokens");
        const auto segment = p.exchange(full, 1);
        require(segment.ne[1] == first.segment && segment.is_contiguous(),
                "communication segments must have equal contiguous extents");
        if (!p.count) continue;
        const auto local = p.local(full, 1);
        require(local.is_contiguous(), "mathematical token columns must be contiguous");
        auto* words = static_cast<std::int32_t*>(local.data);
        std::iota(words, words + local.numel(), width * covered);
        covered += p.count;
    }
    require(covered == columns, "partition did not cover the real sequence");
    for (int i = 0; i < width * columns; ++i)
        require(storage[i] == i, "rank-order gather changed token identity");
    for (int i = width * columns; i < width * first.padded; ++i)
        require(storage[i] == -1, "mathematical work touched communication padding");
}
} // namespace

int main() {
    try {
        for (int n : {1, 2, 3, 128, 129, 255, 256, 257, 4097})
            for (unsigned ranks : {1U, 2U, 4U}) ownership(n, ranks);
        for (const SequenceParallelPolicy policy : {SequenceParallelPolicy{}, {2, 129}, {2, 256}})
            for (int maximum : {128, 129, 255, 256, 257, 4097}) {
                const int local = policy.local_capacity(maximum);
                const int full  = policy.partition(maximum, 0).padded;
                for (int n = 1; n <= maximum; ++n)
                    for (unsigned rank = 0; rank < policy.ranks; ++rank) {
                        const auto p = policy.partition(n, rank);
                        require(p.count <= local && p.padded <= full,
                                "capacity omitted a reachable sequence shape");
                    }
            }
        std::cout << "PASS sequence partition ownership and capacity\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
