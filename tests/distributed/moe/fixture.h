#pragma once

#include "models/common/moe/exchange.h"
#include <array>

namespace ninfer::test::exchange {
inline constexpr int H = 2560, I = 640, E = 512, K = 10;
using models::moe::Parameters;

struct Matrix {
    DeviceBuffer data, index, divisors;
    ExpertMatrixDeviceView view;
    Matrix(Device, bool gate, bool fp8, bool affine, std::span<const std::uint32_t> owned,
           models::IndexRange slice);
};

struct Fixture {
    Matrix gu, down;
    DeviceBuffer router, shared_gu, shared_down;
    Parameters parameters;
    Fixture(Device, std::span<const std::uint32_t> owned, ExpertBankId, bool fp8 = false,
            bool affine = false, models::IndexRange slice = {0, I});
};

struct Column {
    std::array<int, K> experts;
    std::vector<std::uint16_t> input;
    std::array<double, K> weights;
    double shared(int output_row) const;
    double expert(int id, int output_row, bool fp8, bool affine) const;
};

Column column(const std::vector<models::ExpertPartition>& peers, int source, int index,
              int local_hits);
void local_reference(DeviceContext&, const models::moe::ExpertDomain&, ExpertBankId);
} // namespace ninfer::test::exchange
