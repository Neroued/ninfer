#pragma once

#include "core/buffer.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "state_record_fixture.h"
#include <array>
#include <bit>
#include <memory>
#include <vector>

namespace ninfer::tool {

// Shared represented inputs / State backing for public context-materialization tests
// and benchmarks. Mathematical references remain in the correctness suite.
struct Bf16ContextFixture {
    static constexpr int Layers = 5, Hidden = 4096, Rows = 512, Heads = 4;
    static constexpr int Capacity = 2048, Padded = 2056, Records = 8;
    static constexpr ops::ContextKVMaterializeProfile Profile{Hidden,      128,    Heads,  Capacity,
                                                              QType::BF16, 1.e-5F, 10000.F};
    static constexpr std::size_t CacheElements = std::size_t(128) * Padded * Heads * Records;
    std::array<DeviceBuffer, Layers> parents, norms, keys, values;
    std::array<std::vector<std::uint16_t>, Layers> parent_bits, norm_bits;
    std::array<ops::ContextKVMaterializeLayerView, Layers> layers;
    std::unique_ptr<StateRecordFixture> records;

    static std::uint16_t bf16(float value) {
        auto bits = std::bit_cast<std::uint32_t>(value);
        bits += 0x7fffU + ((bits >> 16) & 1U);
        return std::uint16_t(bits >> 16);
    }

    static float value(std::uint16_t bits) {
        return std::bit_cast<float>(std::uint32_t(bits) << 16);
    }

    static std::uint32_t mix(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x85ebca6bU;
        x ^= x >> 13;
        x *= 0xc2b2ae35U;
        return x ^ (x >> 16);
    }

    explicit Bf16ContextFixture(Device device) {
        std::vector<Tensor> fields;
        for (int l = 0; l < Layers; ++l) {
            auto& host = parent_bits[l];
            host.resize(std::size_t(3072) * Hidden,
                        0x7fc0U); // Parent Q is deliberately not consumable.
            for (int row = 0; row < 1024; ++row)
                for (int k = 0; k < Hidden; ++k) {
                    const int n = int(mix(std::uint32_t(row) * 73856093U ^
                                          std::uint32_t(k) * 19349663U ^ (l + 1) * 83492791U) %
                                      257) -
                                  128;
                    host[std::size_t(row) * Hidden + k] = bf16(float(n) / 4096.F);
                }
            parents[l] = DeviceBuffer(device, host.size() * 2);
            parents[l].copy_from_host(host.data(), host.size() * 2);
            norm_bits[l].resize(128);
            for (int d = 0; d < 128; ++d)
                norm_bits[l][d] = bf16(.75F + l * .03125F + ((d * 7 + l) % 19) / 128.F);
            norms[l] = DeviceBuffer(device, 256);
            norms[l].copy_from_host(norm_bits[l].data(), 256);
            keys[l]   = DeviceBuffer(device, CacheElements * 2);
            values[l] = DeviceBuffer(device, CacheElements * 2);
            keys[l].fill(0xa5);
            values[l].fill(0xa5);
            auto weight = [&](int first) {
                Weight w;
                w.device        = device;
                w.payload       = parents[l].data();
                w.payload_bytes = parents[l].size();
                w.qtype         = QType::BF16;
                w.ndim          = 2;
                w.n = w.shape[0] = w.padded_shape[0] = Rows;
                w.k = w.shape[1] = w.padded_shape[1] = Hidden;
                w.qdata = static_cast<const std::uint16_t*>(parents[l].data()) +
                          std::size_t(first) * Hidden;
                w.layout = QuantLayout::Contiguous;
                return w;
            };
            layers[l] = {weight(0), weight(Rows), Tensor(norms[l].span(), DType::BF16, {128}), {}};
            fields.emplace_back(keys[l].span(), DType::BF16,
                                std::initializer_list<int>{128, Padded, Heads, Records});
            fields.emplace_back(values[l].span(), DType::FP16,
                                std::initializer_list<int>{128, Padded, Heads, Records});
        }
        records = std::make_unique<StateRecordFixture>(fields, 3);
    }

    void bind(int l, const Tensor& offsets) {
        layers[l].cache = {records->field(2 * l, offsets),
                           records->field(2 * l + 1, offsets),
                           Capacity,
                           Padded,
                           Heads,
                           128};
    }

    void reset() {
        for (int l = 0; l < Layers; ++l) {
            keys[l].fill(0xa5);
            values[l].fill(0xa5);
        }
        records->upload();
    }

    static std::size_t index(int record, int head, int slot, int d) {
        return d + 128ULL * (slot + Padded * (head + Heads * record));
    }
};

} // namespace ninfer::tool
