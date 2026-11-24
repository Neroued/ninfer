#include "core/buffer.h"
#include "core/weight_bank.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value) {
    if (!value) { throw std::runtime_error("weight bank addressing differs"); }
}

template <class F>
void rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid weight bank accepted");
}

void integers() {
    for (const auto format :
         {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16, QType::Q8_G32_FP16}) {
        const auto g =
            weight_geometry(format, QuantLayout::RowSplit, std::array<std::uint64_t, 2>{1536, 130});
        HostBuffer bytes(g.bytes);
        WeightParent parent{g, Device::cpu(), static_cast<const std::byte*>(bytes.data())};
        const auto bank = bind_weight_bank(parent, describe_weight_bank(g, {128, 3, 128, 512}));
        rejects([&] { (void)device_weight_bank(bank); });
        for (unsigned e = 0; e < 3; ++e) {
            const auto w     = matrix_weight(bank, e);
            const auto row   = 128 + e * 512;
            const auto* base = parent.data;
            const bool q8    = format == QType::Q8_G32_FP16;
            require(w.qdata == base + row * (q8 ? 256 : 128));
            require(w.scales == base + g.scale_offset + row * (q8 ? 16 : 8));
            if (format == QType::Q5_G64_FP16) {
                require(w.qhigh == base + g.high_offset + row * 32);
            }
            if (format == QType::Q6_G64_FP16) {
                require(w.qhigh == base + g.high_offset + row * 64);
            }
            require(bank.layout.codes.expert_stride == bank.layout.codes.expert_bytes * 4);
        }
        rejects([&] { (void)describe_weight_bank(g, {128, 4, 128, 512}); });
    }
}

void block_scales() {
    for (const auto nv : {false, true}) {
        const auto g = weight_geometry(nv ? QType::NVFP4 : QType::FP8_E4M3FN_BLOCK128X128_BF16,
                                       nv ? QuantLayout::BatchedBlockScaleK16M128x4
                                          : QuantLayout::BlockScaleM128K128,
                                       std::array<std::uint64_t, 3>{3, 256, 384});
        HostBuffer bytes(g.bytes);
        std::array<float, 3> divisors{2, 3, 5};
        WeightParent parent{g, Device::cpu(), static_cast<const std::byte*>(bytes.data()),
                            nv ? std::span<const float>(divisors) : std::span<const float>{}};
        const auto bank = bind_weight_bank(parent, describe_weight_bank(g, {128, 3, 128, 256}));
        for (unsigned e = 0; e < 3; ++e) {
            const auto w = matrix_weight(bank, e, 7);
            require(w.qdata == parent.data + (e * 256 + 128) * (nv ? 192 : 384));
            require(w.scales ==
                    parent.data + (nv ? 147456 + (e * 256 + 128) * 24 : 294912 + (e * 2 + 1) * 6));
            require(!nv || (w.weight_scale_divisor == divisors[e] && w.input_scale_divisor == 7));
            require(bank.divisor(e) == (nv ? divisors[e] : 1));
        }
        rejects([&] { (void)describe_weight_bank(g, {64, 1, 128, 256}); });
        if (nv) {
            rejects([&] { (void)describe_weight_bank(g, {128, 2, 256, 256}); });
        }
    }
}

void fp8_blocks() {
    for (const auto block : {64U, 128U}) {
        const auto format =
            block == 64 ? QType::FP8_E4M3FN_BLOCK64X64_BF16 : QType::FP8_E4M3FN_BLOCK128X128_BF16;
        const auto layout =
            block == 64 ? QuantLayout::BlockScaleM64K64 : QuantLayout::BlockScaleM128K128;
        // Block64 must retain the actual TP2 dimensions, including K=320.
        const std::uint64_t n = block == 64 ? 640 : 768, k = block == 64 ? 320 : 384;
        const auto g = weight_geometry(format, layout, std::array<std::uint64_t, 3>{3, n, k});
        require(g.code_bytes == 3 * n * k && g.scale_offset == g.code_bytes &&
                g.scale_offset % 256 == 0 && g.scale_bytes == 3 * (n / block) * (k / block) * 2 &&
                g.bytes == g.scale_offset + g.scale_bytes && !g.divisor_count &&
                g.padded_columns == k && g.scale_block_rows == block && g.group_size == block);
        HostBuffer storage(g.bytes);
        WeightParent parent{g, Device::cpu(), static_cast<const std::byte*>(storage.data())};
        const auto bank =
            bind_weight_bank(parent, describe_weight_bank(g, {block, 3, n - block, n}));
        require(bank.layout.codes.expert_bytes == (n - block) * k &&
                bank.layout.codes.expert_stride == n * k &&
                bank.layout.scales.expert_bytes == (n / block - 1) * (k / block) * 2 &&
                bank.layout.scales.expert_stride == (n / block) * (k / block) * 2 &&
                !bank.layout.has_divisors);
        for (unsigned e = 0; e < 3; ++e) {
            const auto w   = matrix_weight(bank, e);
            const auto row = e * n + block;
            require(w.qdata == parent.data + row * k &&
                    w.scales == parent.data + g.code_bytes + row / block * (k / block) * 2 &&
                    w.qtype == format && w.layout == layout && w.n == n - block && w.k == k &&
                    w.padded_shape[1] == k && w.scale_dtype == DType::BF16 &&
                    w.scale_ne[0] == k / block && w.scale_ne[1] == n / block - 1 &&
                    w.scale_nb[0] == 2 && w.scale_nb[1] == k / block * 2 &&
                    w.scale_nb[2] == (n / block - 1) * (k / block) * 2 && !w.weight_scale_divisor &&
                    !w.input_scale_divisor && bank.divisor(e) == 1);
            for (std::uint64_t r = row; r < row + n - block; ++r)
                for (std::uint64_t c = 0; c < k / block; ++c)
                    require(weight_scale_offset(g, r, c) ==
                            g.code_bytes + ((r / block) * (k / block) + c) * 2);
        }
        rejects([&] { (void)describe_weight_bank(g, {block / 2, 1, block, n}); });
        rejects([&] {
            (void)native_weight(
                {{block, k}, {{&parent, k * (block / 2), k * (block / 2 + block)}}});
        });
        for (const auto shape :
             {std::array<std::uint64_t, 2>{n - 1, k}, std::array<std::uint64_t, 2>{n, k - 1}})
            rejects([&] { (void)weight_geometry(format, layout, shape); });
        rejects([&] {
            (void)weight_geometry(format,
                                  block == 64 ? QuantLayout::BlockScaleM128K128
                                              : QuantLayout::BlockScaleM64K64,
                                  std::array<std::uint64_t, 2>{256, 256});
        });
        rejects([&] {
            (void)weight_geometry(format, layout, std::array<std::uint64_t, 4>{1, 3, n, k});
        });
        const auto matrix = weight_geometry(format, layout, std::array<std::uint64_t, 2>{n, k});
        require(matrix.code_bytes == n * k && matrix.scale_bytes == (n / block) * (k / block) * 2);
    }
    const auto gu =
        weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                        std::array<std::uint64_t, 3>{512, 640, 2560});
    const auto down =
        weight_geometry(QType::FP8_E4M3FN_BLOCK64X64_BF16, QuantLayout::BlockScaleM64K64,
                        std::array<std::uint64_t, 3>{512, 2560, 320});
    require(gu.code_bytes == 838860800 && gu.scale_bytes == 512 * 10 * 40 * 2 &&
            down.code_bytes == 419430400 && down.scale_bytes == 512 * 40 * 5 * 2);
}
} // namespace

int main() {
    try {
        integers();
        block_scales();
        fp8_blocks();
        std::cout << "weight bank views passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
