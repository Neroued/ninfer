#include "core/nf4.h"

#include <algorithm>
#include <bit>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value) {
    if (!value) throw std::runtime_error("NF4 test failed");
}

template <class F>
void rejects(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid NF4 input accepted");
}

void validation() {
    using namespace ninfer;
    const auto lut = make_nf4_bf16_lut(2.0F);
    for (std::size_t k : {16, 160, 7280}) {
        std::vector<std::byte> row(k / 2 + k / 16, std::byte{0xf0});
        std::fill(row.begin() + k / 2, row.end(), std::byte{0x38});
        std::vector<std::uint16_t> out(k);
        decode_nf4_row(row, lut, out);
        for (std::size_t i = 0; i < k; ++i) require(out[i] == (i % 2 ? 0x3f00 : 0xbf00));
        std::fill(out.begin(), out.end(), 0xbeef);
        for (auto word : {0x7f, 0x80, 0xff}) {
            row.back() = std::byte(word);
            rejects([&] { decode_nf4_row(row, lut, out); });
            require(std::all_of(out.begin(), out.end(), [](auto w) { return w == 0xbeef; }));
        }
    }
    for (auto word : {0U, 0x80000000U, 0xbf800000U, 0x7f800000U, 0x7fc00000U, 1U})
        rejects([&] { (void)make_nf4_bf16_lut(std::bit_cast<float>(word)); });
    std::vector<std::uint16_t> out(160);
    std::vector<std::byte> row(90);
    rejects([&] { decode_nf4_row({}, lut, out); });
    rejects([&] { decode_nf4_row(row, lut, {}); });
    rejects([&] { decode_nf4_row(row, lut, std::span(out).first(159)); });
}
} // namespace

int main(int argc, char**) {
    try {
        if (argc == 1) {
            validation();
            return 0;
        }
        // Binary test protocol: divisor word -> success word and LUT, or zero on rejection.
        std::uint32_t divisor;
        while (std::cin.read(reinterpret_cast<char*>(&divisor), sizeof(divisor))) {
            ninfer::Nf4Bf16Lut lut;
            std::uint32_t valid = 1;
            try {
                lut = ninfer::make_nf4_bf16_lut(std::bit_cast<float>(divisor));
            } catch (const std::invalid_argument&) { valid = 0; }
            std::cout.write(reinterpret_cast<const char*>(&valid), sizeof(valid));
            if (valid)
                std::cout.write(reinterpret_cast<const char*>(lut.words.data()), sizeof(lut));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
