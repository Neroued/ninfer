#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "core/nf4.h"
#include "ninfer/ops/ple.h"

#include <algorithm>
#include <bit>
#include <memory>
#include <set>

namespace {
using namespace ninfer;
using namespace ninfer::test;

std::uint32_t row_id(const ops::NgramHashParameters& hash, int h, int x, int previous, int older,
                     int eos) {
    if (previous == eos) older = eos;
    auto word =
        std::uint64_t(x) * hash.multipliers[0] ^ std::uint64_t(previous) * hash.multipliers[1];
    if (h >= 8) word ^= std::uint64_t(older) * hash.multipliers[2];
    auto row = std::bit_cast<std::int64_t>(word) % std::int64_t(hash.vocab_sizes[h]);
    if (row < 0) row += hash.vocab_sizes[h];
    return std::uint32_t(row + hash.offsets[h]);
}

struct Shard {
    std::uint32_t first, rows;
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint16_t> lut;
    GuardedDeviceBuffer pages, decode;
    Tensor decode_view;
    int leading;

    Shard(std::uint32_t begin, std::uint32_t count, int number, bool unaligned)
        : first(begin), rows(count),
          bytes(((std::uint64_t(count) + 44) / 45) * 4096 + unaligned, 0xa5),
          lut(2048 + unaligned, 0x3f80), pages(bytes.size()), decode(lut.size() * 2),
          leading(unaligned) {
        const auto table = make_nf4_bf16_lut(0.75F + number * 0.03125F);
        std::copy(table.words.begin(), table.words.end(), lut.begin() + leading);
        for (std::uint32_t r = 0; r < count; ++r) {
            auto* p = bytes.data() + leading + (r / 45) * 4096 + (r % 45) * 90;
            for (int c = 0; c < 80; ++c) p[c] = (r * 71 + c * 29 + number * 13) & 255;
            for (int g = 0; g < 10; ++g) p[80 + g] = g ? (r * 17 + g * 13 + number) % 127 : 0;
            p[0] = 0x80; // The first word pair contains -0 and +0.
        }
        pages.copy_from_host(bytes.data(), bytes.size());
        decode.copy_from_host(lut.data(), lut.size() * 2);
        decode_view = Tensor(decode.span().subspan(leading * 2, 4096), DType::BF16, {16, 128});
    }

    ops::Nf4EmbeddingShard view() const {
        return {pages.span().subspan(leading), decode_view, first, rows};
    }

    std::uint16_t at(std::uint32_t row, int feature) const {
        const auto r      = row - first;
        const auto base   = leading + (r / 45) * 4096 + (r % 45) * 90;
        const auto packed = bytes[base + feature / 2];
        const auto code   = (packed >> (4 * (feature % 2))) & 15;
        return lut[leading + bytes[base + 80 + feature / 16] * 16 + code];
    }

    int preserved() const {
        return verify_exact("lookup pages", from_device<std::uint8_t>(pages.data(), bytes.size()),
                            bytes) +
               verify_exact("lookup LUT", from_device<std::uint16_t>(decode.data(), lut.size()),
                            lut) +
               pages.verify_guards("lookup pages") + decode.verify_guards("lookup LUT");
    }
};

struct Fixture {
    int t, b, stride, tail_stride, slots, eos = 7;
    std::vector<int> ids, tail, valid, selected;
    GuardedDeviceBuffer d_ids, d_tail, d_valid, d_selected, output;
    Tensor input, counts, result;
    ops::NgramTokenTailView before;
    ops::NgramHashParameters hash;
    std::vector<std::unique_ptr<Shard>> table;

    Fixture(int tokens, int batch, bool padded, bool wide = false)
        : t(tokens), b(batch), stride(t + (padded ? 3 : 0)), tail_stride(padded ? 5 : 2),
          slots(b + 2), ids(stride * b), tail(tail_stride * slots), valid(b), selected(b),
          d_ids(ids.size() * 4), d_tail(tail.size() * 4), d_valid(b * 4), d_selected(b * 4),
          output(std::size_t(2560) * t * b * 2), input(d_ids.span(), DType::I32, {t, b}),
          counts(d_valid.span(), DType::I32, {b}), result(output.span(), DType::BF16, {2560, t, b}),
          before{Tensor(d_tail.span(), DType::I32, {2, slots}),
                 Tensor(d_selected.span(), DType::I32, {b})} {
        input.nb[1]            = stride * 4;
        before.token_ids.nb[1] = tail_stride * 4;
        hash.multipliers = {0xd6e8feb86659fd93ULL, 0xa5a3564e27f8862bULL, 0x9e3779b97f4a7c15ULL};
        if (wide) {
            hash.multipliers = {UINT64_MAX, 1ULL << 63, 3};
            constexpr std::uint32_t moduli[]{1,     2,     UINT32_MAX, 1U << 31,
                                             65535, 65536, 65537,      1234567};
            for (int h = 0; h < 16; ++h) {
                hash.vocab_sizes[h] = moduli[h % 8];
                hash.offsets[h]     = h % 8 == 2 ? 1 : 0;
            }
            std::set<std::uint32_t> needed;
            for (int phase = 0; phase < 2; ++phase) {
                reset(phase, true);
                for (int col = 0; col < t; ++col)
                    for (int h = 0; h < 16; ++h) needed.insert(hash_for(h, col, 0));
            }
            int number = 0;
            for (auto row : needed)
                table.push_back(std::make_unique<Shard>(row, 1, number++, padded));
        } else {
            for (int h = 0; h < 16; ++h) {
                hash.vocab_sizes[h] = 701 - h * 2;
                hash.offsets[h]     = h * 744;
            }
            for (int i = 0; i < 128; ++i)
                table.push_back(std::make_unique<Shard>(i * 93, 93, i, padded));
        }
        reset(0, wide);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void reset(int phase, bool wide = false) {
        std::fill(ids.begin(), ids.end(), -777);
        std::fill(tail.begin(), tail.end(), -999);
        for (int s = 0; s < slots; ++s) {
            tail[s * tail_stride]     = (s * 13 + phase) % 251;
            tail[s * tail_stride + 1] = s % 2 ? eos : 19;
        }
        for (int row = 0; row < b; ++row) {
            valid[row]    = row == 0 && b > 1 && phase == 0 ? 0 : std::max(1, t - row % t);
            selected[row] = valid[row] ? (b - row + phase) % slots : -1;
            for (int col = 0; col < valid[row]; ++col)
                ids[row * stride + col] =
                    col % 4 == 1 ? eos : (col * 97 + row * 17 + phase * 33) % 251;
        }
        if (wide) {
            valid[0]    = t;
            selected[0] = 0;
            tail[0]     = phase ? INT32_MAX : 0;
            tail[1]     = phase ? 1 : 0;
            const int values[2][3]{{1, 0, INT32_MAX}, {INT32_MAX, 7, 1}};
            std::copy_n(values[phase], t, ids.begin());
        }
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        d_tail.copy_from_host(tail.data(), tail.size() * 4);
        d_valid.copy_from_host(valid.data(), valid.size() * 4);
        d_selected.copy_from_host(selected.data(), selected.size() * 4);
        CUDA_CHECK(cudaMemset(output.data(), 0xa5, output.bytes()));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    std::uint32_t hash_for(int h, int col, int row) const {
        const int slot     = selected[row];
        const int previous = col ? ids[row * stride + col - 1] : tail[slot * tail_stride + 1];
        const int older =
            col > 1 ? ids[row * stride + col - 2] : tail[slot * tail_stride + (col == 1)];
        return row_id(hash, h, ids[row * stride + col], previous, older, eos);
    }

    std::vector<ops::Nf4EmbeddingShard> views(int partition) const {
        std::vector<ops::Nf4EmbeddingShard> out;
        for (std::size_t i = 0; i < table.size(); ++i) {
            const bool take = partition == 0 || (partition == 1 && i < table.size() / 2) ||
                              (partition == 2 && i >= table.size() / 2) ||
                              (partition == 3 && i % 2 == 0);
            if (take) out.push_back(table[i]->view());
        }
        return out;
    }

    std::vector<std::uint16_t> reference(std::span<const ops::Nf4EmbeddingShard> local) const {
        std::vector<std::uint16_t> out(std::size_t(2560) * t * b);
        for (int row = 0; row < b; ++row)
            for (int col = 0; col < valid[row]; ++col)
                for (int h = 0; h < 16; ++h) {
                    const auto index   = hash_for(h, col, row);
                    const Shard* owner = nullptr;
                    for (const auto& shard : local)
                        if (std::uint64_t(index) >= shard.row_begin &&
                            std::uint64_t(index) < std::uint64_t(shard.row_begin) + shard.rows) {
                            for (const auto& candidate : table)
                                if (candidate->first == shard.row_begin) owner = candidate.get();
                            break;
                        }
                    if (owner)
                        for (int c = 0; c < 160; ++c)
                            out[(std::size_t(row) * t + col) * 2560 + h * 160 + c] =
                                owner->at(index, c);
                }
        return out;
    }

    int verify(std::span<const ops::Nf4EmbeddingShard> local) const {
        int failures = verify_exact(
            "ngram embedding", from_device<std::uint16_t>(output.data(), std::size_t(2560) * t * b),
            reference(local));
        failures += output.verify_guards("embedding") + d_ids.verify_guards("IDs") +
                    d_tail.verify_guards("tail") + d_valid.verify_guards("counts") +
                    d_selected.verify_guards("slots");
        failures += verify_exact("IDs preserved", from_device<int>(d_ids.data(), ids.size()), ids);
        failures +=
            verify_exact("tail preserved", from_device<int>(d_tail.data(), tail.size()), tail);
        failures +=
            verify_exact("counts preserved", from_device<int>(d_valid.data(), valid.size()), valid);
        failures += verify_exact("slots preserved",
                                 from_device<int>(d_selected.data(), selected.size()), selected);
        for (const auto& shard : table) failures += shard->preserved();
        return failures;
    }

    void launch(std::span<const ops::Nf4EmbeddingShard> local, CudaExecutionView execution) {
        ops::ngram_embedding_lookup(input, counts, before, hash, eos, local, result, execution);
    }
};

int run(int tokens, int batch, int partition, bool padded, bool wide = false) {
    Fixture f(tokens, batch, padded, wide);
    auto& device     = cuda_environment().context;
    const auto local = f.views(partition);
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] { f.launch(local, device.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        f.reset(phase, wide);
        graph.launch(device.stream());
        device.synchronize();
        failures += f.verify(local);
    }
    return failures;
}

int combine() {
    Fixture f(6, 4, true);
    auto& device     = cuda_environment().context;
    const auto first = f.views(1), second = f.views(2);
    f.launch(first, device.execution_view());
    device.synchronize();
    auto a = from_device<std::uint16_t>(f.output.data(), std::size_t(2560) * f.t * f.b);
    f.launch(second, device.execution_view());
    device.synchronize();
    auto b              = from_device<std::uint16_t>(f.output.data(), a.size());
    const auto expected = f.reference(f.views(0));
    int failures = 0, signed_zeros = 0;
    for (std::size_t i = 0; i < a.size(); i += 2) {
        const auto pack = [i](const auto& words) {
            return std::uint32_t(words[i]) | (std::uint32_t(words[i + 1]) << 16);
        };
        failures += pack(a) + pack(b) != pack(expected);
        signed_zeros += expected[i] == 0x8000;
    }
    return failures + (signed_zeros == 0);
}

int invalid() {
    Fixture f(1, 1, false);
    auto local         = f.views(0);
    const auto cuda    = cuda_environment().context.execution_view();
    const auto rejects = [&](auto change) {
        auto copy = local;
        change(copy);
        try {
            f.launch(copy, cuda);
        } catch (const std::invalid_argument&) { return 0; }
        return 1;
    };
    return rejects([](auto& views) { views[1].row_begin = 1; }) +
           rejects([](auto& views) { views[0].pages.bytes = 4096; }) +
           rejects([](auto& views) { views[0].decode.ne[0] = 8; });
}
} // namespace

void add_ple_lookup_cases(ninfer::test::OpSuite& suite) {
    for (auto [t, b] :
         {std::pair{1, 1}, {1, 8}, {6, 4}, {7, 1}, {8, 1}, {9, 1}, {17, 8}, {128, 1}, {4096, 1}})
        for (int partition : {0, 1, 3, 4})
            suite.add("ngram_embedding_lookup/t" + std::to_string(t) + "/b" + std::to_string(b) +
                          "/partition" + std::to_string(partition),
                      [=] { return run(t, b, partition, t % 2); },
                      {{"format", "nf4"}, {"execution", "graph"}});
    suite.add("ngram_embedding_lookup/wide_hash", [] { return run(3, 1, 0, true, true); },
              {{"format", "nf4"}, {"execution", "graph"}});
    suite.add("ngram_embedding_lookup/combine_words", combine, {{"format", "nf4"}});
    suite.add("ngram_embedding_lookup/contract", invalid, {{"format", "nf4"}});
}
