#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/sampling/random_reference.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;

constexpr int rows = 5, domain = 248077, physical = 248320;

int first_token(int row, int col) { return (8 * row + col) * 4 + 1; }

int draft_index(int row, int col) { return (row + col) % 2; }

double first_q(int row, int col) { return (row + col) % 3 == 0 ? .75 : .5; }

// Target mass is exactly 1/4 on four consecutive IDs. A sparse proposal puts its
// two masses on the first two IDs. The one-hot proposal puts all mass on its draft.
std::vector<int> reference(int row, int extent, bool sparse, const ops::SamplingConfig& cfg) {
    using sampling_reference::uniform;
    std::vector<int> tokens;
    for (int col = 0; col <= extent; ++col) {
        const int first = first_token(row, col);
        const int slot  = draft_index(row, col);
        const int pos   = 101 + row + col;
        std::array<double, 4> q{};
        if (sparse) {
            q[0] = first_q(row, col);
            q[1] = 1. - q[0];
        } else {
            q[slot] = 1.;
        }
        if (col < extent) {
            const bool pass =
                cfg.temperature <= 0
                    ? slot == 0
                    : uniform(cfg.seed, pos, ops::kSamplePurposeSpeculativeAccept) < .25 / q[slot];
            if (pass) {
                tokens.push_back(first + slot);
                continue;
            }
        }
        int pick = 0;
        if (cfg.temperature > 0) {
            const bool bonus = col == extent;
            std::array<double, 4> weights{};
            double mass = 0.;
            for (int j = 0; j < 4; ++j) {
                weights[j] = bonus ? .25 : std::max(0., .25 - q[j]);
                mass += weights[j];
            }
            const auto purpose = bonus ? ops::kSamplePurposeSpeculativeBonus
                                       : ops::kSamplePurposeSpeculativeCorrection;
            double goal        = mass * uniform(cfg.seed, pos, purpose);
            while (pick < 3 && goal >= weights[pick]) goal -= weights[pick++];
        }
        tokens.push_back(first + pick);
        break;
    }
    return tokens;
}

int rectangular_accept(int steps, int width) {
    auto& device    = cuda_environment().context;
    const auto cuda = device.execution_view();
    const int limit = std::min(steps, width - 1);
    const std::vector<int> extents{limit, limit, 0, INT_MIN, 1};
    const std::vector<int> valid{limit + 1, limit + 1, 1, 0, 2};
    const std::vector<int> live{1, 1, 1, 0, 1}, positions{100, 101, 102, INT_MIN, 104};

    // IDs have padded columns. q stores batch before columns, with different padding.
    constexpr int ids_col = 32, q_row = 24;
    const int ids_row       = steps * ids_col + 9;
    const int q_col         = rows * q_row + 7;
    constexpr int draft_col = rows + 1, draft_row = 1;
    std::vector<int> ids((rows - 1) * ids_row + (steps - 1) * ids_col + 16, INT_MIN),
        drafts((rows - 1) * draft_row + (steps - 1) * draft_col + 1, INT_MIN);
    std::vector<float> q(steps * q_col, std::numeric_limits<float>::quiet_NaN());
    std::vector<int> target(rows * width, INT_MIN);
    std::vector<std::uint16_t> logits(std::size_t(physical) * width * rows, f32_to_bf16(-64));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < valid[row]; ++col) {
            const int first = first_token(row, col);
            const auto at   = (std::size_t(row) * width + col) * physical;
            std::fill_n(logits.begin() + at + first, 4, f32_to_bf16(0));
            target[row * width + col] = first;
            if (col >= extents[row]) continue;
            drafts[col * draft_col + row * draft_row] = first + draft_index(row, col);
            const int id_base                         = row * ids_row + col * ids_col;
            const int q_base                          = col * q_col + row * q_row;
            for (int j = 0; j < 16; ++j) {
                ids[id_base + j] = first + j;
                q[q_base + j]    = j == 0 ? first_q(row, col) : j == 1 ? 1 - first_q(row, col) : 0.;
            }
        }
    }
    auto xbuf = to_device(logits), tbuf = to_device(target), dbuf = to_device(drafts);
    auto ibuf = to_device(ids), qbuf = to_device(q), ebuf = to_device(extents),
         pbuf     = to_device(positions);
    auto live_buf = to_device(live), vbuf = to_device(valid);
    Tensor x(xbuf.span(), DType::BF16, {physical, width, rows}),
        t(tbuf.span(), DType::I32, {width, rows});
    Tensor d(dbuf.span(), DType::I32, {steps, rows});
    d.nb[0] = 4 * draft_col;
    d.nb[1] = 4 * draft_row;
    Tensor choices(ibuf.span(), DType::I32, {16, steps, rows}),
        probs(qbuf.span(), DType::FP32, {16, steps, rows});
    choices.nb[1] = 4 * ids_col;
    choices.nb[2] = 4 * ids_row;
    probs.nb[1]   = 4 * q_col;
    probs.nb[2]   = 4 * q_row;
    Tensor e(ebuf.span(), DType::I32, {rows}), p(pbuf.span(), DType::I32, {rows}),
        active(live_buf.span(), DType::I32, {rows}), lengths(vbuf.span(), DType::I32, {rows});
    GuardedDeviceBuffer output(rows * width * 4), count(rows * 4), keys_buf(20 * width * rows * 8);
    Tensor y(output.span(), DType::I32, {width, rows}), counts(count.span(), DType::I32, {rows});
    Tensor keys(keys_buf.span(), DType::I64, {20, width, rows, 1});
    const auto bytes = std::max(
        {ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(domain, 1, 15, 1, rows),
         ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(domain, {false}, 1, 15, 1,
                                                                        rows),
         ops::vocabulary_candidates_workspace_capacity_bytes(physical, width * rows, 20)});
    const auto greedy_bytes = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
        domain, width - 1, width - 1, rows, rows);
    const auto sparse_bytes = ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
        domain, {false}, width - 1, width - 1, rows, rows);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena arena(scratch.span());
    std::vector<ops::SamplingConfig> cfg(rows);
    DeviceBuffer config_buf(cuda.device(), cfg.size() * sizeof(cfg[0]));
    const auto* params = static_cast<const ops::SamplingConfig*>(config_buf.data());
    int errors         = 0;
    auto check = [&](const char* label, bool sparse, std::size_t expected_peak, auto launch) {
        output.fill(0xa5);
        count.fill(0xa5);
        arena.reset_peak();
        launch();
        device.synchronize();
        if (arena.peak_used() != expected_peak) {
            std::cerr << label << " workspace peak " << arena.peak_used() << ", expected "
                      << expected_peak << '\n';
            ++errors;
        }
        std::vector<int> expected(width * rows), n(rows);
        for (int b = 0; b < rows; ++b) {
            if (!live[b]) continue;
            const auto seq = reference(b, extents[b], sparse, cfg[b]);
            std::copy(seq.begin(), seq.end(), expected.begin() + b * width);
            n[b] = seq.size();
        }
        errors += verify_exact(label, from_device<int>(output.data(), expected.size()), expected);
        errors += verify_exact(label, from_device<int>(count.data(), rows), n);
    };
    for (int phase = 0; phase < 9; ++phase) {
        const bool raw = phase == 0;
        for (int b = 0; b < rows; ++b) {
            cfg[b].temperature = raw || b == 1 ? 0.F : 1.F;
            cfg[b].top_k       = 4;
            cfg[b].seed        = 17 * phase + b;
        }
        cfg[3].mask = {reinterpret_cast<const std::uint32_t*>(std::uintptr_t{4}), 1};
        config_buf.copy_from_host(cfg.data(), config_buf.size());
        check("local one-hot", false, greedy_bytes, [&] {
            ops::speculative_accept_greedy_drafts(t, x, d, e, p, y, counts, domain, params, active,
                                                  arena, cuda);
        });
        check("local sparse", true, raw ? 0 : sparse_bytes, [&] {
            ops::speculative_accept_sparse_drafts(t, x, d, choices, probs, e, p, y, counts, domain,
                                                  params, {raw}, active, arena, cuda);
        });
        ops::vocabulary_candidates(x, 0, domain, params, lengths, d, keys, arena, cuda);
        check("vocabulary one-hot", false, 0, [&] {
            ops::accept_vocabulary_candidates(keys, params, d, e, p, y, counts, active, cuda);
        });
        check("vocabulary sparse", true, 0, [&] {
            ops::accept_vocabulary_candidates_sparse(keys, params, d, choices, probs, e, p, y,
                                                     counts, active, cuda);
        });
    }
    auto reject = [&](const Tensor& ids, const Tensor& q, Tensor out) {
        try {
            ops::speculative_accept_sparse_drafts(t, x, d, ids, q, e, p, out, counts, domain,
                                                  params, {false}, active, arena, cuda);
            ++errors;
        } catch (const std::invalid_argument&) {}
        try {
            ops::accept_vocabulary_candidates_sparse(keys, params, d, ids, q, e, p, out, counts,
                                                     active, cuda);
            ++errors;
        } catch (const std::invalid_argument&) {}
    };
    auto bad_ids  = choices;
    bad_ids.nb[1] = 4; // adjacent lists overlap
    reject(bad_ids, probs, y);
    auto bad_q  = probs;
    bad_q.nb[0] = 8;
    reject(choices, bad_q, y);
    bad_q = probs;
    --bad_q.ne[1];
    reject(choices, bad_q, y);
    auto alias = y;
    alias.data = static_cast<float*>(probs.data) + (steps - 1) * q_col + 2 * q_row;
    reject(choices, probs, alias);
    return errors + output.verify_guards("width tokens") + count.verify_guards("width counts") +
           scratch.verify_guards("width scratch") + keys_buf.verify_guards("width keys");
}
} // namespace

void add_verification_width_cases(ninfer::test::OpSuite& suite) {
    suite.add("accept/rectangular/proposal7_width4", [] { return rectangular_accept(7, 4); });
    suite.add("accept/rectangular/proposal2_width8", [] { return rectangular_accept(2, 8); });
}
