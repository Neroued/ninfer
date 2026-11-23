#include "ops/suite.h"
#include "ops/op_tester.h"
#include "ops/sampling/random_reference.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
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

// Both target and proposal see the previous token. A low-confidence step is excluded
// using only its conditional distribution, before its proposed token can affect the extent.
int causal_stopping(bool sparse, bool sharded) {
    constexpr int proposal_steps = 3, width = proposal_steps + 1;
    constexpr int vocab = 16, alternate = 8, features = 256;
    const bool full_sparse    = sparse && !sharded;
    const int batch           = full_sparse ? 1024 : 32768;
    const int trials          = full_sparse ? 16 : 1;
    const int lanes           = proposal_steps * batch;
    const int physical_rows   = full_sparse ? physical : vocab;
    const int token_domain    = full_sparse ? domain : vocab;
    constexpr auto absent     = std::uint16_t{0xc47a}; // BF16 -1000
    constexpr float threshold = .7f;
    auto& device              = cuda_environment().context;
    const auto cuda           = device.execution_view();
    auto buffer = [&](std::size_t bytes) { return DeviceBuffer(cuda.device(), bytes); };

    std::vector<int> anchors(batch), positions(batch), live(batch, 1), extents(batch), valid(batch);
    auto anchor_buf = to_device(anchors), position_buf = to_device(positions),
         live_buf = to_device(live), extent_buf = to_device(extents), valid_buf = to_device(valid);
    Tensor anchor(anchor_buf.span(), DType::I32, {batch});
    Tensor position(position_buf.span(), DType::I32, {batch});
    Tensor live_rows(live_buf.span(), DType::I32, {batch});
    Tensor extent(extent_buf.span(), DType::I32, {batch});
    Tensor valid_columns(valid_buf.span(), DType::I32, {batch});
    const int stride = (token_domain + 31) / 32;
    std::vector<std::uint32_t> masks(width * stride);
    for (int t = 0; t < width; ++t) masks[t * stride] = 1U | (1U << alternate);
    auto mask_buf = to_device(masks);
    std::vector<ops::SamplingConfig> target_config(batch), proposal_config(batch);
    for (int b = 0; b < batch; ++b) {
        target_config[b] = {.temperature = 1.0f, .top_k = 2, .seed = 0xa7248b100ULL + b};
        if (full_sparse)
            target_config[b].mask = {static_cast<const std::uint32_t*>(mask_buf.data()), stride};
        proposal_config[b] = {
            .temperature = sparse ? 1.0f : 0.0f, .top_k = 2, .seed = target_config[b].seed};
    }
    auto target_cfg_buf = to_device(target_config), proposal_cfg_buf = to_device(proposal_config);
    const auto* target_cfg   = static_cast<const ops::SamplingConfig*>(target_cfg_buf.data());
    const auto* proposal_cfg = static_cast<const ops::SamplingConfig*>(proposal_cfg_buf.data());

    std::vector<int> candidate_ids(vocab * lanes);
    std::vector<float> unary(vocab * lanes, -1000.0f);
    std::vector<std::uint16_t> hidden(features * std::size_t(lanes));
    std::vector<std::uint16_t> predecessor(features * vocab), successor(features * vocab);
    predecessor[0]       = f32_to_bf16(1.0f);
    successor[0]         = f32_to_bf16(1.5f);
    const auto uses_edge = [](int b, int step) { return b % 3 != 1 || step != 1; };
    if (sparse)
        for (int b = 0; b < batch; ++b)
            for (int step = 0; step < proposal_steps; ++step) {
                const int col = b * proposal_steps + step;
                for (int id = 0; id < vocab; ++id) candidate_ids[col * vocab + id] = id;
                unary[col * vocab] = unary[col * vocab + alternate] = 0.0f;
                hidden[std::size_t(col) * features] = f32_to_bf16(uses_edge(b, step) ? 1.0f : 0.0f);
            }
    auto ids_buf = to_device(candidate_ids), unary_buf = to_device(unary),
         hidden_buf = to_device(hidden), pred_buf = to_device(predecessor),
         succ_buf = to_device(successor);
    Tensor choices(ids_buf.span(), DType::I32, {vocab, proposal_steps, batch});
    Tensor scores(unary_buf.span(), DType::FP32, {vocab, proposal_steps, batch});
    Tensor projected(hidden_buf.span(), DType::BF16, {features, proposal_steps, batch});
    Tensor pred(pred_buf.span(), DType::BF16, {features, vocab});
    Tensor succ(succ_buf.span(), DType::BF16, {features, vocab});

    const int proposal_cap = sparse ? 20 : 1;
    const auto bytes       = std::max(
        {ops::argmax_with_probability_workspace_capacity_bytes(vocab, lanes),
               ops::candidate_selector_path_workspace_capacity_bytes(
             ops::CandidateSelectorInput::Candidates, proposal_steps, proposal_steps, batch, batch),
               ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
             vocab, proposal_steps, proposal_steps, batch, batch),
         full_sparse ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                           token_domain, {false}, proposal_steps, proposal_steps, batch, batch)
                           : 0,
               ops::vocabulary_candidates_workspace_capacity_bytes(vocab / 2, width * batch, 20),
               ops::vocabulary_candidates_with_lse_workspace_capacity_bytes(vocab / 2, lanes,
                                                                            proposal_cap),
               std::size_t(1)});
    auto scratch = buffer(bytes);
    MemoryArena arena(scratch.span());
    auto draft_buf = buffer(lanes * 4), q_buf = buffer(vocab * lanes * 4),
         confidence_buf = buffer(lanes * 4), shard_p_buf = buffer(lanes * 4),
         shard_ids_buf = buffer(lanes * 4);
    Tensor drafts(draft_buf.span(), DType::I32, {proposal_steps, batch});
    Tensor q(q_buf.span(), DType::FP32, {vocab, proposal_steps, batch});
    Tensor confidence(confidence_buf.span(), DType::FP32, {proposal_steps, batch});
    Tensor draft_flat(draft_buf.span(), DType::I32, {lanes});
    Tensor confidence_flat(confidence_buf.span(), DType::FP32, {lanes});
    Tensor sharded_p(shard_p_buf.span(), DType::FP32, {lanes});
    Tensor sharded_ids(shard_ids_buf.span(), DType::I32, {lanes});
    auto draft_valid_buf = to_device(std::vector<int>(lanes, 1));
    auto draft_pos_buf   = buffer(lanes * 4),
         draft_cfg_buf   = buffer(lanes * sizeof(ops::SamplingConfig));
    Tensor draft_valid(draft_valid_buf.span(), DType::I32, {lanes});
    Tensor draft_pos(draft_pos_buf.span(), DType::I32, {lanes});
    const auto* draft_cfg = static_cast<const ops::SamplingConfig*>(draft_cfg_buf.data());
    auto input_buf        = buffer(std::size_t(physical_rows) * width * batch * 2);
    auto records_buf      = buffer(std::size_t(proposal_cap + 1) * lanes * 2 * 8);
    auto key_buf          = buffer(std::size_t(20) * width * batch * 2 * 8);
    auto tokens_buf = buffer(width * batch * 4), counts_buf = buffer(batch * 4),
         target_ids_buf = buffer(width * batch * 4);
    Tensor records(records_buf.span(), DType::I64, {proposal_cap + 1, 1, lanes, 2});
    Tensor keys(key_buf.span(), DType::I64, {20, width, batch, 2});
    Tensor target(input_buf.span(), DType::BF16, {physical_rows, width, batch});
    Tensor target_flat(input_buf.span(), DType::BF16, {vocab, width * batch});
    Tensor target_ids(target_ids_buf.span(), DType::I32, {width, batch});
    Tensor target_flat_ids(target_ids_buf.span(), DType::I32, {width * batch});
    Tensor licensed(tokens_buf.span(), DType::I32, {width, batch});
    Tensor counts(counts_buf.span(), DType::I32, {batch});
    if (full_sparse) {
        input_buf.fill();
        target_ids_buf.fill();
        device.synchronize();
    }

    std::vector<std::uint16_t> logits(std::size_t(vocab) * width * batch);
    // Place token 0 and token 8 on different ranks, emulating one all-gather on this device.
    const auto upload = [&](int columns) {
        if (!sharded) {
            input_buf.copy_from_host(logits.data(), std::size_t(columns) * vocab * 2);
            return;
        }
        constexpr int shard = vocab / 2;
        std::vector<std::uint16_t> packed(std::size_t(columns) * vocab);
        for (int rank = 0; rank < 2; ++rank)
            for (int col = 0; col < columns; ++col)
                std::copy_n(logits.begin() + col * vocab + rank * shard, shard,
                            packed.begin() + (rank * columns + col) * shard);
        input_buf.copy_from_host(packed.data(), packed.size() * 2);
    };
    const auto shard_logits = [&](int rank, int steps, int rows) {
        return Tensor(input_buf.span().subspan(std::size_t(rank) * steps * rows * vocab),
                      DType::BF16, {vocab / 2, steps, rows});
    };
    std::vector<int> sequences(batch);
    std::array<int, width> observed_extents{};
    std::array<int, 1 << width> observed{};
    int errors           = 0;
    const auto run_trial = [&] {
        int rounds = 0;
        for (; rounds < width && std::any_of(live.begin(), live.end(), [](int x) { return x; });
             ++rounds) {
            anchor_buf.copy_from_host(anchors.data(), batch * 4);
            position_buf.copy_from_host(positions.data(), batch * 4);
            live_buf.copy_from_host(live.data(), batch * 4);
            std::vector<int> proposal_tokens;
            if (sparse) {
                // The proposal counter starts at the next logical position.
                auto next = positions;
                for (auto& p : next) ++p;
                auto next_buf = to_device(next);
                Tensor next_pos(next_buf.span(), DType::I32, {batch});
                ops::candidate_selector_path(choices, scores, projected, anchor, pred, succ, vocab,
                                             next_pos, proposal_cfg, drafts, q, confidence,
                                             live_rows, arena, cuda);
                device.synchronize();
                proposal_tokens = from_device<int>(draft_buf, lanes);
            }
            if (!sparse || sharded) {
                std::fill_n(logits.begin(), std::size_t(vocab) * lanes, absent);
                std::vector<ops::SamplingConfig> flat_cfg(lanes);
                std::vector<int> flat_pos(lanes);
                for (int b = 0; b < batch; ++b)
                    for (int step = 0; step < proposal_steps; ++step) {
                        const int col  = b * proposal_steps + step;
                        const int prev = step == 0 ? anchors[b]
                                         : sparse  ? proposal_tokens[col - 1]
                                                   : 0;
                        logits[col * vocab] =
                            f32_to_bf16(prev == 0 && uses_edge(b, step) ? 1.5f : 0.0f);
                        logits[col * vocab + alternate] = 0;
                        flat_cfg[col]                   = proposal_config[b];
                        flat_pos[col]                   = positions[b] + step + 1;
                    }
                upload(lanes);
                if (sharded) {
                    draft_cfg_buf.copy_from_host(flat_cfg.data(),
                                                 flat_cfg.size() * sizeof(flat_cfg[0]));
                    draft_pos_buf.copy_from_host(flat_pos.data(), flat_pos.size() * 4);
                    for (int rank = 0; rank < 2; ++rank) {
                        auto record = records.slice(3, rank, 1);
                        ops::vocabulary_candidates_with_lse(shard_logits(rank, 1, lanes),
                                                            rank * (vocab / 2), vocab, draft_cfg,
                                                            draft_valid, {}, record, arena, cuda);
                    }
                    auto& ids = sparse ? sharded_ids : draft_flat;
                    ops::sample_vocabulary_candidates(records, draft_cfg, draft_pos,
                                                      ops::kSamplePurposeDFlash2Proposal, ids,
                                                      sharded_p, {}, cuda);
                } else {
                    const Tensor input(input_buf.span(), DType::BF16, {vocab, lanes});
                    ops::argmax_with_probability(input, draft_flat, confidence_flat, vocab, arena,
                                                 cuda);
                }
                device.synchronize();
                if (!sparse) proposal_tokens = from_device<int>(draft_buf, lanes);
            }
            const auto p = from_device<float>(sharded ? shard_p_buf : confidence_buf, lanes);
            const auto selector_p = sparse && sharded ? from_device<float>(confidence_buf, lanes)
                                                      : std::vector<float>{};
            std::fill(logits.begin(), logits.end(), absent);
            for (int b = 0; b < batch; ++b) {
                extents[b] = valid[b] = 0;
                if (!live[b]) continue;
                const int base  = b * proposal_steps;
                int prev        = anchors[b];
                const int limit = std::min(proposal_steps, width - positions[b] - 1);
                while (extents[b] < limit) {
                    const int col = base + extents[b];
                    if (!selector_p.empty() && std::abs(p[col] - selector_p[col]) > 2e-6) ++errors;
                    if (!(p[col] >= threshold)) break;
                    ++extents[b];
                }
                ++observed_extents[extents[b]];
                valid[b] = extents[b] + 1;
                for (int step = 0; step < valid[b]; ++step) {
                    const auto col          = (b * width + step) * vocab;
                    logits[col]             = f32_to_bf16(prev == 0 ? 0.0f : .75f);
                    logits[col + alternate] = f32_to_bf16(prev == 0 ? 1.0f : 0.0f);
                    if (step < extents[b]) prev = proposal_tokens[base + step];
                }
            }
            extent_buf.copy_from_host(extents.data(), batch * 4);
            valid_buf.copy_from_host(valid.data(), batch * 4);
            if (full_sparse) {
                // Update only the leading scores in each full-vocabulary column.
                // The target mask admits only tokens 0 and 8.
                CUDA_CHECK(cudaMemcpy2DAsync(input_buf.data(), physical_rows * 2, logits.data(),
                                             vocab * 2, vocab * 2, width * batch,
                                             cudaMemcpyHostToDevice, cuda.stream()));
            } else {
                upload(width * batch);
            }
            if (sharded) {
                for (int rank = 0; rank < 2; ++rank) {
                    auto out = keys.slice(3, rank, 1);
                    ops::vocabulary_candidates(shard_logits(rank, width, batch), rank * (vocab / 2),
                                               vocab, target_cfg, valid_columns, drafts, out, arena,
                                               cuda);
                }
                if (sparse)
                    ops::accept_vocabulary_candidates_sparse(keys, target_cfg, drafts, choices, q,
                                                             extent, position, licensed, counts,
                                                             live_rows, cuda);
                else
                    ops::accept_vocabulary_candidates(keys, target_cfg, drafts, extent, position,
                                                      licensed, counts, live_rows, cuda);
            } else if (sparse) {
                ops::speculative_accept_sparse_drafts(
                    target_ids, target, drafts, choices, q, extent, position, licensed, counts,
                    token_domain, target_cfg, {false}, live_rows, arena, cuda);
            } else {
                ops::argmax(target_flat, target_flat_ids, vocab, cuda);
                ops::speculative_accept_greedy_drafts(target_ids, target, drafts, extent, position,
                                                      licensed, counts, vocab, target_cfg,
                                                      live_rows, arena, cuda);
            }
            device.synchronize();
            const auto emitted = from_device<int>(tokens_buf, width * batch);
            const auto lengths = from_device<int>(counts_buf, batch);
            for (int b = 0; b < batch; ++b) {
                if (!live[b]) continue;
                if (lengths[b] < 1 || lengths[b] > extents[b] + 1) return false;
                for (int i = 0; i < lengths[b]; ++i) {
                    const int token = emitted[b * width + i];
                    if (token != 0 && token != alternate) return false;
                    sequences[b] = 2 * sequences[b] + (token == alternate);
                    anchors[b]   = token;
                }
                positions[b] += lengths[b];
                live[b] = positions[b] < width;
            }
        }
        return rounds >= 2 && std::none_of(live.begin(), live.end(), [](int x) { return x; });
    };
    for (int trial = 0; trial < trials; ++trial) {
        if (trial != 0) {
            std::fill(anchors.begin(), anchors.end(), 0);
            std::fill(positions.begin(), positions.end(), 0);
            std::fill(live.begin(), live.end(), 1);
            std::fill(sequences.begin(), sequences.end(), 0);
            for (int b = 0; b < batch; ++b) {
                target_config[b].seed += batch;
                proposal_config[b].seed = target_config[b].seed;
            }
            target_cfg_buf.copy_from_host(target_config.data(), target_cfg_buf.size());
            proposal_cfg_buf.copy_from_host(proposal_config.data(), proposal_cfg_buf.size());
        }
        if (!run_trial()) {
            std::cerr << "invalid licensed sequence in causal stopping trial\n";
            return errors + 1;
        }
        for (int seq : sequences) ++observed[seq];
    }
    if (observed_extents[0] == 0 || observed_extents[1] == 0 ||
        observed_extents[proposal_steps] == 0) {
        std::cerr << "causal stopping fixture did not exercise variable extents across rounds\n";
        return errors + 1;
    }
    const int samples = batch * trials;
    const std::array<double, 2> next_one{1.0 / (1.0 + std::exp(-1.0)), 1.0 / (1.0 + std::exp(.75))};
    for (int seq = 0; seq < int(observed.size()); ++seq) {
        int prev   = 0;
        double law = 1.0;
        for (int bit = width - 1; bit >= 0; --bit) {
            const int token = (seq >> bit) & 1;
            law *= token ? next_one[prev] : 1.0 - next_one[prev];
            prev = token;
        }
        const double sigma = std::sqrt(samples * law * (1.0 - law));
        if (std::abs(observed[seq] - samples * law) > 6.0 * sigma + 1.0) {
            std::cerr << "causal stopping sequence " << seq << " observed " << observed[seq]
                      << ", analytical expectation " << samples * law << '\n';
            ++errors;
        }
    }
    return errors;
}
} // namespace

void add_verification_width_cases(ninfer::test::OpSuite& suite) {
    suite.add("accept/rectangular/proposal7_width4", [] { return rectangular_accept(7, 4); });
    suite.add("accept/rectangular/proposal2_width8", [] { return rectangular_accept(2, 8); });
    for (bool sparse : {false, true})
        for (bool sharded : {false, true})
            suite.add(std::string("accept/causal_stopping/") + (sharded ? "sharded_" : "local_") +
                          (sparse ? "sparse" : "greedy"),
                      [=] { return causal_stopping(sparse, sharded); });
}
