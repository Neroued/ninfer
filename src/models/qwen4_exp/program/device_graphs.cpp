#include "models/qwen4_exp/program/device_graphs.h"
#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

DeviceGraphs::DeviceGraphs(RankProgramImpl& owner)
    : owner_(owner), bounds_(decode_context_bounds(owner.shape.max_context)) {}

execution::TextRun DeviceGraphs::main_run(std::uint32_t batch, bool verify, std::uint32_t visible) {
    auto& p           = owner_;
    auto& r           = p.resources;
    const auto& io    = r.plan.persistent.buffers;
    const auto tokens = verify ? p.shape.draft_tokens + 1 : 1;
    execution::TextExecution execution{r.workspace,
                                       *r.states,
                                       *r.main,
                                       nullptr,
                                       r.tensor(io.main_qsa.keys),
                                       r.tensor(io.main_qsa.coordinates),
                                       r.tensor(io.ple_records),
                                       &*p.parallel};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(*r.gdn_records);
    return execution::TextRun(p.parameters, execution, r.main_batch(tokens, batch, visible), update,
                              p.cuda);
}

void DeviceGraphs::body(Batch& profiles, Round* round, std::uint32_t batch, std::uint32_t depth,
                        DeviceRound mode) {
    auto& p        = owner_;
    auto& r        = p.resources;
    const auto& io = r.plan.persistent.buffers;
    const auto& c  = p.parameters.profile->config.text;
    const int b = batch, t = depth ? p.shape.draft_tokens + 1 : 1;
    const auto visible = round ? bounds_.front() : bounds_.back();
    if (depth) {
        for (std::uint32_t step = 0; step < depth; ++step) {
            DecodeGraphChild child;
            p.draft_forward(step, batch, visible, round ? &profiles.drafts[step].front() : nullptr,
                            round ? &child : nullptr);
            if (round) round->draft.push_back(child);
            p.save_draft(step, batch);
        }
        const auto& mtp = *io.mtp;
        auto anchors    = r.tensor(mtp.anchors).slice(0, 0, batch);
        auto extents    = r.tensor(mtp.extents).slice(0, 0, batch);
        auto drafts     = r.packed(mtp.drafts, {t - 1, b});
        auto ids        = r.main_batch(t, b, visible).ids;
        ops::prepare_verify_ids(anchors, drafts, extents, ids, p.cuda);
        p.copy_verify_inputs(batch, round != nullptr);
    }
    const auto call = r.main_batch(t, b, visible);
    p.prepare_device_ple(call);
    {
        auto run = main_run(batch, depth != 0, visible);
        run.stem();
        std::size_t mixer = 0;
        for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
            if (layer == c.ple.layer) {
                p.ple_ready.wait(p.cuda.stream());
                run.consume_ple(p.device_embedding(t, b));
            }
            if (round && std::holds_alternative<execution::QsaParameters>(
                             p.parameters.text.layers[layer].mixer)) {
                auto child = profiles.main[depth != 0].front()[mixer++].append_to(p.cuda.stream());
                round->main.push_back(child);
                run.finish_mixer(layer);
            } else
                run.mixer(layer);
            run.resident_moe(layer);
            run.combine(layer);
        }
        execution::TextOutputs output{r.packed(io.pending_sample, {dimension(c.hidden), t, b}), {}};
        if (io.pending_conditioning)
            output.conditioning =
                r.packed(*io.pending_conditioning, {dimension(c.residual_width()), t, b});
        Tensor logits;
        if (mode != DeviceRound::Control)
            logits =
                r.packed(*io.logits, {dimension(p.parameters.profile->local.vocabulary), t, b});
        run.finish(output, logits);
    }
    if (mode == DeviceRound::Greedy || mode == DeviceRound::Sample) {
        const auto k = mode == DeviceRound::Greedy ? 1 : 20;
        if (depth)
            p.vocabulary_accept(batch, k);
        else
            p.vocabulary_sample(batch, false, k);
        p.copy_sample_result(batch, depth != 0);
    }
}

void DeviceGraphs::prepare(GraphInputs& inputs) {
    auto& p        = owner_;
    auto& r        = p.resources;
    const auto& io = r.plan.persistent.buffers;
    const auto drafts =
        p.parameters.mtp ? std::min(p.shape.draft_tokens, p.shape.max_context - 1) : 0;
    batches_.resize(p.shape.max_concurrency);
    const auto write = [&](const Tensor& target, std::int32_t value) {
        std::vector<std::int32_t> values(target.numel(), value);
        copy_bytes({target.data, target.bytes(), target.device},
                   {values.data(), values.size() * sizeof(value), Device::cpu()}, p.cuda);
    };
    for (std::uint32_t b = 1; b <= p.shape.max_concurrency; ++b) {
        auto& batch = batches_[b - 1];
        batch.drafts.resize(drafts);
        batch.rounds.resize(drafts + 1);
        for (std::uint32_t verify = 0; verify < (drafts ? 2U : 1U); ++verify) {
            const auto t = verify ? p.shape.draft_tokens + 1 : 1;
            inputs.batch(r.main_batch(t, b, bounds_.back()), bounds_.back());
            for (auto visible : bounds_) {
                auto& profile = batch.main[verify].emplace_back();
                auto run      = main_run(b, verify, visible);
                for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
                    if (!std::holds_alternative<execution::QsaParameters>(
                            p.parameters.text.layers[layer].mixer))
                        continue;
                    profile.emplace_back().capture(p.cuda.stream(),
                                                   [&] { run.local_mixer(layer); });
                }
            }
        }
        for (std::uint32_t step = 0; step < drafts; ++step) {
            inputs.batch(r.mtp_batch(io.mtp->draft_steps[step], 1, b, bounds_.back()),
                         bounds_.back());
            for (auto visible : bounds_) {
                auto run = p.draft_run(step, b, visible);
                batch.drafts[step].emplace_back().capture(p.cuda.stream(),
                                                          [&] { run.local_mixer(); });
            }
        }
        for (std::uint32_t depth = 0; depth <= drafts; ++depth) {
            const auto t = depth ? p.shape.draft_tokens + 1 : 1;
            inputs.batch(r.main_batch(t, b, bounds_.back()), bounds_.back());
            write(r.tensor(io.batch.valid_columns).slice(0, 0, b), depth + 1);
            if (depth) {
                write(r.tensor(io.mtp->extents).slice(0, 0, b), depth);
                write(r.tensor(io.mtp->anchors).slice(0, 0, b),
                      p.parameters.profile->config.text.ple.eos);
            }
            body(batch, nullptr, b, depth, DeviceRound::Sample);
            p.device.synchronize();
            for (std::uint32_t m = 0; m < (depth ? 3U : 4U); ++m) {
                auto& round = batch.rounds[depth][m];
                round.installed_draft.resize(depth);
                round.graph.prepare(
                    p.cuda, [&] { body(batch, &round, b, depth, static_cast<DeviceRound>(m)); },
                    p.parallel ? &*p.parallel : nullptr);
            }
            // Replay a complete round after construction; all ranks enter the same variant.
            replay(b, DeviceRound::Sample, std::vector<std::uint32_t>(depth, bounds_.back()),
                   bounds_.back());
            p.device.synchronize();
        }
        for (int verify = 0; verify < (drafts ? 2 : 1); ++verify)
            for (int sampled = 0; sampled < 2; ++sampled) {
                batch.decisions[verify * 2 + sampled].prepare(
                    p.cuda,
                    [&] {
                        if (verify)
                            p.vocabulary_accept(b, sampled ? 20 : 1);
                        else
                            p.vocabulary_sample(b, false, sampled ? 20 : 1);
                        p.copy_sample_result(b, verify);
                    },
                    p.parallel ? &*p.parallel : nullptr);
            }
    }
    if (drafts)
        for (std::uint32_t b = 1; b <= p.shape.max_concurrency; ++b)
            for (int rebuild = 0; rebuild < 2; ++rebuild)
                batches_[b - 1].finish[rebuild].prepare(
                    p.cuda, [&] { p.finish_verify(b, rebuild); },
                    p.parallel ? &*p.parallel : nullptr);
    workspace_peak_ = r.workspace.peak_used();
}

std::size_t DeviceGraphs::bound(std::uint32_t visible) const {
    const auto found = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (found == bounds_.end()) throw std::logic_error("decode graph context exceeds its capacity");
    return found - bounds_.begin();
}

void DeviceGraphs::replay(std::uint32_t batch, DeviceRound mode,
                          std::span<const std::uint32_t> draft_visible, std::uint32_t visible) {
    auto& profiles  = batches_.at(batch - 1);
    auto& round     = profiles.rounds.at(draft_visible.size())[static_cast<int>(mode)];
    const auto main = bound(visible);
    if (main != round.installed_main) {
        const auto& selected = profiles.main[!draft_visible.empty()][main];
        for (std::size_t i = 0; i < round.main.size(); ++i)
            round.graph.executable.update_child(round.main[i], selected[i]);
        round.installed_main = main;
    }
    for (std::size_t step = 0; step < draft_visible.size(); ++step) {
        const auto selected = bound(draft_visible[step]);
        if (selected != round.installed_draft[step]) {
            round.graph.executable.update_child(round.draft[step], profiles.drafts[step][selected]);
            round.installed_draft[step] = selected;
        }
    }
    round.graph.launch(owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
}

void DeviceGraphs::decide(std::uint32_t batch, bool verify, std::int32_t candidates) {
    batches_.at(batch - 1).decisions[verify * 2 + (candidates == 20)].launch(
        owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
}

void DeviceGraphs::finish(std::uint32_t batch, bool rebuild_history) {
    batches_.at(batch - 1).finish[rebuild_history].launch(
        owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
}

} // namespace ninfer::models::qwen4_exp::detail
