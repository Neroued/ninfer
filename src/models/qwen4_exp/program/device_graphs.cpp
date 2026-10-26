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
                                       p.parallel ? &*p.parallel : nullptr,
                                       p.parallel ? &p.ple_ready : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(*r.gdn_records);
    return execution::TextRun(p.parameters, execution, r.main_batch(tokens, batch, visible), update,
                              p.cuda);
}

void DeviceGraphs::body(Batch& profiles, Round* round, std::uint32_t batch, bool verify) {
    auto& p          = owner_;
    auto& r          = p.resources;
    const auto& io   = r.plan.persistent.buffers;
    const auto& c    = p.parameters.profile->config.text;
    const auto depth = verify ? p.shape.draft_tokens : 0;
    const int b = batch, t = verify ? p.shape.draft_tokens + 1 : 1;
    const auto visible = round ? bounds_.front() : bounds_.back();
    p.install_decode(batch);
    if (depth) {
        for (std::uint32_t step = 0; step < depth; ++step) {
            DecodeGraphChild child;
            p.draft_forward(step, batch, visible, round ? &profiles.drafts[step].front() : nullptr,
                            round ? &child : nullptr);
            if (round) round->draft.push_back(child);
        }
        const auto& mtp = *io.mtp;
        auto anchors    = r.tensor(io.anchors).slice(0, 0, batch);
        auto extents    = r.tensor(mtp.extents).slice(0, 0, batch);
        auto drafts     = p.draft_proposals(b);
        auto ids        = r.main_batch(t, b, visible).ids;
        ops::prepare_verify_ids(anchors, drafts, extents, ids, p.cuda);
        p.copy_verify_inputs(batch, round ? &round->readback : nullptr);
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
        run.finish(output);
        const auto project = [&] {
            auto sample = output.sample.view({dimension(c.hidden), t * b});
            auto logits =
                r.packed(*io.logits, {dimension(p.parameters.profile->local.vocabulary), t * b});
            execution::project_head(sample, p.parameters.text.head, logits, r.workspace, p.cuda);
        };
        if (round)
            round->head.capture(p.cuda.stream(), project);
        else
            project();
    }
}

void DeviceGraphs::prepare(GraphInputs& inputs) {
    auto& p           = owner_;
    auto& r           = p.resources;
    const auto& io    = r.plan.persistent.buffers;
    const auto drafts = p.parameters.mtp ? p.shape.draft_tokens : 0;
    batches_.resize(p.shape.max_concurrency);
    const auto write = [&](const Tensor& target, std::int32_t value) {
        std::vector<std::int32_t> values(target.numel(), value);
        copy_bytes({target.data, target.bytes(), target.device},
                   {values.data(), values.size() * sizeof(value), Device::cpu()}, p.cuda);
    };
    for (std::uint32_t b = 1; b <= p.shape.max_concurrency; ++b) {
        auto& batch = batches_[b - 1];
        batch.drafts.resize(drafts);
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
        for (int verify = 0; verify < (drafts ? 2 : 1); ++verify) {
            const auto depth = verify ? drafts : 0;
            const auto t     = verify ? drafts + 1 : 1;
            inputs.batch(r.main_batch(t, b, bounds_.back()), bounds_.back());
            write(r.tensor(io.batch.valid_columns).slice(0, 0, b),
                  std::min(t, p.shape.max_context));
            if (verify) {
                write(r.tensor(io.mtp->extents).slice(0, 0, b),
                      std::min(drafts, p.shape.max_context - 1));
                write(r.tensor(io.anchors).slice(0, 0, b),
                      p.parameters.profile->config.text.ple.eos);
            }
            const auto decision = [&] {
                if (verify)
                    p.accept_forward(b);
                else
                    p.sample_forward(b, false);
                p.copy_sample_result(b, verify);
            };
            inputs.stage();
            body(batch, nullptr, b, verify);
            decision();
            p.device.synchronize();
            auto& round = batch.rounds[verify];
            round.installed_draft.resize(depth);
            round.graph.prepare(
                p.cuda, [&] { body(batch, &round, b, verify); },
                p.parallel ? &*p.parallel : nullptr);
            batch.decisions[verify].prepare(p.cuda, decision, p.parallel ? &*p.parallel : nullptr);
            replay(b, std::vector<std::uint32_t>(depth, bounds_.back()), bounds_.back(), true,
                   false);
            decide(b, verify);
            p.device.synchronize();
        }
    }
    workspace_peak_ = r.workspace.peak_used();
}

std::size_t DeviceGraphs::bound(std::uint32_t visible) const {
    const auto found = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (found == bounds_.end()) throw std::logic_error("decode graph context exceeds its capacity");
    return found - bounds_.begin();
}

void DeviceGraphs::replay(std::uint32_t batch, std::span<const std::uint32_t> draft_visible,
                          std::uint32_t visible, bool head, bool read_drafts) {
    auto& profiles  = batches_.at(batch - 1);
    auto& round     = profiles.rounds[!draft_visible.empty()];
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
    round.head.set_enabled(round.graph.executable, head);
    round.readback.set_enabled(round.graph.executable, read_drafts);
    round.graph.launch(owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
}

void DeviceGraphs::decide(std::uint32_t batch, bool verify) {
    batches_.at(batch - 1).decisions[verify].launch(owner_.cuda,
                                                    owner_.parallel ? &*owner_.parallel : nullptr);
}

} // namespace ninfer::models::qwen4_exp::detail
