#include "models/qwen4_exp/program/ple_pipeline.h"
#include "core/nvtx.h"
#include "core/transfer.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen4_exp {

PlePipeline::PlePipeline(Resources& r, const DeviceContext& device)
    : resources_(r), cuda_(device.execution_view()), completed_(device) {
    appends_.reserve(2 * r.plan.shape.draft_tokens);
    for (std::uint32_t i = 0; i < 2 * r.plan.shape.draft_tokens; ++i) appends_.emplace_back(device);
}

PlePipeline::~PlePipeline() {
    try {
        drain();
    } catch (...) { std::terminate(); }
}

bool PlePipeline::matches(const Window& w, std::uint32_t tokens,
                          std::span<const ple::InputRow> rows) const {
    if (w.cancelled || w.consumed || w.tokens != tokens || w.rows.size() != rows.size())
        return false;
    std::size_t offset = 0;
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto& a = w.rows[b];
        const auto& c = rows[b].identity;
        if (a.sequence != c.sequence || a.epoch != c.epoch || a.frontier != c.frontier ||
            a.raw_tail != c.raw_tail || rows[b].ids.size() != tokens ||
            !std::equal(rows[b].ids.begin(), rows[b].ids.end(), w.ids.begin() + offset) ||
            w.known[b] != tokens)
            return false;
        offset += tokens;
    }
    return true;
}

std::optional<ple::Ticket>
PlePipeline::prepare(std::uint32_t tokens, std::span<const ple::InputRow> rows, bool required) {
    nvtx::ScopedRange prepare(nvtx::Name::PlePrepare, nvtx::Category::Runtime,
                              tokens * rows.size());
    progress();
    for (auto& w : windows_) {
        if (w && matches(*w, tokens, rows)) {
            if (required) resources_.ple_prepare->prioritize(w->ticket);
            return w->ticket;
        }
    }
    auto ticket = resources_.ple_prepare->prepare(tokens, rows, required);
    if (!ticket) return {};
    auto& w = windows_.at(ticket->slot);
    if (w) throw std::logic_error("PLE producer reused a live window");
    w.emplace();
    w->ticket = *ticket;
    w->tokens = tokens;
    w->ids.resize(tokens * rows.size());
    for (std::size_t b = 0; b < rows.size(); ++b) {
        w->rows.push_back(rows[b].identity);
        std::copy(rows[b].ids.begin(), rows[b].ids.end(), w->ids.begin() + b * tokens);
        w->known[b] = rows[b].ids.size();
    }
    return ticket;
}

ple::Ticket PlePipeline::begin(std::uint32_t tokens, std::span<const ple::InputRow> rows) {
    for (;;) {
        if (auto ticket = prepare(tokens, rows, true)) return *ticket;
        for (auto& w : windows_) {
            if (w && !w->consumed && !w->cancelled) {
                resources_.ple_prepare->cancel(w->ticket);
                w->cancelled = true;
            }
        }
        std::this_thread::yield();
    }
}

ple::Ticket PlePipeline::require(std::uint32_t tokens, std::span<const ple::InputRow> rows) {
    const auto ticket = begin(tokens, rows);
    if (!window(ticket).seal_requested) {
        std::array<std::uint32_t, kMaximumConcurrency> valid{};
        for (std::size_t b = 0; b < rows.size(); ++b) valid[b] = rows[b].ids.size();
        seal(ticket, {valid.data(), rows.size()});
    }
    return ticket;
}

void PlePipeline::append_device(ple::Ticket ticket, const Tensor& ids,
                                std::span<const std::uint32_t> extents, std::uint32_t step) {
    auto& w      = window(ticket);
    const auto k = resources_.plan.shape.draft_tokens;
    if (step >= k || extents.size() != w.rows.size() || ids.numel() != w.rows.size() ||
        ids.dtype != DType::I32 || !ids.is_contiguous() || w.seal_requested)
        throw std::logic_error("invalid PLE draft handoff");
    auto& transfer = appends_.at(ticket.slot * k + step);
    if (transfer.pending) throw std::logic_error("PLE draft handoff is still live");
    transfer.produced.record(cuda_.stream());
    const auto upload = resources_.ple_transfer();
    transfer.produced.wait(upload.stream());
    copy_bytes_async(append_destination(ticket.slot, step, extents.size()),
                     {ids.data, ids.bytes(), cuda_.device()}, upload);
    transfer.copied.record(upload.stream());
    append_submitted(ticket, extents, step);
}

MemorySpan PlePipeline::append_destination(std::uint32_t slot, std::uint32_t step,
                                           std::uint32_t batch) {
    const auto index  = slot * resources_.plan.shape.draft_tokens + step;
    const auto offset = resources_.plan.persistent.buffers.mtp->ingress.bytes +
                        index * resources_.plan.shape.max_concurrency * sizeof(std::int32_t);
    return resources_.mtp_ingress().subspan(offset, batch * sizeof(std::int32_t));
}

void PlePipeline::capture_append(std::uint32_t slot, const Tensor& ids, std::uint32_t step,
                                 CudaExecutionView cuda) {
    auto& transfer = appends_.at(slot * resources_.plan.shape.draft_tokens + step);
    copy_bytes_async(append_destination(slot, step, ids.numel()),
                     {ids.data, ids.bytes(), cuda.device()}, cuda);
    transfer.copied.record_external(cuda.stream());
}

void PlePipeline::append_submitted(ple::Ticket ticket, std::span<const std::uint32_t> extents,
                                   std::uint32_t step) {
    auto& w      = window(ticket);
    const auto k = resources_.plan.shape.draft_tokens;
    if (step >= k || extents.size() != w.rows.size() || w.seal_requested)
        throw std::logic_error("invalid PLE draft handoff");
    auto& transfer = appends_.at(ticket.slot * k + step);
    if (transfer.pending) throw std::logic_error("PLE draft handoff is still live");
    transfer.ticket = ticket;
    transfer.batch  = extents.size();
    for (std::size_t b = 0; b < extents.size(); ++b) transfer.active[b] = step < extents[b];
    transfer.pending = true;
    ++w.append_count;
}

void PlePipeline::seal(ple::Ticket ticket, std::span<const std::uint32_t> valid) {
    auto& w = window(ticket);
    if (w.seal_requested || valid.size() != w.rows.size())
        throw std::logic_error("invalid PLE seal");
    std::copy(valid.begin(), valid.end(), w.valid.begin());
    w.seal_requested = true;
    progress();
}

std::span<const std::int32_t> PlePipeline::input(ple::Ticket ticket, std::uint32_t row) {
    const auto& w = window(ticket);
    if (!w.sealed || row >= w.rows.size()) throw std::logic_error("PLE IDs are not complete");
    return std::span<const std::int32_t>(w.ids).subspan(row * w.tokens, w.valid[row]);
}

void PlePipeline::prefetch(std::uint32_t tokens, std::span<const ple::InputRow> rows) {
    if (auto ticket = prepare(tokens, rows, false); ticket && !window(*ticket).seal_requested) {
        std::array<std::uint32_t, kMaximumConcurrency> valid{};
        for (std::size_t b = 0; b < rows.size(); ++b) valid[b] = rows[b].ids.size();
        seal(*ticket, {valid.data(), rows.size()});
    }
}

PlePipeline::Window& PlePipeline::window(ple::Ticket ticket) {
    auto& w = windows_.at(ticket.slot);
    if (!w || w->ticket != ticket) throw std::logic_error("stale Program PLE ticket");
    return *w;
}

void PlePipeline::progress() {
    for (std::size_t i = 0; i < appends_.size(); ++i) {
        auto& transfer = appends_[i];
        if (!transfer.pending || !transfer.copied.ready()) continue;
        auto& w = window(transfer.ticket);
        if (i % resources_.plan.shape.draft_tokens != w.next_append) continue;
        if (!w.cancelled && !w.error) {
            const auto offset = resources_.plan.persistent.buffers.mtp->ingress.bytes +
                                i * resources_.plan.shape.max_concurrency * sizeof(std::int32_t);
            const auto* ids = reinterpret_cast<const std::int32_t*>(
                resources_.mtp_ingress()
                    .subspan(offset, transfer.batch * sizeof(std::int32_t))
                    .data);
            std::array<ple::Append, kMaximumConcurrency> additions;
            std::size_t count = 0;
            for (std::uint32_t b = 0; b < transfer.batch; ++b)
                if (transfer.active[b]) {
                    if (w.known[b] >= w.tokens)
                        throw std::logic_error("PLE draft exceeds its window");
                    w.ids[b * w.tokens + w.known[b]++] = ids[b];
                    additions[count++]                 = {b, {ids + b, 1}};
                }
            if (count) resources_.ple_prepare->append(w.ticket, {additions.data(), count});
        }
        --w.append_count;
        ++w.next_append;
        transfer.pending = false;
    }
    for (auto& entry : windows_) {
        if (!entry) continue;
        auto& w = *entry;
        if (w.seal_requested && !w.sealed && !w.append_count && !w.cancelled && !w.error) {
            resources_.ple_prepare->seal(w.ticket, {w.valid.data(), w.rows.size()});
            w.sealed = true;
        }
        const auto status = resources_.ple_prepare->poll(w.ticket);
        if (status.error) w.error = status.error;
        if (!w.cancelled && !w.error && !w.uploaded &&
            status.status == ple::disk::PrepareStatus::HostReady) {
            resources_.ple_upload->upload(resources_.ple_prepare->embedding(w.ticket),
                                          resources_.ple_transfer());
            w.uploaded = true;
        }
        if ((w.consumed || w.cancelled) && !w.append_count && status.drained &&
            resources_.ple_upload->try_release(w.ticket)) {
            resources_.ple_prepare->release(w.ticket);
            entry.reset();
        }
    }
}

ple::DeviceEmbedding PlePipeline::consume(ple::Ticket ticket) {
    nvtx::ScopedRange wait(nvtx::Name::PleWait, nvtx::Category::Runtime);
    for (;;) {
        progress();
        auto& w = window(ticket);
        if (w.error) std::rethrow_exception(w.error);
        if (w.cancelled) throw std::logic_error("consuming a cancelled PLE window");
        if (w.uploaded) {
            auto value = resources_.ple_upload->acquire(ticket, cuda_);
            w.acquired = true;
            return value;
        }
        std::this_thread::yield();
    }
}

void PlePipeline::consumed(ple::Ticket ticket) {
    auto& w = window(ticket);
    resources_.ple_upload->consumer_submitted(ticket, cuda_);
    w.consumed = true;
}

void PlePipeline::abandon_consumer(Window& w) {
    if (!w.acquired || w.consumed) return;
    auto input = resources_.ple_upload->binding(w.ticket.slot, w.tokens, w.rows.size());
    input.consumed.record(cuda_.stream());
    consumed(w.ticket);
}

void PlePipeline::wait_compute() {
    completed_.record(cuda_.stream());
    while (!completed_.ready()) {
        progress();
        std::this_thread::yield();
    }
    progress();
}

void PlePipeline::cancel(std::uint64_t sequence) {
    for (auto& w : windows_) {
        if (w && !w->cancelled && std::any_of(w->rows.begin(), w->rows.end(), [&](const auto& row) {
                return row.sequence == sequence;
            })) {
            abandon_consumer(*w);
            resources_.ple_prepare->cancel(w->ticket);
            w->cancelled = true;
        }
    }
    progress();
}

void PlePipeline::drain() {
    for (auto& w : windows_) {
        if (w && !w->cancelled) {
            abandon_consumer(*w);
            resources_.ple_prepare->cancel(w->ticket);
            w->cancelled = true;
        }
    }
    while (std::any_of(windows_.begin(), windows_.end(), [](const auto& w) { return bool(w); })) {
        progress();
        std::this_thread::yield();
    }
}

} // namespace ninfer::models::qwen4_exp
