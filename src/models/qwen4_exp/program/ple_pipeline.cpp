#include "models/qwen4_exp/program/ple_pipeline.h"
#include "models/qwen4_exp/program/resources.h"
#include "core/nvtx.h"
#include "core/transfer.h"

#include <algorithm>
#include <string>
#include <thread>

namespace ninfer::models::qwen4_exp {
namespace {
void synchronize(cudaStream_t stream) {
    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("PLE stream synchronization failed: ") +
                                 cudaGetErrorString(status));
}
} // namespace

PlePipeline::PlePipeline(Resources& r, const DeviceContext& device)
    : resources_(r), cuda_(device.execution_view()) {
    appends_.reserve(2 * r.plan.shape.draft_tokens);
    for (std::uint32_t i = 0; i < 2 * r.plan.shape.draft_tokens; ++i) appends_.emplace_back(device);
}

PlePipeline::~PlePipeline() {
    try {
        drain();
    } catch (const std::logic_error&) { std::terminate(); } catch (...) {
        // drain has already stopped every Host producer and attempted all CUDA consumers.
    }
}

bool PlePipeline::matches(const Window& w, std::uint32_t tokens,
                          std::span<const ple::InputRow> rows) const {
    if (w.cancelled || w.acquired || w.tokens != tokens || w.rows.size() != rows.size())
        return false;
    std::size_t offset = 0;
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto& a = w.rows[b];
        const auto& c = rows[b].identity;
        if (a.sequence != c.sequence || a.epoch != c.epoch || a.frontier != c.frontier ||
            a.raw_tail != c.raw_tail || w.known[b] != rows[b].ids.size() ||
            !std::equal(rows[b].ids.begin(), rows[b].ids.end(), w.ids.begin() + offset) ||
            (w.seal_requested && w.valid[b] != rows[b].ids.size()))
            return false;
        offset += tokens;
    }
    return true;
}

std::optional<ple::Ticket> PlePipeline::prepare(std::uint32_t tokens,
                                                std::span<const ple::InputRow> rows,
                                                std::optional<runtime::ExecutionTicket> owner) {
    if (drained_) throw std::logic_error("PLE pipeline is shut down");
    if (!tokens || rows.empty() || rows.size() > kMaximumConcurrency ||
        std::uint64_t(tokens) * rows.size() > resources_.plan.ple.max_positions)
        throw std::invalid_argument("invalid PLE input shape");
    for (const auto& row : rows)
        if (row.ids.size() > tokens) throw std::invalid_argument("invalid PLE input prefix");
    nvtx::ScopedRange prepare(nvtx::Name::PlePrepare, nvtx::Category::Runtime,
                              tokens * rows.size());
    progress();
    for (auto& w : windows_) {
        if (!w || !matches(*w, tokens, rows) || (owner && w->owner && w->owner != owner)) continue;
        if (owner) {
            if (w->error) std::rethrow_exception(w->error);
            try {
                resources_.ple_prepare->prioritize(w->ticket);
            } catch (...) {
                const auto status = resources_.ple_prepare->poll(w->ticket);
                if (status.error) std::rethrow_exception(status.error);
                throw;
            }
            w->owner = owner;
        }
        return w->ticket;
    }
    Window prepared;
    prepared.owner  = owner;
    prepared.tokens = tokens;
    prepared.ids.resize(std::size_t(tokens) * rows.size());
    prepared.rows.reserve(rows.size());
    for (std::size_t b = 0; b < rows.size(); ++b) {
        prepared.rows.push_back(rows[b].identity);
        std::copy(rows[b].ids.begin(), rows[b].ids.end(), prepared.ids.begin() + b * tokens);
        prepared.known[b] = rows[b].ids.size();
    }
    auto ticket = resources_.ple_prepare->prepare(tokens, rows, owner.has_value());
    if (!ticket) {
        if (owner)
            for (auto& w : windows_)
                if (w && !w->owner) cancel(*w);
        return {};
    }
    auto& w = windows_.at(ticket->slot);
    if (w) throw std::logic_error("PLE producer reused a live window");
    prepared.ticket = *ticket;
    w.emplace(std::move(prepared));
    return ticket;
}

std::optional<ple::Ticket> PlePipeline::try_begin(runtime::ExecutionTicket owner,
                                                  std::uint32_t tokens,
                                                  std::span<const ple::InputRow> rows) {
    return prepare(tokens, rows, owner);
}

std::optional<ple::Ticket> PlePipeline::try_require(runtime::ExecutionTicket owner,
                                                    std::uint32_t tokens,
                                                    std::span<const ple::InputRow> rows) {
    const auto ticket = try_begin(owner, tokens, rows);
    if (ticket && !window(*ticket).seal_requested) {
        std::array<std::uint32_t, kMaximumConcurrency> valid{};
        for (std::size_t b = 0; b < rows.size(); ++b) valid[b] = rows[b].ids.size();
        seal(*ticket, {valid.data(), rows.size()});
    }
    return ticket;
}

void PlePipeline::append_device(ple::Ticket ticket, const Tensor& ids,
                                std::span<const std::uint32_t> extents, std::uint32_t step) {
    auto& w      = window(ticket);
    const auto k = resources_.plan.shape.draft_tokens;
    if (step >= k || extents.size() != w.rows.size() ||
        ids.numel() != static_cast<std::int64_t>(w.rows.size()) || ids.dtype != DType::I32 ||
        !ids.is_contiguous())
        throw std::logic_error("invalid PLE draft handoff");
    auto& transfer = appends_.at(ticket.slot * k + step);
    append_submitted(ticket, extents, step);
    const auto upload = resources_.ple_transfer();
    try {
        transfer.produced.record(cuda_.stream());
        transfer.produced.wait(upload.stream());
        copy_bytes_async(append_destination(ticket.slot, step, extents.size()),
                         {ids.data, ids.bytes(), cuda_.device()}, upload);
        transfer.copied.record(upload.stream());
    } catch (...) {
        // A failed event record can follow successful DMA submission. Drain that transfer
        // before releasing its pinned destination; ordinary execution never waits here.
        w.error = std::current_exception();
        try {
            synchronize(upload.stream());
        } catch (...) { std::rethrow_exception(w.error); }
        transfer.pending = false;
        --w.append_count;
        std::rethrow_exception(w.error);
    }
}

MemorySpan PlePipeline::append_destination(std::uint32_t slot, std::uint32_t step,
                                           std::uint32_t batch) {
    const auto index  = slot * resources_.plan.shape.draft_tokens + step;
    const auto offset = index * resources_.plan.shape.max_concurrency * sizeof(std::int32_t);
    return resources_.ple_handoff().subspan(offset, batch * sizeof(std::int32_t));
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
    if (step >= k || extents.size() != w.rows.size() || w.seal_requested || w.cancelled ||
        !w.owner || step != w.next_append + w.append_count)
        throw std::logic_error("invalid PLE draft handoff");
    if (w.error) std::rethrow_exception(w.error);
    for (const auto extent : extents)
        if (extent > k) throw std::logic_error("invalid PLE draft extent");
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
    if (w.seal_requested || w.cancelled || valid.size() != w.rows.size())
        throw std::logic_error("invalid PLE seal");
    if (w.error) std::rethrow_exception(w.error);
    for (const auto length : valid)
        if (length > w.tokens) throw std::logic_error("PLE seal exceeds its window");
    std::copy(valid.begin(), valid.end(), w.valid.begin());
    w.seal_requested = true;
    progress();
}

bool PlePipeline::input_ready(ple::Ticket ticket) {
    const auto& w = window(ticket);
    if (w.error) std::rethrow_exception(w.error);
    return w.sealed && !w.cancelled;
}

std::span<const std::int32_t> PlePipeline::input(ple::Ticket ticket, std::uint32_t row) {
    const auto& w = window(ticket);
    if (w.error) std::rethrow_exception(w.error);
    if (!w.sealed || w.cancelled || row >= w.rows.size())
        throw std::logic_error("PLE IDs are not complete");
    return std::span<const std::int32_t>(w.ids).subspan(row * w.tokens, w.valid[row]);
}

void PlePipeline::prefetch(std::uint32_t tokens, std::span<const ple::InputRow> rows) {
    if (auto ticket = prepare(tokens, rows, {});
        ticket && !window(*ticket).owner && !window(*ticket).seal_requested) {
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
    if (drained_) return;
    for (auto& w : windows_) {
        if (!w) continue;
        const auto status = resources_.ple_prepare->poll(w->ticket);
        if (status.error) w->error = status.error;
    }
    for (std::size_t i = 0; i < appends_.size(); ++i) {
        auto& transfer = appends_[i];
        if (!transfer.pending || !transfer.copied.ready()) continue;
        auto& w = window(transfer.ticket);
        if (i % resources_.plan.shape.draft_tokens != w.next_append) continue;
        if (!w.cancelled && !w.error) {
            try {
                const auto offset =
                    i * resources_.plan.shape.max_concurrency * sizeof(std::int32_t);
                const auto* ids = reinterpret_cast<const std::int32_t*>(
                    resources_.ple_handoff()
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
            } catch (...) {
                const auto status = resources_.ple_prepare->poll(w.ticket);
                w.error           = status.error ? status.error : std::current_exception();
            }
        }
        --w.append_count;
        ++w.next_append;
        transfer.pending = false;
    }
    for (auto& entry : windows_) {
        if (!entry) continue;
        auto& w = *entry;
        if (w.seal_requested && !w.sealed && !w.append_count && !w.cancelled && !w.error) {
            try {
                resources_.ple_prepare->seal(w.ticket, {w.valid.data(), w.rows.size()});
                w.sealed = true;
            } catch (...) {
                const auto status = resources_.ple_prepare->poll(w.ticket);
                w.error           = status.error ? status.error : std::current_exception();
            }
        }
        const auto status = resources_.ple_prepare->poll(w.ticket);
        if (status.error) w.error = status.error;
        if (!w.cancelled && !w.error && !w.uploaded &&
            status.status == ple::disk::PrepareStatus::HostReady) {
            try {
                resources_.ple_upload->upload(resources_.ple_prepare->embedding(w.ticket),
                                              resources_.ple_transfer());
                w.uploaded = true;
            } catch (...) {
                const auto failed = resources_.ple_prepare->poll(w.ticket);
                w.error           = failed.error ? failed.error : std::current_exception();
            }
        }
        if ((w.consumed || w.cancelled) && !w.append_count && status.drained &&
            resources_.ple_upload->try_release(w.ticket)) {
            resources_.ple_prepare->release(w.ticket);
            entry.reset();
        }
    }
}

std::optional<ple::DeviceEmbedding> PlePipeline::try_consume(ple::Ticket ticket) {
    progress();
    auto& w = window(ticket);
    if (w.error) std::rethrow_exception(w.error);
    if (w.cancelled || w.acquired || !w.owner)
        throw std::logic_error("invalid PLE window consumption");
    if (!w.uploaded) return {};
    auto value = resources_.ple_upload->acquire(ticket, cuda_);
    w.acquired = true;
    return value;
}

void PlePipeline::consumer_submitted(ple::Ticket ticket) {
    auto& w = window(ticket);
    resources_.ple_upload->consumer_submitted(ticket, cuda_);
    w.consumed = true;
}

void PlePipeline::abandon_consumer(Window& w) {
    if (!w.acquired || w.consumed) return;
    auto input = resources_.ple_upload->binding(w.ticket.slot, w.tokens, w.rows.size());
    input.consumed.record(cuda_.stream());
    consumer_submitted(w.ticket);
}

void PlePipeline::cancel(Window& w) {
    if (w.cancelled) return;
    abandon_consumer(w);
    resources_.ple_prepare->cancel(w.ticket);
    w.cancelled = true;
}

void PlePipeline::cancel(runtime::ExecutionTicket owner) {
    for (auto& w : windows_)
        if (w && w->owner == owner) cancel(*w);
    progress();
}

void PlePipeline::cancel_sequence(std::uint64_t sequence) {
    for (auto& w : windows_) {
        if (w && !w->owner && std::any_of(w->rows.begin(), w->rows.end(), [&](const auto& row) {
                return row.sequence == sequence;
            }))
            cancel(*w);
    }
    progress();
}

bool PlePipeline::retired(runtime::ExecutionTicket owner) const {
    return std::none_of(windows_.begin(), windows_.end(),
                        [&](const auto& w) { return w && w->owner == owner; });
}

bool PlePipeline::idle() const {
    return std::none_of(windows_.begin(), windows_.end(), [](const auto& w) { return bool(w); });
}

void PlePipeline::drain() {
    if (drained_) return;
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    for (auto& w : windows_) {
        if (!w) continue;
        attempt([&] { abandon_consumer(*w); });
        // A failed CUDA completion record must not prevent cancellation of the CPU writer.
        if (resources_.ple_prepare) attempt([&] { resources_.ple_prepare->cancel(w->ticket); });
        w->cancelled = true;
    }
    if (resources_.ple_prepare) {
        while (!failure && !idle()) {
            attempt([&] { progress(); });
            if (!failure && !idle()) std::this_thread::yield();
        }
    }
    // Joins the producer even if CUDA events can no longer be queried. UploadSlots retains
    // every borrowed Host output until its own consumers have been drained below.
    resources_.ple_prepare.reset();
    for (auto& append : appends_)
        if (append.pending) attempt([&] { append.copied.synchronize(); });
    attempt([&] { synchronize(resources_.ple_transfer().stream()); });
    attempt([&] { synchronize(cuda_.stream()); });
    for (auto& w : windows_) {
        if (!w) continue;
        attempt([&] { (void)resources_.ple_upload->try_release(w->ticket); });
        w.reset();
    }
    for (auto& append : appends_) append.pending = false;
    drained_ = true;
    if (failure) std::rethrow_exception(failure);
}

} // namespace ninfer::models::qwen4_exp
