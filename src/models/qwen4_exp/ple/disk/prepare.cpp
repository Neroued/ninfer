#include "models/qwen4_exp/ple/disk/prepare.h"

#include "models/qwen4_exp/ple/hash.h"

#include "ninfer/types.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::ple::disk {
namespace {

PrepareOptions checked_options(PrepareOptions options, std::span<const MemorySpan> outputs,
                               std::int32_t eos) {
    if (!options.max_positions || options.max_positions > UINT32_MAX / kHeads || outputs.empty() ||
        outputs.size() > UINT32_MAX || eos < 0)
        throw std::invalid_argument("invalid PLE preparation capacity");
    const auto bytes = std::size_t(options.max_positions) * kEmbeddingBytes;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto output = outputs[i];
        validate_memory(output);
        if (output.device != Device::cpu() || output.bytes < bytes ||
            reinterpret_cast<std::uintptr_t>(output.data) % alignof(std::uint16_t))
            throw std::invalid_argument("invalid PLE Host output buffer");
        const auto begin = reinterpret_cast<std::uintptr_t>(output.data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(outputs[j].data);
            if (begin < other + bytes && other < begin + bytes)
                throw std::invalid_argument("PLE Host outputs overlap");
        }
    }
    return options;
}

void check_ids(std::span<const std::int32_t> ids) {
    for (auto id : ids)
        if (id < 0) throw std::invalid_argument("negative PLE token ID");
}

} // namespace

struct PrepareService::Impl {
    struct Window {
        MemorySpan output;
        std::vector<std::int32_t> ids;
        std::array<SequenceIdentity, 8> identities{};
        std::array<std::uint32_t, 8> known{}, planned{}, valid{};
        std::uint64_t generation = 0;
        std::uint32_t tokens = 0, batch = 0;
        PrepareStatus status = PrepareStatus::Preparing;
        std::exception_ptr error;
        bool allocated = false, released = false, bound = false, required = true, promoted = false;
        bool sealed = false, seal_applied = false, cancelled = false, cancel_applied = false;
        bool drained = true;
    };

    const Table& table;
    const std::int32_t eos;
    const PrepareOptions options;
    mutable std::mutex mutex;
    std::vector<Window> windows;
    RowReader reader;
    std::vector<std::thread> workers;
    std::exception_ptr fatal;
    bool stopping = false;

    Impl(const Table& table, std::int32_t eos, PrepareOptions options,
         std::span<const MemorySpan> outputs)
        : table(table), eos(eos), options(checked_options(std::move(options), outputs, eos)),
          windows(outputs.size()),
          reader(table, outputs.size(), this->options.max_positions * kHeads,
                 this->options.io_depth, this->options.cache_bytes) {
        for (std::size_t i = 0; i < windows.size(); ++i) {
            windows[i].output = outputs[i];
            windows[i].ids.resize(this->options.max_positions);
        }
        std::optional<CpuTopology> topology;
        if (!this->options.affinity.empty()) {
            topology.emplace();
            validate_cpu_affinity(topology->info(), this->options.affinity);
        }
        workers = cpu_detail::start_workers(
            1,
            [&](std::uint32_t) {
                if (topology) bind_current_thread(*topology, this->options.affinity);
            },
            [](void* self, std::uint32_t) noexcept { static_cast<Impl*>(self)->run(); }, this);
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        reader.wake();
        cpu_detail::join_workers(workers);
    }

    Window& get(Ticket ticket) {
        if (ticket.slot >= windows.size()) throw std::invalid_argument("invalid PLE ticket slot");
        auto& w = windows[ticket.slot];
        if (!w.allocated || w.released || w.generation != ticket.generation)
            throw std::invalid_argument("stale PLE ticket");
        return w;
    }

    static WindowSnapshot snapshot(const Window& w) {
        return {{w.tokens, w.batch, w.valid, w.identities}, w.status, w.drained, w.error};
    }

    bool tick() {
        // This bounded critical section serializes control changes with all E writers.
        // No mutex is held while waiting for disk completion or caller input.
        std::lock_guard lock(mutex);
        for (std::uint32_t i = 0; i < windows.size(); ++i) {
            auto& w = windows[i];
            if (!w.allocated) continue;
            if (w.released) {
                if (w.bound) reader.release(i);
                w.bound = w.allocated = false;
                continue;
            }
            if (stopping) w.cancelled = true;
            if (!w.bound && !w.cancelled) {
                reader.begin(i,
                             {static_cast<std::uint16_t*>(w.output.data),
                              std::size_t(w.tokens) * w.batch * kEmbeddingElements},
                             w.required);
                w.bound    = true;
                w.promoted = w.required;
            }
            if (w.cancelled && !w.cancel_applied) {
                if (w.bound) reader.cancel(i, 0, w.tokens * w.batch * kHeads);
                w.cancel_applied = true;
                w.status         = w.error ? PrepareStatus::Failed : PrepareStatus::Cancelled;
            }
            if (w.cancelled) continue;
            if (w.required && !w.promoted) {
                reader.prioritize(i);
                w.promoted = true;
            }
            if (w.sealed && !w.seal_applied) {
                for (std::uint32_t b = 0; b < w.batch; ++b) {
                    const auto first = b * w.tokens + w.valid[b];
                    const auto end   = (b + 1) * w.tokens;
                    reader.cancel(i, first * kHeads, end * kHeads);
                    auto* output = static_cast<std::uint16_t*>(w.output.data);
                    std::fill(output + std::size_t(first) * kEmbeddingElements,
                              output + std::size_t(end) * kEmbeddingElements, 0);
                }
                w.seal_applied = true;
            }
        }
        reader.pump();
        std::size_t budget = 32;
        for (bool required : {true, false}) {
            for (std::uint32_t i = 0; i < windows.size() && budget; ++i) {
                auto& w = windows[i];
                if (!w.allocated || w.released || w.cancelled || w.required != required) continue;
                for (std::uint32_t b = 0; b < w.batch && budget; ++b) {
                    const auto end   = w.sealed ? w.valid[b] : w.known[b];
                    const auto start = w.planned[b];
                    if (start >= end) continue;
                    const auto count = std::min<std::size_t>(budget, end - start);
                    const auto* ids  = w.ids.data() + b * w.tokens;
                    const std::array tail{start >= 2 ? ids[start - 2]
                                                     : (start == 1 ? w.identities[b].raw_tail[1]
                                                                   : w.identities[b].raw_tail[0]),
                                          start ? ids[start - 1] : w.identities[b].raw_tail[1]};
                    std::array<std::uint32_t, 32 * kHeads> rows;
                    hash_tokens({ids + start, count}, tail, eos, table.parameters().hash,
                                std::span(rows).first(count * kHeads));
                    for (std::size_t r = 0; r < count * kHeads; ++r)
                        reader.request(i, (b * w.tokens + start) * kHeads + r, rows[r]);
                    w.planned[b] += count;
                    budget -= count;
                }
            }
        }
        reader.pump();
        bool more = false;
        for (std::uint32_t i = 0; i < windows.size(); ++i) {
            auto& w = windows[i];
            if (!w.allocated) continue;
            if (w.bound && !w.error && reader.error(i)) {
                w.error     = reader.error(i);
                w.status    = PrepareStatus::Failed;
                w.cancelled = true;
                more        = true;
            }
            w.drained    = (!w.cancelled || w.cancel_applied) && (!w.bound || reader.drained(i));
            bool planned = true;
            for (std::uint32_t b = 0; b < w.batch; ++b)
                planned &= w.planned[b] >= (w.sealed ? w.valid[b] : w.known[b]);
            if (!w.cancelled && w.sealed && planned && !reader.pending(i))
                w.status = PrepareStatus::HostReady;
            if (!w.cancelled && !planned) more = true;
            if (stopping && w.drained) {
                if (w.bound) reader.release(i);
                w.bound = w.allocated = false;
            }
        }
        return more || reader.runnable();
    }

    void run() noexcept {
        try {
            for (;;) {
                const bool more = tick();
                {
                    std::lock_guard lock(mutex);
                    if (stopping && std::none_of(windows.begin(), windows.end(),
                                                 [](const auto& w) { return w.allocated; }))
                        return;
                }
                if (!more) reader.wait();
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            fatal = std::current_exception();
            for (auto& w : windows) {
                if (!w.allocated) continue;
                w.error  = fatal;
                w.status = PrepareStatus::Failed;
                // The worker exits here. Outstanding kernel reads only target reader-owned slabs.
                w.drained = true;
            }
        }
    }
};

PrepareService::PrepareService(const Table& table, std::int32_t eos, PrepareOptions options,
                               std::span<const MemorySpan> outputs)
    : impl_(std::make_unique<Impl>(table, eos, std::move(options), outputs)) {}

PrepareService::~PrepareService() = default;

std::optional<Ticket> PrepareService::prepare(std::uint32_t tokens, std::span<const InputRow> rows,
                                              bool required) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    if (p.fatal) std::rethrow_exception(p.fatal);
    if (!tokens || rows.empty() || rows.size() > kMaximumConcurrency ||
        std::uint64_t(tokens) * rows.size() > p.options.max_positions)
        throw std::invalid_argument("invalid PLE window shape");
    for (const auto& row : rows) {
        if (row.ids.size() > tokens || row.identity.raw_tail[0] < 0 || row.identity.raw_tail[1] < 0)
            throw std::invalid_argument("invalid PLE input prefix");
        check_ids(row.ids);
    }
    std::size_t free  = 0;
    bool has_required = false;
    for (const auto& w : p.windows) {
        free += !w.allocated;
        has_required |= w.allocated && !w.released && !w.cancelled && w.required;
    }
    if (!free || (!required && !has_required && free < 2)) return {};
    for (std::uint32_t slot = 0; slot < p.windows.size(); ++slot) {
        auto& w = p.windows[slot];
        if (w.allocated) continue;
        ++w.generation;
        w.tokens    = tokens;
        w.batch     = rows.size();
        w.allocated = true;
        w.released = w.bound = w.promoted = w.sealed = w.seal_applied = false;
        w.cancelled = w.cancel_applied = false;
        w.required                     = required;
        w.drained                      = false;
        w.status                       = PrepareStatus::Preparing;
        w.error                        = {};
        w.known = w.planned = w.valid = {};
        w.identities                  = {};
        for (std::size_t b = 0; b < rows.size(); ++b) {
            w.identities[b] = rows[b].identity;
            w.known[b]      = rows[b].ids.size();
            std::copy(rows[b].ids.begin(), rows[b].ids.end(), w.ids.begin() + b * tokens);
        }
        p.reader.wake();
        return Ticket{slot, w.generation};
    }
    throw std::logic_error("PLE free slot disappeared");
}

void PrepareService::prioritize(Ticket ticket) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    auto& w = p.get(ticket);
    if (w.cancelled || w.status == PrepareStatus::Failed)
        throw std::invalid_argument("terminal PLE window cannot be prioritized");
    w.required = true;
    p.reader.wake();
}

void PrepareService::append(Ticket ticket, std::span<const Append> rows) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    auto& w = p.get(ticket);
    if (w.sealed || w.cancelled || w.status != PrepareStatus::Preparing)
        throw std::invalid_argument("PLE window is closed to append");
    std::array<bool, 8> seen{};
    for (const auto& row : rows) {
        if (row.batch_row >= w.batch || seen[row.batch_row] ||
            row.ids.size() > w.tokens - w.known[row.batch_row])
            throw std::invalid_argument("invalid PLE appended prefix");
        seen[row.batch_row] = true;
        check_ids(row.ids);
    }
    for (const auto& row : rows) {
        const auto b = row.batch_row;
        std::copy(row.ids.begin(), row.ids.end(), w.ids.begin() + b * w.tokens + w.known[b]);
        w.known[b] += row.ids.size();
    }
    p.reader.wake();
}

void PrepareService::seal(Ticket ticket, std::span<const std::uint32_t> valid) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    auto& w = p.get(ticket);
    if (w.sealed || w.cancelled || w.status != PrepareStatus::Preparing || valid.size() != w.batch)
        throw std::invalid_argument("invalid PLE seal");
    for (std::size_t b = 0; b < valid.size(); ++b)
        if (valid[b] > w.known[b]) throw std::invalid_argument("PLE seal exceeds known IDs");
    std::copy(valid.begin(), valid.end(), w.valid.begin());
    w.sealed = true;
    p.reader.wake();
}

void PrepareService::cancel(Ticket ticket) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    auto& w = p.get(ticket);
    if (w.cancelled) return;
    w.cancelled      = true;
    w.status         = w.error ? PrepareStatus::Failed : PrepareStatus::Cancelled;
    w.cancel_applied = bool(p.fatal);
    w.drained        = bool(p.fatal);
    p.reader.wake();
}

WindowSnapshot PrepareService::poll(Ticket ticket) const {
    std::lock_guard lock(impl_->mutex);
    return Impl::snapshot(impl_->get(ticket));
}

HostEmbedding PrepareService::embedding(Ticket ticket) const {
    std::lock_guard lock(impl_->mutex);
    const auto& w = impl_->get(ticket);
    if (w.cancelled || w.status != PrepareStatus::HostReady)
        throw std::logic_error("PLE embedding is not HostReady");
    return {ticket, w.output.subspan(0, std::size_t(w.tokens) * w.batch * kEmbeddingBytes),
            Impl::snapshot(w).input};
}

void PrepareService::release(Ticket ticket) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    auto& w = p.get(ticket);
    if (w.status == PrepareStatus::Preparing || !w.drained ||
        (w.cancelled && !w.cancel_applied && !p.fatal))
        throw std::logic_error("PLE window still has Host work");
    w.released = true;
    if (p.fatal) w.allocated = false;
    p.reader.wake();
}

ReadStats PrepareService::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->reader.stats();
}

std::size_t PrepareService::memory_bytes() const noexcept {
    const auto& p = *impl_;
    return sizeof(Impl) +
           p.windows.size() * (sizeof(Impl::Window) +
                               std::size_t(p.options.max_positions) * sizeof(std::int32_t)) +
           p.reader.memory_bytes();
}

std::size_t PrepareService::memory_bytes(TableGeometry table, const PrepareOptions& options,
                                         std::uint32_t slots) {
    if (!options.max_positions || options.max_positions > UINT32_MAX / kHeads || !slots)
        throw std::invalid_argument("invalid PLE preparation capacity");
    return sizeof(Impl) +
           std::size_t(slots) *
               (sizeof(Impl::Window) + std::size_t(options.max_positions) * sizeof(std::int32_t)) +
           RowReader::memory_bytes(table, slots, options.max_positions * kHeads, options.io_depth,
                                   options.cache_bytes);
}

} // namespace ninfer::models::qwen4_exp::ple::disk
