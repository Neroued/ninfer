#include "models/qwen4_exp/ple/disk/row_reader.h"

#include "core/buffer.h"
#include "core/io/read_queue.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <system_error>

namespace ninfer::models::qwen4_exp::ple::disk {
namespace {

constexpr std::uint32_t none = std::numeric_limits<std::uint32_t>::max();

std::uint64_t mix(std::uint64_t word) {
    word ^= word >> 30;
    word *= 0xbf58476d1ce4e5b9ULL;
    word ^= word >> 27;
    word *= 0x94d049bb133111ebULL;
    return word ^ (word >> 31);
}

std::uint32_t capacity(std::uint64_t value) {
    if (!value || value >= none / 2) throw std::invalid_argument("PLE read capacity is too large");
    return static_cast<std::uint32_t>(value);
}

// Fixed index into separately owned records. Backshift deletion avoids accumulating tombstones.
class Index {
public:
    explicit Index(std::size_t size) : slots_(std::bit_ceil(size * 2), none) {}

    template <class Match>
    std::uint32_t find(std::uint64_t hash, Match match) const {
        for (auto p = hash & mask(); slots_[p] != none; p = (p + 1) & mask())
            if (match(slots_[p])) return slots_[p];
        return none;
    }

    void insert(std::uint64_t hash, std::uint32_t value) {
        auto p = hash & mask();
        while (slots_[p] != none) p = (p + 1) & mask();
        slots_[p] = value;
    }

    template <class Hash>
    void erase(std::uint64_t hash, std::uint32_t value, Hash hash_of) {
        auto gap = hash & mask();
        while (slots_[gap] != value) {
            if (slots_[gap] == none) throw std::logic_error("missing PLE index entry");
            gap = (gap + 1) & mask();
        }
        for (auto p = (gap + 1) & mask(); slots_[p] != none; p = (p + 1) & mask()) {
            const auto home = hash_of(slots_[p]) & mask();
            if (((p - home) & mask()) >= ((p - gap) & mask())) {
                slots_[gap] = slots_[p];
                gap         = p;
            }
        }
        slots_[gap] = none;
    }

    std::size_t bytes() const { return slots_.size() * sizeof(slots_[0]); }

private:
    std::size_t mask() const { return slots_.size() - 1; }

    std::vector<std::uint32_t> slots_;
};

struct CacheEntry {
    std::uint32_t row = none;
    std::array<std::byte, kRowBytes> bytes;
};

struct alignas(128) CacheSet {
    std::array<CacheEntry, 4> ways;
    std::uint32_t next = 0;
};

} // namespace

struct RowReader::Impl {
    struct Target {
        std::uint32_t row = none, previous = none, next = none;
        bool active = false;
    };

    struct Row {
        std::uint32_t id = 0, head = none, parts = 0, active = 0;
        const Nf4Bf16Lut* lut = nullptr;
        std::array<std::byte, kRowBytes> bytes;
        std::exception_ptr error;
        bool ready = false, abandoned = false, indexed = false;
    };

    struct Fragment {
        std::uint32_t page = none, next = none, row = none, page_offset = 0;
        std::uint16_t row_offset = 0, bytes = 0;
    };

    struct Page {
        std::uint64_t offset = 0;
        std::uint32_t file = 0, head = none, previous = none, next = none;
        bool queued = false, required = false;
    };

    struct Owner {
        std::span<std::uint16_t> output;
        std::vector<std::uint32_t> targets;
        std::size_t pending = 0, unready = 0;
        std::exception_ptr error;
        bool required = false, bound = false;
    };

    const Table& table;
    std::uint32_t owner_rows, part_capacity, read_bytes = 4096;
    std::vector<Owner> owners;
    std::vector<Target> targets;
    std::vector<Row> rows;
    std::vector<Fragment> fragments;
    std::vector<Page> pages;
    std::vector<std::uint32_t> free_rows, free_pages, inflight, ready_rows;
    std::size_t ready_begin = 0, ready_end = 0;
    std::array<std::uint32_t, 2> queue_head{none, none}, queue_tail{none, none};
    Index row_index, page_index;
    std::vector<CacheSet> cache;
    std::vector<artifact::ReadSegment> segments;
    HostBuffer slab;
    std::vector<MemorySpan> buffers;
    std::vector<io::ReadCompletion> completions;
    ReadStats stats;
    std::unique_ptr<io::DirectReadQueue> queue;

    struct Layout {
        std::uint32_t owners, owner_rows, rows, fragments, parts, depth;
        std::size_t cache_sets, segments, alignment = 4096, read_bytes = 4096;

        [[nodiscard]] std::size_t bytes() const {
            return sizeof(Impl) + owners * sizeof(Owner) +
                   rows * (sizeof(Target) + sizeof(Row) + 3 * sizeof(std::uint32_t)) +
                   fragments * (sizeof(Fragment) + sizeof(Page) + sizeof(std::uint32_t)) +
                   (std::bit_ceil(std::size_t(rows) * 2) +
                    std::bit_ceil(std::size_t(fragments) * 2)) *
                       sizeof(std::uint32_t) +
                   cache_sets * sizeof(CacheSet) + segments * sizeof(artifact::ReadSegment) +
                   depth * (read_bytes + sizeof(MemorySpan) + sizeof(io::ReadCompletion) +
                            sizeof(std::uint32_t));
        }
    };

    static Layout describe(TableGeometry table, std::uint32_t count, std::uint32_t per_owner,
                           std::uint32_t depth, std::size_t cache_bytes) {
        if (!depth || depth > 65535 || !table.max_row_segments || !count || !per_owner ||
            (cache_bytes && cache_bytes < sizeof(CacheSet)))
            throw std::invalid_argument("invalid PLE row reader resources");
        // Later segments start at a V3 file's 4 KiB-aligned payload prefix. Only the
        // first segment can straddle a read page; each later segment is at most 90 bytes.
        Layout out{count,
                   per_owner,
                   capacity(std::uint64_t(count) * per_owner),
                   0,
                   static_cast<std::uint32_t>(
                       std::min<std::size_t>(kRowBytes, table.max_row_segments + 1)),
                   depth,
                   cache_bytes / sizeof(CacheSet),
                   table.max_row_segments};
        out.fragments  = capacity(std::uint64_t(out.rows) * out.parts);
        out.alignment  = table.memory_alignment;
        out.read_bytes = table.read_alignment;
        return out;
    }

    Impl(const Table& table, std::uint32_t count, std::uint32_t per_owner, std::uint32_t depth,
         std::size_t cache_bytes)
        : Impl(table, describe(table.geometry(), count, per_owner, depth, cache_bytes)) {}

    Impl(const Table& table, const Layout& layout)
        : table(table), owner_rows(layout.owner_rows), part_capacity(layout.parts),
          read_bytes(layout.read_bytes), owners(layout.owners), targets(layout.rows),
          rows(layout.rows), fragments(layout.fragments), pages(layout.fragments),
          inflight(layout.depth, none), ready_rows(layout.rows), row_index(layout.rows),
          page_index(layout.fragments), cache(layout.cache_sets), segments(layout.segments),
          completions(layout.depth) {
        for (auto& owner : owners) owner.targets.reserve(owner_rows);
        free_rows.resize(rows.size());
        free_pages.resize(pages.size());
        std::iota(free_rows.rbegin(), free_rows.rend(), 0);
        std::iota(free_pages.rbegin(), free_pages.rend(), 0);
        slab = HostBuffer(std::size_t(layout.depth) * read_bytes, layout.alignment);
        buffers.reserve(layout.depth);
        for (std::uint32_t i = 0; i < layout.depth; ++i)
            buffers.push_back(slab.span().subspan(std::size_t(i) * read_bytes, read_bytes));
        queue = std::make_unique<io::DirectReadQueue>(table.files(), buffers, layout.depth);
    }

    Owner& owner(std::uint32_t id) {
        if (id >= owners.size() || !owners[id].bound)
            throw std::invalid_argument("unbound PLE row owner");
        return owners[id];
    }

    std::uint64_t page_hash(std::uint32_t id) const {
        return mix(pages[id].offset ^ (std::uint64_t(pages[id].file) << 32));
    }

    void unlink_page(std::uint32_t id) {
        auto& p      = pages[id];
        const auto q = unsigned(p.required);
        if (p.previous == none)
            queue_head[q] = p.next;
        else
            pages[p.previous].next = p.next;
        if (p.next == none)
            queue_tail[q] = p.previous;
        else
            pages[p.next].previous = p.previous;
        p.previous = p.next = none;
        p.queued            = false;
    }

    void queue_page(std::uint32_t id) {
        auto& p      = pages[id];
        const auto q = unsigned(p.required);
        p.previous   = queue_tail[q];
        p.next       = none;
        if (p.previous == none)
            queue_head[q] = id;
        else
            pages[p.previous].next = id;
        queue_tail[q] = id;
        p.queued      = true;
    }

    void promote(std::uint32_t id) {
        auto& p = pages[id];
        if (p.required) return;
        const bool queued = p.queued;
        if (queued) unlink_page(id);
        p.required = true;
        if (queued) queue_page(id);
    }

    bool wanted(std::uint32_t page) const {
        for (auto f = pages[page].head; f != none; f = fragments[f].next)
            if (rows[fragments[f].row].active) return true;
        return false;
    }

    const CacheEntry* cached(std::uint32_t row) const {
        if (cache.empty()) return nullptr;
        for (const auto& entry : cache[mix(row) % cache.size()].ways)
            if (entry.row == row) return &entry;
        return nullptr;
    }

    void cache_row(const Row& row) {
        if (cache.empty()) return;
        auto& set = cache[mix(row.id) % cache.size()];
        for (auto& entry : set.ways)
            if (entry.row == row.id) return;
        auto& entry = set.ways[set.next++ % set.ways.size()];
        entry.row   = row.id;
        entry.bytes = row.bytes;
    }

    void unindex(std::uint32_t id) {
        auto& row = rows[id];
        if (!row.indexed) return;
        row_index.erase(mix(row.id), id, [&](auto i) { return mix(rows[i].id); });
        row.indexed = false;
    }

    void finish_page(std::uint32_t id, const std::byte* data, std::size_t bytes,
                     std::exception_ptr error = {}) {
        auto part = pages[id].head;
        while (part != none) {
            auto& f   = fragments[part];
            auto& row = rows[f.row];
            if (!data && !error) {
                row.abandoned = true;
                unindex(f.row);
            } else if (error || std::size_t(f.page_offset) + f.bytes > bytes) {
                if (!row.error)
                    row.error = error ? error
                                      : std::make_exception_ptr(std::runtime_error(
                                            "PLE read ended before required row bytes"));
            } else {
                std::memcpy(row.bytes.data() + f.row_offset, data + f.page_offset, f.bytes);
            }
            if (--row.parts == 0) ready_rows[ready_end++ % ready_rows.size()] = f.row;
            f.page = none;
            part   = f.next;
        }
        page_index.erase(page_hash(id), id, [&](auto i) { return page_hash(i); });
        pages[id] = {};
        free_pages.push_back(id);
    }

    void decode(std::uint32_t id) {
        auto& row = rows[id];
        std::array<std::uint16_t, kRowElements> decoded;
        if (!row.abandoned && !row.error) {
            try {
                decode_nf4_row(row.bytes, *row.lut, decoded);
                cache_row(row);
            } catch (...) { row.error = std::current_exception(); }
        }
        if (row.error || row.abandoned) unindex(id);
        for (auto t = row.head; t != none; t = targets[t].next) {
            auto& target = targets[t];
            auto& owner  = owners[t / owner_rows];
            --owner.unready;
            if (!target.active) continue;
            if (row.error)
                owner.error = row.error;
            else if (row.abandoned)
                throw std::logic_error("active PLE row was abandoned");
            else
                std::copy(decoded.begin(), decoded.end(),
                          owner.output.begin() + std::size_t(t % owner_rows) * kRowElements);
            --owner.pending;
            target.active = false;
        }
        row.active = 0;
        row.ready  = true;
    }

    void pump() {
        const auto n = queue->poll(completions);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& c = completions[i];
            const auto id = inflight[c.buffer_index];
            if (id == none || c.tag != id) throw std::logic_error("unmatched PLE page completion");
            stats.read_bytes += c.bytes;
            finish_page(id, static_cast<const std::byte*>(buffers[c.buffer_index].data), c.bytes,
                        c.error ? std::make_exception_ptr(std::system_error(
                                      c.error, std::generic_category(), "PLE page read"))
                                : std::exception_ptr{});
            inflight[c.buffer_index] = none;
        }
        std::size_t skipped = 0;
        for (std::uint32_t buffer = 0; buffer < buffers.size(); ++buffer) {
            if (inflight[buffer] != none) continue;
            while (queue_head[0] != none || queue_head[1] != none) {
                const auto id     = queue_head[1] != none ? queue_head[1] : queue_head[0];
                const auto p      = pages[id];
                const bool needed = wanted(id);
                unlink_page(id);
                if (!needed) {
                    finish_page(id, nullptr, 0);
                    if (++skipped == 256) break;
                    continue;
                }
                if (queue->try_enqueue({p.file, buffer, p.offset, read_bytes, id}) !=
                    io::ReadAdmission::Accepted)
                    throw std::logic_error("PLE read slot admission failed");
                inflight[buffer] = id;
                ++stats.reads;
                break;
            }
            if (skipped == 256) break;
        }
        queue->submit();
        for (std::size_t budget = 0; ready_begin != ready_end && budget < 128; ++budget)
            decode(ready_rows[ready_begin++ % ready_rows.size()]);
    }
};

RowReader::RowReader(const Table& table, std::uint32_t owners, std::uint32_t rows_per_owner,
                     std::uint32_t io_depth, std::size_t cache_bytes)
    : impl_(std::make_unique<Impl>(table, owners, rows_per_owner, io_depth, cache_bytes)) {}

RowReader::~RowReader() = default;

void RowReader::begin(std::uint32_t id, std::span<std::uint16_t> output, bool required) {
    auto& p = *impl_;
    if (id >= p.owners.size() || p.owners[id].bound || output.empty() ||
        output.size() % kRowElements || output.size() / kRowElements > p.owner_rows)
        throw std::invalid_argument("invalid PLE row output");
    auto& o    = p.owners[id];
    o.output   = output;
    o.required = required;
    o.bound    = true;
    o.error    = {};
}

void RowReader::request(std::uint32_t owner_id, std::uint32_t destination, std::uint32_t row_id) {
    auto& p     = *impl_;
    auto& owner = p.owner(owner_id);
    if (destination >= owner.output.size() / kRowElements || row_id >= p.table.parameters().rows)
        throw std::invalid_argument("invalid PLE row request");
    const auto target_id = owner_id * p.owner_rows + destination;
    auto& target         = p.targets[target_id];
    if (target.row != none) throw std::invalid_argument("PLE destination already requested");
    ++p.stats.rows;
    auto id = p.row_index.find(mix(row_id), [&](auto i) { return p.rows[i].id == row_id; });
    if (id == none) {
        if (p.free_rows.empty()) throw std::logic_error("PLE row capacity exhausted");
        id = p.free_rows.back();
        p.free_rows.pop_back();
        auto& row         = p.rows[id];
        const auto source = p.table.row(row_id);
        row.id            = row_id;
        row.lut           = source.lut;
        row.indexed       = true;
        p.row_index.insert(mix(row_id), id);
        if (const auto* hit = p.cached(row_id)) {
            row.bytes = hit->bytes;
            row.ready = true;
            ++p.stats.cache_hits;
        } else {
            const auto count =
                artifact::map_file_range(source.object, source.offset, kRowBytes, p.segments);
            std::uint32_t part = 0;
            for (std::size_t s = 0; s < count; ++s) {
                const auto& segment  = p.segments[s];
                std::uint64_t copied = 0;
                while (copied < segment.bytes) {
                    const auto position = segment.file_offset + copied;
                    const auto offset   = position / p.read_bytes * p.read_bytes;
                    const auto prefix   = std::uint32_t(position - offset);
                    const auto bytes =
                        std::min<std::uint64_t>(p.read_bytes - prefix, segment.bytes - copied);
                    const auto hash = mix(offset ^ (std::uint64_t(segment.file_index) << 32));
                    auto page       = p.page_index.find(hash, [&](auto i) {
                        return p.pages[i].offset == offset && p.pages[i].file == segment.file_index;
                    });
                    if (page == none) {
                        if (p.free_pages.empty())
                            throw std::logic_error("PLE page capacity exhausted");
                        page = p.free_pages.back();
                        p.free_pages.pop_back();
                        p.pages[page].offset   = offset;
                        p.pages[page].file     = segment.file_index;
                        p.pages[page].required = owner.required;
                        p.page_index.insert(hash, page);
                        p.queue_page(page);
                    } else {
                        ++p.stats.shared_pages;
                        if (owner.required) p.promote(page);
                    }
                    if (part == p.part_capacity)
                        throw std::logic_error("PLE row fragment capacity exceeded");
                    const auto f       = id * p.part_capacity + part++;
                    p.fragments[f]     = {page,
                                          p.pages[page].head,
                                          id,
                                          prefix,
                                          std::uint16_t(segment.destination_offset + copied),
                                          std::uint16_t(bytes)};
                    p.pages[page].head = f;
                    ++row.parts;
                    copied += bytes;
                }
            }
        }
    } else {
        ++p.stats.shared_rows;
        if (owner.required)
            for (std::uint32_t f = 0; f < p.part_capacity; ++f) {
                const auto page = p.fragments[id * p.part_capacity + f].page;
                if (page != none) p.promote(page);
            }
    }
    auto& row = p.rows[id];
    target    = {id, none, row.head, !row.ready};
    if (row.head != none) p.targets[row.head].previous = target_id;
    row.head = target_id;
    owner.targets.push_back(target_id);
    if (row.ready) {
        decode_nf4_row(row.bytes, *row.lut,
                       owner.output.subspan(std::size_t(destination) * kRowElements, kRowElements));
    } else {
        ++row.active;
        ++owner.pending;
        ++owner.unready;
    }
}

void RowReader::cancel(std::uint32_t id, std::uint32_t begin, std::uint32_t end) {
    auto& p     = *impl_;
    auto& owner = p.owner(id);
    if (begin > end || end > owner.output.size() / kRowElements)
        throw std::invalid_argument("invalid PLE cancellation range");
    if (begin == end) return;
    for (auto t : owner.targets) {
        auto& target     = p.targets[t];
        const auto index = t % p.owner_rows;
        if (index >= begin && index < end && target.active) {
            target.active = false;
            --p.rows[target.row].active;
            --owner.pending;
            if (p.rows[target.row].active) continue;
            for (std::uint32_t f = 0; f < p.part_capacity; ++f) {
                const auto page = p.fragments[target.row * p.part_capacity + f].page;
                if (page == none || !p.pages[page].queued || p.wanted(page)) continue;
                p.unlink_page(page);
                p.finish_page(page, nullptr, 0);
            }
        }
    }
}

void RowReader::prioritize(std::uint32_t id) {
    auto& p        = *impl_;
    auto& owner    = p.owner(id);
    owner.required = true;
    for (auto t : owner.targets) {
        if (!p.targets[t].active) continue;
        for (std::uint32_t f = 0; f < p.part_capacity; ++f) {
            const auto page = p.fragments[p.targets[t].row * p.part_capacity + f].page;
            if (page != none) p.promote(page);
        }
    }
}

void RowReader::release(std::uint32_t id) {
    auto& p     = *impl_;
    auto& owner = p.owner(id);
    if (owner.pending || owner.unready) throw std::logic_error("PLE row owner still has writers");
    for (auto t : owner.targets) {
        const auto target = p.targets[t];
        auto& row         = p.rows[target.row];
        if (target.previous == none)
            row.head = target.next;
        else
            p.targets[target.previous].next = target.next;
        if (target.next != none) p.targets[target.next].previous = target.previous;
        if (row.head == none) {
            p.unindex(target.row);
            row = {};
            p.free_rows.push_back(target.row);
        }
        p.targets[t] = {};
    }
    owner.targets.clear();
    owner.output = {};
    owner.error  = {};
    owner.bound  = false;
}

void RowReader::pump() { impl_->pump(); }

bool RowReader::runnable() const noexcept {
    const auto& p = *impl_;
    return p.ready_begin != p.ready_end || ((p.queue_head[0] != none || p.queue_head[1] != none) &&
                                            p.queue->outstanding() < p.buffers.size());
}

void RowReader::wait() { (void)impl_->queue->wait(); }

void RowReader::wake() { impl_->queue->wake(); }

std::size_t RowReader::pending(std::uint32_t owner) const { return impl_->owner(owner).pending; }

bool RowReader::drained(std::uint32_t owner) const { return impl_->owner(owner).unready == 0; }

std::exception_ptr RowReader::error(std::uint32_t owner) const { return impl_->owner(owner).error; }

ReadStats RowReader::stats() const noexcept { return impl_->stats; }

std::size_t RowReader::memory_bytes() const noexcept {
    const auto& p = *impl_;
    return sizeof(Impl) + p.owners.size() * sizeof(Impl::Owner) +
           p.targets.size() * (sizeof(Impl::Target) + sizeof(std::uint32_t)) +
           p.rows.size() * (sizeof(Impl::Row) + 2 * sizeof(std::uint32_t)) +
           p.fragments.size() * sizeof(Impl::Fragment) +
           p.pages.size() * (sizeof(Impl::Page) + sizeof(std::uint32_t)) + p.row_index.bytes() +
           p.page_index.bytes() + p.cache.size() * sizeof(CacheSet) +
           p.segments.size() * sizeof(artifact::ReadSegment) + p.slab.size() +
           p.buffers.size() *
               (sizeof(MemorySpan) + sizeof(io::ReadCompletion) + sizeof(std::uint32_t));
}

std::size_t RowReader::memory_bytes(TableGeometry table, std::uint32_t owners, std::uint32_t rows,
                                    std::uint32_t depth, std::size_t cache) {
    return Impl::describe(table, owners, rows, depth, cache).bytes();
}

} // namespace ninfer::models::qwen4_exp::ple::disk
