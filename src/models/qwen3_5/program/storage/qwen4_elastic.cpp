#include "models/qwen3_5/program/storage/qwen4_elastic.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::detail {
namespace {

std::uint32_t pages_of(std::uint32_t tokens) {
    return (tokens + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U) /
           static_cast<std::uint32_t>(kPagedKVPageSize);
}

void add_pool(const qwen3_5::PagedKVCacheLayout& pool, std::uint32_t step_pages,
              std::vector<ElasticRangeSpec>& out) {
    const std::uint32_t pages = pool.pages.spec.page_group_count;
    for (const DeviceKVPlaneLayout& plane : pool.pages.planes) {
        if (plane.geometry.host) { continue; } // active KV in RAM: not in the elastic arena
        const LayoutRegion& r = plane.storage.region;
        if (pages == 0 || r.bytes % pages != 0) {
            throw std::logic_error("elastic KV: a KV plane is not page-major");
        }
        const std::size_t page_bytes = r.bytes / pages;
        out.push_back(
            ElasticRangeSpec{r.offset, r.bytes, page_bytes * std::min(step_pages, pages)});
    }
}

} // namespace

std::vector<ElasticRangeSpec> qwen4_elastic_ranges(const PersistentLayout& layout,
                                                   std::uint32_t step_pages) {
    if (!layout.qwen4 || layout.qwen4->expert_slots <= 0 || !layout.decoder.text_kv.elastic_pages ||
        layout.elastic_bytes == 0) {
        throw std::logic_error("elastic KV: the layout has no elastic KV pages or expert pool");
    }
    std::vector<ElasticRangeSpec> out;
    add_pool(layout.decoder.text_kv, step_pages, out);
    if (layout.decoder.mtp_kv) { add_pool(*layout.decoder.mtp_kv, step_pages, out); }
    const Qwen4PersistentLayout& q = *layout.qwen4;
    for (std::int32_t plane = 0; plane < 4; ++plane) {
        // The tail the planner laid out (uniform NVFP4 planes, or the GGUF arena's columns).
        const auto first      = q.elastic_first[static_cast<std::size_t>(plane)];
        const auto bytes      = q.elastic_bytes[static_cast<std::size_t>(plane)];
        const LayoutRegion& r = q.cache_pool[static_cast<std::size_t>(plane)];
        if (first + bytes > r.bytes) {
            throw std::logic_error("elastic KV: expert pool too small");
        }
        out.push_back(ElasticRangeSpec{r.offset + first, bytes, bytes});
    }
    return out;
}

Qwen4ElasticVram::Qwen4ElasticVram(std::vector<Rank> ranks, execution::Qwen4OffloadExperts& experts,
                                   std::uint32_t step_pages, std::size_t kv_budget_bytes)
    : experts_(experts), step_pages_(step_pages), kv_budget_(kv_budget_bytes) {
    if (ranks.empty() || step_pages == 0 || ranks[0].text == nullptr) {
        throw std::invalid_argument("elastic KV: no rank or no grow step");
    }
    for (const Rank& rank : ranks) {
        if (rank.arena == nullptr || rank.layout == nullptr || rank.text == nullptr ||
            (rank.mtp == nullptr) != (ranks[0].mtp == nullptr)) {
            throw std::invalid_argument("elastic KV: incomplete rank");
        }
        RankState r;
        r.memory          = rank;
        std::size_t index = 0;
        const auto planes = [&](const qwen3_5::PagedKVCacheLayout& pool, std::vector<Plane>& out) {
            const std::uint32_t pages = pool.pages.spec.page_group_count;
            for (const DeviceKVPlaneLayout& plane : pool.pages.planes) {
                if (plane.geometry.host) { continue; } // Host planes have no elastic range
                out.push_back(Plane{index++, plane.storage.region.bytes / pages});
            }
        };
        planes(rank.layout->decoder.text_kv, r.text);
        if (rank.layout->decoder.mtp_kv) { planes(*rank.layout->decoder.mtp_kv, r.mtp); }
        for (std::size_t q = 0; q < 4; ++q) {
            r.expert[q]       = index++;
            r.column_bytes[q] = static_cast<std::size_t>(
                experts_.elastic_column_bytes(static_cast<std::int32_t>(q)));
        }
        if (index != rank.arena->range_count()) {
            throw std::logic_error("elastic KV: arena ranges do not match the layout");
        }
        r.budget = rank.arena->physical_bytes();
        ranks_.push_back(std::move(r));
    }
    text_capacity_ = ranks_[0].memory.text->capacity_pages();
    mtp_capacity_  = ranks_[0].memory.mtp != nullptr ? ranks_[0].memory.mtp->capacity_pages() : 0;
    stats_.text_pages = std::min(step_pages_, text_capacity_);
    stats_.mtp_pages  = std::min(step_pages_, mtp_capacity_);
    stats_.columns    = experts_.active_columns();
    stats_.base_slots = experts_.base_slots();
    stats_.kv_bytes   = kv_bytes(stats_.text_pages, stats_.mtp_pages);
    // The arenas were built with exactly these prefixes (qwen4_elastic_ranges).
    for (const RankState& r : ranks_) {
        const auto t = targets(r, stats_.text_pages, stats_.mtp_pages, stats_.columns);
        if (r.memory.arena->physical_delta(t) != 0) {
            throw std::logic_error("elastic KV: arena prefixes differ from the initial backing");
        }
    }
    for (std::size_t i = 0; i < ranks_.size(); ++i) {
        RankState& r     = ranks_[i];
        const bool owner = i == 0;
        r.memory.text->set_backing(stats_.text_pages, [this, owner](std::uint32_t pages) {
            if (!owner) { throw std::logic_error("elastic KV: a mirror pool handed out a page"); }
            grow(false, pages);
        });
        if (r.memory.mtp != nullptr) {
            r.memory.mtp->set_backing(stats_.mtp_pages, [this, owner](std::uint32_t pages) {
                if (!owner) {
                    throw std::logic_error("elastic KV: a mirror pool handed out a page");
                }
                grow(true, pages);
            });
        }
    }
    std::fprintf(
        stderr,
        "elastic KV: grow step %u pages, Text pool %u pages, MTP pool %u pages, %d elastic "
        "expert columns, budget %zu bytes on rank 0\n",
        step_pages_, text_capacity_, mtp_capacity_, stats_.columns, ranks_[0].budget);
}

std::uint32_t Qwen4ElasticVram::step_up(std::uint32_t pages,
                                        std::uint32_t capacity) const noexcept {
    const std::uint32_t stepped = (pages + step_pages_ - 1U) / step_pages_ * step_pages_;
    return std::min(std::max(stepped, std::min(step_pages_, capacity)), capacity);
}

std::vector<std::size_t> Qwen4ElasticVram::targets(const RankState& r, std::uint32_t text_pages,
                                                   std::uint32_t mtp_pages,
                                                   std::int32_t columns) const {
    std::vector<std::size_t> t(r.memory.arena->range_count(), 0);
    for (const Plane& p : r.text) { t[p.range] = p.page_bytes * text_pages; }
    for (const Plane& p : r.mtp) { t[p.range] = p.page_bytes * mtp_pages; }
    for (std::size_t q = 0; q < 4; ++q) {
        t[r.expert[q]] = r.column_bytes[q] * static_cast<std::size_t>(columns);
    }
    return t;
}

bool Qwen4ElasticVram::fits(std::uint32_t text_pages, std::uint32_t mtp_pages,
                            std::int32_t columns) const {
    for (const RankState& r : ranks_) {
        const std::int64_t after =
            static_cast<std::int64_t>(r.memory.arena->physical_bytes()) +
            r.memory.arena->physical_delta(targets(r, text_pages, mtp_pages, columns));
        if (after > static_cast<std::int64_t>(r.budget)) { return false; }
    }
    return true;
}

std::size_t Qwen4ElasticVram::kv_bytes(std::uint32_t text_pages, std::uint32_t mtp_pages) const {
    std::size_t bytes = 0;
    for (const Plane& p : ranks_[0].text) { bytes += p.page_bytes * text_pages; }
    for (const Plane& p : ranks_[0].mtp) { bytes += p.page_bytes * mtp_pages; }
    return bytes;
}

void Qwen4ElasticVram::apply(std::uint32_t text_pages, std::uint32_t mtp_pages,
                             std::int32_t columns, const char* why) {
    const std::int32_t before = experts_.active_columns();
    // Shrink the cache before its columns lose their memory, grow it after they have it back.
    if (columns < before) { experts_.set_active_columns(columns); }
    for (RankState& r : ranks_) {
        r.memory.arena->set_mapped(targets(r, text_pages, mtp_pages, columns), r.memory.stream);
    }
    if (columns > before) { experts_.set_active_columns(columns); }
    for (RankState& r : ranks_) {
        r.memory.text->set_backed_pages(text_pages);
        if (r.memory.mtp != nullptr) { r.memory.mtp->set_backed_pages(mtp_pages); }
    }
    stats_.shrinks += std::max(0, before - columns);
    stats_.returns += std::max(0, columns - before);
    stats_.text_pages = text_pages;
    stats_.mtp_pages  = mtp_pages;
    stats_.columns    = columns;
    stats_.kv_bytes   = kv_bytes(text_pages, mtp_pages);
    std::fprintf(stderr,
                 "elastic KV (%s): Text KV %u pages (%u tokens), MTP KV %u pages, %zu KV bytes per "
                 "GPU; expert columns %d -> %d (%d slots/layer/GPU); rank 0 physical %zu of %zu "
                 "bytes\n",
                 why, text_pages, text_pages * static_cast<std::uint32_t>(kPagedKVPageSize),
                 mtp_pages, stats_.kv_bytes, before, columns, stats_.base_slots + columns,
                 ranks_[0].memory.arena->physical_bytes(), ranks_[0].budget);
}

void Qwen4ElasticVram::grow(bool mtp_pool, std::uint32_t pages) {
    std::uint32_t text = stats_.text_pages;
    std::uint32_t mtp  = stats_.mtp_pages;
    if (mtp_pool) {
        mtp = std::max(mtp, step_up(pages, mtp_capacity_));
    } else {
        text = std::max(text, step_up(pages, text_capacity_));
    }
    if ((mtp_pool ? mtp : text) < pages) {
        throw std::logic_error("elastic KV: page beyond the pool");
    }
    if (kv_budget_ > 0 && kv_bytes(text, mtp) > kv_budget_) {
        throw std::runtime_error("elastic KV: the request needs more KV than kv_budget_bytes (" +
                                 std::to_string(kv_bytes(text, mtp)) + " > " +
                                 std::to_string(kv_budget_) + " bytes)");
    }
    std::int32_t columns = stats_.columns;
    while (columns > 0 && !fits(text, mtp, columns)) { --columns; }
    if (!fits(text, mtp, columns)) {
        throw std::runtime_error("elastic KV: device memory exhausted at " +
                                 std::to_string(text * kPagedKVPageSize) + " tokens of KV");
    }
    ++stats_.grows;
    apply(text, mtp, columns, "grow");
}

Qwen4ElasticVram::Backing Qwen4ElasticVram::plan(std::uint32_t text_tokens,
                                                 std::uint32_t mtp_tokens) const {
    const RankState& r0 = ranks_[0];
    Backing out;
    out.text = step_up(std::max(r0.memory.text->allocated_extent(), pages_of(text_tokens)),
                       text_capacity_);
    if (r0.memory.mtp != nullptr) {
        out.mtp = step_up(std::max(r0.memory.mtp->allocated_extent(), pages_of(mtp_tokens)),
                          mtp_capacity_);
    }
    if (kv_budget_ > 0 && kv_bytes(out.text, out.mtp) > kv_budget_) {
        throw std::runtime_error("elastic KV: the prompt needs more KV than kv_budget_bytes");
    }
    out.columns = stats_.columns;
    while (out.columns > 0 && !fits(out.text, out.mtp, out.columns)) { --out.columns; }
    if (!fits(out.text, out.mtp, out.columns)) {
        throw std::runtime_error("elastic KV: device memory exhausted at " +
                                 std::to_string(out.text * kPagedKVPageSize) + " tokens of KV");
    }
    while (out.columns < experts_.elastic_columns() && fits(out.text, out.mtp, out.columns + 1)) {
        ++out.columns;
    }
    return out;
}

void Qwen4ElasticVram::rebalance(std::uint32_t text_tokens, std::uint32_t mtp_tokens) {
    const Backing next = plan(text_tokens, mtp_tokens);
    if (next.text == stats_.text_pages && next.mtp == stats_.mtp_pages &&
        next.columns == stats_.columns) {
        return;
    }
    apply(next.text, next.mtp, next.columns, "rebalance");
}

bool Qwen4ElasticVram::shrinkable() const {
    const Backing next = plan(0, 0);
    return next.text < stats_.text_pages || next.mtp < stats_.mtp_pages;
}

void Qwen4ElasticVram::shrink_to_live() {
    const Backing next = plan(0, 0);
    if (next.text >= stats_.text_pages && next.mtp >= stats_.mtp_pages) { return; }
    // Only the KV side shrinks here: a backing that a live page still needs is never released.
    apply(std::min(next.text, stats_.text_pages), std::min(next.mtp, stats_.mtp_pages),
          std::max(next.columns, stats_.columns), "idle KV to Host");
}

} // namespace ninfer::models::qwen3_5::detail
