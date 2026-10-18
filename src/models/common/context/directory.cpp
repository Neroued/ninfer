#include "models/common/context/directory.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::context {

namespace {
void finalize_mappings(ContextActions& actions) {
    struct RowUpdates {
        TableRow row;
        std::vector<MappingUpdate> ranges;
    };

    std::vector<RowUpdates> rows;
    ContextActions ordered;
    ordered.operations.reserve(actions.operations.size());
    for (auto& action : actions.operations) {
        auto* update = std::get_if<MappingUpdate>(&action);
        if (!update) {
            ordered.operations.push_back(std::move(action));
            continue;
        }
        if (update->pages.empty()) continue;
        auto row = std::find_if(rows.begin(), rows.end(),
                                [&](const auto& existing) { return existing.row == update->row; });
        if (row == rows.end()) {
            rows.push_back({update->row, {}});
            row = rows.end() - 1;
        }
        auto& ranges         = row->ranges;
        const auto new_begin = update->logical_begin;
        const auto new_end   = new_begin + update->pages.size();
        auto first           = std::lower_bound(ranges.begin(), ranges.end(), new_begin,
                                                [](const auto& range, auto begin) {
                                          return range.logical_begin + range.pages.size() < begin;
                                      });
        auto last            = first;
        std::size_t begin = new_begin, end = new_end;
        while (last != ranges.end() && last->logical_begin <= end) {
            begin = std::min<std::size_t>(begin, last->logical_begin);
            end   = std::max(end, last->logical_begin + last->pages.size());
            ++last;
        }
        if (first == last) {
            ranges.insert(first, std::move(*update));
            continue;
        }
        MappingUpdate merged{update->row, static_cast<std::uint32_t>(begin),
                             std::vector<PageSlot>(end - begin)};
        for (auto previous = first; previous != last; ++previous)
            std::copy(previous->pages.begin(), previous->pages.end(),
                      merged.pages.begin() + (previous->logical_begin - begin));
        std::copy(update->pages.begin(), update->pages.end(),
                  merged.pages.begin() + (new_begin - begin));
        const auto position = ranges.erase(first, last);
        ranges.insert(position, std::move(merged));
    }
    // Copies address physical slots directly, so only the final mapping is needed after them.
    // Coalescing also bounds pinned upload storage by the total execution-table size.
    for (auto& row : rows)
        for (auto& range : row.ranges) ordered.operations.emplace_back(std::move(range));
    actions = std::move(ordered);
}
} // namespace

KVPoolDirectory::KVPoolDirectory(const KVDirectorySpec& spec, HostDirectory* host)
    : allocation_(spec.pool_id, spec.device_pages),
      pages_(allocation_, spec.logical_pages, spec.geometry, spec.transfer),
      histories_(pages_, spec.table_rows, spec.histories, spec.page_capacity, host) {}

ContextDirectory::ContextDirectory(ContextDirectorySpec spec)
    : host_allocation_(spec.host.pool_id, spec.host.bytes, spec.host.minimum_bytes),
      host_(host_allocation_, spec.host.extent_count),
      states_(spec.state, spec.host.bytes ? &host_allocation_ : nullptr) {
    pools_.reserve(spec.pools.size());
    for (const auto& pool : spec.pools) {
        if (pool.pool_id != pools_.size())
            throw std::invalid_argument("context pool IDs must be contiguous");
        pools_.push_back(
            std::make_unique<KVPoolDirectory>(pool, spec.host.bytes ? &host_ : nullptr));
    }
}

KVPoolDirectory& ContextDirectory::pool(std::uint32_t id) { return *pools_.at(id); }

const KVPoolDirectory& ContextDirectory::pool(std::uint32_t id) const { return *pools_.at(id); }

ContextTransaction ContextDirectory::begin_transaction() { return ContextTransaction(*this); }

ContextTransaction::ContextTransaction(ContextDirectory& directory) : directory_(&directory) {}

ContextTransaction::ContextTransaction(ContextTransaction&& other) noexcept
    : directory_(std::exchange(other.directory_, nullptr)), stage_(other.stage_),
      cancelled_(other.cancelled_), adopted_(other.adopted_), actions_(std::move(other.actions_)),
      state_operations_(std::move(other.state_operations_)), state_(std::move(other.state_)),
      state_copies_(std::move(other.state_copies_)), bindings_(std::move(other.bindings_)),
      backups_(std::move(other.backups_)), state_readers_(std::move(other.state_readers_)),
      page_readers_(std::move(other.page_readers_)), retired_rows_(std::move(other.retired_rows_)),
      growth_(std::move(other.growth_)), commits_(std::move(other.commits_)) {}

ContextTransaction::~ContextTransaction() {
    if (!directory_) return;
    if (in_flight()) std::terminate();
    release_reservations();
}

void ContextTransaction::require_reserved() const {
    if (stage_ != Stage::Reserved) throw std::logic_error("context actions are already prepared");
}

void ContextTransaction::own_state(StateId image) {
    require_reserved();
    if (state_) throw std::logic_error("context transaction already owns a State destination");
    state_.emplace(directory_->states(), image);
}

StateDirectory::Destination& ContextTransaction::state_destination() {
    if (!state_) throw std::logic_error("context transaction has no State destination");
    return *state_;
}

void ContextTransaction::add_state_copy(StateDirectory::Transfer transfer,
                                        bool keep_source_replica) {
    require_reserved();
    state_copies_.push_back({std::move(transfer), keep_source_replica});
}

void ContextTransaction::add_state_group(StateId source, std::size_t offset, std::size_t bytes) {
    require_reserved();
    const auto image_bytes = directory_->states().image_bytes();
    if (offset > image_bytes || bytes > image_bytes - offset)
        throw std::invalid_argument("State copy group is outside the image");
    if (!bytes) return;
    auto reader = directory_->states().read(source);
    const StateCopy copy{reader.device_slot(),
                         directory_->states().physical_slot(state_destination().image()), offset,
                         bytes};
    state_operations_.reserve(state_operations_.size() + 1);
    state_readers_.push_back(std::move(reader));
    state_operations_.emplace_back(copy);
}

void ContextTransaction::clear_state() {
    require_reserved();
    state_operations_.emplace_back(
        StateClear{directory_->states().physical_slot(state_destination().image())});
}

void ContextTransaction::add_binding(std::uint32_t pool, HistoryBinding binding) {
    require_reserved();
    if (this->binding(pool)) throw std::logic_error("duplicate context pool binding");
    if (binding.history().pool_id != pool)
        throw std::invalid_argument("context binding names another pool");
    bindings_.push_back({pool, std::move(binding)});
}

HistoryBinding* ContextTransaction::binding(std::uint32_t pool) noexcept {
    for (auto& binding : bindings_)
        if (binding.pool == pool) return &binding.value;
    return nullptr;
}

const HistoryBinding* ContextTransaction::binding(std::uint32_t pool) const noexcept {
    for (const auto& binding : bindings_)
        if (binding.pool == pool) return &binding.value;
    return nullptr;
}

void ContextTransaction::add_backup(std::uint32_t pool, std::vector<PageId> pages,
                                    HostPageReservation destination, bool drop_device) {
    require_reserved();
    if (!destination.valid() || pages.size() != directory_->host().page_count(destination))
        throw std::invalid_argument("context backup membership is inconsistent");
    for (const auto page : pages)
        if (page.pool_id != pool || !directory_->pool(pool).pages().valid(page))
            throw std::invalid_argument("context backup contains a foreign page");
    backups_.push_back({pool, std::move(pages), std::move(destination), drop_device});
}

void ContextTransaction::retain(PageDirectory& pages, PageId page) {
    require_reserved();
    page_readers_.push_back(pages.read_device(page, 0));
}

void ContextTransaction::retain(StateDirectory& states, StateId image) {
    require_reserved();
    state_readers_.push_back(states.read(image));
}

void ContextTransaction::copy_page(PageDirectory& pages, PageId source, PageId destination) {
    require_reserved();
    const PageCopy copy{pages.physical(source), pages.physical(destination)};
    auto reader = pages.read_device(source, 0);
    actions_.operations.reserve(actions_.operations.size() + 1);
    page_readers_.push_back(std::move(reader));
    actions_.operations.emplace_back(copy);
}

void ContextTransaction::update_mapping(TableRow row, std::uint32_t begin,
                                        std::span<const PageSlot> pages) {
    require_reserved();
    if (!directory_->pool(row.pool_id).histories().valid(row))
        throw std::invalid_argument("context mapping row is stale");
    for (auto page : pages)
        if (page.pool_id != row.pool_id ||
            !directory_->pool(row.pool_id).allocation().is_live(page))
            throw std::invalid_argument("context mapping contains a stale page");
    actions_.operations.emplace_back(MappingUpdate{row, begin, {pages.begin(), pages.end()}});
}

void ContextTransaction::retire(RowLease&& row) {
    require_reserved();
    retired_rows_.reserve(retired_rows_.size() + 1);
    retired_rows_.push_back(std::move(row));
}

void ContextTransaction::reserve_after_retirement(HistoryDirectory& histories, HistoryId history,
                                                  std::uint32_t pages) {
    require_reserved();
    for (auto& item : growth_) {
        if (item.histories == &histories && item.history == history) {
            item.pages = pages;
            return;
        }
    }
    growth_.push_back({&histories, history, pages});
}

void ContextTransaction::prepare_actions() {
    require_reserved();
    for (const auto& copy : state_copies_)
        actions_.operations.emplace_back(copy.transfer.description());
    for (const auto& action : state_operations_) actions_.operations.push_back(action);
    for (const auto& backup : backups_) {
        const auto range = directory_->host().writable_view(backup.destination);
        actions_.operations.emplace_back(
            PageTransfer{StateCopyDirection::DeviceToHost, range.extent,
                         directory_->host().device_sources(backup.destination)});
    }
    for (auto& pending : bindings_) {
        auto& binding       = pending.value;
        const auto restores = binding.restores();
        const auto stride   = binding.pages().host_page_layout().page_stride;
        for (std::size_t begin = 0; begin < restores.size();) {
            const auto first = binding.pages().host_replica(restores[begin].page());
            auto end         = begin + 1;
            while (end < restores.size()) {
                const auto next = binding.pages().host_replica(restores[end].page());
                if (next.extent != first.extent ||
                    next.page_offset != first.page_offset + end - begin)
                    break;
                ++end;
            }
            const auto range = directory_->host().view(first.extent);
            if (first.page_offset > range.page_count ||
                end - begin > range.page_count - first.page_offset)
                throw std::logic_error("context restore exceeds its Host extent");
            HostExtent host = range.extent;
            host.offset += static_cast<std::size_t>(first.page_offset) * stride;
            host.bytes = (end - begin) * stride;
            PageTransfer copy{
                .direction = StateCopyDirection::HostToDevice, .host = host, .pages = {}};
            copy.pages.reserve(end - begin);
            for (auto i = begin; i < end; ++i) copy.pages.push_back(restores[i].device());
            actions_.operations.emplace_back(std::move(copy));
            begin = end;
        }
        binding.prepare_tail_copy();
        if (binding.needs_tail_copy())
            actions_.operations.emplace_back(
                PageCopy{binding.tail_source(), binding.tail_destination()});
        binding.prepare_mapping(*this);
    }
    finalize_mappings(actions_);
    stage_ = Stage::Prepared;
}

void ContextTransaction::commit_history(HistoryDirectory& histories, HistoryId history,
                                        std::uint32_t frontier, std::uint32_t retain_through) {
    require_reserved();
    commits_.push_back({&histories, history, frontier, retain_through});
}

void ContextTransaction::mark_submitted() {
    if (stage_ != Stage::Prepared) throw std::logic_error("context transaction is not prepared");
    stage_ = Stage::Submitted;
}

void ContextTransaction::adopt() noexcept {
    assert(stage_ == Stage::Submitted && !adopted_);
    for (const auto& item : commits_) item.histories->commit_frontier(item.history, item.frontier);
    adopted_ = true;
}

void ContextTransaction::mark_completed() {
    if (stage_ != Stage::Submitted) throw std::logic_error("context transaction was not submitted");
    stage_ = Stage::Completed;
}

bool ContextTransaction::in_flight() const noexcept { return stage_ == Stage::Submitted; }

void ContextTransaction::publish() {
    if (stage_ != Stage::Completed || cancelled_)
        throw std::logic_error("context publication requires an uncancelled completed transaction");
    state_readers_.clear();
    page_readers_.clear();
    for (auto& copy : state_copies_) copy.transfer.publish(copy.keep_source);
    state_copies_.clear();
    for (auto& backup : backups_) {
        (void)directory_->host().publish(std::move(backup.destination));
        if (!backup.drop_device) continue;
        auto& pages = directory_->pool(backup.pool).pages();
        for (const auto page : backup.logical) {
            if (!pages.valid(page)) continue;
            if (!pages.drop_device_replica(page))
                throw std::logic_error("published context backup still has a Device reader");
        }
    }
    backups_.clear();
    for (auto& binding : bindings_) binding.value.publish_restores();
    for (const auto& item : growth_) item.histories->reserve_growth(item.history, item.pages);
    growth_.clear();
    assert(commits_.empty() || adopted_);
    for (const auto& item : commits_) {
        const auto pages =
            item.retain_through > item.frontier
                ? item.histories->write_pages(item.history, item.frontier, item.retain_through)
                : 0;
        item.histories->reserve_growth(item.history, pages);
    }
    commits_.clear();
    retired_rows_.clear();
    stage_ = Stage::Published;
}

void ContextTransaction::release_reservations() {
    actions_.operations.clear();
    state_operations_.clear();
    state_readers_.clear();
    page_readers_.clear();
    state_copies_.clear();
    backups_.clear();
    bindings_.clear();
    state_.reset();
    retired_rows_.clear();
    growth_.clear();
    commits_.clear();
}

void ContextTransaction::discard() {
    if (in_flight()) throw std::logic_error("cannot discard an in-flight context transaction");
    release_reservations();
    stage_ = Stage::Discarded;
}

void ContextTransaction::reset_plan() {
    if (stage_ != Stage::Reserved && stage_ != Stage::Prepared)
        throw std::logic_error("cannot reset a submitted context transaction");
    release_reservations();
    stage_     = Stage::Reserved;
    cancelled_ = false;
    adopted_   = false;
}

std::uint32_t ContextTransaction::spill_page_groups() const noexcept {
    std::uint32_t count = 0;
    for (const auto& backup : backups_)
        if (backup.drop_device) count += backup.logical.size();
    return count;
}

std::uint32_t ContextTransaction::tail_copy_page_groups() const noexcept {
    std::uint32_t count = 0;
    for (const auto& binding : bindings_) count += binding.value.needs_tail_copy();
    return count;
}

} // namespace ninfer::models::context
