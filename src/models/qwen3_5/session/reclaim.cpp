#include "models/qwen3_5/session/internal.h"

#include "models/common/context/physical_reclaim.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <unordered_map>

namespace ninfer::models::qwen3_5::detail {
namespace {

struct StateDemotion {
    context::StateId handle;
    std::uint64_t content_epoch = 0;
    bool host_resident          = false;
    std::vector<CheckpointHandle> references;

    friend bool operator==(const StateDemotion&, const StateDemotion&) = default;
};

struct PageDemotion {
    context::PageId handle;
    std::uint32_t position          = 0;
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
    bool host_covers_prefix         = false;
    std::vector<CheckpointHandle> references;

    friend bool operator==(const PageDemotion&, const PageDemotion&) = default;
};

using PhysicalFacts  = context::PhysicalReclaim<ContextResources>;
using PhysicalOwners = PhysicalFacts::Group;

PhysicalFacts physical_facts(const SessionImpl& program, std::span<const CheckpointHandle> allowed,
                             std::span<const CheckpointHandle> excluded,
                             ContextResources shortage) {
    const std::array pools{
        context::ReclaimPool{program.text_kv_histories, program.text_kv_pages,
                             program.text_kv_pages->host_page_layout().page_stride},
        context::ReclaimPool{program.backend_kv_histories, program.backend_kv_pages,
                             program.backend_kv_pages
                                 ? program.backend_kv_pages->host_page_layout().page_stride
                                 : 0}};
    return PhysicalFacts(program.checkpoints, *program.state_store,
                         program.state_store->image_bytes(), pools, allowed, excluded, shortage);
}

struct ReleasedPages {
    std::vector<context::PageId> pages;

    bool empty() const noexcept { return pages.empty(); }

    template <typename Predicate>
    bool all(Predicate&& predicate) const {
        return std::all_of(pages.begin(), pages.end(), predicate);
    }
};

struct ReleaseScratch {
    std::vector<const context::HistoryPrefix*> main, backend;
    std::vector<CheckpointHandle> records;

    struct Image {
        context::StateId handle;
        std::uint32_t count = 0;
    };

    std::vector<Image> images;
};

std::optional<ContextResources> released_resources(const SessionImpl& program,
                                                   std::span<const CheckpointHandle> handles,
                                                   ContextResources shortage,
                                                   ReleaseScratch& scratch,
                                                   ReleasedPages& released_pages) {
    scratch.main.clear();
    scratch.backend.clear();
    scratch.records.clear();
    scratch.images.clear();
    for (auto handle : handles) {
        if (!program.can_release_checkpoint(handle)) return std::nullopt;
        if (std::find(scratch.records.begin(), scratch.records.end(), handle) !=
            scratch.records.end())
            continue;
        scratch.records.push_back(handle);
        const auto& record = program.checkpoint(handle);
        const auto image   = record.continuation().image;
        auto found         = std::find_if(scratch.images.begin(), scratch.images.end(),
                                          [&](const auto& entry) { return entry.handle == image; });
        if (found == scratch.images.end())
            scratch.images.push_back({image, 1});
        else
            ++found->count;
        scratch.main.push_back(&record.text_prefix());
        if (const auto* prefix = record.backend_prefix()) scratch.backend.push_back(prefix);
    }
    ContextResources result;
    for (const auto& image : scratch.images) {
        if (image.count != program.state_store->checkpoint_references(image.handle) ||
            !program.state_store->can_release_after_checkpoint_references(image.handle,
                                                                          image.count))
            continue;
        if (shortage.host_bytes && program.state_store->host_resident(image.handle))
            result.host_bytes += program.state_store->image_bytes();
        if (shortage.state_slots && program.state_store->device_resident(image.handle))
            ++result.state_slots;
    }
    const auto count = [&](const context::HistoryDirectory& histories,
                           const context::PageDirectory& pages, const auto& removed, bool main,
                           std::size_t stride) {
        const bool device = main ? shortage.kv_page_groups[kTextKvPool.index] != 0
                                 : shortage.kv_page_groups[kBackendKvPool.index] != 0;
        if (!shortage.host_bytes && !device) return;
        for (auto page : histories.released_pages(removed)) {
            if (shortage.host_bytes && pages.host_resident(page)) result.host_bytes += stride;
            if (device && pages.device_resident(page)) {
                ++(main ? result.kv_page_groups[kTextKvPool.index]
                        : result.kv_page_groups[kBackendKvPool.index]);
                released_pages.pages.push_back(page);
            }
        }
    };
    count(*program.text_kv_histories, *program.text_kv_pages, scratch.main, true,
          program.text_kv_pages->host_page_layout().page_stride);
    if (program.backend_kv_histories)
        count(*program.backend_kv_histories, *program.backend_kv_pages, scratch.backend, false,
              program.backend_kv_pages->host_page_layout().page_stride);
    return result;
}

} // namespace

// A quote carries content identities and complete optional holders. It holds no execution
// lease: start_demote revalidates it before acquiring real destinations.
struct DemotionPlan {
    const SessionImpl* owner                  = nullptr;
    runtime::ContextTransferResource resource = runtime::ContextTransferResource{};
    std::vector<CheckpointHandle> allowed;
    std::vector<CheckpointHandle> excluded;
    std::optional<StateDemotion> state;
    std::vector<PageDemotion> pages;
    // Each range retains one component's deep-first selection order. Transfers traverse
    // each range forward in logical position without reversing the component ordering.
    std::vector<std::uint32_t> page_group_ends;
};

std::uint32_t SessionImpl::checkpoint_recovery_frontier(CheckpointHandle retained,
                                                        const RequestBasePlan& base,
                                                        std::uint32_t target) const {
    if (!valid_checkpoint(retained) || checkpoint(retained).frontier() > target ||
        !checkpoint_matches(retained, base)) {
        return 0;
    }
    return checkpoint(retained).frontier();
}

struct ReclaimPlan {
    ReclaimPlan(const SessionImpl& program, std::span<const CheckpointHandle> allowed,
                std::span<const CheckpointHandle> excluded, ContextResources requested)
        : owner(&program), shortage(requested),
          facts(physical_facts(program, allowed, excluded, requested)) {}

    const SessionImpl* owner;
    ContextResources shortage;
    PhysicalFacts facts;
};

struct ReleasePlan {
    std::shared_ptr<const ReclaimPlan> evaluation;
    std::vector<CheckpointHandle> sources;
    ContextResources released;
    ReleasedPages pages;
};

std::optional<ContextDemotion> quote_demotion(const SessionImpl& program,
                                              const PhysicalOwners& owners,
                                              std::span<const CheckpointHandle> excluded,
                                              ContextResources shortage) {
    if (shortage.host_bytes ||
        (!shortage.state_slots && !shortage.kv_page_groups[kTextKvPool.index] &&
         !shortage.kv_page_groups[kBackendKvPool.index])) {
        return std::nullopt;
    }
    const auto resource = shortage.state_slots ? runtime::ContextTransferResource{}
                          : shortage.kv_page_groups[kTextKvPool.index]
                              ? runtime::ContextTransferResource{kTextKvPool}
                              : runtime::ContextTransferResource{kBackendKvPool};
    auto plan           = std::make_shared<DemotionPlan>();
    plan->owner         = &program;
    plan->resource      = resource;
    plan->allowed       = owners.sources;
    plan->excluded.assign(excluded.begin(), excluded.end());
    ContextDemotion candidate;
    if (resource == runtime::ContextTransferResource{}) {
        for (const auto state : owners.states) {
            if (program.state_store->role(state) != context::StateRole::CheckpointImmutable ||
                !program.state_store->can_release_after_checkpoint_references(
                    state, program.state_store->checkpoint_references(state))) {
                continue;
            }
            const auto host = program.state_store->host_resident(state);
            plan->state     = StateDemotion{state, program.state_store->content_epoch(state), host,
                                        owners.sources};
            candidate.released.state_slots = 1;
            candidate.host_bytes           = host ? 0 : program.state_store->image_bytes();
            break;
        }
        if (!plan->state) { return std::nullopt; }
    } else {
        const auto& pages   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? *program.text_kv_pages
                                  : *program.backend_kv_pages;
        const auto& objects = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? owners.pages[kTextKvPool.index]
                                  : owners.pages[kBackendKvPool.index];
        const auto needed   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? shortage.kv_page_groups[kTextKvPool.index]
                                  : shortage.kv_page_groups[kBackendKvPool.index];
        const auto stride   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? program.text_kv_pages->host_page_layout().page_stride
                                  : program.backend_kv_pages->host_page_layout().page_stride;
        for (const auto& object : objects) {
            const auto page = object.handle;
            if (pages.active_history_references(page) || pages.source_pins(page) ||
                !pages.can_pin_source(page)) {
                continue;
            }
            const auto host = pages.host_replica_current(page);
            plan->pages.push_back({page, object.position, pages.content_epoch(page),
                                   pages.committed_columns(page), host, owners.sources});
            if (!host) { candidate.host_bytes += stride; }
            if (plan->pages.size() == needed) { break; }
        }
        if (plan->pages.empty()) { return std::nullopt; }
        (resource == runtime::ContextTransferResource{kTextKvPool}
             ? candidate.released.kv_page_groups[kTextKvPool.index]
             : candidate.released.kv_page_groups[kBackendKvPool.index]) =
            static_cast<std::uint32_t>(plan->pages.size());
    }
    candidate.sources = owners.sources;
    candidate.impl    = std::move(plan);
    return candidate;
}

std::optional<ContextRelease>
quote_release(const std::shared_ptr<ReclaimPlan>& evaluation,
              std::span<const CheckpointHandle> sources,
              context::ReleaseInventory<ContextResources>::Selection& selection) {

    auto plan           = std::make_shared<ReleasePlan>();
    auto probe          = selection.probe(sources);
    const auto released = &selection.released();
    for (const auto group : selection.freed())
        for (std::size_t pool = 0; pool < ContextResources::pool_capacity; ++pool)
            if (evaluation->shortage.kv_page_groups[pool])
                for (const auto& page : evaluation->facts.groups[group].pages[pool])
                    if (page.releasable &&
                        (pool == kTextKvPool.index
                             ? evaluation->owner->text_kv_pages->device_resident(page.handle)
                             : evaluation->owner->backend_kv_pages->device_resident(page.handle)))
                        plan->pages.pages.push_back(page.handle);
    const auto shortage = evaluation->shortage;
    const auto useful   = shortage.host_bytes    ? released->host_bytes
                          : shortage.state_slots ? released->state_slots
                          : shortage.kv_page_groups[kTextKvPool.index]
                              ? released->kv_page_groups[kTextKvPool.index]
                          : shortage.kv_page_groups[kBackendKvPool.index]
                              ? released->kv_page_groups[kBackendKvPool.index]
                              : 0;
    if (!useful) { return std::nullopt; }
    plan->evaluation = evaluation;
    plan->sources.assign(sources.begin(), sources.end());
    plan->released = *released;
    return ContextRelease{plan->sources, *released, std::move(plan)};
}

std::optional<ContextResources>
SessionImpl::checkpoint_release_resources(std::span<const CheckpointHandle> handles,
                                          ContextResources shortage) const {
    ReleaseScratch scratch;
    ReleasedPages pages;
    return released_resources(*this, handles, shortage, scratch, pages);
}

ContextReclaimPlan SessionImpl::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                             std::span<const CheckpointHandle> excluded,
                                             ContextResources shortage) const {
    ContextReclaimPlan result;
    if (has_context_operation()) { return result; }
    result.impl_   = std::make_shared<ReclaimPlan>(*this, allowed, excluded, shortage);
    auto selection = result.impl_->facts.inventory->select();
    for (const auto& group : result.impl_->facts.groups) {
        if (auto quote = quote_demotion(*this, group, excluded, shortage)) {
            result.demotions.push_back(std::move(*quote));
        }
        if (auto quote = quote_release(result.impl_, group.sources, selection)) {
            result.releases.push_back(std::move(*quote));
        }
    }
    return result;
}

context::ReleasePlan<ContextRelease, ContextResources>
SessionImpl::plan_releases(std::span<const CheckpointHandle> allowed,
                           std::span<const CheckpointHandle> excluded,
                           ContextResources shortage) const {
    auto evaluation = std::make_shared<ReclaimPlan>(*this, allowed, excluded, shortage);
    context::ReleasePlan<ContextRelease, ContextResources> result{{}, evaluation->facts.inventory};
    if (has_context_operation()) return result;
    auto selection = evaluation->facts.inventory->select();
    for (const auto& group : evaluation->facts.groups)
        if (auto quote = quote_release(evaluation, group.sources, selection))
            result.releases.push_back(std::move(*quote));
    return result;
}

struct DemotionBatch {
    DemotionBatch(std::shared_ptr<const ReclaimPlan> evaluation, ContextResources requested)
        : evaluation(std::move(evaluation)), shortage(requested) {
        if (!this->evaluation || shortage.state_slots || shortage.host_bytes ||
            (!shortage.kv_page_groups[kTextKvPool.index] &&
             !shortage.kv_page_groups[kBackendKvPool.index])) {
            valid = false;
            return;
        }
        const auto original = this->evaluation->shortage;
        resource            = shortage.kv_page_groups[kTextKvPool.index]
                                  ? runtime::ContextTransferResource{kTextKvPool}
                                  : runtime::ContextTransferResource{kBackendKvPool};
        valid               = !original.state_slots && !original.host_bytes &&
                (resource == runtime::ContextTransferResource{kTextKvPool}
                     ? original.kv_page_groups[kTextKvPool.index] != 0
                     : !original.kv_page_groups[kTextKvPool.index] &&
                           original.kv_page_groups[kBackendKvPool.index] != 0);
    }

    std::shared_ptr<const ReclaimPlan> evaluation;
    ContextResources shortage;
    runtime::ContextTransferResource resource = runtime::ContextTransferResource{kTextKvPool};
    bool valid                                = true;
    std::vector<std::shared_ptr<const DemotionPlan>> components;
    std::vector<const PageDemotion*> ordered;
    std::vector<std::uint32_t> group_ends;
    std::unordered_map<std::uint32_t, const PageDemotion*> selected;

    const context::PageDirectory& page_store() const {
        return resource == runtime::ContextTransferResource{kTextKvPool}
                   ? *evaluation->owner->text_kv_pages
                   : *evaluation->owner->backend_kv_pages;
    }

    bool append(const ContextDemotion& quote) {
        if (!valid) { return false; }
        const auto& program = *evaluation->owner;
        if (program.has_context_operation() || !quote.impl || quote.impl->owner != &program ||
            quote.impl->state || quote.impl->resource != resource || quote.impl->pages.empty() ||
            quote.sources != quote.impl->allowed ||
            (!components.empty() && components.front()->excluded != quote.impl->excluded)) {
            valid = false;
            return false;
        }
        for (const auto source : quote.sources) {
            if (!program.valid_checkpoint(source) || program.checkpoints[source.index].pins) {
                valid = false;
                return false;
            }
        }
        const auto& pages     = page_store();
        const auto& component = *quote.impl;
        std::size_t group     = 0;
        auto end              = component.page_group_ends.empty()
                                    ? static_cast<std::uint32_t>(component.pages.size())
                                    : component.page_group_ends.front();
        for (std::uint32_t i = 0; i < component.pages.size(); ++i) {
            const auto& page = component.pages[i];
            if (!pages.valid(page.handle) || !pages.device_resident(page.handle) ||
                pages.active_history_references(page.handle) || pages.source_pins(page.handle) ||
                !pages.can_pin_source(page.handle) ||
                pages.content_epoch(page.handle) != page.content_epoch ||
                pages.committed_columns(page.handle) != page.committed_columns ||
                pages.host_replica_current(page.handle) != page.host_covers_prefix) {
                valid = false;
                return false;
            }
            const auto [entry, added] =
                selected.emplace(pages.descriptor_index(page.handle), &page);
            if (!added && *entry->second != page) {
                valid = false;
                return false;
            }
            if (added) { ordered.push_back(&page); }
            if (i + 1 == end) {
                const auto count = static_cast<std::uint32_t>(ordered.size());
                if (count && (group_ends.empty() || group_ends.back() != count)) {
                    group_ends.push_back(count);
                }
                if (++group < component.page_group_ends.size()) {
                    end = component.page_group_ends[group];
                }
            }
        }
        components.push_back(quote.impl);
        return true;
    }

    bool covers(const ContextRelease& quote) const {
        if (!valid || ordered.empty() || !quote.impl || quote.impl->evaluation != evaluation ||
            quote.sources != quote.impl->sources || quote.released != quote.impl->released ||
            quote.impl->pages.empty()) {
            return false;
        }
        const auto requested = quote.impl->evaluation->shortage;
        if (requested.state_slots || requested.host_bytes ||
            (resource == runtime::ContextTransferResource{kTextKvPool}
                 ? !requested.kv_page_groups[kTextKvPool.index]
                 : requested.kv_page_groups[kTextKvPool.index] ||
                       !requested.kv_page_groups[kBackendKvPool.index])) {
            return false;
        }
        const auto& program = *evaluation->owner;
        if (program.has_context_operation()) { return false; }
        for (const auto source : quote.sources) {
            if (!program.valid_checkpoint(source) || program.checkpoints[source.index].pins) {
                return false;
            }
        }
        const auto& pages = page_store();
        return quote.impl->pages.all([&](auto page) {
            if (!pages.valid(page) || !pages.device_resident(page)) { return false; }
            const auto found = selected.find(pages.descriptor_index(page));
            return found != selected.end() && found->second->handle == page;
        });
    }

    std::optional<ContextDemotion> finish() const {
        if (!valid || ordered.empty() || evaluation->owner->has_context_operation()) {
            return std::nullopt;
        }
        const auto& program = *evaluation->owner;
        const auto needed   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? shortage.kv_page_groups[kTextKvPool.index]
                                  : shortage.kv_page_groups[kBackendKvPool.index];
        const auto stride   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? program.text_kv_pages->host_page_layout().page_stride
                                  : program.backend_kv_pages->host_page_layout().page_stride;
        const auto capacity = program.directory.available_host_pages(resource.pool->index, needed);
        std::uint32_t missing = 0;
        auto plan             = std::make_shared<DemotionPlan>();
        plan->owner           = &program;
        plan->resource        = resource;
        plan->excluded        = components.front()->excluded;
        ContextDemotion result;
        for (const auto* page : ordered) {
            const auto bytes = page->host_covers_prefix ? 0 : stride;
            if (plan->pages.size() == needed || (bytes && missing == capacity)) break;
            missing += bytes != 0;
            plan->pages.push_back(*page);
            result.host_bytes += bytes;
            plan->allowed.insert(plan->allowed.end(), page->references.begin(),
                                 page->references.end());
        }
        if (plan->pages.empty()) { return std::nullopt; }
        const auto count = static_cast<std::uint32_t>(plan->pages.size());
        for (const auto end : group_ends) {
            if (end < count) { plan->page_group_ends.push_back(end); }
        }
        plan->page_group_ends.push_back(count);
        std::sort(plan->allowed.begin(), plan->allowed.end(),
                  [](auto a, auto b) { return a.index < b.index; });
        plan->allowed.erase(std::unique(plan->allowed.begin(), plan->allowed.end()),
                            plan->allowed.end());
        result.sources                                               = plan->allowed;
        (resource == runtime::ContextTransferResource{kTextKvPool}
             ? result.released.kv_page_groups[kTextKvPool.index]
             : result.released.kv_page_groups[kBackendKvPool.index]) = count;
        result.impl                                                  = std::move(plan);
        return result;
    }
};

context::HostWriteRequirements
SessionImpl::demotion_host_requirements(const ContextDemotion& quote) const {
    if (!quote.impl || quote.impl->owner != this)
        throw std::invalid_argument("demotion quote belongs to another session");
    const auto& plan = *quote.impl;
    context::HostWriteRequirements write;
    write.kv_pages.resize(directory.pool_count());
    if (plan.state)
        write.state_images = plan.state->host_resident ? 0 : 1;
    else
        for (const auto& page : plan.pages)
            if (!page.host_covers_prefix) ++write.kv_pages[plan.resource.pool->index];
    return write;
}

runtime::ContextStart SessionImpl::start_demote(const ContextDemotion& quote,
                                                std::span<const CheckpointHandle> host_victims) {
    if (!quote.impl || quote.impl->owner != this || has_context_operation()) {
        return runtime::ContextStart::Stale;
    }
    const auto& plan = *quote.impl;
    if (quote.sources != plan.allowed) { return runtime::ContextStart::Stale; }
    for (const auto source : plan.allowed) {
        if (!valid_checkpoint(source) || checkpoints[source.index].pins ||
            std::find(plan.excluded.begin(), plan.excluded.end(), source) != plan.excluded.end()) {
            return runtime::ContextStart::Stale;
        }
    }
    // Validate actual current checkpoint holders before acquiring any destination or lease.
    // KV aliases preserve logical positions: fork/view copy a prefix in place, COW creates
    // a new handle, growth appends fresh handles, and truncation only removes a suffix.
    if (plan.state) {
        if (quote.released != ContextResources{.state_slots = 1}) {
            return runtime::ContextStart::Stale;
        }
        const auto& expected = *plan.state;
        std::size_t matched  = 0;
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || slot.value->continuation().image != expected.handle) { continue; }
            if (matched == expected.references.size() ||
                expected.references[matched++] != checkpoints.handle(index)) {
                return runtime::ContextStart::Stale;
            }
        }
        if (matched != expected.references.size() ||
            state_store->role(expected.handle) != context::StateRole::CheckpointImmutable ||
            !state_store->device_resident(expected.handle) ||
            state_store->content_epoch(expected.handle) != expected.content_epoch ||
            state_store->host_resident(expected.handle) != expected.host_resident ||
            !state_store->can_release_after_checkpoint_references(
                expected.handle, state_store->checkpoint_references(expected.handle)) ||
            quote.host_bytes != (expected.host_resident ? 0 : state_store->image_bytes())) {
            return runtime::ContextStart::Stale;
        }
    } else {
        const bool main       = plan.resource == runtime::ContextTransferResource{kTextKvPool};
        const auto* pages     = main ? text_kv_pages : backend_kv_pages;
        const auto* histories = main ? text_kv_histories : backend_kv_histories;
        if (!pages || !histories || plan.pages.empty()) { return runtime::ContextStart::Stale; }
        ContextResources released;
        (main ? released.kv_page_groups[kTextKvPool.index]
              : released.kv_page_groups[kBackendKvPool.index]) =
            static_cast<std::uint32_t>(plan.pages.size());
        if (quote.released != released) { return runtime::ContextStart::Stale; }

        struct Holder {
            CheckpointHandle checkpoint;
            context::HistoryId address;
            std::uint32_t pages;
        };

        std::vector<Holder> holders;
        holders.reserve(checkpoints.size());
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || (!main && !slot.value->history().backend)) { continue; }
            const auto& record = *slot.value;
            holders.push_back({checkpoints.handle(index),
                               main ? record.history().text : *record.history().backend,
                               context::kv_pages_for_frontier(main ? record.frontier()
                                                                   : record.backend_frontier())});
        }
        const auto stride      = main ? text_kv_pages->host_page_layout().page_stride
                                      : backend_kv_pages->host_page_layout().page_stride;
        std::size_t host_bytes = 0;
        std::vector<std::uint32_t> descriptors;
        descriptors.reserve(plan.pages.size());
        for (const auto& page : plan.pages) {
            if (!pages->valid(page.handle) || !pages->device_resident(page.handle) ||
                pages->active_history_references(page.handle) || pages->source_pins(page.handle) ||
                !pages->can_pin_source(page.handle) ||
                pages->content_epoch(page.handle) != page.content_epoch ||
                pages->committed_columns(page.handle) != page.committed_columns ||
                pages->host_replica_current(page.handle) != page.host_covers_prefix) {
                return runtime::ContextStart::Stale;
            }
            std::size_t matched = 0;
            for (const auto& holder : holders) {
                if (holder.pages <= page.position ||
                    histories->logical_page(holder.address, page.position) != page.handle) {
                    continue;
                }
                if (matched == page.references.size() ||
                    page.references[matched++] != holder.checkpoint) {
                    return runtime::ContextStart::Stale;
                }
            }
            if (matched != page.references.size()) { return runtime::ContextStart::Stale; }
            descriptors.push_back(pages->descriptor_index(page.handle));
            if (!page.host_covers_prefix) { host_bytes += stride; }
        }
        std::sort(descriptors.begin(), descriptors.end());
        if (std::adjacent_find(descriptors.begin(), descriptors.end()) != descriptors.end() ||
            host_bytes != quote.host_bytes) {
            return runtime::ContextStart::Stale;
        }
    }

    for (const auto victim : host_victims)
        if (!can_release_checkpoint(victim) ||
            std::find(plan.allowed.begin(), plan.allowed.end(), victim) != plan.allowed.end())
            return runtime::ContextStart::Stale;
    std::optional<context::HostPlacement> placement;
    if (host_victims.empty()) {
        placement = directory.plan_host(demotion_host_requirements(quote));
    } else {
        const auto releases = plan_releases(host_victims, plan.allowed, {.host_bytes = 1});
        auto selected       = releases.inventory->select();
        auto probe          = selected.probe(host_victims);
        placement           = directory.plan_host(demotion_host_requirements(quote),
                                                  releases.inventory->host_extents(selected.freed()),
                                                  releases.inventory->host_pages(selected.freed()));
    }
    if (!placement) return runtime::ContextStart::Stale;
    auto execution_reservation = executor().try_reserve(runtime::FrameKind::Context);
    if (!execution_reservation) return runtime::ContextStart::Deferred;
    for (const auto victim : host_victims)
        if (!release_checkpoint(victim))
            throw std::logic_error("prepared Host victim changed before demotion");
    auto targets = directory.host_allocation().reserve(*placement);
    if (!targets) throw std::logic_error("prepared Host placement changed before demotion");
    auto transaction = std::make_unique<PendingContext>(directory, DemoteOperation{});
    transaction->reservation.emplace(std::move(*execution_reservation));
    auto& migration = transaction->transaction;
    if (plan.state) {
        if (plan.state->host_resident) {
            if (!state_store->drop_device_replica(plan.state->handle))
                throw std::logic_error("prepared state replica could not be released");
            return runtime::ContextStart::Started;
        }
        auto transfer = state_store->reserve_device_to_host(plan.state->handle,
                                                            directory.take_host_image(*targets));
        if (!transfer) throw std::logic_error("prepared state demotion lost its destination");
        migration.add_state_copy(std::move(*transfer), false);
    } else {
        auto& pages = plan.resource == runtime::ContextTransferResource{kTextKvPool}
                          ? *text_kv_pages
                          : *backend_kv_pages;
        std::vector<context::PageId> missing;
        std::size_t begin       = 0;
        const auto append_group = [&](std::size_t end) {
            for (auto index = end; index != begin; --index) {
                const auto& page = plan.pages[index - 1];
                if (!page.host_covers_prefix) { missing.push_back(page.handle); }
            }
            begin = end;
        };
        if (plan.page_group_ends.empty()) {
            append_group(plan.pages.size());
        } else {
            for (const auto end : plan.page_group_ends) { append_group(end); }
        }
        if (!migration.add_backups(plan.resource.pool->index, std::move(missing), *targets, true))
            throw std::logic_error("prepared KV demotion lost its destination");
        // Complete Host replicas already preserve these contents. Once the entire batch is
        // reserved, release their Device copies synchronously, as for a single demotion.
        for (const auto& page : plan.pages) {
            if (page.host_covers_prefix && !pages.drop_device_replica(page.handle)) {
                throw std::logic_error("quoted redundant Device page changed during demotion");
            }
        }
        if (!migration.spill_page_groups()) return runtime::ContextStart::Started;
    }

    context_operation_ = std::move(transaction);
    try {
        (void)submit_context(*context_operation_, ContextWork{});
    } catch (...) {
        abort_context();
        throw;
    }
    return runtime::ContextStart::Started;
}

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5 {

ContextDemotionBatch::ContextDemotionBatch(std::unique_ptr<detail::DemotionBatch> impl)
    : impl_(std::move(impl)) {}

ContextDemotionBatch::ContextDemotionBatch(ContextDemotionBatch&&) noexcept            = default;
ContextDemotionBatch& ContextDemotionBatch::operator=(ContextDemotionBatch&&) noexcept = default;
ContextDemotionBatch::~ContextDemotionBatch()                                          = default;

bool ContextDemotionBatch::append(const ContextDemotion& quote) {
    return impl_ && impl_->append(quote);
}

bool ContextDemotionBatch::covers(const ContextRelease& quote) const {
    return impl_ && impl_->covers(quote);
}

std::optional<ContextDemotion> ContextDemotionBatch::finish() const {
    return impl_ ? impl_->finish() : std::nullopt;
}

ContextDemotionBatch ContextReclaimPlan::begin_kv_batch(ContextResources shortage) const {
    return ContextDemotionBatch(std::make_unique<detail::DemotionBatch>(impl_, shortage));
}

} // namespace ninfer::models::qwen3_5
