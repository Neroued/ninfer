#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/transactions/reclaim.h"

#include "core/device.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <unordered_map>

namespace ninfer::models::qwen3_5::detail {
namespace {

struct StateDemotion {
    StateImageHandle handle;
    std::uint64_t content_epoch = 0;
    bool host_resident          = false;
    std::vector<CheckpointHandle> references;

    friend bool operator==(const StateDemotion&, const StateDemotion&) = default;
};

struct PageDemotion {
    context::LogicalKVPageHandle handle;
    std::uint32_t position          = 0;
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
    bool host_covers_prefix         = false;
    std::vector<CheckpointHandle> references;

    friend bool operator==(const PageDemotion&, const PageDemotion&) = default;
};

bool checkpoint_prefix(const CheckpointState& earlier, const CheckpointState& later) {
    const auto frontier = earlier.frontier();
    if (frontier > later.frontier() || earlier.key.identity_tag != later.key.identity_tag ||
        earlier.snapshot.identity().digests.at(frontier) !=
            later.snapshot.identity().digests.at(frontier)) {
        return false;
    }
    if (earlier.snapshot.identity_owner() == later.snapshot.identity_owner()) { return true; }
    return std::equal(earlier.snapshot.identity().ledger.begin(),
                      earlier.snapshot.identity().ledger.begin() + frontier,
                      later.snapshot.identity().ledger.begin()) &&
           earlier.snapshot.identity().prefix_identity.prefix_equals(
               later.snapshot.identity().prefix_identity, frontier);
}

struct PhysicalPage {
    context::LogicalKVPageHandle handle;
    std::uint32_t position = 0;
};

struct PhysicalOwners {
    std::vector<CheckpointHandle> sources;
    std::vector<StateImageHandle> states;
    std::vector<PhysicalPage> main;
    std::vector<PhysicalPage> backend;
};

constexpr auto kNoPhysicalPage = std::numeric_limits<std::uint32_t>::max();

struct PhysicalPoolFacts {
    struct History {
        context::KVHistoryHandle address;
        std::vector<std::uint32_t> checkpoints;
    };

    const context::KVHistoryStore* history_store = nullptr;
    const context::LogicalKVPageStore* store     = nullptr;
    std::vector<PhysicalPage> pages;
    std::vector<std::uint32_t> descriptors;
    std::vector<History> histories;

    std::uint32_t page(context::LogicalKVPageHandle handle) {
        if (descriptors.empty()) descriptors.assign(store->capacity(), kNoPhysicalPage);
        auto& index = descriptors[store->descriptor_index(handle)];
        if (index == kNoPhysicalPage) {
            index = pages.size();
            pages.push_back({handle});
        }
        return index;
    }
};

// A read-only inventory of actual prefix holders. It does not retain content or infer
// ownership from copies of C++ pointers; the stores answer release queries directly.
struct PhysicalFacts {
    PhysicalFacts(const ProgramImpl& program, ContextResources shortage)
        : host_kv(program.host_kv_extents && program.host_kv_extents->occupied()) {
        const auto prepare = [&](PhysicalPoolFacts& pool, bool is_main) {
            pool.history_store =
                is_main ? program.text_kv_histories.get() : program.backend_kv_histories.get();
            pool.store = is_main ? program.text_kv_pages.get() : program.backend_kv_pages.get();
            if (!pool.history_store) return;
            for (std::uint32_t index = 0; index < program.checkpoints.size(); ++index) {
                const auto& slot = program.checkpoints[index];
                if (!slot.value) continue;
                const auto* prefix =
                    is_main ? &slot.value->text_prefix() : slot.value->backend_prefix();
                if (!prefix) continue;
                const auto handle = prefix->history();
                auto found =
                    std::find_if(pool.histories.begin(), pool.histories.end(),
                                 [&](const auto& history) { return history.address == handle; });
                if (found == pool.histories.end()) {
                    pool.histories.push_back({handle, {index}});
                } else
                    found->checkpoints.push_back(index);
            }
        };
        if (shortage.host_bytes ||
            (!shortage.state_slots && shortage.kv_page_groups[kTextKvPool.index]))
            prepare(main, true);
        if (shortage.host_bytes ||
            (!shortage.state_slots && !shortage.kv_page_groups[kTextKvPool.index] &&
             shortage.kv_page_groups[kBackendKvPool.index]))
            prepare(backend, false);
    }

    bool host_kv;
    PhysicalPoolFacts main, backend;
};

std::vector<PhysicalOwners> physical_owner_sets(const ProgramImpl& program, PhysicalFacts& facts,
                                                std::span<const CheckpointHandle> allowed,
                                                std::span<const CheckpointHandle> excluded,
                                                ContextResources shortage) {
    std::vector<bool> eligible(program.checkpoints.size(), false);
    for (const auto handle : allowed) {
        if (program.valid_checkpoint(handle) && !program.checkpoints[handle.index].pins &&
            std::find(excluded.begin(), excluded.end(), handle) == excluded.end()) {
            eligible[handle.index] = true;
        }
    }
    std::map<std::vector<std::uint32_t>, PhysicalOwners> unique;
    for (std::uint32_t index = 0; index < eligible.size(); ++index) {
        if (eligible[index]) { unique.try_emplace(std::vector{index}); }
    }
    const auto permitted = [&](const auto& references) {
        return std::all_of(references.begin(), references.end(),
                           [&](auto index) { return eligible[index]; });
    };
    const bool host = shortage.host_bytes != 0;
    if (host || shortage.state_slots) {
        struct StateOwners {
            StateImageHandle handle;
            std::vector<std::uint32_t> checkpoints;
        };

        std::vector<StateOwners> states(program.state_store->capacity());
        for (std::uint32_t index = 0; index < program.checkpoints.size(); ++index) {
            const auto& slot = program.checkpoints[index];
            if (!slot.value) { continue; }
            const auto state = slot.value->continuation().image;
            if (host ? program.state_store->host_resident(state)
                     : program.state_store->device_resident(state)) {
                auto& owners  = states[program.state_store->descriptor_index(state)];
                owners.handle = state;
                owners.checkpoints.push_back(index);
            }
        }
        for (const auto& state : states) {
            if (!state.checkpoints.empty() && permitted(state.checkpoints)) {
                unique[state.checkpoints].states.push_back(state.handle);
            }
        }
    }
    const auto collect_pages = [&](bool main_pool) {
        auto& pool = main_pool ? facts.main : facts.backend;
        if (!pool.store || !pool.history_store || (host && !facts.host_kv)) { return; }
        struct PageOwners {
            std::uint32_t holders  = kNoPhysicalPage;
            std::uint32_t position = 0, last_owner = 0;
        };
        std::vector<PageOwners> owners;
        owners.reserve(pool.store->occupied());
        std::vector<std::vector<std::uint32_t>> holder_sets;
        std::unordered_map<std::uint64_t, std::uint32_t> merged_sets;
        std::vector<std::uint32_t> boundaries;
        for (std::size_t index = 0; index < pool.histories.size(); ++index) {
            const auto& history = pool.histories[index];
            const auto address  = pool.histories[index].address;
            if (!address.valid()) { continue; }
            boundaries.clear();
            boundaries.push_back(0);
            for (const auto owner : history.checkpoints) {
                const auto& record = *program.checkpoints[owner].value;
                boundaries.push_back(context::kv_pages_for_frontier(
                    main_pool ? record.frontier() : record.backend_frontier()));
            }
            std::sort(boundaries.begin(), boundaries.end());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
            for (std::size_t interval = 1; interval < boundaries.size(); ++interval) {
                const auto begin = boundaries[interval - 1], end = boundaries[interval];
                std::vector<std::uint32_t> holders;
                for (const auto owner : history.checkpoints) {
                    const auto& record = *program.checkpoints[owner].value;
                    if (context::kv_pages_for_frontier(
                            main_pool ? record.frontier() : record.backend_frontier()) > begin) {
                        holders.push_back(owner);
                    }
                }
                const auto last_owner   = holders.back();
                const auto holder_index = static_cast<std::uint32_t>(holder_sets.size());
                holder_sets.push_back(std::move(holders));
                for (auto position = begin; position < end; ++position) {
                    const auto handle = pool.history_store->logical_page(address, position);
                    if (!(host ? pool.store->host_resident(handle)
                               : pool.store->device_resident(handle))) {
                        continue;
                    }
                    const auto page_index = pool.page(handle);
                    if (owners.size() <= page_index) { owners.resize(page_index + 1); }
                    auto& entry = owners[page_index];
                    if (entry.holders == kNoPhysicalPage) {
                        entry.holders = holder_index;
                    } else {
                        const auto key =
                            (static_cast<std::uint64_t>(entry.holders) << 32U) | holder_index;
                        auto [merged, added] = merged_sets.try_emplace(
                            key, static_cast<std::uint32_t>(holder_sets.size()));
                        if (added) {
                            const auto& left  = holder_sets[entry.holders];
                            const auto& right = holder_sets[holder_index];
                            std::vector<std::uint32_t> joined;
                            joined.reserve(left.size() + right.size());
                            std::set_union(left.begin(), left.end(), right.begin(), right.end(),
                                           std::back_inserter(joined));
                            holder_sets.push_back(std::move(joined));
                        }
                        entry.holders = merged->second;
                    }
                    if (last_owner >= entry.last_owner) {
                        entry.last_owner = last_owner;
                        entry.position   = position;
                    }
                }
            }
        }
        for (const auto page_index : pool.descriptors) {
            if (page_index == kNoPhysicalPage) { continue; }
            const auto& entry = owners[page_index];
            if (entry.holders == kNoPhysicalPage) { continue; }
            const auto& holders = holder_sets[entry.holders];
            if (permitted(holders)) {
                auto& group = unique[holders];
                (main_pool ? group.main : group.backend)
                    .push_back({pool.pages[page_index].handle, entry.position});
            }
        }
    };
    if (host || (!shortage.state_slots && shortage.kv_page_groups[kTextKvPool.index])) {
        collect_pages(true);
    }
    if (host || (!shortage.state_slots && !shortage.kv_page_groups[kTextKvPool.index] &&
                 shortage.kv_page_groups[kBackendKvPool.index])) {
        collect_pages(false);
    }
    std::vector<PhysicalOwners> result;
    result.reserve(unique.size());
    for (auto& [indices, owners] : unique) {
        owners.sources.reserve(indices.size());
        for (const auto index : indices) {
            owners.sources.push_back(program.checkpoints.handle(index));
        }
        const auto deeper_first = [](const auto& left, const auto& right) {
            return left.position > right.position;
        };
        std::sort(owners.main.begin(), owners.main.end(), deeper_first);
        std::sort(owners.backend.begin(), owners.backend.end(), deeper_first);
        result.push_back(std::move(owners));
    }
    return result;
}

struct ReleasedPages {
    std::vector<context::LogicalKVPageHandle> pages;

    bool empty() const noexcept { return pages.empty(); }

    template <typename Predicate>
    bool all(Predicate&& predicate) const {
        return std::all_of(pages.begin(), pages.end(), predicate);
    }
};

struct ReleaseScratch {
    std::vector<const context::KVPrefixRef*> main, backend;
    std::vector<CheckpointHandle> records;

    struct Image {
        StateImageHandle handle;
        std::uint32_t count = 0;
    };

    std::vector<Image> images;
};

std::optional<ContextResources> released_resources(const ProgramImpl& program,
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
            result.host_bytes += program.state_images->storage().image_bytes();
        if (shortage.state_slots && program.state_store->device_resident(image.handle))
            ++result.state_slots;
    }
    const auto count = [&](const context::KVHistoryStore& histories,
                           const context::LogicalKVPageStore& pages, const auto& removed, bool main,
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
          program.text_host_kv_page_stride);
    if (program.backend_kv_histories)
        count(*program.backend_kv_histories, *program.backend_kv_pages, scratch.backend, false,
              program.backend_host_kv_page_stride);
    return result;
}

} // namespace

// A quote carries content identities and complete optional holders. It holds no execution
// lease: start_demote revalidates it before acquiring real destinations.
struct DemotionPlan {
    const ProgramImpl* owner                  = nullptr;
    runtime::ContextTransferResource resource = runtime::ContextTransferResource{};
    std::vector<CheckpointHandle> allowed;
    std::vector<CheckpointHandle> excluded;
    std::optional<StateDemotion> state;
    std::vector<PageDemotion> pages;
    // Each range retains one component's deep-first selection order. Transfers traverse
    // each range forward in logical position without reversing the component ordering.
    std::vector<std::uint32_t> page_group_ends;
};

std::uint32_t ProgramImpl::checkpoint_recovery_frontier(CheckpointHandle retained,
                                                        const RequestBasePlan& base,
                                                        std::uint32_t target) const {
    if (!valid_checkpoint(retained) || checkpoint(retained).frontier() > target ||
        !checkpoint_matches(retained, base)) {
        return 0;
    }
    return checkpoint(retained).frontier();
}

struct ReclaimPlan {
    ReclaimPlan(const ProgramImpl& program, ContextResources requested)
        : owner(&program), shortage(requested), facts(program, requested) {}

    const ProgramImpl* owner;
    ContextResources shortage;
    PhysicalFacts facts;
    mutable std::unordered_map<std::uint64_t, bool> prefixes;

    bool prefix(CheckpointHandle earlier, CheckpointHandle later) const {
        const auto& a = owner->checkpoint(earlier);
        const auto& b = owner->checkpoint(later);
        if (a.frontier() > b.frontier() || a.key.identity_tag != b.key.identity_tag ||
            a.snapshot.identity().digests.at(a.frontier()) !=
                b.snapshot.identity().digests.at(a.frontier())) {
            return false;
        }
        if (a.snapshot.identity_owner() == b.snapshot.identity_owner()) { return true; }
        const auto key   = (static_cast<std::uint64_t>(earlier.index) << 32U) | later.index;
        const auto found = prefixes.find(key);
        if (found != prefixes.end()) { return found->second; }
        const bool matches = checkpoint_prefix(a, b);
        prefixes.emplace(key, matches);
        return matches;
    }
};

struct ReleasePlan {
    std::shared_ptr<const ReclaimPlan> evaluation;
    std::vector<CheckpointHandle> sources;
    ContextResources released;
    ReleasedPages pages;
};

std::uint64_t
ProgramImpl::checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                                      std::span<const CheckpointHandle> surviving) const {
    return recovery_loss(*this, removed, surviving, [&](auto a, auto b) {
        return checkpoint_prefix(checkpoint(a), checkpoint(b));
    });
}

std::optional<ContextDemotion> quote_demotion(const ProgramImpl& program,
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
            if (program.state_store->role(state) != context::StateImageRole::CheckpointImmutable ||
                !program.state_store->can_release_after_checkpoint_references(
                    state, program.state_store->checkpoint_references(state))) {
                continue;
            }
            const auto host = program.state_store->host_resident(state);
            plan->state     = StateDemotion{state, program.state_store->content_epoch(state), host,
                                        owners.sources};
            candidate.released.state_slots = 1;
            candidate.host_bytes = host ? 0 : program.state_images->storage().image_bytes();
            break;
        }
        if (!plan->state) { return std::nullopt; }
    } else {
        const auto& pages   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? *program.text_kv_pages
                                  : *program.backend_kv_pages;
        const auto& objects = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? owners.main
                                  : owners.backend;
        const auto needed   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? shortage.kv_page_groups[kTextKvPool.index]
                                  : shortage.kv_page_groups[kBackendKvPool.index];
        const auto stride   = resource == runtime::ContextTransferResource{kTextKvPool}
                                  ? program.text_host_kv_page_stride
                                  : program.backend_host_kv_page_stride;
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

std::optional<ContextRelease> quote_release(const std::shared_ptr<ReclaimPlan>& evaluation,
                                            std::span<const CheckpointHandle> sources,
                                            ReleaseScratch& scratch) {
    auto plan = std::make_shared<ReleasePlan>();
    const auto released =
        released_resources(*evaluation->owner, sources, evaluation->shortage, scratch, plan->pages);
    if (!released) { return std::nullopt; }
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
ProgramImpl::checkpoint_release_resources(std::span<const CheckpointHandle> handles,
                                          ContextResources shortage) const {
    ReleaseScratch scratch;
    ReleasedPages pages;
    return released_resources(*this, handles, shortage, scratch, pages);
}

ContextReclaimPlan ProgramImpl::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                             std::span<const CheckpointHandle> excluded,
                                             ContextResources shortage) const {
    ContextReclaimPlan result;
    if (context_operation_) { return result; }
    result.impl_ = std::make_shared<ReclaimPlan>(*this, shortage);
    const auto owners =
        physical_owner_sets(*this, result.impl_->facts, allowed, excluded, shortage);
    ReleaseScratch scratch;
    for (const auto& group : owners) {
        if (auto quote = quote_demotion(*this, group, excluded, shortage)) {
            result.demotions.push_back(std::move(*quote));
        }
        if (auto quote = quote_release(result.impl_, group.sources, scratch)) {
            result.releases.push_back(std::move(*quote));
        }
    }
    return result;
}

std::vector<ContextRelease> ProgramImpl::plan_releases(std::span<const CheckpointHandle> allowed,
                                                       std::span<const CheckpointHandle> excluded,
                                                       ContextResources shortage) const {
    if (context_operation_) { return {}; }
    auto evaluation   = std::make_shared<ReclaimPlan>(*this, shortage);
    const auto owners = physical_owner_sets(*this, evaluation->facts, allowed, excluded, shortage);
    ReleaseScratch scratch;
    std::vector<ContextRelease> result;
    for (const auto& group : owners) {
        if (auto quote = quote_release(evaluation, group.sources, scratch)) {
            result.push_back(std::move(*quote));
        }
    }
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

    const context::LogicalKVPageStore& page_store() const {
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
        const auto& program  = *evaluation->owner;
        const auto needed    = resource == runtime::ContextTransferResource{kTextKvPool}
                                   ? shortage.kv_page_groups[kTextKvPool.index]
                                   : shortage.kv_page_groups[kBackendKvPool.index];
        const auto stride    = resource == runtime::ContextTransferResource{kTextKvPool}
                                   ? program.text_host_kv_page_stride
                                   : program.backend_host_kv_page_stride;
        const auto usage     = program.physical_usage();
        const auto available = usage.capacity.host_bytes - usage.occupied.host_bytes;
        auto plan            = std::make_shared<DemotionPlan>();
        plan->owner          = &program;
        plan->resource       = resource;
        plan->excluded       = components.front()->excluded;
        ContextDemotion result;
        for (const auto* page : ordered) {
            const auto bytes = page->host_covers_prefix ? 0 : stride;
            if (plan->pages.size() == needed || bytes > available - result.host_bytes) { break; }
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

bool ProgramImpl::start_demote(const ContextDemotion& quote) {
    if (!quote.impl || quote.impl->owner != this || context_operation_) { return false; }
    const auto& plan = *quote.impl;
    if (quote.sources != plan.allowed) { return false; }
    for (const auto source : plan.allowed) {
        if (!valid_checkpoint(source) || checkpoints[source.index].pins ||
            std::find(plan.excluded.begin(), plan.excluded.end(), source) != plan.excluded.end()) {
            return false;
        }
    }
    // Validate actual current checkpoint holders before acquiring any destination or lease.
    // KV aliases preserve logical positions: fork/view copy a prefix in place, COW creates
    // a new handle, growth appends fresh handles, and truncation only removes a suffix.
    if (plan.state) {
        if (quote.released != ContextResources{.state_slots = 1}) { return false; }
        const auto& expected = *plan.state;
        std::size_t matched  = 0;
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || slot.value->continuation().image != expected.handle) { continue; }
            if (matched == expected.references.size() ||
                expected.references[matched++] != checkpoints.handle(index)) {
                return false;
            }
        }
        if (matched != expected.references.size() ||
            state_store->role(expected.handle) != context::StateImageRole::CheckpointImmutable ||
            !state_store->device_resident(expected.handle) ||
            state_store->content_epoch(expected.handle) != expected.content_epoch ||
            state_store->host_resident(expected.handle) != expected.host_resident ||
            !state_store->can_release_after_checkpoint_references(
                expected.handle, state_store->checkpoint_references(expected.handle)) ||
            quote.host_bytes !=
                (expected.host_resident ? 0 : state_images->storage().image_bytes())) {
            return false;
        }
    } else {
        const bool main       = plan.resource == runtime::ContextTransferResource{kTextKvPool};
        const auto* pages     = main ? text_kv_pages.get() : backend_kv_pages.get();
        const auto* histories = main ? text_kv_histories.get() : backend_kv_histories.get();
        if (!pages || !histories || plan.pages.empty()) { return false; }
        ContextResources released;
        (main ? released.kv_page_groups[kTextKvPool.index]
              : released.kv_page_groups[kBackendKvPool.index]) =
            static_cast<std::uint32_t>(plan.pages.size());
        if (quote.released != released) { return false; }

        struct Holder {
            CheckpointHandle checkpoint;
            context::KVHistoryHandle address;
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
        const auto stride      = main ? text_host_kv_page_stride : backend_host_kv_page_stride;
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
                return false;
            }
            std::size_t matched = 0;
            for (const auto& holder : holders) {
                if (holder.pages <= page.position ||
                    histories->logical_page(holder.address, page.position) != page.handle) {
                    continue;
                }
                if (matched == page.references.size() ||
                    page.references[matched++] != holder.checkpoint) {
                    return false;
                }
            }
            if (matched != page.references.size()) { return false; }
            descriptors.push_back(pages->descriptor_index(page.handle));
            if (!page.host_covers_prefix) { host_bytes += stride; }
        }
        std::sort(descriptors.begin(), descriptors.end());
        if (std::adjacent_find(descriptors.begin(), descriptors.end()) != descriptors.end() ||
            host_bytes != quote.host_bytes) {
            return false;
        }
    }

    ContextOperation transaction(*migration_context_, host_kv_extents.get(), DemoteOperation{});
    auto& migration = transaction.migration;
    if (plan.state) {
        if (plan.state->host_resident) {
            return state_store->drop_device_replica(plan.state->handle);
        }
        auto transfer = state_store->reserve_device_to_host(plan.state->handle);
        if (!transfer) { return false; }
        migration.add_state_copy(std::move(*transfer), false);
    } else {
        auto& pages = plan.resource == runtime::ContextTransferResource{kTextKvPool}
                          ? *text_kv_pages
                          : *backend_kv_pages;
        std::vector<context::LogicalKVPageHandle> missing;
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
        if (!missing.empty() && !host_kv_extents) { return false; }
        std::size_t offset = 0;
        while (offset < missing.size()) {
            auto count = missing.size() - offset;
            std::optional<context::HostKVExtentReservation> destination;
            while (count) {
                auto reservation =
                    host_kv_extents->prepare(pages, std::span(missing).subspan(offset, count));
                if (reservation) {
                    destination.emplace(std::move(*reservation));
                    break;
                }
                count /= 2;
            }
            // All destinations and source leases belong to this local transaction until
            // every component is ready. A failed reservation unwinds without dropping pages.
            if (!destination) { return false; }
            migration.add_backup(*plan.resource.pool, pages,
                                 std::vector<context::LogicalKVPageHandle>(
                                     missing.begin() + offset, missing.begin() + offset + count),
                                 std::move(*destination), true);
            offset += count;
        }
        // Complete Host replicas already preserve these contents. Once the entire batch is
        // reserved, release their Device copies synchronously, as for a single demotion.
        for (const auto& page : plan.pages) {
            if (page.host_covers_prefix && !pages.drop_device_replica(page.handle)) {
                throw std::logic_error("quoted redundant Device page changed during demotion");
            }
        }
        if (!migration.spill_page_groups()) return true;
    }

    context_operation_.emplace(std::move(transaction));
    try {
        context_operation_->migration.submit();
    } catch (...) {
        abort_context();
        throw;
    }
    return true;
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

std::uint64_t ContextReclaimPlan::recovery_loss(std::span<const CheckpointHandle> removed,
                                                std::span<const CheckpointHandle> surviving) const {
    if (!impl_) { throw std::logic_error("recovery query has no Native evaluation"); }
    return detail::recovery_loss(*impl_->owner, removed, surviving,
                                 [&](auto a, auto b) { return impl_->prefix(a, b); });
}

} // namespace ninfer::models::qwen3_5
