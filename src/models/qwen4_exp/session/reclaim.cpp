#include "models/qwen4_exp/session/internal.h"

#include "models/common/context/physical_reclaim.h"
#include "models/common/qwen/recovery_store.h"

#include <algorithm>
#include <unordered_set>

namespace ninfer::models::qwen4_exp::detail {
namespace {
template <class T>
void append_unique(std::vector<T>& out, const T& value) {
    if (std::find(out.begin(), out.end(), value) == out.end()) out.push_back(value);
}

context::PhysicalReclaim<ContextResources>
physical_facts(const SessionImpl& program, std::span<const CheckpointHandle> allowed,
               std::span<const CheckpointHandle> excluded, ContextResources shortage) {
    std::array<context::ReclaimPool, ContextResources::pool_capacity> pools;
    for (std::uint16_t i = 0; i < program.directory.pool_count(); ++i) {
        const auto& pool = program.directory.pool(i);
        pools[i]         = {&pool.histories(), &pool.pages(), program.pools[i].host_page_stride};
    }
    return {program.checkpoints,
            program.directory.states(),
            program.directory.states().image_bytes(),
            pools,
            allowed,
            excluded,
            shortage};
}

void seal(Demotion& plan) {
    const auto& p            = *plan.owner;
    plan.checkpoint_revision = p.checkpoints.revision();
    for (const auto image : plan.images)
        plan.image_epochs.push_back(p.directory.states().content_epoch(image));
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
        for (const auto page : plan.pages[pool]) {
            const auto& store = p.directory.pool(pool).pages();
            plan.page_versions[pool].push_back(
                {store.content_epoch(page), store.committed_columns(page)});
        }
}

ContextRelease
quote_release(const SessionImpl& p, const context::PhysicalReclaim<ContextResources>& facts,
              std::span<const CheckpointHandle> sources,
              const context::ReleaseInventory<ContextResources>::Selection& selection,
              ContextResources shortage) {
    auto proof   = std::make_shared<Release>();
    proof->owner = &p;
    for (const auto group : selection.freed())
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
            if (shortage.kv_page_groups[pool])
                for (const auto& page : facts.groups[group].pages[pool])
                    if (page.releasable &&
                        p.directory.pool(pool).pages().device_resident(page.handle))
                        proof->pages[pool].push_back(page.handle);
    return {{sources.begin(), sources.end()}, selection.released(), std::move(proof)};
}

bool valid_demotion(const SessionImpl& p, const Demotion& plan, bool refresh_holders = true) {
    if (!refresh_holders && plan.checkpoint_revision != p.checkpoints.revision()) return false;
    for (const auto h : plan.records)
        if (!p.checkpoints.erasable(h)) return false;
    ContextResources needed{.state_slots = static_cast<std::uint32_t>(plan.images.size())};
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
        needed.kv_page_groups[pool] = plan.pages[pool].size();
    std::vector<bool> images;
    std::array<std::vector<bool>, 2> pages;
    if (refresh_holders) {
        const auto facts = physical_facts(p, plan.records, {}, needed);
        images.resize(p.directory.states().capacity());
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
            pages[pool].resize(p.directory.pool(pool).pages().capacity());
        for (const auto& group : facts.groups) {
            for (const auto image : group.states)
                images[p.directory.states().descriptor_index(image)] = true;
            for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
                for (const auto& page : group.pages[pool])
                    pages[pool][p.directory.pool(pool).pages().descriptor_index(page.handle)] =
                        true;
        }
    }
    const auto& states = p.directory.states();
    if (plan.images.size() != plan.image_epochs.size()) return false;
    for (std::size_t i = 0; i < plan.images.size(); ++i) {
        const auto image = plan.images[i];
        if (!states.valid(image) || (refresh_holders && !images[states.descriptor_index(image)]) ||
            !states.device_resident(image) || states.content_epoch(image) != plan.image_epochs[i] ||
            states.role(image) != context::StateRole::CheckpointImmutable ||
            !states.can_release_after_checkpoint_references(image,
                                                            states.checkpoint_references(image)))
            return false;
    }
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
        const auto& store = p.directory.pool(pool).pages();
        if (plan.pages[pool].size() != plan.page_versions[pool].size()) return false;
        for (std::size_t i = 0; i < plan.pages[pool].size(); ++i) {
            const auto page    = plan.pages[pool][i];
            const auto version = plan.page_versions[pool][i];
            if (!store.valid(page) ||
                (refresh_holders && !pages[pool][store.descriptor_index(page)]) ||
                !store.device_resident(page) || !store.can_pin_source(page) ||
                store.source_pins(page) || store.active_history_references(page) ||
                store.content_epoch(page) != version.epoch ||
                store.committed_columns(page) != version.columns)
                return false;
        }
    }
    return true;
}
} // namespace

std::vector<Image> SessionImpl::checkpoint_images(std::span<const CheckpointHandle> records) const {
    std::vector<Image> out;
    std::unordered_set<std::uint32_t> seen;
    for (const auto h : records)
        if (checkpoints.valid(h)) {
            const auto image = checkpoints.at(h).snapshot.continuation().image;
            if (seen.insert(directory.states().descriptor_index(image)).second)
                out.push_back(image);
        }
    return out;
}

std::vector<context::PageId>
SessionImpl::checkpoint_pages(std::span<const CheckpointHandle> records, std::uint32_t pool) const {
    std::vector<context::PageId> out;
    std::unordered_set<std::uint32_t> seen;
    for (const auto h : records)
        if (checkpoints.valid(h)) {
            const auto& prefix = checkpoints.at(h).snapshot.prefix(pool);
            for (std::uint32_t i = 0; i < (prefix.frontier() + 63) / 64; ++i) {
                const auto page =
                    directory.pool(pool).histories().logical_page(prefix.history(), i);
                if (seen.insert(directory.pool(pool).pages().descriptor_index(page)).second)
                    out.push_back(page);
            }
        }
    return out;
}

ContextResources SessionImpl::footprint(std::span<const CheckpointHandle> input,
                                        bool released) const {
    if (released &&
        std::any_of(input.begin(), input.end(), [&](auto h) { return !checkpoints.erasable(h); }))
        return {};
    std::vector<CheckpointHandle> records;
    for (const auto h : input)
        if (checkpoints.valid(h)) append_unique(records, h);
    ContextResources out;
    const auto& states = directory.states();
    for (const auto image : checkpoint_images(records)) {
        const auto references = std::count_if(records.begin(), records.end(), [&](auto h) {
            return checkpoints.at(h).snapshot.continuation().image == image;
        });
        if (released && (references != states.checkpoint_references(image) ||
                         !states.can_release_after_checkpoint_references(image, references)))
            continue;
        out.state_slots += states.device_resident(image);
        if (states.host_resident(image)) out.host_bytes += directory.states().image_bytes();
    }
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        const auto& storage = directory.pool(pool);
        std::vector<context::PageId> pages;
        if (released) {
            std::vector<const context::HistoryPrefix*> prefixes;
            for (const auto h : records)
                prefixes.push_back(&checkpoints.at(h).snapshot.prefix(pool));
            pages = storage.histories().released_pages(prefixes);
        } else
            pages = checkpoint_pages(records, pool);
        for (const auto page : pages) {
            if (released && storage.pages().source_pins(page)) continue;
            out.kv_page_groups[pool] += storage.pages().device_resident(page);
            if (storage.pages().host_resident(page)) out.host_bytes += pools[pool].host_page_stride;
        }
    }
    return out;
}

std::size_t SessionImpl::host_released(std::span<const CheckpointHandle> records) const {
    return footprint(records, true).host_bytes;
}

std::shared_ptr<Demotion> SessionImpl::demotion(std::span<const CheckpointHandle> records,
                                                ContextResources shortage) const {
    if (!directory.host_allocation().capacity_bytes()) return {};
    auto out   = std::make_shared<Demotion>();
    out->owner = this;
    for (const auto h : records)
        if (checkpoints.valid(h)) append_unique(out->records, h);
    if (shortage.state_slots)
        for (const auto image : checkpoint_images(out->records))
            if (directory.states().device_resident(image) && !directory.states().source_pins(image))
                out->images.push_back(image);
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool)
        if (shortage.kv_page_groups[pool]) {
            const auto& pages = directory.pool(pool).pages();
            for (const auto page : checkpoint_pages(out->records, pool))
                if (pages.device_resident(page) && !pages.source_pins(page) &&
                    !pages.active_history_references(page))
                    out->pages[pool].push_back(page);
        }
    seal(*out);
    return out->empty() ? nullptr : out;
}

ContextDemotion SessionImpl::describe_demotion(std::shared_ptr<const Demotion> plan) const {
    ContextDemotion out;
    out.sources              = plan->records;
    out.released.state_slots = plan->images.size();
    for (const auto image : plan->images)
        if (!directory.states().host_resident(image))
            out.host_bytes += directory.states().image_bytes();
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        out.released.kv_page_groups[pool] = plan->pages[pool].size();
        for (const auto page : plan->pages[pool])
            if (!directory.pool(pool).pages().host_replica_current(page))
                out.host_bytes += pools[pool].host_page_stride;
    }
    out.impl = std::move(plan);
    return out;
}

bool SessionImpl::prepare_demotion(context::ContextTransaction& migration, const Demotion& plan) {
    std::vector<Image> redundant;
    for (const auto image : plan.images) {
        auto& states = directory.states();
        if (!states.valid(image) || !states.device_resident(image) || states.source_pins(image))
            return false;
        if (states.host_resident(image))
            redundant.push_back(image);
        else {
            auto transfer = states.reserve_device_to_host(image);
            if (!transfer) return false;
            migration.add_state_copy(std::move(*transfer), false);
        }
    }
    std::array<std::vector<context::PageId>, 2> redundant_pages;
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        auto& store = directory.pool(pool).pages();
        std::vector<context::PageId> pages;
        for (auto page : plan.pages[pool]) {
            if (!store.valid(page) || !store.device_resident(page) || store.source_pins(page) ||
                store.active_history_references(page))
                return false;
            if (store.host_replica_current(page))
                redundant_pages[pool].push_back(page);
            else
                pages.push_back(page);
        }
        if (!pages.empty()) {
            auto target = directory.host().prepare(store, pages);
            if (!target) return false;
            migration.add_backup(pool, std::move(pages), std::move(*target), true);
        }
    }
    for (const auto image : redundant)
        if (!directory.states().drop_device_replica(image))
            throw std::logic_error("state demotion lost exclusivity");
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool)
        for (const auto page : redundant_pages[pool])
            if (!directory.pool(pool).pages().drop_device_replica(page))
                throw std::logic_error("KV demotion lost exclusivity");
    return true;
}

struct DemotionBatch {
    const SessionImpl* owner;
    ContextResources shortage;
    std::optional<runtime::KvPoolId> pool;
    bool valid;
    std::vector<std::shared_ptr<const Demotion>> components;
    std::unordered_set<std::uint32_t> pages;
};

context::ReleasePlan<ContextRelease, ContextResources>
SessionImpl::plan_releases(std::span<const CheckpointHandle> allowed,
                           std::span<const CheckpointHandle> excluded,
                           ContextResources shortage) const {
    auto facts = detail::physical_facts(*this, allowed, excluded, shortage);
    context::ReleasePlan<ContextRelease, ContextResources> out{{}, facts.inventory};
    if (this->operation) return out;
    auto selection = facts.inventory->select();
    for (const auto& group : facts.groups) {
        auto probe = selection.probe(group.sources);
        if (selection.released().useful(shortage))
            out.releases.push_back(
                detail::quote_release(*this, facts, group.sources, selection, shortage));
    }
    return out;
}

ContextReclaimPlan SessionImpl::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                             std::span<const CheckpointHandle> excluded,
                                             ContextResources shortage) const {
    auto& p = *this;
    ContextReclaimPlan out;
    out.impl = std::make_shared<detail::Reclaim>(detail::Reclaim{&p, shortage});
    if (p.operation) return out;
    auto facts     = detail::physical_facts(p, allowed, excluded, shortage);
    auto selection = facts.inventory->select();
    for (const auto& group : facts.groups) {
        auto probe = selection.probe(group.sources);
        if (selection.released().useful(shortage))
            out.releases.push_back(
                detail::quote_release(p, facts, group.sources, selection, shortage));
        if (shortage.host_bytes || !p.directory.host_allocation().capacity_bytes()) continue;
        auto plan     = std::make_shared<detail::Demotion>();
        plan->owner   = &p;
        plan->records = group.sources;
        if (shortage.state_slots) {
            for (const auto image : group.states)
                if (p.directory.states().role(image) == context::StateRole::CheckpointImmutable &&
                    p.directory.states().can_release_after_checkpoint_references(
                        image, p.directory.states().checkpoint_references(image))) {
                    plan->images.push_back(image);
                    break;
                }
        } else if (const auto pool = shortage.first_kv_deficit()) {
            const auto& pages = p.directory.pool(pool->index).pages();
            for (const auto& page : group.pages[pool->index]) {
                if (!pages.device_resident(page.handle) || pages.source_pins(page.handle) ||
                    pages.active_history_references(page.handle) ||
                    !pages.can_pin_source(page.handle))
                    continue;
                plan->pages[pool->index].push_back(page.handle);
                if (plan->pages[pool->index].size() == shortage.kv_page_groups[pool->index]) break;
            }
        }
        detail::seal(*plan);
        if (!plan->empty()) out.demotions.push_back(p.describe_demotion(std::move(plan)));
    }
    return out;
}

bool SessionImpl::start_demote(const ContextDemotion& quote) {
    auto& p = *this;
    if (p.operation || !quote.impl || quote.impl->owner != &p ||
        !p.directory.host_allocation().capacity_bytes() || quote.sources != quote.impl->records ||
        !detail::valid_demotion(p, *quote.impl) ||
        quote.host_bytes > p.directory.host_allocation().free_bytes())
        return false;
    const auto current = p.describe_demotion(quote.impl);
    if (current.host_bytes != quote.host_bytes || current.released != quote.released) return false;
    auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    p.operation = std::make_unique<detail::ContextOperation>(
        p.directory, detail::DemoteOperation{quote.impl}, std::move(*execution));
    try {
        auto& migration = p.operation->migration;
        if (!p.prepare_demotion(migration, *quote.impl)) {
            p.cancel_operation();
            return false;
        }
        p.submit_context();
        return true;
    } catch (...) {
        p.cancel_operation();
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

ContextResources Session::checkpoint_footprint(std::span<const CheckpointHandle> h) const {
    return impl_->footprint(h, false);
}

std::size_t Session::host_bytes_released(std::span<const CheckpointHandle> h) const {
    return impl_->host_released(h);
}

context::RecoveryEvaluation<>
Session::evaluate_recovery(std::span<const CheckpointHandle> advertised) const {
    impl_->recovery_index.synchronize(qwen::RecoveryStoreView{impl_->checkpoints});
    return impl_->recovery_index.evaluate(advertised);
}

CheckpointSummary Session::checkpoint_summary(CheckpointHandle h) const {
    CheckpointSummary out;
    static_cast<CheckpointMetadata&>(out) = checkpoint_metadata(h);
    out.key                               = checkpoint_key(h, out.frontier);
    out.evictable_resources               = impl_->footprint({&h, 1}, true);
    return out;
}

context::ReleasePlan<ContextRelease, ContextResources>
Session::plan_releases(std::span<const CheckpointHandle> allowed,
                       std::span<const CheckpointHandle> excluded,
                       ContextResources shortage) const {
    return impl_->plan_releases(allowed, excluded, shortage);
}

ContextReclaimPlan Session::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                         std::span<const CheckpointHandle> excluded,
                                         ContextResources shortage) const {
    return impl_->plan_reclaim(allowed, excluded, shortage);
}

ContextDemotionBatch ContextReclaimPlan::begin_kv_batch(ContextResources shortage) const {
    const auto pool = shortage.first_kv_deficit();
    auto batch      = std::make_shared<detail::DemotionBatch>();
    batch->owner    = impl->owner;
    batch->shortage = shortage;
    batch->pool     = pool;
    batch->valid    = !shortage.state_slots && !shortage.host_bytes && pool &&
                   !impl->shortage.state_slots && !impl->shortage.host_bytes &&
                   impl->shortage.first_kv_deficit() == pool;
    return ContextDemotionBatch(std::move(batch));
}

bool ContextDemotionBatch::append(const ContextDemotion& quote) {
    if (!plan_->valid) return false;
    const auto pool = plan_->pool->index;
    if (!quote.impl || quote.impl->owner != plan_->owner || !quote.impl->images.empty() ||
        quote.sources != quote.impl->records || quote.impl->pages[pool].empty() ||
        !quote.impl->pages[1 - pool].empty() ||
        !detail::valid_demotion(*plan_->owner, *quote.impl, false)) {
        plan_->valid = false;
        return false;
    }
    const auto& store = plan_->owner->directory.pool(pool).pages();
    for (const auto page : quote.impl->pages[pool])
        plan_->pages.insert(store.descriptor_index(page));
    plan_->components.push_back(quote.impl);
    return true;
}

bool ContextDemotionBatch::covers(const ContextRelease& release) const {
    if (!plan_->valid || !release.impl || release.impl->owner != plan_->owner) return false;
    const auto pool = plan_->pool->index;
    if (release.impl->pages[pool].empty() || !release.impl->pages[1 - pool].empty()) return false;
    const auto& store = plan_->owner->directory.pool(pool).pages();
    return std::all_of(
        release.impl->pages[pool].begin(), release.impl->pages[pool].end(), [&](auto page) {
            return store.valid(page) && plan_->pages.contains(store.descriptor_index(page));
        });
}

std::optional<ContextDemotion> ContextDemotionBatch::finish() const {
    if (!plan_->valid || plan_->components.empty()) return {};
    const auto& p               = *plan_->owner;
    const auto pool             = plan_->pool->index;
    const auto& store           = p.directory.pool(pool).pages();
    const auto free             = p.directory.host_allocation().free_bytes();
    const auto stride           = p.pools[pool].host_page_stride;
    auto result                 = std::make_shared<detail::Demotion>();
    result->owner               = &p;
    result->checkpoint_revision = p.checkpoints.revision();
    std::unordered_set<std::uint32_t> selected;
    std::size_t bytes = 0;
    bool complete     = false;
    for (const auto& component : plan_->components) {
        if (!detail::valid_demotion(p, *component, false)) return {};
        bool used = false;
        for (std::size_t i = 0; i < component->pages[pool].size(); ++i) {
            const auto page = component->pages[pool][i];
            if (!selected.insert(store.descriptor_index(page)).second) continue;
            const auto needed = store.host_replica_current(page) ? 0 : stride;
            if (needed > free - bytes) {
                complete = true;
                break;
            }
            bytes += needed;
            result->pages[pool].push_back(page);
            result->page_versions[pool].push_back(component->page_versions[pool][i]);
            used = true;
            if (result->pages[pool].size() == plan_->shortage.kv_page_groups[pool]) {
                complete = true;
                break;
            }
        }
        if (used)
            for (const auto h : component->records) detail::append_unique(result->records, h);
        if (complete) break;
    }
    if (result->empty()) return {};
    return p.describe_demotion(std::move(result));
}

bool Session::start_demote(const ContextDemotion& quote) { return impl_->start_demote(quote); }

std::size_t Session::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                            std::optional<SequenceHandle> keep) {
    auto& p = *impl_;
    if (!p.directory.host_allocation().capacity_bytes()) return 0;
    const auto before = p.directory.host_allocation().occupied_bytes();
    const auto images = p.checkpoint_images(excluded);
    std::vector<CheckpointHandle> all;
    for (std::uint32_t i = 0; i < p.checkpoints.size(); ++i)
        if (p.checkpoints.valid(p.checkpoints.handle(i))) all.push_back(p.checkpoints.handle(i));
    for (const auto image : p.checkpoint_images(all)) {
        if (std::find(images.begin(), images.end(), image) != images.end() ||
            (keep && p.valid(*keep) && p.sequence(*keep).continuation.image == image))
            continue;
        (void)p.directory.states().drop_host_replica(image);
    }
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
        const auto protected_pages = p.checkpoint_pages(excluded, pool);
        auto& store                = p.directory.pool(pool).pages();
        std::vector<context::PageId> drop;
        for (const auto page : p.checkpoint_pages(all, pool))
            if (std::find(protected_pages.begin(), protected_pages.end(), page) ==
                    protected_pages.end() &&
                store.host_resident(page) &&
                store.device_covers(page, store.required_columns(page)) &&
                p.directory.host().can_release_page_replica(store, page))
                drop.push_back(page);
        if (!drop.empty()) (void)p.directory.host().release_page_replicas(store, drop);
    }
    return before - p.directory.host_allocation().occupied_bytes();
}

} // namespace ninfer::models::qwen4_exp
