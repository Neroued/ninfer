#include "models/common/context/recovery_value.h"
#include "models/common/context/identity.h"
#include "models/common/context/recovery.h"
#include "models/common/context/release_inventory.h"
#include "models/common/context/directory.h"

#include <array>
#include <iostream>
#include <random>

namespace {
namespace context = ninfer::models::context;
using Handle      = context::CheckpointHandle;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Record {
    std::vector<std::uint32_t> tokens;
    std::uint32_t domain = 0;
};

struct View {
    context::MetadataBudget metadata{context::CheckpointStore<Record>::metadata_bytes(128) +
                                     context::RecoveryIndex<>::metadata_bytes(128) * 4};
    context::CheckpointStore<Record> store{128, metadata};
    bool collide = false;

    const std::uint64_t& revision() const { return store.revision(); }

    std::size_t capacity() const { return store.size(); }

    Handle handle(std::uint32_t i) const { return store.handle(i); }

    bool valid(Handle h) const { return store.valid(h); }

    std::uint32_t frontier(Handle h) const { return store.at(h).tokens.size(); }

    context::RecoveryKey key(Handle h, std::uint32_t f) const {
        const auto& value    = store.at(h);
        std::uint64_t digest = 1;
        for (std::size_t i = 0; i < f && !collide; ++i) digest = digest * 131 + value.tokens.at(i);
        return {{digest, digest ^ 42}, value.domain};
    }

    bool prefix(Handle a, Handle b) const {
        const auto& x = store.at(a);
        const auto& y = store.at(b);
        return x.domain == y.domain && x.tokens.size() <= y.tokens.size() &&
               std::equal(x.tokens.begin(), x.tokens.end(), y.tokens.begin());
    }

    Handle add(std::vector<std::uint32_t> tokens, std::uint32_t domain = 0) {
        const auto h = store.reserve();
        require(h.has_value(), "fixture exhausted its checkpoint slots");
        store.publish(*h, {std::move(tokens), domain});
        return *h;
    }
};

// Independent exhaustive definition: group positive recovery degradation, not removed aliases.
std::uint64_t oracle(const View& view, std::span<const Handle> input,
                     std::span<const Handle> advertised) {
    std::vector<Handle> removed;
    for (const auto h : input)
        if (view.valid(h) && std::find(removed.begin(), removed.end(), h) == removed.end())
            removed.push_back(h);
    const auto degraded = [&](Handle h) {
        return std::none_of(advertised.begin(), advertised.end(), [&](auto kept) {
            return view.valid(kept) &&
                   std::find(removed.begin(), removed.end(), kept) == removed.end() &&
                   view.frontier(kept) == view.frontier(h) && view.prefix(kept, h);
        });
    };
    std::uint64_t result = 0;
    for (const auto branch : removed) {
        if (!degraded(branch)) continue;
        if (std::any_of(removed.begin(), removed.end(), [&](auto other) {
                return other != branch && degraded(other) && view.prefix(branch, other) &&
                       (view.frontier(branch) < view.frontier(other) || other.index < branch.index);
            }))
            continue;
        std::uint32_t worst = 0;
        for (const auto point : removed) {
            if (!view.prefix(point, branch)) continue;
            std::uint32_t restored = 0;
            for (const auto kept : advertised)
                if (view.valid(kept) &&
                    std::find(removed.begin(), removed.end(), kept) == removed.end() &&
                    view.prefix(kept, point))
                    restored = std::max(restored, view.frontier(kept));
            worst = std::max(worst, view.frontier(point) - restored);
        }
        result += worst;
    }
    return result;
}

void redundant_aliases_do_not_split_loss() {
    View view;
    const auto p = view.add({1});
    const auto a = view.add({1, 2}), aa = view.add({1, 2});
    const auto b = view.add({1, 3}), bb = view.add({1, 3});
    context::RecoveryIndex index(view.metadata);
    index.synchronize(view);
    const std::array all{p, a, aa, b, bb};
    auto evaluation = index.evaluate(all);
    require(evaluation.loss(std::array{p}) == 1 && evaluation.loss(std::array{a, b}) == 0 &&
                evaluation.loss(std::array{p, a, b}) == 1,
            "redundant aliases split a shared ancestor's recovery loss");
    require(evaluation.loss(std::array{p, a, aa, b}) == 2,
            "removing the last equivalent source lost its real degradation");
}

void catalog_reclamation() {
    View view;
    const auto old    = view.add({1, 2, 3});
    const auto bytes  = view.store.released_metadata(std::array{view.store.retirement(old)});
    const auto before = view.metadata.allocated();
    require(view.store.erase(old) && before == view.metadata.allocated() + bytes && bytes != 0,
            "catalog quote differs from actual empty-segment retirement");
    const auto fresh = view.add({4, 5, 6});
    require(!view.store.valid(old) && view.store.valid(fresh) && fresh.generation != old.generation,
            "catalog segment reuse revived a stale checkpoint");
    view.store.clear();
    context::MetadataAccount other(view.metadata);
    other.acquire(view.metadata.available());
    require(!view.store.reserve(), "optional checkpoint capture ignored shared metadata pressure");
}

void known_losses() {
    for (const bool reverse : {false, true}) {
        View view;
        const auto root = view.add(std::vector<std::uint32_t>(100, 1));
        auto left = view.store.at(root).tokens, right = left;
        left.resize(200, 2);
        right.resize(200, 3);
        const auto a = view.add(reverse ? right : left);
        const auto b = view.add(reverse ? left : right);
        left.resize(150);
        right.resize(150);
        const auto x = view.add(left), y = view.add(right);
        context::RecoveryIndex index(view.metadata);
        index.synchronize(view);
        const std::array removed{root, a, b};
        const std::array one{x};
        const std::array both{x, y};
        auto first  = index.evaluate(one);
        auto second = index.evaluate(both);
        require(first.loss(removed) == 300 && second.loss(removed) == 200,
                "branch loss depended on checkpoint order or dropped the common ancestor");
        const std::array chain{root, reverse ? b : a};
        require(first.loss(chain) == 100, "intermediate fallback hid a shallow recovery point");
        const auto comparisons = index.work().exact_comparisons;
        for (int i = 0; i < 50; ++i) require(first.loss(removed) == 300, "query state leaked");
        require(index.work().exact_comparisons == comparisons, "query rescanned token identities");
    }
}

void lifecycle() {
    View view;
    view.collide = true;
    context::RecoveryIndex index(view.metadata);
    const auto deep = view.add({1, 2, 3, 4});
    index.synchronize(view);
    const auto shallow   = view.add({1, 2});
    const auto unrelated = view.add({9, 8});
    const auto domain    = view.add({1, 2}, 1);
    const auto alias     = view.add({1, 2});
    index.synchronize(view);
    const std::array advertised{shallow, alias, unrelated, domain};
    auto evaluation = index.evaluate(advertised);
    require(evaluation.prefix(shallow, deep) && !evaluation.prefix(unrelated, deep) &&
                !evaluation.prefix(domain, deep),
            "digest collision or domain invented a prefix");
    const std::array removed{shallow, shallow, deep};
    require(evaluation.loss(removed) == 2, "surviving alias did not provide fallback");
    require(view.store.erase(shallow), "could not retire a checkpoint");
    bool stale = false;
    try {
        (void)evaluation.loss(removed);
    } catch (const std::logic_error&) { stale = true; }
    require(stale, "evaluation survived checkpoint mutation");
    const auto replacement = view.add({5, 6, 7});
    require(replacement.index == shallow.index && replacement != shallow, "slot was not reused");
    index.synchronize(view);
    const std::array kept{alias};
    auto refreshed = index.evaluate(kept);
    require(!refreshed.contains(shallow) && refreshed.contains(replacement) &&
                !refreshed.prefix(replacement, deep) && refreshed.loss(removed) == 2,
            "reused slot inherited the retired prefix");
    require(view.store.erase(alias), "could not retire final alias");
    index.synchronize(view);
    auto empty = index.evaluate({});
    require(empty.loss(std::span(&deep, 1)) == 4, "retired ancestor remained a fallback");
    const auto syncs = index.work().synchronizations;
    index.synchronize(view);
    require(index.work().synchronizations == syncs, "unchanged store rebuilt its index");
    View other;
    const auto foreign = other.add({1, 2, 3, 4});
    require(empty.loss(std::span(&foreign, 1)) == 0, "foreign owner entered recovery loss");
}

void exhaustive_and_random() {
    std::mt19937 random(20261019);
    for (unsigned fixture = 0; fixture < 80; ++fixture) {
        View view;
        view.collide = fixture % 4 == 0;
        context::RecoveryIndex index(view.metadata);
        std::vector<Handle> handles;
        for (unsigned i = 0; i < 32; ++i) {
            std::vector<std::uint32_t> tokens;
            if (i && random() % 3) tokens = view.store.at(handles[random() % i]).tokens;
            for (auto add = random() % 5; add; --add) tokens.push_back(random() % 8);
            handles.push_back(view.add(std::move(tokens), random() % 3));
            if (i % 3 == 0) index.synchronize(view);
        }
        index.synchronize(view);
        for (unsigned query = 0; query < 50; ++query) {
            std::vector<Handle> removed, kept;
            for (const auto h : handles) {
                if (random() % 2) {
                    removed.push_back(h);
                    if (random() % 3 == 0) removed.push_back(h);
                }
                if (random() % 3) kept.push_back(h);
            }
            removed.push_back({});
            kept.push_back({});
            auto evaluation     = index.evaluate(kept);
            const auto expected = oracle(view, removed, kept);
            require(evaluation.loss(removed) == expected,
                    "indexed score differs from exhaustive oracle");
            std::shuffle(removed.begin(), removed.end(), random);
            require(evaluation.loss(removed) == expected,
                    "input permutation changed recovery loss");
            std::vector<Handle> unavailable;
            for (const auto h : handles)
                if (random() % 4 == 0) unavailable.push_back(h);
            auto reduced = kept;
            std::erase_if(reduced, [&](auto h) {
                return std::find(unavailable.begin(), unavailable.end(), h) != unavailable.end();
            });
            require(evaluation.loss(removed, unavailable) == oracle(view, removed, reduced),
                    "admission gain used a removed fallback or scored the victim as incoming "
                    "value");
        }
    }
}

bool same_release(const context::ContextReleaseSet& a, const context::ContextReleaseSet& b) {
    return a.records.size() == b.records.size() &&
           std::all_of(a.records.begin(), a.records.end(), [&](const auto& record) {
               return std::find(b.records.begin(), b.records.end(), record) != b.records.end();
           });
}

void physical_selection() {
    using Resources = context::ContextReleaseSet;
    std::mt19937 random(403);
    for (unsigned trial = 0; trial < 60; ++trial) {
        context::ReleaseInventory<Resources> inventory(12);
        std::vector<Handle> handles;
        for (unsigned i = 0; i < 12; ++i) {
            handles.push_back({&inventory, i, 1});
            inventory.include(handles.back());
        }
        for (unsigned i = 0; i < 20; ++i) {
            std::vector<Handle> holders;
            for (const auto h : handles)
                if (random() % 5 == 0) holders.push_back(h);
            if (holders.empty()) holders.push_back(handles[random() % handles.size()]);
            Resources released;
            released.records.push_back(
                {{unsigned(random() % 2), context::ContextMemory::Device}, {0, i, 1}});
            if (random() % 2)
                released.records.push_back({{0, context::ContextMemory::Host}, {1, i, 1}});
            inventory.add(std::move(holders), std::move(released));
        }
        auto selection    = inventory.select();
        const auto oracle = [&] {
            Resources expected;
            for (const auto& object : inventory.objects())
                if (std::all_of(object.holders.begin(), object.holders.end(), [&](auto h) {
                        return std::find(selection.handles().begin(), selection.handles().end(),
                                         h) != selection.handles().end();
                    })) {
                    expected.merge(object.released);
                }
            return expected;
        };
        for (unsigned round = 0; round < 24; ++round) {
            const auto before = selection.released();
            const std::array additions{handles[random() % 12], handles[random() % 12],
                                       handles[random() % 12]};
            const bool commit = random() % 4 == 0;
            {
                auto probe = selection.probe(additions);
                require(same_release(selection.released(), oracle()),
                        "incremental release counted a partial or duplicate holder set");
                if (commit) probe.commit();
            }
            require(same_release(selection.released(), oracle()) &&
                        (commit || same_release(selection.released(), before)),
                    "physical trial did not restore its release facts");
        }
    }
}

void marginal_dependencies() {
    View view;
    std::vector<Handle> all;
    for (const auto& key : std::vector<std::vector<std::uint32_t>>{
             {1}, {1, 2}, {1, 3}, {1, 2, 4}, {1, 2, 4}, {1, 3, 5}, {9}, {9, 8}})
        all.push_back(view.add(key));
    context::RecoveryIndex index(view.metadata);
    index.synchronize(view);
    auto evaluation = index.evaluate(all);
    for (unsigned mask = 0; mask < (1U << all.size()); ++mask) {
        std::vector<Handle> removed;
        for (unsigned i = 0; i < all.size(); ++i)
            if (mask & (1U << i)) removed.push_back(all[i]);
        for (unsigned a = 0; a < all.size(); ++a) {
            const std::array additions{all[a], all[(a + 3) % all.size()]};
            const auto dependencies = evaluation.dependencies(removed, additions);
            auto trial              = removed;
            trial.insert(trial.end(), additions.begin(), additions.end());
            const auto marginal = oracle(view, trial, all) - oracle(view, removed, all);
            for (const auto b : all) {
                if (dependencies.intersects(std::span<const Handle>(&b, 1))) continue;
                auto changed = removed, combined = trial;
                changed.push_back(b);
                combined.push_back(b);
                require(oracle(view, combined, all) - oracle(view, changed, all) == marginal,
                        "unrelated removal changed cached marginal recovery loss");
            }
        }
    }
}

void recovery_workspace_returns_idle_high_water() {
    context::MetadataBudget catalog(32ULL << 20);
    context::RecoveryMetadata workspace(catalog, 2);
    context::RecoveryIndex<> index(catalog, &workspace);
    context::CheckpointStore<Record> store(512, catalog, &index);
    const auto idle = workspace.budget().limit();
    std::vector<Handle> handles;
    for (unsigned i = 0; i < 257; ++i) {
        const auto h = store.reserve();
        require(h.has_value(), "workspace growth exhausted the fixture");
        store.publish(*h, {{i + 1}, 0});
        handles.push_back(*h);
    }
    const auto peak = workspace.budget().limit();
    require(peak > idle, "catalog did not grow its protected workspace");
    {
        context::MetadataAccount active(workspace.budget());
        active.acquire(1024);
        for (auto h : handles) require(store.erase(h), "retirement failed");
        require(store.metadata_extent() == 0 && workspace.budget().limit() == peak,
                "active decision lost the workspace promised before checkpoint retirement");
    }
    require(workspace.budget().limit() == idle,
            "idle decision workspace retained its high-water quota");
    const auto h = store.reserve();
    require(h && workspace.budget().limit() > idle && workspace.budget().limit() < peak,
            "recycled low slots regrew the obsolete high-water workspace");
    store.cancel(*h);
    require(workspace.budget().limit() == idle, "cancelled catalog growth retained its quota");
}

void reserved_decision_metadata() {
    context::MetadataBudget parent(1024);
    {
        context::MetadataBudget child(400, &parent);
        require(parent.reserved() == 400, "child did not protect its quota");
        child.acquire(120);
        child.reserve(80);
        child.consume(80);
        require(parent.allocated() == 200 && parent.reserved() == 200,
                "child spending was not visible to parent accounting");
        child.release(200);
        bool failed = false;
        try {
            child.grow(1025);
        } catch (const context::MetadataExhausted&) { failed = true; }
        require(failed && child.limit() == 400 && parent.reserved() == 400,
                "failed quota growth changed accounting");
    }
    require(parent.available() == 1024, "child quota leaked on destruction");
    const auto required = context::recovery_value_metadata_bytes(8, 2) +
                          context::CheckpointStore<Record>::metadata_bytes(8) +
                          context::RecoveryIndex<>::metadata_bytes(8);
    context::MetadataBudget catalog(required);
    context::RecoveryMetadata protected_work(catalog, 2);
    context::RecoveryIndex<> index(catalog, &protected_work);
    context::CheckpointStore<Record> store(8, catalog, &index);
    auto handle = store.reserve();
    require(handle.has_value(), "descriptor could not reserve its recovery workspace");
    store.publish(*handle, {{1, 2, 3}, 0});
    context::MetadataAccount other(catalog);
    other.acquire(catalog.available());
    context::MetadataAccount decision(protected_work.budget());
    decision.acquire(protected_work.budget().available());
    require(catalog.available() == 0 && decision.bytes() > 0,
            "catalog pressure consumed its reserved eviction workspace");
    require(store.erase(*handle), "decision metadata prevented checkpoint retirement");
}

void shared_identity_budget_survives_retirement() {
    struct Identity {
        std::vector<std::uint32_t> tokens;
        context::IdentityMetadata retention;

        std::size_t metadata_bytes() const {
            return sizeof(*this) + tokens.capacity() * sizeof(std::uint32_t);
        }
    };

    context::MetadataBudget directory(8ULL << 20);
    context::RecoveryMetadata recovery(directory, 2, 1, 1, 2048);
    auto& budget = recovery.identity_budget();
    auto source  = std::make_shared<Identity>();
    source->tokens.reserve(256);
    source->tokens.assign(4, 7);
    const std::shared_ptr<const Identity> immutable = source;
    auto first                                      = source->retention.retain(budget, immutable);
    require(first && budget.allocated() >= source->tokens.capacity() * sizeof(std::uint32_t),
            "identity cache omitted unused vector capacity");
    const auto charged = budget.allocated();
    auto alias         = source->retention.retain(budget, immutable);
    require(alias && budget.allocated() == charged,
            "shared checkpoint identity was charged more than once");

    auto other = std::make_shared<Identity>(*source);
    other->tokens.reserve(256);
    require(!other->retention.retain(budget, std::shared_ptr<const Identity>(other)) &&
                budget.allocated() == charged,
            "optional identity admission exceeded the bounded pool or inherited a copy's lease");

    context::MetadataAccount remaining(budget);
    remaining.acquire(budget.available());
    auto decision = first->retention.retain(budget, first);
    require(decision != nullptr, "required recovery could not borrow a paid identity at capacity");
    remaining.release(remaining.bytes());
    first.reset();
    alias.reset();
    require(budget.allocated() == charged,
            "checkpoint retirement returned bytes still retained by a recovery decision");
    decision.reset();
    require(budget.allocated() == 0 && source->tokens.size() == 4,
            "expired weak identity cache retained its quota while the request remained alive");

    auto replacement = other->retention.retain(budget, std::shared_ptr<const Identity>(other));
    require(replacement != nullptr, "retired decision quota was not reusable");
    std::weak_ptr<const Identity> lifetime = replacement;
    other.reset();
    require(!lifetime.expired(), "cache lease did not own its complete identity backing");
    replacement.reset();
    require(lifetime.expired() && budget.allocated() == 0,
            "last cache lease failed to release its backing and quota");
}

} // namespace

int main() {
    try {
        shared_identity_budget_survives_retirement();
        recovery_workspace_returns_idle_high_water();
        reserved_decision_metadata();
        catalog_reclamation();
        known_losses();
        redundant_aliases_do_not_split_loss();
        lifecycle();
        exhaustive_and_random();
        physical_selection();
        marginal_dependencies();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
