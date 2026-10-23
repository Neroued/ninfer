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
    context::MetadataBudget metadata{context::CheckpointStore<Record>::metadata_bytes(128)};
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

// Independent exhaustive definition: one maximum degradation per maximal removed branch.
std::uint64_t oracle(const View& view, std::span<const Handle> input,
                     std::span<const Handle> advertised) {
    std::vector<Handle> removed;
    for (const auto h : input)
        if (view.valid(h) && std::find(removed.begin(), removed.end(), h) == removed.end())
            removed.push_back(h);
    std::uint64_t result = 0;
    for (const auto branch : removed) {
        if (std::any_of(removed.begin(), removed.end(), [&](auto other) {
                return other != branch && view.prefix(branch, other) &&
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
        context::RecoveryIndex index;
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
    context::RecoveryIndex index;
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
        context::RecoveryIndex index;
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
            require(
                evaluation.loss(removed, unavailable) == oracle(view, removed, reduced),
                "admission gain used a removed fallback or scored the victim as incoming value");
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
} // namespace

int main() {
    try {
        catalog_reclamation();
        known_losses();
        lifecycle();
        exhaustive_and_random();
        physical_selection();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
