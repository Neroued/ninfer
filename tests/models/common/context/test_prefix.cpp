#include "models/common/context/prefix.h"
#include "models/common/context/recovery_value.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
namespace c = ninfer::models::context;
using ninfer::TokenId;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

c::InputIdentityView view(const std::vector<TokenId>& tokens) {
    return {.tokens = c::TokenView(tokens)};
}

void shared_growth() {
    c::MetadataBudget budget(32ULL << 20);
    c::PrefixStore store(budget);
    std::vector<TokenId> tokens(150'000);
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = i % 179;
    std::vector<c::PrefixRef> retained;
    for (unsigned i = 0; i < 30; ++i) {
        retained.push_back(store.prepare(7, view(tokens), tokens.size()));
        tokens.insert(tokens.end(), 1'000, 11 + i);
    }
    const auto usage = store.usage();
    require(usage.stored_tokens == 179'000, "growing requests duplicated their common prefix");
    require(usage.owned_bytes < (2ULL << 20), "text identity retained expanded coordinates");
    const auto first = retained.front().id(), last = retained.back().id();
    require(store.is_prefix(first, last), "growth lost the exact prefix relation");
    require(!store.is_prefix(last, first), "later frontier was treated as an earlier State");
    auto alias = retained.front();
    retained.clear();
    require(store.usage().stored_tokens == 150'000, "early endpoint retained its entire suffix");
    require(store.matches(alias.id(), view(tokens)), "suffix release changed a live endpoint");
    alias.reset();
    require(!store.valid(first) && store.usage().stored_tokens == 0,
            "last prefix owner did not release its content");
}

void exact_matches_and_endpoints() {
    c::MetadataBudget budget(8ULL << 20);
    c::PrefixStore store(budget);
    std::mt19937 rng(1488);
    std::vector<TokenId> base(1'024);
    for (auto& token : base) token = rng() % 128;
    std::vector<std::vector<TokenId>> values;
    std::vector<c::PrefixRef> refs;
    for (unsigned i = 0; i < 80; ++i) {
        auto input = base;
        input.resize(17 + rng() % 1'008);
        if (i % 3) input[rng() % input.size()] += 128;
        refs.push_back(store.prepare(3, view(input), input.size()));
        values.push_back(std::move(input));
    }
    for (std::size_t a = 0; a < values.size(); ++a)
        for (std::size_t b = 0; b < values.size(); ++b) {
            const bool expected = values[a].size() <= values[b].size() &&
                                  std::equal(values[a].begin(), values[a].end(), values[b].begin());
            require(store.is_prefix(refs[a].id(), refs[b].id()) == expected,
                    "prefix ancestry differs from exact token comparison");
            require(store.matches(refs[a].id(), view(values[b])) == expected,
                    "checkpoint match differs from exact token comparison");
        }
    auto other_domain = store.prepare(4, view(values.front()), values.front().size());
    require(!store.is_prefix(other_domain.id(), refs.front().id()), "identity domains were mixed");
    const auto before = store.usage();
    (void)store.match(3, view(base), base.size());
    require(store.usage().owned_bytes == before.owned_bytes,
            "read-only match allocated cache data");
    for (std::size_t i = 0; i < refs.size(); i += 2) refs[i].reset();
    for (std::size_t i = 1; i < refs.size(); i += 2)
        require(store.matches(refs[i].id(), view(values[i])), "compaction invalidated an endpoint");
}

void coordinates_and_media() {
    c::MetadataBudget budget(8ULL << 20);
    c::PrefixStore store(budget);
    std::vector<TokenId> tokens(701, 9);
    std::array<std::vector<std::int32_t>, 3> positions;
    auto input = view(tokens);
    for (unsigned a = 0; a < 3; ++a) {
        positions[a].resize(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) positions[a][i] = i;
        input.positions[a] = positions[a];
    }
    ninfer::models::frontend::VisionItem item;
    item.content_digest[0] = 71;
    item.grid              = {1, 16, 16};
    item.token_spans       = {{270, 128}};
    const std::array media{c::MediaIdentityView{&item}};
    input.media = media;
    auto full   = store.prepare(0, input, tokens.size());
    auto inside = store.prepare(0, input, 300);
    require(store.is_prefix(inside.id(), full.id()), "identity lookup rejected a media interior");
    const auto original_key = full.key();
    positions[1][350] += 1;
    require(store.match(0, input, tokens.size()).frontier == 350,
            "different coordinates reused model input");
    auto different = store.prepare(0, input, tokens.size());
    require(different.key() != original_key, "coordinate change did not affect fingerprint");
    positions[1][350] -= 1;
    item.content_digest[0] += 1;
    require(store.match(0, input, tokens.size()).frontier == 270,
            "different media contents reused placeholders");
    item.content_digest[0] -= 1;
    auto prefix = store.prepare(0, input, 200);
    full.reset();
    inside.reset();
    different.reset();
    require(!store.replay_media(prefix.id()), "early checkpoint retained a later media item");
    require(store.usage().stored_tokens == 200, "media suffix survived its final owner");
}

void bounded_fragments_and_queries() {
    c::MetadataBudget budget(8ULL << 20);
    c::PrefixStore store(budget);
    std::vector<TokenId> tokens(2048, 5);
    std::vector<std::int32_t> coordinates(tokens.size());
    for (std::size_t i = 0; i < coordinates.size(); ++i) coordinates[i] = i * i % 1031;
    auto input      = view(tokens);
    input.positions = {coordinates, coordinates, coordinates};
    c::PrefixRef current;
    for (std::uint32_t f = 1; f <= tokens.size(); ++f) {
        auto next = store.prepare(0, input, f, &current);
        current   = std::move(next);
    }
    require(store.usage().nodes <= 2 * tokens.size() / c::PrefixStore::block_tokens + 2,
            "short irregular captures accumulated permanent unary fragments");
    c::PrefixQuery query(store, 0, input);
    require(query.matches(current.id()), "query lost the exact irregular input");
    for (auto f : {1500U, 31U, 512U, 2048U}) {
        auto split = store.prepare(0, input, f);
        require(query.matches(split.id()) && query.key(f) == split.key(),
                "sparse demand key or cached identity proof changed after a split");
    }
    require(query.matches(current.id()), "endpoint changed after branch compaction");
}

void failed_preparation_releases_suffix() {
    c::MetadataBudget budget(2ULL << 20);
    c::PrefixStore store(budget);
    std::vector<TokenId> tokens(100000, 8);
    auto prefix = store.prepare(0, view(tokens), 512);
    c::MetadataAccount pressure(budget);
    pressure.acquire(budget.available() - 16384);
    bool rejected = false;
    try {
        (void)store.prepare(0, view(tokens), tokens.size(), &prefix);
    } catch (const c::MetadataExhausted&) { rejected = true; }
    require(rejected && store.usage().stored_tokens == 512 &&
                store.matches(prefix.id(), view(tokens)),
            "failed preparation retained a suffix or damaged the published prefix");
    pressure.release(pressure.bytes());
    auto complete = store.prepare(0, view(tokens), tokens.size(), &prefix);
    require(store.is_prefix(prefix.id(), complete.id()),
            "failed preparation corrupted subsequent reuse");
}

void exhausted_initialization_can_retry() {
    const std::vector<TokenId> tokens(1024, 7);
    for (std::size_t available = 0; available <= 4096; available += 64) {
        c::MetadataBudget budget(1ULL << 20);
        c::PrefixStore store(budget);
        c::MetadataAccount pressure(budget);
        pressure.acquire(budget.available() - available);
        try {
            (void)store.prepare(0, view(tokens), tokens.size());
        } catch (const c::MetadataExhausted&) {}
        pressure.release(pressure.bytes());
        auto prefix = store.prepare(0, view(tokens), tokens.size());
        require(store.matches(prefix.id(), view(tokens)),
                "partial metadata initialization corrupted a subsequent preparation");
    }
}

void recovery_geometry_does_not_pin_tokens() {
    c::MetadataBudget budget(4ULL << 20);
    c::PrefixStore store(budget);
    std::vector<TokenId> tokens(700, 2);
    std::array<ninfer::models::frontend::VisionItem, 3> items;
    std::vector<c::MediaIdentityView> media;
    for (std::uint32_t i = 0; i < items.size(); ++i) {
        items[i].grid        = {1, 8, 8};
        items[i].token_spans = {{100 + i * 200, 64}};
        media.push_back({&items[i]});
    }
    auto input      = view(tokens);
    input.media     = media;
    auto prefix     = store.prepare(0, input, tokens.size());
    const auto work = c::recovery_work(budget, store, prefix.id(), 128, 128);
    prefix.reset();
    require(store.usage().stored_tokens == 0, "recovery geometry retained retired token blocks");
    const auto later = work->rebuild(250);
    require(later.tokens == 450 && later.vision_items == 2 && later.vision_patches == 128,
            "retirement changed the quoted media reconstruction work");
}

void media_fragment_compaction() {
    constexpr std::uint32_t length = 1536;
    c::MetadataBudget budget(c::PrefixStore::metadata_bytes(length, 33, {true, 3, 3}));
    c::PrefixStore store(budget);
    std::vector<TokenId> tokens(length, 9);
    std::vector<std::int32_t> positions(length);
    std::vector<std::uint8_t> types(length);
    std::vector<ninfer::models::frontend::VisionItem> items(length / 3);
    std::vector<c::MediaIdentityView> media;
    for (std::uint32_t i = 0; i < length; ++i) {
        positions[i] = i * i % 97;
        types[i]     = i % 3 == 1;
    }
    for (std::uint32_t i = 0; i < items.size(); ++i) {
        items[i].token_spans = {{3 * i + 1, 1}};
        items[i].timestamps  = {double(i)};
        media.push_back({&items[i]});
    }
    auto input      = view(tokens);
    input.positions = {positions, positions, positions};
    input.types     = types;
    input.media     = media;
    c::PrefixRef current;
    std::vector<c::PrefixRef> endpoints;
    for (std::uint32_t f = 1; f <= length; ++f) {
        current = store.prepare(0, input, f, &current);
        if (f % 48 == 0) endpoints.push_back(current);
    }
    const auto key = current.key();
    std::mt19937 random(448);
    std::shuffle(endpoints.begin(), endpoints.end(), random);
    for (auto& endpoint : endpoints) {
        endpoint.reset();
        require(store.matches(current.id(), input) && current.key() == key,
                "media compaction changed a surviving endpoint");
    }
    std::size_t media_count = 0;
    for (auto item = store.replay_media(current.id()); item; item = item->previous()) ++media_count;
    require(media_count == items.size(), "compaction lost media events");
    current.reset();
    require(store.usage().stored_tokens == 0, "media compaction retained an unowned path");
}

void ledger_and_commit() {
    std::vector<TokenId> prompt(300, 4);
    c::RequestLedger ledger(prompt);
    ledger.prepare_append(10);
    const std::array<TokenId, 3> output{7, 8, 9};
    ledger.append(output);
    std::vector<TokenId> slice;
    ledger.copy(298, 303, slice);
    require(slice == std::vector<TokenId>({4, 4, 7, 8, 9}),
            "ledger range crosses input incorrectly");
    const std::array<TokenId, 2> pending{10, 11};
    require(ledger.view(pending).back() == 11 && ledger.size() == 303,
            "staged output changed the committed ledger");
    c::MetadataBudget budget(1ULL << 20);
    c::PrefixStore store(budget);
    c::PrefixBuilder builder(store, 0);
    auto input  = c::InputIdentityView{.tokens = ledger.view()};
    auto update = builder.prepare(input, 302);
    require(builder.frontier() == 0, "prepared prefix advanced before commit");
    builder.commit(std::move(update));
    require(builder.frontier() == 302 && builder.prefix().frontier() == 256,
            "private decode tail was permanently published");
    auto captured = builder.try_checkpoint(input, 302);
    require(captured.frontier() == 302 && captured.frontier() + 1 == ledger.size(),
            "checkpoint included the pending input token");
    auto discarded = builder.prepare(input, 303, true);
    discarded      = {};
    require(builder.frontier() == 302, "cancelled prefix update changed State frontier");
}

void generated_coordinates_match_prepared_input() {
    c::MetadataBudget budget(1ULL << 20);
    c::PrefixStore store(budget);
    const std::vector<TokenId> prompt{1, 2, 3, 4};
    const std::array<TokenId, 3> proposed{5, 6, 7};
    const auto accepted = std::span(proposed).first(2);
    std::array<std::vector<std::int32_t>, 3> positions{{{0, 1, 1, 2}, {0, 1, 2, 2}, {0, 2, 1, 2}}};
    c::InputIdentityView generated{.tokens          = c::TokenView(prompt, {}, accepted),
                                   .generated_delta = -1};
    for (unsigned axis = 0; axis < 3; ++axis) generated.positions[axis] = positions[axis];
    const auto original = store.prepare(0, generated, generated.size());
    auto rebuilt_tokens = prompt;
    rebuilt_tokens.insert(rebuilt_tokens.end(), accepted.begin(), accepted.end());
    auto rebuilt_positions = positions;
    auto rebuilt           = view(rebuilt_tokens);
    for (unsigned axis = 0; axis < 3; ++axis) {
        rebuilt_positions[axis].insert(rebuilt_positions[axis].end(), {3, 4});
        rebuilt.positions[axis] = rebuilt_positions[axis];
    }
    const auto other = store.prepare(0, rebuilt, rebuilt.size());
    require(original.id() == other.id(),
            "accepted output and independently prepared MRoPE input differ");
    rebuilt_positions[1].back() = 5;
    require(store.match(0, rebuilt, rebuilt.size()).frontier == 5,
            "generated coordinate mismatch reused a full endpoint");
}
} // namespace

int main() {
    shared_growth();
    exact_matches_and_endpoints();
    coordinates_and_media();
    ledger_and_commit();
    bounded_fragments_and_queries();
    failed_preparation_releases_suffix();
    exhausted_initialization_can_retry();
    recovery_geometry_does_not_pin_tokens();
    media_fragment_compaction();
    generated_coordinates_match_prepared_input();
    std::cout << "shared prefix and request ledger checks passed\n";
}
