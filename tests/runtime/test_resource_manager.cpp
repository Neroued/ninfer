#include "runtime/engine/context_cache/resource_manager.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Usage = ninfer::runtime::ContextResourceUsage;
using Key   = std::vector<std::uint32_t>;
using Role  = ninfer::runtime::CheckpointRole;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Handle {
    std::uint32_t index;
    friend bool operator==(Handle, Handle) = default;
};

struct Base {
    struct Summary {
        std::uint32_t prompt_tokens;
        bool publish_continuation = true;
    };

    struct Advice {
        bool update_session_index = true;
        std::optional<std::uint64_t> session_key;
    };

    Key tokens;
    Advice advice;
    std::vector<std::uint32_t> captures;

    [[nodiscard]] const auto& capture_frontiers() const { return captures; }

    [[nodiscard]] Summary summary() const { return {static_cast<std::uint32_t>(tokens.size())}; }

    [[nodiscard]] const Advice& context_cache() const { return advice; }

    [[nodiscard]] std::optional<Key> prefix_shortlist_key(std::uint32_t frontier) const {
        if (frontier > tokens.size()) { return std::nullopt; }
        return Key(tokens.begin(), tokens.begin() + frontier);
    }
};

struct Source {
    std::optional<Handle> checkpoint;
    std::uint32_t reused_tokens = 0;
    ninfer::runtime::PrefillWork remaining_work;
    std::vector<ninfer::runtime::ContextTransferRequirement> transfers;
    bool consume_source = false;
    bool take_private   = false;
    std::vector<Handle> retired_points;
    std::vector<Handle> private_points;
};

// Concrete immutable contents and their Host aliases, without cache policy or an allocator.
// The production stores have separate tests for actual allocation and transfer lifetimes.
struct Program {
    struct Content {
        Key tokens;
        std::vector<std::size_t> host_blocks;
        bool evictable           = true;
        bool alive               = true;
        bool device_state        = true;
        bool leased              = false;
        bool release_blocked     = false;
        bool demotion_blocked    = false;
        std::uint32_t main_pages = 1;
        std::optional<std::uint32_t> demotion_page_limit;
        std::optional<std::size_t> demotion_bytes;
        Role role = Role::SharedPrefix;
    };

    struct HostBlock {
        std::size_t bytes;
        bool external_reference;
    };

    struct Metadata {
        std::uint32_t frontier;
        Role role;
        bool leased;
    };

    struct Summary {
        Key key;
        std::uint32_t frontier;
        bool fully_device;
        Usage evictable_resources;
        bool leased;
        Role role;
    };

    struct PhysicalUsage {
        Usage capacity;
        Usage occupied;
    };

    std::size_t host_capacity;
    std::vector<Content> contents;
    std::vector<HostBlock> blocks;
    std::vector<Handle> released;
    std::vector<Handle> demoted;
    std::vector<std::vector<Handle>> demotion_submissions;
    std::uint32_t main_capacity    = 1024;
    bool transfer_in_progress      = false;
    bool combined_demotion_blocked = false;

    std::size_t host_block(std::size_t bytes, bool external_reference = false) {
        blocks.push_back({bytes, external_reference});
        return blocks.size() - 1;
    }

    Handle add(Key tokens, std::vector<std::size_t> host_blocks = {},
               Role role = Role::SharedPrefix) {
        const Handle handle{static_cast<std::uint32_t>(contents.size())};
        contents.push_back({std::move(tokens), std::move(host_blocks)});
        contents.back().role = role;
        require(physical_usage().occupied.host_bytes <= host_capacity,
                "fixture exceeded its physical Host capacity");
        return handle;
    }

    [[nodiscard]] bool references(const Content& content, std::size_t block) const {
        return content.alive && std::find(content.host_blocks.begin(), content.host_blocks.end(),
                                          block) != content.host_blocks.end();
    }

    [[nodiscard]] Usage checkpoint_footprint(std::span<const Handle> handles) const {
        Usage result;
        std::vector<std::size_t> unique;
        for (const auto handle : handles) {
            const auto& content = contents.at(handle.index);
            require(content.alive, "cache queried a released checkpoint");
            result.state_slots += content.device_state;
            result.main_kv_pages += content.main_pages;
            for (const auto block : content.host_blocks) {
                if (std::find(unique.begin(), unique.end(), block) == unique.end()) {
                    unique.push_back(block);
                    result.host_bytes += blocks.at(block).bytes;
                }
            }
        }
        return result;
    }

    [[nodiscard]] Metadata checkpoint_metadata(Handle handle) const {
        const auto& content = contents.at(handle.index);
        require(content.alive, "cache queried a released checkpoint");
        return {static_cast<std::uint32_t>(content.tokens.size()), content.role, content.leased};
    }

    [[nodiscard]] Summary checkpoint_summary(Handle handle) const {
        const auto& content = contents.at(handle.index);
        const auto usage    = checkpoint_footprint({&handle, 1});
        return {content.tokens,
                static_cast<std::uint32_t>(content.tokens.size()),
                content.host_blocks.empty(),
                content.evictable ? usage : Usage{},
                content.leased,
                content.role};
    }

    [[nodiscard]] Key checkpoint_key(Handle handle, std::uint32_t frontier) const {
        const auto& content = contents.at(handle.index);
        require(content.alive && frontier <= content.tokens.size(), "invalid prefix identity read");
        return {content.tokens.begin(), content.tokens.begin() + frontier};
    }

    [[nodiscard]] PhysicalUsage physical_usage() const {
        PhysicalUsage result{.capacity = {.state_slots      = 64,
                                          .main_kv_pages    = main_capacity,
                                          .backend_kv_pages = 1024,
                                          .host_bytes       = host_capacity}};
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (blocks[block].external_reference ||
                std::any_of(contents.begin(), contents.end(),
                            [&](const auto& content) { return references(content, block); })) {
                result.occupied.host_bytes += blocks[block].bytes;
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t host_bytes_released(std::span<const Handle> victims) const {
        std::size_t bytes = 0;
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (blocks[block].external_reference) { continue; }
            bool referenced = false;
            bool survives   = false;
            for (std::uint32_t index = 0; index < contents.size(); ++index) {
                if (!references(contents[index], block)) { continue; }
                referenced = true;
                if (!contents[index].evictable ||
                    std::find(victims.begin(), victims.end(), Handle{index}) == victims.end()) {
                    survives = true;
                }
            }
            if (referenced && !survives) { bytes += blocks[block].bytes; }
        }
        return bytes;
    }

    [[nodiscard]] bool valid_checkpoint(Handle handle) const {
        return handle.index < contents.size() && contents[handle.index].alive;
    }

    bool release_checkpoint(Handle handle) {
        auto& content = contents.at(handle.index);
        if (!content.alive || !content.evictable || content.release_blocked) { return false; }
        content.alive = false;
        released.push_back(handle);
        return true;
    }

    [[nodiscard]] std::optional<Source> inspect_source(const Base& base,
                                                       std::optional<Handle> handle,
                                                       bool consume                    = false,
                                                       std::span<const Handle> carried = {},
                                                       std::span<const Handle> retired = {}) const {
        Source source{.checkpoint = handle};
        source.consume_source = consume;
        source.take_private   = consume;
        source.retired_points.assign(retired.begin(), retired.end());
        for (const auto point : carried) {
            if (!handle || !valid_checkpoint(point)) { continue; }
            const auto& content = contents.at(point.index);
            if (content.role != Role::Continuation &&
                content.tokens.size() <= contents.at(handle->index).tokens.size()) {
                source.private_points.push_back(point);
            }
        }
        if (handle) {
            const auto summary = checkpoint_summary(*handle);
            if (base.prefix_shortlist_key(summary.frontier) != summary.key) { return std::nullopt; }
            source.reused_tokens = summary.frontier;
            const auto footprint = checkpoint_footprint({&*handle, 1});
            if (footprint.host_bytes) {
                source.transfers.push_back({
                    .direction = ninfer::runtime::ContextTransferDirection::HostToDevice,
                    .work      = {.payload_bytes = footprint.host_bytes},
                });
            }
        }
        source.remaining_work.tokens = base.tokens.size() - source.reused_tokens;
        return source;
    }

    [[nodiscard]] bool has_context_transaction() const { return transfer_in_progress; }

    bool release_redundant_host(std::span<const Handle>) { return false; }

    struct Demotion {
        Handle handle;
        std::size_t host_bytes;
        std::vector<Handle> sources;
        Usage released;
        std::vector<std::pair<Handle, std::uint32_t>> main_parts;
    };

    struct Release {
        std::vector<Handle> sources;
        Usage released;
    };

    struct ReclaimPlan {
        struct Batch {
            const Program* program;
            Usage shortage;
            std::vector<Demotion> ordered;

            bool append(const Demotion& demotion) {
                ordered.push_back(demotion);
                return true;
            }

            [[nodiscard]] bool covers(const Release& release) const {
                return program->covers_release(ordered, release.sources, shortage);
            }

            [[nodiscard]] std::optional<Demotion> finish() const {
                return program->combined_demotion(ordered, shortage);
            }
        };

        const Program* program;
        std::vector<Demotion> demotions;
        std::vector<Release> releases;

        [[nodiscard]] Batch begin_kv_batch(Usage shortage) const { return {program, shortage}; }

        [[nodiscard]] std::uint64_t recovery_loss(std::span<const Handle> removed,
                                                  std::span<const Handle> surviving) const {
            return program->checkpoint_recovery_loss(removed, surviving);
        }
    };

    [[nodiscard]] ReclaimPlan plan_reclaim(std::span<const Handle> allowed,
                                           std::span<const Handle> excluded, Usage shortage) const {
        return {this, demotion_options(allowed, excluded, shortage),
                plan_releases(allowed, excluded, shortage)};
    }

    [[nodiscard]] std::vector<Demotion> demotion_options(std::span<const Handle> allowed,
                                                         std::span<const Handle> excluded,
                                                         Usage shortage) const {
        std::vector<Demotion> result;
        for (const auto handle : allowed) {
            const auto& content = contents.at(handle.index);
            if (!content.alive || !content.evictable || content.leased || !content.demotion_bytes ||
                std::find(excluded.begin(), excluded.end(), handle) != excluded.end()) {
                continue;
            }
            Usage released;
            if (shortage.state_slots && content.device_state) {
                released.state_slots = 1;
            } else if (!shortage.state_slots && shortage.main_kv_pages) {
                released.main_kv_pages =
                    std::min({shortage.main_kv_pages, content.main_pages,
                              content.demotion_page_limit.value_or(content.main_pages)});
            }
            const auto units = released.state_slots + released.main_kv_pages;
            if (!units) { continue; }
            result.push_back(
                {handle,
                 *content.demotion_bytes * units,
                 {handle},
                 released,
                 released.main_kv_pages
                     ? std::vector<std::pair<Handle, std::uint32_t>>{{handle,
                                                                      released.main_kv_pages}}
                     : std::vector<std::pair<Handle, std::uint32_t>>{}});
        }
        return result;
    }

    [[nodiscard]] std::optional<Demotion> combined_demotion(std::span<const Demotion> ordered,
                                                            Usage shortage) const {
        if (ordered.empty() || shortage.state_slots || !shortage.main_kv_pages) {
            return std::nullopt;
        }
        Demotion combined{ordered.front().handle, 0, {}, {}};
        auto remaining   = shortage.main_kv_pages;
        const auto usage = physical_usage();
        auto host_left   = usage.capacity.host_bytes - usage.occupied.host_bytes;
        // The fixture supplies disjoint, explicit page quantities. Physical aliases,
        // reservations and transfer leases belong to the Native contract tests.
        for (const auto& plan : ordered) {
            if (plan.main_parts.empty()) { break; }
            for (const auto& [handle, pages] : plan.main_parts) {
                const auto bytes = *contents.at(handle.index).demotion_bytes;
                const auto take  = static_cast<std::uint32_t>(std::min<std::size_t>(
                    std::min(remaining, pages), bytes ? host_left / bytes : remaining));
                if (take) {
                    combined.main_parts.emplace_back(handle, take);
                    combined.sources.push_back(handle);
                    combined.released.main_kv_pages += take;
                    combined.host_bytes += take * bytes;
                    remaining -= take;
                    host_left -= take * bytes;
                }
                if (take < pages || !remaining) {
                    return combined.main_parts.empty() ? std::nullopt
                                                       : std::optional(std::move(combined));
                }
            }
        }
        return combined.main_parts.empty() ? std::nullopt : std::optional(std::move(combined));
    }

    [[nodiscard]] bool covers_release(std::span<const Demotion> ordered,
                                      std::span<const Handle> removed, Usage shortage) const {
        if (shortage.state_slots || shortage.host_bytes || !shortage.main_kv_pages) {
            return false;
        }
        std::vector<std::uint32_t> covered(contents.size());
        for (const auto& plan : ordered) {
            for (const auto& [handle, pages] : plan.main_parts) {
                covered.at(handle.index) = std::max(covered.at(handle.index), pages);
            }
        }
        bool releases_pages = false;
        for (const auto handle : removed) {
            const auto& content = contents.at(handle.index);
            if (!content.alive || !content.evictable || content.leased || content.release_blocked ||
                covered.at(handle.index) < content.main_pages) {
                return false;
            }
            releases_pages = releases_pages || content.main_pages != 0;
        }
        return releases_pages;
    }

    [[nodiscard]] std::vector<Release> plan_releases(std::span<const Handle> allowed,
                                                     std::span<const Handle> excluded,
                                                     Usage shortage) const {
        const auto eligible = [&](Handle handle) {
            const auto& content = contents.at(handle.index);
            return content.alive && content.evictable && !content.leased &&
                   !content.release_blocked &&
                   std::find(allowed.begin(), allowed.end(), handle) != allowed.end() &&
                   std::find(excluded.begin(), excluded.end(), handle) == excluded.end();
        };
        std::vector<Release> result;
        const auto append = [&](std::vector<Handle> sources) {
            if (sources.empty() || !std::all_of(sources.begin(), sources.end(), eligible)) {
                return;
            }
            Usage released;
            for (const auto handle : sources) {
                const auto& content = contents.at(handle.index);
                released.state_slots += content.device_state;
                released.main_kv_pages += content.main_pages;
            }
            released.host_bytes = host_bytes_released(sources);
            const bool useful   = shortage.state_slots        ? released.state_slots != 0
                                  : shortage.main_kv_pages    ? released.main_kv_pages != 0
                                  : shortage.backend_kv_pages ? false
                                                              : released.host_bytes != 0;
            if (!useful) { return; }
            if (std::none_of(result.begin(), result.end(),
                             [&](const auto& action) { return action.sources == sources; })) {
                result.push_back({std::move(sources), released});
            }
        };
        for (const auto handle : allowed) { append({handle}); }
        if (shortage.host_bytes) {
            for (std::size_t block = 0; block < blocks.size(); ++block) {
                std::vector<Handle> sources;
                for (std::uint32_t i = 0; i < contents.size(); ++i) {
                    if (references(contents[i], block)) { sources.push_back({i}); }
                }
                append(std::move(sources));
            }
        }
        return result;
    }

    [[nodiscard]] bool prefix(Handle earlier, Handle later) const {
        const auto& a = contents.at(earlier.index).tokens;
        const auto& b = contents.at(later.index).tokens;
        return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
    }

    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const Handle> removed,
                                                         std::span<const Handle> surviving) const {
        std::vector<Handle> deepest;
        for (const auto handle : removed) {
            const auto deeper = std::any_of(removed.begin(), removed.end(), [&](auto other) {
                return other != handle && prefix(handle, other) &&
                       (contents.at(handle.index).tokens.size() <
                            contents.at(other.index).tokens.size() ||
                        other.index < handle.index);
            });
            if (!deeper) { deepest.push_back(handle); }
        }
        std::uint64_t loss = 0;
        for (const auto end : deepest) {
            std::size_t line_loss = 0;
            for (const auto handle : removed) {
                if (!prefix(handle, end)) { continue; }
                std::size_t fallback = 0;
                for (const auto kept : surviving) {
                    if (std::find(removed.begin(), removed.end(), kept) == removed.end() &&
                        prefix(kept, handle)) {
                        fallback = std::max(fallback, contents.at(kept.index).tokens.size());
                    }
                }
                line_loss = std::max(line_loss, contents.at(handle.index).tokens.size() - fallback);
            }
            loss += line_loss;
        }
        return loss;
    }

    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(Handle retained, const Base& base,
                                                             std::uint32_t target) const {
        const auto& tokens = contents.at(retained.index).tokens;
        return tokens.size() <= target && base.prefix_shortlist_key(tokens.size()) == tokens
                   ? static_cast<std::uint32_t>(tokens.size())
                   : 0;
    }

    bool start_demote(const Demotion& plan) {
        if ((combined_demotion_blocked && plan.main_parts.size() > 1) ||
            std::any_of(plan.sources.begin(), plan.sources.end(),
                        [&](auto handle) { return contents.at(handle.index).demotion_blocked; })) {
            return false;
        }
        const auto usage = physical_usage();
        if (plan.host_bytes > usage.capacity.host_bytes - usage.occupied.host_bytes) {
            return false;
        }
        if (!plan.main_parts.empty()) {
            for (const auto& [handle, pages] : plan.main_parts) {
                auto& content    = contents.at(handle.index);
                const auto bytes = *content.demotion_bytes * pages;
                if (bytes) { content.host_blocks.push_back(host_block(bytes)); }
                content.main_pages -= pages;
                demoted.push_back(handle);
            }
        } else {
            auto& content = contents.at(plan.handle.index);
            if (plan.host_bytes) { content.host_blocks.push_back(host_block(plan.host_bytes)); }
            if (plan.released.state_slots) { content.device_state = false; }
            demoted.push_back(plan.handle);
        }
        demotion_submissions.push_back(plan.sources);
        return true;
    }
};

struct Model {
    using Program          = ::Program;
    using RequestBasePlan  = Base;
    using CheckpointHandle = Handle;
    using SourceCandidate  = Source;
    using CacheSessionKey  = std::uint64_t;
};

struct Fixture {
    using Cache = ninfer::runtime::ResourceManager<Model>;
    using Token = Cache::OwnerToken;
    Program program;
    Cache cache;
    std::uint64_t next_order = 0;

    explicit Fixture(std::size_t capacity)
        : program{.host_capacity = capacity},
          cache(true, {.prefill = {.token_ns_q32 = ninfer::runtime::kContextCostQ32One}}) {}

    Cache::SourceChoice source(const Base& base, std::optional<Handle> desired,
                               std::optional<Token> resume = {}) {
        auto choices     = cache.candidates(program, base, UINT32_MAX, resume);
        const auto found = std::find_if(choices.begin(), choices.end(), [&](const auto& choice) {
            return choice.source.checkpoint == desired;
        });
        require(found != choices.end(), "requested fixture source is not a legal candidate");
        return *found;
    }

    Token begin(const Base& base, std::optional<Handle> desired = {},
                std::span<const Handle> carried = {}, std::optional<Token> resume = {}) {
        return cache.adopt(program, source(base, desired, resume), base, ++next_order, carried);
    }

    Token publish(Handle handle, std::optional<std::uint64_t> session = {}) {
        const Base base{.tokens = program.contents[handle.index].tokens,
                        .advice = {.session_key = session}};
        const auto token = begin(base);
        cache.publish(program, token, handle);
        cache.finish(program, token, base, next_order);
        return token;
    }

    void use_shared(Handle handle) {
        const Base base{.tokens = program.contents[handle.index].tokens};
        const auto token = begin(base, handle);
        cache.abandon(program, token);
    }
};

void test_cold_writeback_protects_reused_history() {
    Fixture f(64);
    const auto cold     = f.program.add({1}, {f.program.host_block(16)});
    const auto hot      = f.program.add({2}, {f.program.host_block(32)});
    const auto incoming = f.program.add({3, 4, 5});
    f.publish(cold);
    f.publish(hot);
    f.publish(incoming);
    f.use_shared(hot);
    require(!f.cache.host_victims(f.program, 48, incoming),
            "ordinary writeback sacrificed reused history");
    const auto victims = f.cache.host_victims(f.program, 32, incoming);
    require(victims && *victims == std::vector<Handle>{cold},
            "ordinary writeback failed to use permitted cold history");
}

void test_host_preflight_counts_aliases_and_has_no_partial_deletion() {
    Fixture f(64);
    const auto block = f.program.host_block(48);
    (void)f.program.host_block(16, true);
    const auto first    = f.program.add({1}, {block});
    const auto second   = f.program.add({2}, {block});
    const auto incoming = f.program.add({3, 4, 5});
    f.publish(first);
    f.publish(second);
    f.publish(incoming);
    require(!f.cache.host_victims(f.program, 48, incoming, {}, {&second, 1}),
            "Host preflight ignored a surviving alias");
    require(f.program.released.empty(), "failed preflight deleted history");
    const auto victims = f.cache.host_victims(f.program, 48, incoming);
    require(victims && *victims == std::vector<Handle>({first, second}),
            "Host preflight failed to count the last shared reference");
    require(f.cache.erase(f.program, first) && f.program.physical_usage().occupied.host_bytes == 64,
            "first alias release fabricated free physical bytes");
    require(f.cache.erase(f.program, second) &&
                f.program.physical_usage().occupied.host_bytes == 16,
            "last alias did not release its physical bytes");
}

void test_candidates_keep_all_frontiers_and_independent_contents() {
    Fixture f(64);
    const auto shallow    = f.program.add({1, 2});
    const auto middle     = f.program.add({1, 2, 3, 4}, {f.program.host_block(16)});
    const auto deep       = f.program.add({1, 2, 3, 4, 5, 6});
    const auto recomputed = f.program.add({1, 2, 3, 4, 5, 6});
    f.publish(shallow);
    f.publish(middle);
    f.publish(deep);
    f.publish(recomputed);
    const Base base{.tokens = {1, 2, 3, 4, 5, 6, 7}};
    const auto all = f.cache.candidates(f.program, base);
    require(all.size() == 5 && all[0].source.checkpoint == deep &&
                all[1].source.checkpoint == recomputed && all[2].source.checkpoint == middle &&
                all[3].source.checkpoint == shallow && !all[4].source.checkpoint,
            "source selection discarded a shallower tier peer or independent state");
    const auto bounded = f.cache.candidates(f.program, base, 3);
    require(bounded.size() == 2 && bounded[0].source.checkpoint == shallow,
            "maximum recovery frontier discarded a valid shallower point");
    require(f.cache.erase(f.program, deep), "fixture could not retire one same-key source");
    const auto surviving = f.cache.candidates(f.program, base);
    require(surviving.front().source.checkpoint == recomputed,
            "removing one same-key checkpoint erased its surviving index peer");
}

void test_private_continuation_advances_with_adoption_heat() {
    Fixture f(128);
    (void)f.program.host_block(96, true);
    const auto old       = f.program.add({1, 2}, {f.program.host_block(16)}, Role::Continuation);
    const auto old_owner = f.publish(old);
    const auto entrant   = f.program.add({9});
    f.publish(entrant);
    const Base next{.tokens = {1, 2, 3}};
    auto choice = f.source(next, old);
    require(choice.take_over && choice.owner == old_owner && choice.source.consume_source,
            "anonymous private continuation cannot transfer its writer");
    require(f.program.release_checkpoint(old), "fixture consume failed");
    const auto owner = f.cache.adopt(f.program, choice, next, 20, {}, {&old, 1});
    require(owner == old_owner && f.cache.retention(owner).reused,
            "consumed handle retirement lost the adopted continuation relationship");
    const auto advanced =
        f.program.add({1, 2, 3, 4}, {f.program.host_block(16)}, Role::Continuation);
    f.cache.publish(f.program, owner, advanced);
    f.cache.finish(f.program, owner, next, 20);
    require(!f.cache.host_victims(f.program, 32, entrant),
            "new endpoint lost actual private continuation reuse protection");
    const auto replay = f.program.add({1, 2, 3}, {f.program.host_block(16)}, Role::InputReplay);
    f.cache.publish(f.program, owner, replay);
    require(f.cache.retention(owner).reused && !f.cache.host_victims(f.program, 32, entrant),
            "input replay and endpoint did not share their owner's retention priority");
}

void test_branch_adoption_time_is_independent_of_finish_order() {
    Fixture f(256);
    const auto parent   = f.program.add({1, 2}, {f.program.host_block(16)}, Role::Continuation);
    const auto original = f.publish(parent, 42);
    const Base branch{.tokens = {1, 2, 3},
                      .advice = {.update_session_index = false, .session_key = 42}};
    const auto first_choice = f.source(branch, parent);
    require(!first_choice.take_over && !first_choice.source.consume_source,
            "branch disabling session publication consumed its named predecessor");
    const auto first       = f.cache.adopt(f.program, first_choice, branch, 10);
    const auto first_used  = f.cache.retention(first).last_demand;
    const auto second      = f.cache.adopt(f.program, f.source(branch, parent), branch, 11);
    const auto second_used = f.cache.retention(second).last_demand;
    require(first != original && second != original && first != second && first_used < second_used,
            "independent branches did not acquire distinct real adoption times");
    const auto a = f.program.add({1, 2, 3}, {f.program.host_block(16)}, Role::Continuation);
    const auto b = f.program.add({1, 2, 4}, {f.program.host_block(16)}, Role::Continuation);
    f.cache.publish(f.program, second, b);
    f.cache.finish(f.program, second, branch, 11);
    f.cache.publish(f.program, first, a);
    f.cache.finish(f.program, first, branch, 10);
    require(f.cache.retention(first).last_demand == first_used &&
                f.cache.retention(second).last_demand == second_used &&
                f.program.valid_checkpoint(parent),
            "late completion fabricated recency or destroyed the independent parent");
    const auto victims = f.cache.host_victims(f.program, 224, std::nullopt);
    require(victims && std::find(victims->begin(), victims->end(), a) != victims->end() &&
                std::find(victims->begin(), victims->end(), parent) == victims->end(),
            "writeback protection used completion order rather than actual adoption order");
}

void test_shared_source_does_not_inherit_private_heat() {
    Fixture f(128);
    (void)f.program.host_block(64, true);
    const auto shared = f.program.add({1, 2}, {f.program.host_block(16)});
    f.publish(shared);
    const Base base{.tokens = {1, 2, 3}, .advice = {.session_key = 42}};
    const auto owner = f.begin(base, shared);
    require(!f.cache.retention(owner).reused, "shared-prefix adoption heated a new private suffix");
    const auto endpoint = f.program.add({1, 2, 3}, {f.program.host_block(16)}, Role::Continuation);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, base, 20);
    const auto incoming = f.program.add({9, 8});
    f.publish(incoming);
    const auto victims = f.cache.host_victims(f.program, 48, incoming);
    require(victims && *victims == std::vector<Handle>{endpoint},
            "shared adoption failed to protect the shared point independently of its cold child");
}

void test_private_role_wins_same_position_without_erasing_shared() {
    Fixture f(64);
    const auto shared = f.program.add({1, 2});
    const auto replay = f.program.add({1, 2}, {}, Role::InputReplay);
    f.publish(shared);
    const auto owner = f.publish(replay);
    const Base base{.tokens = {1, 2, 3}};
    const auto choices = f.cache.candidates(f.program, base);
    require(choices.front().owner == owner && choices.front().source.checkpoint == replay,
            "physical position matching selected a public role before a valid private source");
    require(f.cache.recycle_input(f.program, owner) && f.program.valid_checkpoint(shared),
            "private input replacement removed its independent shared alias");
    require(f.cache.candidates(f.program, base).front().source.checkpoint == shared,
            "private retirement lost the remaining shared index entry");
}

void test_resume_preserves_owner_and_does_not_create_a_hit() {
    Fixture f(64);
    const Base base{.tokens = {1, 2, 3}};
    const auto owner  = f.begin(base);
    const auto replay = f.program.add({1, 2}, {f.program.host_block(16)}, Role::InputReplay);
    f.cache.publish(f.program, owner, replay);
    const auto choice  = f.source(base, replay, owner);
    const auto resumed = f.cache.adopt(f.program, choice, base, 1, {&replay, 1});
    require(resumed == owner && !f.cache.retention(owner).reused &&
                f.cache.points(owner) == std::vector<Handle>{replay},
            "own pause/replay changed continuation identity or counted as cross-request reuse");
    require(f.cache.erase(f.program, replay), "active relation pinned all optional input history");
    const auto root = f.source(base, {}, owner);
    require(f.cache.adopt(f.program, root, base, 1) == owner,
            "evicting all optional checkpoints destroyed an active continuation token");
    const auto shared = f.program.add({8});
    f.cache.publish(f.program, owner, shared);
    f.cache.abandon(f.program, owner);
    require(f.program.valid_checkpoint(shared), "cancellation removed an independent public entry");
}

void test_rewind_evaluation_preserves_sources_until_binding() {
    Fixture f(64);
    const Base base{.tokens = {1, 2, 3, 4, 5, 6}};
    const auto owner    = f.begin(base);
    const auto replay   = f.program.add({1, 2}, {}, Role::InputReplay);
    const auto endpoint = f.program.add({1, 2, 3, 4, 5, 6}, {}, Role::Continuation);
    const auto anchor   = f.program.add({1}, {}, Role::LongAnchor);
    const auto deeper   = f.program.add({1, 2, 3, 4}, {}, Role::LongAnchor);
    const auto leased   = f.program.add({1, 2, 3, 4, 5}, {}, Role::LongAnchor);
    const auto shared   = f.program.add({1, 2, 3});
    f.cache.publish(f.program, owner, replay);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.publish(f.program, owner, anchor);
    f.cache.publish(f.program, owner, deeper);
    f.cache.publish(f.program, owner, leased);
    f.cache.publish(f.program, owner, shared);
    f.cache.finish(f.program, owner, base, 1);
    f.program.contents[leased.index].leased = true;
    const Base rewritten{.tokens = {1, 2, 5}};
    auto choice = f.source(rewritten, replay);
    require(f.cache.prepare_source(f.program, rewritten, choice), "rewind source became invalid");
    require(f.program.valid_checkpoint(endpoint) && f.program.valid_checkpoint(deeper) &&
                choice.source.retired_points == std::vector<Handle>({endpoint, deeper}) &&
                f.program.valid_checkpoint(replay) && f.program.valid_checkpoint(anchor) &&
                f.program.valid_checkpoint(leased) && f.program.valid_checkpoint(shared) &&
                !f.cache.retention(owner).reused,
            "read-only rewind evaluation retired history or granted retirement of a reader");
    require(choice.source.private_points == std::vector<Handle>({replay, anchor}),
            "rewind preparation bypassed Native inspection of the remaining compatible points");

    auto early = f.source(rewritten, anchor);
    require(f.cache.prepare_source(f.program, rewritten, early),
            "anchor rewind source became invalid");
    require(f.program.valid_checkpoint(replay) && f.program.valid_checkpoint(anchor) &&
                early.source.retired_points == std::vector<Handle>({replay, endpoint, deeper}) &&
                f.program.valid_checkpoint(leased) && f.program.valid_checkpoint(shared) &&
                early.source.private_points == std::vector<Handle>{anchor},
            "anchor rewind retained a deeper input replay or released its leased history");
}

void test_waiting_source_survives_catalog_replacement_and_transfers_ownership() {
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3}};
    const auto point = f.program.add(base.tokens, {}, Role::Continuation);
    const auto owner = f.publish(point);
    auto choice      = f.source(base, point);
    f.cache.retain_source(f.program, 10, choice);
    f.cache.abandon(f.program, owner);
    require(f.program.valid_checkpoint(point), "catalog removal destroyed a waiting source");
    require(f.cache.candidates(f.program, base).size() == 1,
            "a waiting-only source became an advertised cache entry");
    auto retained = f.cache.retained_source(10);
    require(retained && f.cache.prepare_source(f.program, base, *retained, 10) &&
                !retained->take_over && retained->source.consume_source,
            "sole waiting owner could not consume its point independently of the old session");
    f.cache.release_source(f.program, 10);
    require(!f.program.valid_checkpoint(point), "last waiting reference leaked its checkpoint");
}

void test_waiting_sources_preserve_other_readers_and_obey_revocation_order() {
    using ninfer::runtime::ReclaimPurpose;
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3}};
    const auto point = f.program.add(base.tokens, {}, Role::Continuation);
    f.publish(point);
    auto choice = f.source(base, point);
    f.cache.retain_source(f.program, 10, choice);
    f.cache.retain_source(f.program, 20, choice);
    require(f.cache.prepare_source(f.program, base, choice, 10) && !choice.source.consume_source,
            "a binder consumed another waiting request's immutable source");
    require(!f.cache.erase(f.program, point, {ReclaimPurpose::OptionalWrite}) &&
                !f.cache.erase(f.program, point, {ReclaimPurpose::FreshAdmission, 20}),
            "optional write or younger admission revoked older waiting ownership");
    require(f.cache.prepare_source(
                f.program, base, choice, 10,
                ninfer::runtime::ReclaimRights{ReclaimPurpose::FreshAdmission, 10}) &&
                choice.source.consume_source && f.cache.source_revocations(20) == 0,
            "read-only consumption grant either failed or revoked before acceptance");
    f.cache.binding_started(10, {}, point);
    require(!f.cache.retained_source(20) && f.cache.source_revocations(20) == 1 &&
                f.cache.source_revocations(10) == 0,
            "binding did not distinguish handed-off ownership from a revoked waiter");
    f.cache.release_source(f.program, 10);
    f.cache.release_source(f.program, 20);
    require(f.program.valid_checkpoint(point), "releasing waiters destroyed the catalog owner");
}

void test_waiting_retention_allows_demotion_but_not_optional_deletion() {
    using ninfer::runtime::ReclaimPurpose;
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3}};
    const auto point = f.program.add(base.tokens, {}, Role::Continuation);
    f.program.contents[point.index].demotion_bytes = 16;
    f.publish(point);
    f.cache.retain_source(f.program, 10, f.source(base, point));
    auto cursor =
        f.cache.begin_reclaim(f.program, std::nullopt, {ReclaimPurpose::FreshAdmission, 20});
    const auto progress = f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor);
    require(progress != ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.valid_checkpoint(point) && !f.program.demoted.empty() &&
                f.cache.source_revocations(10) == 0,
            "waiting ownership pinned a Device replica or vanished during demotion");
    f.program.transfer_in_progress = false;
    require(!f.cache.erase(f.program, point, {ReclaimPurpose::OptionalWrite}),
            "optional replacement erased a retained Host source");
    require(f.cache.erase(f.program, point, {ReclaimPurpose::Execution, 20}) &&
                f.cache.source_revocations(10) == 1 && !f.cache.retained_source(10),
            "necessary progress could not revoke a waiting source");
    f.cache.release_source(f.program, 10);
}

void test_shared_host_release_checks_every_waiting_owner() {
    using ninfer::runtime::ReclaimPurpose;
    Fixture f(64);
    const auto block = f.program.host_block(64);
    const auto a     = f.program.add({1}, {block});
    const auto b     = f.program.add({2}, {block});
    f.publish(a);
    f.publish(b);
    f.cache.retain_source(f.program, 10, f.source(Base{.tokens = {1}}, a));
    f.cache.retain_source(f.program, 20, f.source(Base{.tokens = {2}}, b));
    auto fresh =
        f.cache.begin_reclaim(f.program, std::nullopt, {ReclaimPurpose::FreshAdmission, 15});
    require(f.cache.reclaim(f.program, {.host_bytes = 64}, {}, fresh) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.released.empty() && f.cache.source_revocations(10) == 0 &&
                f.cache.source_revocations(20) == 0,
            "partial revocation fabricated Host capacity still owned by an older waiter");
    auto required = f.cache.begin_reclaim(f.program);
    require(f.cache.reclaim(f.program, {.host_bytes = 64}, {}, required) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.physical_usage().occupied.host_bytes == 0 &&
                f.cache.source_revocations(10) == 1 && f.cache.source_revocations(20) == 1,
            "required Host release did not revoke its complete physical holder union");
    f.cache.release_source(f.program, 10);
    f.cache.release_source(f.program, 20);
}

void test_waiting_only_source_is_not_a_public_capture_fallback() {
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3}};
    const auto hidden = f.program.add(base.tokens, {}, Role::Continuation);
    const auto owner  = f.publish(hidden);
    f.cache.retain_source(f.program, 10, f.source(base, hidden));
    f.cache.abandon(f.program, owner);
    const auto cold = f.program.add({9});
    f.publish(cold);
    const auto admission = f.cache.capture_admission(f.program, 0, base, 3);
    auto cursor          = f.cache.begin_reclaim(f.program, admission);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(cold) && f.program.valid_checkpoint(hidden),
            "an undiscoverable waiting point suppressed a useful public capture");
    f.cache.release_source(f.program, 10);
}

void test_waiting_owner_cannot_take_over_an_advanced_session() {
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3}, .advice = {.session_key = 42}};
    const auto point = f.program.add(base.tokens, {}, Role::Continuation);
    const auto owner = f.publish(point, 42);
    auto choice      = f.source(base, point);
    f.cache.retain_source(f.program, 10, choice);
    const auto child = f.cache.adopt(f.program, choice, base, 20);
    const auto next  = f.program.add({1, 2, 3, 4}, {}, Role::Continuation);
    f.cache.publish(f.program, child, next);
    f.cache.finish(f.program, child, base, 20);
    require(f.cache.prepare_source(f.program, base, choice, 10) && !choice.take_over &&
                choice.source.consume_source && f.program.valid_checkpoint(point),
            "waiting source overwrote the newer session or lost its own independent contents");
    const auto branch = f.cache.adopt(f.program, choice, base, 10);
    require(branch != owner && f.program.valid_checkpoint(next),
            "a stale continuation alias replaced the advanced owner's recovery points");
    f.cache.release_source(f.program, 10);
}

void test_older_waiter_does_not_take_over_a_newer_catalog_version() {
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3}, .advice = {.session_key = 42}};
    const auto input = f.program.add(base.tokens, {}, Role::InputReplay);
    const auto owner = f.publish(input, 42);
    auto old         = f.source(base, input);
    f.cache.retain_source(f.program, 10, old);
    const std::array carried{input};
    require(f.cache.adopt(f.program, old, base, 20, carried) == owner,
            "newer session could not preserve its existing input point");
    const auto endpoint = f.program.add({1, 2, 3, 4}, {}, Role::Continuation);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, base, 20);
    const auto choices = f.cache.candidates(f.program, base, UINT32_MAX, std::nullopt, 10, 10);
    const auto found   = std::find_if(choices.begin(), choices.end(),
                                      [&](const auto& c) { return c.source.checkpoint == input; });
    require(found != choices.end() && !found->take_over && !found->source.consume_source &&
                found->source.retired_points.empty(),
            "re-querying a newer catalog bypassed the waiting request's publication order");
    const auto branch = f.cache.adopt(f.program, *found, base, 10);
    f.cache.finish(f.program, branch, base, 10);
    require(branch != owner && f.program.valid_checkpoint(endpoint),
            "older completion overwrote a newer session's recovery point");
    f.cache.release_source(f.program, 10);
}

void test_session_submission_order_controls_only_the_named_hint() {
    Fixture f(128);
    const Base base{.tokens = {1, 2}, .advice = {.session_key = 42}};
    const auto old_owner = f.begin(base);
    const auto new_owner = f.begin(base);
    const auto old       = f.program.add({1, 2}, {}, Role::Continuation);
    const auto newer     = f.program.add({1, 2}, {}, Role::Continuation);
    f.cache.publish(f.program, new_owner, newer);
    f.cache.finish(f.program, new_owner, base, 20);
    f.cache.publish(f.program, old_owner, old);
    f.cache.finish(f.program, old_owner, base, 10);
    const auto candidates = f.cache.candidates(f.program, base);
    require(candidates.front().source.checkpoint == newer && candidates.front().session_hint &&
                f.program.valid_checkpoint(old),
            "an older late finish replaced the named hint or deleted anonymous history");
    require(!f.cache.retention(old_owner).reused && !f.cache.retention(new_owner).reused,
            "session publication manufactured a cache hit");
}

void test_reclaim_respects_native_lease_and_stops_host_eviction_at_capacity() {
    Fixture f(64);
    (void)f.program.host_block(32, true);
    const auto source                               = f.program.add({1, 9});
    f.program.contents[source.index].demotion_bytes = 16;
    const auto first                              = f.program.add({2}, {f.program.host_block(16)});
    const auto second                             = f.program.add({3}, {f.program.host_block(16)});
    f.program.contents[first.index].device_state  = false;
    f.program.contents[second.index].device_state = false;
    f.publish(source);
    f.publish(first);
    f.publish(second);
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{first} &&
                f.program.demoted == std::vector<Handle>{source} &&
                f.program.valid_checkpoint(second),
            "demotion preflight deleted more Host history than its fixed transfer required");

    Fixture pinned(0);
    const auto leased    = pinned.program.add({1});
    const auto available = pinned.program.add({2});
    pinned.publish(leased);
    pinned.publish(available);
    pinned.program.contents[leased.index].evictable = false;
    require(pinned.cache.reclaim(pinned.program, {.state_slots = 1}, {&available, 1}) ==
                ninfer::runtime::ReclaimProgress::Blocked,
            "reclaim removed an excluded or pinned Native checkpoint");
    require(pinned.cache.reclaim(pinned.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                pinned.program.released == std::vector<Handle>{available},
            "reclaim failed to release the available unpinned history");
}

void test_explicit_anchor_updates_remain_bounded_across_requests() {
    Fixture f(128);
    const Base base{.tokens = {1, 2, 3, 4, 5, 6, 7}};
    const auto owner = f.begin(base);
    std::vector<Handle> anchors;
    for (std::uint32_t end = 1; end <= 4; ++end) {
        anchors.push_back(f.program.add(Key(base.tokens.begin(), base.tokens.begin() + end), {},
                                        Role::LongAnchor));
        f.cache.publish(f.program, owner, anchors.back());
    }
    const auto replay   = f.program.add({1, 2}, {}, Role::InputReplay);
    const auto endpoint = f.program.add({1, 2, 3, 4}, {}, Role::Continuation);
    const auto shared   = f.program.add({1, 2});
    f.cache.publish(f.program, owner, replay);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.publish(f.program, owner, shared);
    f.cache.finish(f.program, owner, base, 1);
    const auto carried    = f.cache.points(owner);
    const auto next_owner = f.begin(base, endpoint, carried);
    const auto next       = f.program.add({1, 2, 3, 4, 5}, {}, Role::LongAnchor);
    f.cache.publish(f.program, next_owner, next);
    require(!f.program.valid_checkpoint(anchors[0]) && f.program.valid_checkpoint(next) &&
                f.program.valid_checkpoint(replay) && f.program.valid_checkpoint(shared),
            "fifth cross-request anchor failed to replace only the oldest explicit anchor");
    const auto updated = f.program.add({1, 2, 3}, {}, Role::LongAnchor);
    f.cache.publish(f.program, next_owner, updated);
    require(!f.program.valid_checkpoint(anchors[2]) && f.program.valid_checkpoint(anchors[1]) &&
                f.program.valid_checkpoint(updated),
            "same-frontier anchor update retired a different explicit position");
    f.program.contents[anchors[1].index].evictable = false;
    const auto blocked = f.program.add({1, 2, 3, 4, 5, 6}, {}, Role::LongAnchor);
    f.cache.publish(f.program, next_owner, blocked);
    require(!f.program.valid_checkpoint(blocked) && f.program.valid_checkpoint(anchors[1]) &&
                f.program.valid_checkpoint(anchors[3]) && f.program.valid_checkpoint(next),
            "leased oldest anchor caused a request failure or a search for a different victim");
}

void test_forked_hot_records_do_not_bypass_the_soft_capacity_target() {
    Fixture f(64);
    f.program.main_capacity = 100;
    (void)f.program.host_block(32, true);
    const auto parent = f.program.add({1, 2}, {f.program.host_block(16)}, Role::Continuation);
    f.program.contents[parent.index].main_pages = 40;
    f.publish(parent, 42);
    const Base branch{.tokens = {1, 2, 3},
                      .advice = {.update_session_index = false, .session_key = 42}};
    const auto owner = f.begin(branch, parent);
    const auto child = f.program.add({1, 2, 3}, {f.program.host_block(16)}, Role::Continuation);
    f.program.contents[child.index].main_pages = 40;
    f.cache.publish(f.program, owner, child);
    f.cache.finish(f.program, owner, branch, 2);
    const auto entrant = f.program.add({9, 8, 7});
    f.publish(entrant);
    const auto victims = f.cache.host_victims(f.program, 16, entrant);
    require(victims && *victims == std::vector<Handle>{parent},
            "equal-adoption parent and child bypassed the per-resource protection soft target");
}

void test_large_hotspot_is_preserved_per_resource_pool() {
    Fixture f(64);
    f.program.main_capacity = 100;
    (void)f.program.host_block(32, true);
    const auto main                             = f.program.add({1}, {f.program.host_block(16)});
    f.program.contents[main.index].main_pages   = 80;
    f.program.contents[main.index].device_state = false;
    const auto host                             = f.program.add({2}, {f.program.host_block(16)});
    f.program.contents[host.index].main_pages   = 0;
    f.program.contents[host.index].device_state = false;
    const auto entrant                          = f.program.add({3});
    f.publish(main);
    f.publish(host);
    f.publish(entrant);
    f.use_shared(main);
    f.use_shared(host);
    require(!f.cache.host_victims(f.program, 16, entrant),
            "a newer Host-only hit demoted the sole Main KV hotspot");
}

void test_one_off_fanout_competes_with_repeated_continuation() {
    Fixture f(64);
    const auto shared_prefix   = f.program.host_block(16);
    const auto dialogue_prefix = f.program.host_block(16);
    const auto parent          = f.program.add({1, 2}, {shared_prefix}, Role::Continuation);
    const auto dialogue        = f.program.add({9, 8}, {dialogue_prefix}, Role::Continuation);
    f.publish(parent, 42);
    const auto dialogue_owner = f.publish(dialogue);
    const Base branch{.tokens = {1, 2, 3},
                      .advice = {.update_session_index = false, .session_key = 42}};
    const auto warm = f.begin(branch, parent);
    f.cache.abandon(f.program, warm);

    // The slow fanout branch adopts first, but publishes after both dialogue continuations.
    const auto slow_owner        = f.begin(branch, parent);
    const auto slow_order        = f.next_order;
    const auto slow_used         = f.cache.retention(slow_owner).last_demand;
    const auto continue_dialogue = [&](Handle previous, Key tokens) {
        const Base next{.tokens = std::move(tokens)};
        const auto owner = f.begin(next, previous);
        require(owner == dialogue_owner, "dialogue continuation unexpectedly forked its owner");
        const auto endpoint = f.program.add(next.tokens, {dialogue_prefix}, Role::Continuation);
        f.cache.publish(f.program, owner, endpoint);
        f.cache.finish(f.program, owner, next, f.next_order);
        return endpoint;
    };
    const auto first_dialogue = continue_dialogue(dialogue, {9, 8, 1});
    const auto fast_owner     = f.begin(branch, parent);
    const auto fast =
        f.program.add({1, 2, 3, 4}, {shared_prefix, f.program.host_block(16)}, Role::Continuation);
    f.cache.publish(f.program, fast_owner, fast);
    f.cache.finish(f.program, fast_owner, branch, f.next_order);
    const auto recent_dialogue = continue_dialogue(first_dialogue, {9, 8, 1, 2});
    const auto recent_used     = f.cache.retention(dialogue_owner).last_demand;
    const auto slow =
        f.program.add({1, 2, 3, 5}, {shared_prefix, f.program.host_block(16)}, Role::Continuation);
    f.cache.publish(f.program, slow_owner, slow);
    f.cache.finish(f.program, slow_owner, branch, slow_order);

    const std::array<Handle, 4> all{parent, fast, recent_dialogue, slow};
    require(f.program.checkpoint_footprint(all).host_bytes == 64 &&
                f.program.physical_usage().occupied.host_bytes == 64,
            "fanout counted its shared prefix once per logical branch");
    require(slow_used < recent_used && f.cache.retention(slow_owner).last_demand == slow_used &&
                !f.cache.retention(slow_owner).reused && f.cache.retention(dialogue_owner).reused,
            "late fanout completion displaced the recently adopted dialogue's soft protection");
    auto cursor                = f.cache.begin_reclaim(f.program);
    const auto released_before = f.program.released.size();
    require(
        f.cache.reclaim(f.program, {.host_bytes = 16}, {}, cursor) ==
                ninfer::runtime::ReclaimProgress::Changed &&
            f.program.released.size() == released_before + 1 && f.program.released.back() == slow &&
            f.program.physical_usage().occupied.host_bytes == 48 &&
            f.program.valid_checkpoint(parent) && f.program.valid_checkpoint(fast) &&
            f.program.valid_checkpoint(recent_dialogue) &&
            f.cache.retention(dialogue_owner).last_demand == recent_used,
        "fixed pressure evicted recent dialogue or its shared data before older adopted fanout");
}

void test_reclaim_cursor_keeps_order_across_hits_and_transfers() {
    Fixture f(0);
    const auto first  = f.program.add({1});
    const auto second = f.program.add({2});
    f.publish(first);
    f.publish(second);
    auto cursor                    = f.cache.begin_reclaim(f.program);
    f.program.transfer_in_progress = true;
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Transferring &&
                f.program.released.empty(),
            "an in-flight transfer restarted or consumed the fixed reclaim decision");
    f.program.transfer_in_progress = false;
    f.use_shared(first);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{first} &&
                f.program.valid_checkpoint(second),
            "a hit during the decision reordered its fixed victim list");
}

void test_reclaim_cursor_reconsiders_a_previously_excluded_source() {
    Fixture f(0);
    const auto source = f.program.add({1});
    const auto other  = f.program.add({2});
    f.publish(source);
    f.publish(other);
    auto cursor = f.cache.begin_reclaim(f.program);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {&source, 1}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{other},
            "reclaim deleted its current source");
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>({other, source}),
            "root fallback could not reclaim the preceding candidate's excluded source");
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                ninfer::runtime::ReclaimProgress::Blocked,
            "completed physical reclamation manufactured further progress");
}

void test_reclaim_cursor_filters_retired_handles_and_defers_new_history() {
    Fixture f(0);
    const auto retired  = f.program.add({1});
    const auto retained = f.program.add({2});
    f.publish(retired);
    f.publish(retained);
    auto cursor = f.cache.begin_reclaim(f.program);
    require(f.program.release_checkpoint(retired), "fixture could not retire its native handle");
    const auto later = f.program.add({3});
    f.publish(later);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>({retired, retained}),
            "fixed reclaim decision dereferenced a retired native handle");
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.valid_checkpoint(later),
            "fixed reclaim decision silently expanded to newly published history");
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(later),
            "a new necessary-execution decision omitted newly available history");
}

void test_reclaim_cursor_can_demote_remaining_pages_of_the_same_history() {
    Fixture f(128);
    const auto history     = f.program.add({1});
    auto& content          = f.program.contents[history.index];
    content.device_state   = false;
    content.main_pages     = 5;
    content.demotion_bytes = 4;
    f.publish(history);
    auto cursor = f.cache.begin_reclaim(f.program);
    require(f.cache.reclaim(f.program, {.main_kv_pages = 2}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                content.main_pages == 3,
            "first finite demotion did not release only its required pages");
    require(f.cache.reclaim(f.program, {.main_kv_pages = 3}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                content.main_pages == 0 && f.program.physical_usage().occupied.host_bytes == 20,
            "later shortage could not demote the remaining pages of the same history");
    require(f.cache.reclaim(f.program, {.main_kv_pages = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                content.alive && f.program.demoted.size() == 2,
            "already migrated physical content was copied or deleted again");
}

void test_reclaim_cursor_keeps_host_writeback_permissions() {
    Fixture f(64);
    (void)f.program.host_block(32, true);
    const auto incoming                               = f.program.add({1, 9});
    f.program.contents[incoming.index].demotion_bytes = 16;
    const auto donor                             = f.program.add({2}, {f.program.host_block(16)});
    const auto protected_host                    = f.program.add({3}, {f.program.host_block(16)});
    f.program.contents[donor.index].device_state = false;
    f.program.contents[protected_host.index].device_state = false;
    f.publish(incoming);
    f.publish(donor);
    f.publish(protected_host);
    f.use_shared(protected_host);
    auto cursor = f.cache.begin_reclaim(f.program);
    f.use_shared(donor);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demoted == std::vector<Handle>{incoming} &&
                f.program.released == std::vector<Handle>{donor} &&
                f.program.valid_checkpoint(protected_host),
            "Host preflight discarded the decision's fixed victim permissions after a hit");

    Fixture cold(64);
    (void)cold.program.host_block(32, true);
    const auto source                                  = cold.program.add({1});
    cold.program.contents[source.index].demotion_bytes = 16;
    const auto hot = cold.program.add({2}, {cold.program.host_block(32)});
    cold.publish(source);
    cold.publish(hot);
    cold.use_shared(hot);
    auto cold_cursor = cold.cache.begin_reclaim(cold.program);
    cold.use_shared(source);
    require(cold.cache.reclaim(cold.program, {.state_slots = 1}, {}, cold_cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                cold.program.demoted.empty() &&
                cold.program.released == std::vector<Handle>{source} &&
                cold.program.valid_checkpoint(hot),
            "mid-decision promotion expanded a cold writeback's Host eviction permission");
}

void test_failed_demotion_keeps_scanning_independent_history() {
    Fixture f(64);
    const auto blocked                               = f.program.add({1});
    const auto next                                  = f.program.add({2});
    f.program.contents[blocked.index].demotion_bytes = 8;
    f.program.contents[next.index].demotion_bytes    = 8;
    f.publish(blocked);
    f.publish(next);
    f.program.contents[blocked.index].release_blocked  = true;
    f.program.contents[blocked.index].demotion_blocked = true;
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demoted == std::vector<Handle>{next} &&
                f.program.contents[blocked.index].alive && f.program.released.empty(),
            "one blocked demotion prevented independent writeback or discarded its history");
}

void test_failed_demotion_reports_prior_host_release() {
    Fixture f(16);
    const auto victim                             = f.program.add({1}, {f.program.host_block(16)});
    f.program.contents[victim.index].device_state = false;
    f.program.contents[victim.index].main_pages   = 0;
    const auto incoming                           = f.program.add({2, 3});
    f.program.contents[incoming.index].demotion_bytes = 8;
    f.publish(victim);
    f.publish(incoming);
    f.program.contents[incoming.index].release_blocked  = true;
    f.program.contents[incoming.index].demotion_blocked = true;
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{victim} &&
                f.program.contents[incoming.index].alive && f.program.demoted.empty(),
            "failed destination allocation hid a completed Host capacity release");
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                ninfer::runtime::ReclaimProgress::Blocked,
            "unchanged failed demotion manufactured another capacity event");
}

Key tokens(std::uint32_t count, std::uint32_t first = 1) {
    Key result;
    for (std::uint32_t i = 0; i < count; ++i) { result.push_back(first + i); }
    return result;
}

void test_necessary_execution_preserves_long_recovery_across_one_off_requests() {
    Fixture f(0);
    const Base original{.tokens = tokens(100)};
    const auto owner    = f.begin(original);
    const auto input    = f.program.add(tokens(100), {}, Role::InputReplay);
    const auto endpoint = f.program.add(tokens(101), {}, Role::Continuation);
    f.cache.publish(f.program, owner, input);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, original, 1);
    auto short_history = f.program.add(tokens(3, 1000));
    f.publish(short_history);
    require(
        f.cache.reclaim(f.program, {.state_slots = 1}) ==
                ninfer::runtime::ReclaimProgress::Changed &&
            !f.program.valid_checkpoint(endpoint) && f.program.valid_checkpoint(input),
        "necessary execution discarded the long input instead of its one-token endpoint extension");
    for (std::uint32_t request = 0; request < 4; ++request) {
        require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                        ninfer::runtime::ReclaimProgress::Changed &&
                    !f.program.valid_checkpoint(short_history) && f.program.valid_checkpoint(input),
                "one-off short requests displaced the last long recovery point");
        short_history = f.program.add(tokens(3, 2000 + 10 * request));
        f.publish(short_history);
    }
    require(f.cache.erase(f.program, short_history), "could not retire final interference history");
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(input),
            "ordinary recovery value became a pin against necessary execution");
}

void test_fallback_requires_surviving_compatible_earlier_history() {
    Fixture f(64);
    const auto shared_storage = f.program.host_block(32);
    (void)f.program.host_block(32, true);
    const auto input    = f.program.add(tokens(100), {shared_storage}, Role::InputReplay);
    const auto endpoint = f.program.add(tokens(101), {shared_storage}, Role::Continuation);
    f.publish(input);
    f.publish(endpoint);
    const Base smaller{.tokens = tokens(50, 1000)};
    auto admission = f.cache.capture_admission(f.program, 0, smaller, 50);
    auto decision  = f.cache.begin_reclaim(f.program, admission);
    require(!f.cache.host_victims(f.program, 32, {}, {}, {}, admission, &decision),
            "joint E/R deletion counted the two removed records as each other's fallback");
    const auto survivor = f.program.add(tokens(100));
    f.publish(survivor);
    const auto accepted = f.cache.host_victims(f.program, 32, {}, {}, {}, admission, &decision);
    require(accepted && accepted->size() == 2 &&
                std::find(accepted->begin(), accepted->end(), input) != accepted->end() &&
                std::find(accepted->begin(), accepted->end(), endpoint) != accepted->end(),
            "real independent surviving recovery was not used for marginal-loss admission");
    require(f.cache.erase(f.program, survivor) &&
                !f.cache.host_victims(f.program, 32, {}, {}, {}, admission, &decision),
            "a fixed decision reused recovery value after its fallback was released");
}

void test_optional_capture_does_not_inherit_required_eviction_rights() {
    Fixture f(64);
    const auto long_history = f.program.add(tokens(100), {f.program.host_block(64)});
    f.publish(long_history);
    const Base shorter{.tokens = tokens(3, 1000)};
    const auto admission = f.cache.capture_admission(f.program, 0, shorter, 3);
    auto cursor          = f.cache.begin_reclaim(f.program, admission);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                !f.cache.host_victims(f.program, 64, {}, {}, {}, admission) &&
                f.program.valid_checkpoint(long_history),
            "optional short capture erased more recovery work than it added");
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                ninfer::runtime::ReclaimProgress::Changed,
            "necessary execution accidentally inherited optional admission restrictions");
}

void test_physical_host_alias_uses_all_holders_priority() {
    Fixture f(64);
    const auto block = f.program.host_block(64);
    const auto cold  = f.program.add({1}, {block});
    const auto hot   = f.program.add({2}, {block});
    f.publish(cold);
    f.publish(hot);
    f.use_shared(hot);
    const Base incoming{.tokens = tokens(100, 1000)};
    const auto admission = f.cache.capture_admission(f.program, 0, incoming, 100);
    require(!f.cache.host_victims(f.program, 64, {}, {}, {}, admission) &&
                f.program.released.empty(),
            "cold Host alias granted permission to remove its hot physical co-owner");
}

void test_committed_repeated_demand_admits_a_new_short_working_set() {
    Fixture f(0);
    const auto old = f.program.add(tokens(100));
    f.publish(old);
    f.use_shared(old);
    const Base incoming{.tokens = tokens(3, 1000), .captures = {2}};
    const auto permission = [&] {
        return f.cache.capture_admission(f.program, 0, incoming, 2, false);
    };
    f.cache.observe_committed(incoming, 7, 0, 1);
    require(!permission().priority.reused, "work before a finite candidate created demand");
    f.cache.observe_committed(incoming, 7, 1, 3);
    f.cache.observe_committed(incoming, 7, 0, 3);
    require(!permission().priority.reused, "same-request repeated observation created reuse");
    f.cache.observe_committed(incoming, 8, 0, 3);
    const auto admission = permission();
    require(admission.priority.reused, "two completed requests failed to qualify new demand");
    auto cursor = f.cache.begin_reclaim(f.program, admission);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(old),
            "old long history permanently excluded the repeated short working set");
    const Base old_input{.tokens = tokens(100)};
    require(f.cache.capture_admission(f.program, 0, old_input, 100, false).priority.reused,
            "retiring physical history discarded the bounded record of its actual adoption");
    const auto new_point = f.program.add(tokens(2, 1000));
    f.publish(new_point);
    const Base unrelated{.tokens = {9999}};
    auto cold =
        f.cache.begin_reclaim(f.program, f.cache.capture_admission(f.program, 0, unrelated, 1));
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, cold) ==
                ninfer::runtime::ReclaimProgress::Blocked,
            "publishing the newly learned Shared point lost its actual demand evidence");
}

void test_optional_capture_budget_spans_host_and_device_actions() {
    Fixture f(64);
    const auto host = f.program.add(tokens(60, 1000), {f.program.host_block(64)});
    f.program.contents[host.index].device_state = false;
    const auto device                           = f.program.add(tokens(60, 2000));
    f.publish(host);
    f.publish(device);
    const Base incoming{.tokens = tokens(100)};
    const auto admission    = f.cache.capture_admission(f.program, 0, incoming, 100);
    auto decision           = f.cache.begin_reclaim(f.program, admission);
    const auto host_victims = f.cache.host_victims(f.program, 64, {}, {}, {}, admission, &decision);
    require(host_victims && *host_victims == std::vector<Handle>{host},
            "first Host action did not fit optional capture's recovery budget");
    f.cache.commit_host_victims(f.program, *host_victims, decision);
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, decision) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.valid_checkpoint(device),
            "optional capture spent its full recovery benefit again on a second resource pool");
}

void test_snapshot_revocation_follows_ordinary_history_and_engine_order() {
    Fixture f(96);
    const auto ordinary = f.program.add(tokens(20), {f.program.host_block(32)});
    const auto first    = f.program.add(tokens(100, 1000), {f.program.host_block(32)});
    const auto second   = f.program.add({2000}, {f.program.host_block(32)});
    f.publish(ordinary);
    f.use_shared(ordinary);
    const std::array snapshots{first, second};
    const auto one = f.cache.host_victims(f.program, 32, {}, snapshots);
    require(one && *one == std::vector<Handle>{ordinary},
            "a paused Snapshot was revoked while ordinary Host history could satisfy the request");
    const auto two = f.cache.host_victims(f.program, 64, {}, snapshots);
    require(two && *two == std::vector<Handle>({ordinary, first}),
            "Snapshot size or descriptor order overrode the Engine's explicit revocation order");
}

void test_deeper_survivor_cannot_replace_an_earlier_recovery_point() {
    Fixture f(64);
    const auto storage  = f.program.host_block(64);
    const auto earlier  = f.program.add(tokens(50), {storage});
    const auto endpoint = f.program.add(tokens(100), {storage});
    const auto survivor = f.program.add(tokens(99));
    f.publish(earlier);
    f.publish(endpoint);
    f.publish(survivor);
    const Base incoming{.tokens = tokens(20, 1000)};
    require(!f.cache.host_victims(f.program, 64, {}, {}, {},
                                  f.cache.capture_admission(f.program, 0, incoming, 20)),
            "a later recurrent state was incorrectly used to recover an earlier input frontier");
}

void test_inherited_owner_heat_cannot_spend_a_full_other_continuation() {
    Fixture f(64);
    const auto old_input = f.program.add(tokens(2039), {}, Role::InputReplay);
    const auto owner     = f.publish(old_input);
    const auto shared    = f.program.add(tokens(2039));
    f.publish(shared);
    const auto other =
        f.program.add(tokens(2079, 10000), {f.program.host_block(64)}, Role::Continuation);
    f.publish(other);
    const Base next{.tokens = tokens(2091)};
    const auto continuing = f.begin(next, old_input);
    require(continuing == owner && f.cache.retention(owner).reused,
            "fixture did not inherit actual private adoption evidence");
    const auto admission = f.cache.capture_admission(f.program, owner, next, 2091);
    auto decision        = f.cache.begin_reclaim(f.program, admission);
    require(!f.cache.host_victims(f.program, 64, {}, {}, {}, admission, &decision) &&
                f.program.valid_checkpoint(other),
            "a new 52-token refinement borrowed owner heat to erase another complete dialogue");
}

void test_exact_repeated_candidate_can_replace_a_long_hotspot_with_common_fallback() {
    Fixture f(64);
    const auto system = f.program.add(tokens(32));
    const auto old    = f.program.add(tokens(55000, 10000), {f.program.host_block(64)});
    f.publish(system);
    f.publish(old);
    f.use_shared(old);
    const Base incoming{.tokens = tokens(1000), .captures = {1000}};
    f.cache.observe_committed(incoming, 100, 32, 1000);
    f.cache.observe_committed(incoming, 101, 32, 1000);
    const auto admission = f.cache.capture_admission(f.program, 0, incoming, 1000, false);
    auto decision        = f.cache.begin_reclaim(f.program, admission);
    const auto replaced  = f.cache.host_victims(f.program, 64, {}, {}, {}, admission, &decision);
    require(
        replaced && *replaced == std::vector<Handle>{old},
        "a generic system fallback permanently excluded a repeatedly demanded shorter candidate");
}

void test_old_candidate_demand_cannot_borrow_a_new_owner_epoch() {
    Fixture f(64);
    const auto candidate = f.program.add(tokens(100));
    f.publish(candidate);
    f.use_shared(candidate);
    require(f.cache.erase(f.program, candidate), "could not retire candidate contents");
    const auto incumbent = f.program.add(tokens(500, 1000), {f.program.host_block(64)});
    f.publish(incumbent);
    f.use_shared(incumbent);
    const auto source = f.program.add(tokens(10), {}, Role::InputReplay);
    const auto owner  = f.publish(source);
    const Base request{.tokens = tokens(100)};
    require(f.begin(request, source) == owner, "fixture did not establish newer owner adoption");
    const auto admission = f.cache.capture_admission(f.program, owner, request, 100);
    auto decision        = f.cache.begin_reclaim(f.program, admission);
    require(!f.cache.host_victims(f.program, 64, {}, {}, {}, admission, &decision),
            "an older exact candidate borrowed newer owner heat to displace a more recent demand");
}

void test_demotion_competes_with_its_actual_host_recovery_loss() {
    Fixture f(32);
    const auto movable                               = f.program.add(tokens(2000));
    f.program.contents[movable.index].demotion_bytes = 32;
    const auto device                                = f.program.add(tokens(500, 10000));
    const auto host = f.program.add(tokens(1000, 20000), {f.program.host_block(32)});
    f.program.contents[host.index].device_state = false;
    f.publish(movable);
    f.publish(device);
    f.publish(host);
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(device) && f.program.valid_checkpoint(host) &&
                f.program.valid_checkpoint(movable) && f.program.demoted.empty(),
            "zero-loss Device quote hid a more expensive Host history deletion");
}

void test_adjacent_kv_demotions_share_one_submission() {
    Fixture f(64);
    const auto first                                = f.program.add(tokens(100));
    const auto second                               = f.program.add(tokens(200, 1000));
    f.program.contents[first.index].main_pages      = 2;
    f.program.contents[second.index].main_pages     = 3;
    f.program.contents[first.index].demotion_bytes  = 4;
    f.program.contents[second.index].demotion_bytes = 4;
    f.publish(first);
    f.publish(second);
    require(f.cache.reclaim(f.program, {.main_kv_pages = 4}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demotion_submissions ==
                    std::vector<std::vector<Handle>>{{first, second}} &&
                f.program.contents[first.index].main_pages == 0 &&
                f.program.contents[second.index].main_pages == 1 &&
                f.program.physical_usage().occupied.host_bytes == 16 && f.program.released.empty(),
            "adjacent admitted KV transfers did not satisfy the page shortage in one submission");
}

void test_kv_demotion_batch_stops_at_an_intervening_delete() {
    Fixture f(64);
    const auto first                               = f.program.add(tokens(100));
    const auto removable                           = f.program.add(tokens(50, 1000));
    const auto later                               = f.program.add(tokens(100, 2000));
    const auto survivor                            = f.program.add(tokens(50, 1000));
    f.program.contents[first.index].demotion_bytes = 4;
    f.program.contents[later.index].demotion_bytes = 4;
    f.program.contents[survivor.index].leased      = true;
    f.publish(first);
    f.publish(removable);
    f.publish(later);
    f.publish(survivor);
    require(f.cache.reclaim(f.program, {.main_kv_pages = 3}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demotion_submissions == std::vector<std::vector<Handle>>{{first}} &&
                f.program.contents[later.index].main_pages == 1 && f.program.released.empty(),
            "KV transfer batching skipped an intervening lower-ranked delete");
}

void test_redundant_delete_does_not_split_a_kv_batch_and_remains_a_fallback() {
    for (const bool reject_transfer : {false, true}) {
        Fixture f(64);
        const auto first                                 = f.program.add(tokens(100));
        const auto second                                = f.program.add(tokens(100, 1000));
        const auto survivor                              = f.program.add(tokens(100));
        f.program.contents[first.index].demotion_bytes   = 4;
        f.program.contents[second.index].demotion_bytes  = 4;
        f.program.contents[first.index].demotion_blocked = reject_transfer;
        f.program.contents[survivor.index].leased        = true;
        f.publish(first);
        f.publish(second);
        f.publish(survivor);
        require(f.cache.reclaim(f.program, {.main_kv_pages = 2}) ==
                    ninfer::runtime::ReclaimProgress::Changed,
                "redundant-delete fixture failed to make legitimate reclamation progress");
        if (!reject_transfer) {
            require(f.program.demotion_submissions ==
                            std::vector<std::vector<Handle>>{{first, second}} &&
                        f.program.valid_checkpoint(first) && f.program.valid_checkpoint(second) &&
                        f.program.valid_checkpoint(survivor) && f.program.released.empty(),
                    "a covered delete split the transfer or erased retained checkpoint metadata");
        } else {
            require(f.program.demotion_submissions.empty() &&
                        f.program.released == std::vector<Handle>{first} &&
                        f.program.valid_checkpoint(survivor) &&
                        f.program.contents[second.index].main_pages == 1,
                    "failed migration lost its original delete alternative or skipped to a later "
                    "action");
        }
    }
}

void test_partially_covered_delete_still_stops_a_kv_batch() {
    Fixture f(64);
    const auto first                                    = f.program.add(tokens(100));
    const auto second                                   = f.program.add(tokens(100, 1000));
    const auto survivor                                 = f.program.add(tokens(100));
    f.program.contents[first.index].main_pages          = 2;
    f.program.contents[first.index].demotion_page_limit = 1;
    f.program.contents[first.index].demotion_bytes      = 4;
    f.program.contents[second.index].demotion_bytes     = 4;
    f.program.contents[survivor.index].leased           = true;
    f.publish(first);
    f.publish(second);
    f.publish(survivor);
    require(f.cache.reclaim(f.program, {.main_kv_pages = 3}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demotion_submissions == std::vector<std::vector<Handle>>{{first}} &&
                f.program.contents[first.index].main_pages == 1 &&
                f.program.contents[second.index].main_pages == 1 && f.program.released.empty(),
            "partial page coverage hid a delete that could release another required page");
}

void test_kv_demotion_batch_preserves_host_and_outer_permission_limits() {
    Fixture bounded(12);
    const auto first  = bounded.program.add(tokens(100));
    const auto second = bounded.program.add(tokens(100, 1000));
    for (const auto handle : {first, second}) {
        bounded.program.contents[handle.index].main_pages     = 2;
        bounded.program.contents[handle.index].demotion_bytes = 4;
        bounded.publish(handle);
    }
    require(bounded.cache.reclaim(bounded.program, {.main_kv_pages = 4}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                bounded.program.demotion_submissions ==
                    std::vector<std::vector<Handle>>{{first, second}} &&
                bounded.program.contents[first.index].main_pages == 0 &&
                bounded.program.contents[second.index].main_pages == 1 &&
                bounded.program.physical_usage().occupied.host_bytes == 12 &&
                bounded.program.released.empty(),
            "combined KV transfer exceeded available Host space or deleted unquoted history");

    Fixture protected_owner(64);
    const auto replay = protected_owner.program.add(tokens(100), {}, Role::InputReplay);
    const auto owner  = protected_owner.publish(replay);
    const Base continued{.tokens = tokens(200)};
    require(protected_owner.begin(continued, replay, {&replay, 1}) == owner,
            "fixture failed to establish adopted private progress");
    const auto anchor = protected_owner.program.add(tokens(100), {}, Role::LongAnchor);
    protected_owner.cache.publish(protected_owner.program, owner, anchor);
    protected_owner.cache.finish(protected_owner.program, owner, continued,
                                 protected_owner.next_order);
    for (const auto handle : {replay, anchor}) {
        protected_owner.program.contents[handle.index].demotion_bytes = 4;
        // Native offers migration but no deletion in this fixture. Each independent
        // migration leaves an exact fallback; the union removes both Device sources.
        protected_owner.program.contents[handle.index].release_blocked = true;
    }
    const Base incoming{.tokens = tokens(1000, 10000)};
    const auto admission =
        protected_owner.cache.capture_admission(protected_owner.program, 0, incoming, 1000, false);
    auto cursor = protected_owner.cache.begin_reclaim(protected_owner.program, admission);
    require(protected_owner.cache.reclaim(protected_owner.program, {.main_kv_pages = 2}, {},
                                          cursor) == ninfer::runtime::ReclaimProgress::Changed &&
                protected_owner.program.demotion_submissions.size() == 1 &&
                protected_owner.program.demotion_submissions.front().size() == 1 &&
                protected_owner.program.contents[replay.index].main_pages +
                        protected_owner.program.contents[anchor.index].main_pages ==
                    1 &&
                protected_owner.program.released.empty(),
            "individually admitted migrations lent the outer capture combined hot protection");
}

void test_failed_kv_demotion_batch_falls_back_without_host_deletion() {
    Fixture f(64);
    const auto first  = f.program.add(tokens(100));
    const auto second = f.program.add(tokens(200, 1000));
    const auto host   = f.program.add(tokens(10, 2000), {f.program.host_block(32)});
    f.program.contents[host.index].device_state     = false;
    f.program.contents[host.index].main_pages       = 0;
    f.program.contents[first.index].demotion_bytes  = 4;
    f.program.contents[second.index].demotion_bytes = 4;
    f.program.combined_demotion_blocked             = true;
    f.publish(first);
    f.publish(second);
    f.publish(host);
    require(f.cache.reclaim(f.program, {.main_kv_pages = 2}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demotion_submissions == std::vector<std::vector<Handle>>{{first}} &&
                f.program.contents[second.index].main_pages == 1 &&
                f.program.valid_checkpoint(host) && f.program.released.empty() &&
                f.program.physical_usage().occupied.host_bytes == 36,
            "a rejected combined transfer lost the first action or deleted unrelated Host history");
}

void test_optional_capture_cannot_borrow_a_demotion_sources_heat() {
    Fixture f(64);
    f.program.main_capacity = 64;
    const auto host         = f.program.add(tokens(100, 10000), {f.program.host_block(64)});
    f.program.contents[host.index].device_state     = false;
    f.program.contents[host.index].main_pages       = 0;
    const auto source                               = f.program.add(tokens(2000));
    f.program.contents[source.index].main_pages     = 48;
    f.program.contents[source.index].demotion_bytes = 64;
    const auto newer                                = f.program.add(tokens(10, 20000));
    f.publish(host);
    f.publish(source);
    f.publish(newer);
    f.use_shared(host);
    f.use_shared(source);
    // The newer Main KV demand moves source out of the hot segment. Its exact demand
    // remains recorded, while the independent Host-only history remains hot.
    f.use_shared(newer);

    const Base incoming{.tokens = tokens(1000, 30000)};
    const auto admission = f.cache.capture_admission(f.program, 0, incoming, 1000, false);
    auto decision        = f.cache.begin_reclaim(f.program, admission);
    require(!f.cache.host_victims(f.program, 64, {}, {}, {}, admission, &decision),
            "cold optional capture could directly evict the hot Host history");
    // Deleting source costs 2000 tokens and exceeds this capture's gain. Moving source
    // would cost only the Host victim's 100 tokens, but must not borrow its hotter demand.
    require(f.cache.reclaim(f.program, {.state_slots = 1}, {}, decision) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.valid_checkpoint(host) && f.program.valid_checkpoint(source) &&
                f.program.valid_checkpoint(newer) && f.program.released.empty() &&
                f.program.demoted.empty(),
            "cold optional capture borrowed a demotion source's demand to erase hot Host history");
}

void test_unproven_suffix_can_retreat_without_losing_a_new_dialogue() {
    Fixture f(0);
    const auto predecessor = f.program.add(tokens(2079), {}, Role::Continuation);
    const auto owner       = f.publish(predecessor);
    const Base next{.tokens = tokens(2131)};
    require(f.begin(next, predecessor) == owner, "fixture did not continue its owner");
    const auto replay   = f.program.add(tokens(2091), {}, Role::InputReplay);
    const auto endpoint = f.program.add(tokens(2131), {}, Role::Continuation);
    f.cache.publish(f.program, owner, replay);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, next, f.next_order);
    const auto newcomer = f.program.add(tokens(2079, 10000), {}, Role::Continuation);
    f.publish(newcomer);
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(endpoint) && f.program.valid_checkpoint(replay) &&
                f.program.valid_checkpoint(newcomer),
            "an unproven suffix inherited absolute protection over another complete dialogue");
}

void test_free_host_preserves_an_unproven_suffix_instead_of_deleting_it() {
    Fixture f(32);
    const auto predecessor = f.program.add(tokens(100), {}, Role::Continuation);
    const auto owner       = f.publish(predecessor);
    const Base next{.tokens = tokens(150)};
    require(f.begin(next, predecessor) == owner, "fixture did not continue its owner");
    const auto replay   = f.program.add(tokens(110), {}, Role::InputReplay);
    const auto endpoint = f.program.add(tokens(150), {}, Role::Continuation);
    f.program.contents[endpoint.index].demotion_bytes = 32;
    f.cache.publish(f.program, owner, replay);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, next, f.next_order);
    const auto newcomer = f.program.add(tokens(1000, 10000), {}, Role::Continuation);
    f.publish(newcomer);
    const auto released_before = f.program.released.size();
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demoted == std::vector<Handle>{endpoint} &&
                f.program.released.size() == released_before &&
                f.program.valid_checkpoint(newcomer),
            "demotion and deletion assigned different protection to the same unproven suffix");
}

void test_proven_deep_progress_survives_a_shallow_fallback_and_fork() {
    Fixture f(0);
    const Base original{.tokens = tokens(4000), .advice = {.session_key = 42}};
    const auto owner       = f.begin(original);
    const auto replay      = f.program.add(tokens(32), {}, Role::InputReplay);
    const auto predecessor = f.program.add(tokens(4000), {}, Role::Continuation);
    f.cache.publish(f.program, owner, replay);
    f.cache.publish(f.program, owner, predecessor);
    f.cache.finish(f.program, owner, original, f.next_order);
    const Base next{.tokens = tokens(4100), .advice = {.session_key = 42}};
    require(f.begin(next, predecessor, {&replay, 1}) == owner,
            "fixture did not continue deep progress");
    const auto endpoint = f.program.add(tokens(4100), {}, Role::Continuation);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, next, f.next_order);
    const auto cold = f.program.add(tokens(55000, 10000));
    f.publish(cold);
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(cold) && f.program.valid_checkpoint(endpoint),
            "a shallow R32 was treated as coverage of actually adopted E4000 progress");
    const Base fork{.tokens = tokens(64),
                    .advice = {.update_session_index = false, .session_key = 42}};
    const auto donor_demand = f.cache.retention(owner).last_demand;
    const auto child        = f.begin(fork, replay);
    require(child != owner && f.cache.retention(owner).last_demand == donor_demand &&
                f.cache.retention(child).last_demand > donor_demand,
            "a shallow Fork refreshed its donor's deep proof instead of only its child");
    const auto another = f.program.add(tokens(55000, 70000));
    f.publish(another);
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(another) && f.program.valid_checkpoint(endpoint),
            "a shallow Fork reset its unchanged donor's proven deep progress");
    const Base deep_fork{.tokens = tokens(4200),
                         .advice = {.update_session_index = false, .session_key = 42}};
    const auto deep_child = f.begin(deep_fork, endpoint);
    require(deep_child != owner && f.cache.retention(owner).last_demand > donor_demand &&
                f.cache.retention(deep_child).last_demand == f.cache.retention(owner).last_demand,
            "actual deeper adoption failed to refresh the donor and child with the same evidence");
}

void test_consuming_rewind_resets_proven_progress_to_the_adopted_prefix() {
    Fixture f(0);
    const Base original{.tokens = tokens(4000)};
    const auto owner       = f.begin(original);
    const auto replay      = f.program.add(tokens(32), {}, Role::InputReplay);
    const auto predecessor = f.program.add(tokens(4000), {}, Role::Continuation);
    f.cache.publish(f.program, owner, replay);
    f.cache.publish(f.program, owner, predecessor);
    f.cache.finish(f.program, owner, original, f.next_order);
    const Base deep{.tokens = tokens(4100)};
    require(f.begin(deep, predecessor, {&replay, 1}) == owner,
            "fixture failed to prove deep adoption");
    const auto deep_endpoint = f.program.add(tokens(4100), {}, Role::Continuation);
    f.cache.publish(f.program, owner, deep_endpoint);
    f.cache.finish(f.program, owner, deep, f.next_order);
    const Base rewind{.tokens = tokens(64)};
    require(f.begin(rewind, replay) == owner, "rewind did not consume its existing owner");
    const auto new_replay = f.program.add(tokens(48), {}, Role::InputReplay);
    const auto endpoint   = f.program.add(tokens(64), {}, Role::Continuation);
    f.cache.publish(f.program, owner, new_replay);
    f.cache.publish(f.program, owner, endpoint);
    f.cache.finish(f.program, owner, rewind, f.next_order);
    const auto other = f.program.add(tokens(1000, 10000));
    f.publish(other);
    require(f.cache.reclaim(f.program, {.state_slots = 1}) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(endpoint) && f.program.valid_checkpoint(other),
            "an abandoned deep branch remained the proof after a consuming rewind");
}

} // namespace

int main() {
    try {
        test_demotion_competes_with_its_actual_host_recovery_loss();
        test_adjacent_kv_demotions_share_one_submission();
        test_kv_demotion_batch_stops_at_an_intervening_delete();
        test_redundant_delete_does_not_split_a_kv_batch_and_remains_a_fallback();
        test_partially_covered_delete_still_stops_a_kv_batch();
        test_kv_demotion_batch_preserves_host_and_outer_permission_limits();
        test_failed_kv_demotion_batch_falls_back_without_host_deletion();
        test_optional_capture_cannot_borrow_a_demotion_sources_heat();
        test_unproven_suffix_can_retreat_without_losing_a_new_dialogue();
        test_free_host_preserves_an_unproven_suffix_instead_of_deleting_it();
        test_proven_deep_progress_survives_a_shallow_fallback_and_fork();
        test_consuming_rewind_resets_proven_progress_to_the_adopted_prefix();
        test_inherited_owner_heat_cannot_spend_a_full_other_continuation();
        test_exact_repeated_candidate_can_replace_a_long_hotspot_with_common_fallback();
        test_old_candidate_demand_cannot_borrow_a_new_owner_epoch();
        test_optional_capture_budget_spans_host_and_device_actions();
        test_snapshot_revocation_follows_ordinary_history_and_engine_order();
        test_deeper_survivor_cannot_replace_an_earlier_recovery_point();
        test_necessary_execution_preserves_long_recovery_across_one_off_requests();
        test_fallback_requires_surviving_compatible_earlier_history();
        test_optional_capture_does_not_inherit_required_eviction_rights();
        test_physical_host_alias_uses_all_holders_priority();
        test_committed_repeated_demand_admits_a_new_short_working_set();
        test_cold_writeback_protects_reused_history();
        test_host_preflight_counts_aliases_and_has_no_partial_deletion();
        test_candidates_keep_all_frontiers_and_independent_contents();
        test_private_continuation_advances_with_adoption_heat();
        test_branch_adoption_time_is_independent_of_finish_order();
        test_shared_source_does_not_inherit_private_heat();
        test_private_role_wins_same_position_without_erasing_shared();
        test_resume_preserves_owner_and_does_not_create_a_hit();
        test_rewind_evaluation_preserves_sources_until_binding();
        test_waiting_source_survives_catalog_replacement_and_transfers_ownership();
        test_waiting_sources_preserve_other_readers_and_obey_revocation_order();
        test_waiting_retention_allows_demotion_but_not_optional_deletion();
        test_shared_host_release_checks_every_waiting_owner();
        test_waiting_only_source_is_not_a_public_capture_fallback();
        test_waiting_owner_cannot_take_over_an_advanced_session();
        test_older_waiter_does_not_take_over_a_newer_catalog_version();
        test_session_submission_order_controls_only_the_named_hint();
        test_reclaim_respects_native_lease_and_stops_host_eviction_at_capacity();
        test_explicit_anchor_updates_remain_bounded_across_requests();
        test_forked_hot_records_do_not_bypass_the_soft_capacity_target();
        test_large_hotspot_is_preserved_per_resource_pool();
        test_one_off_fanout_competes_with_repeated_continuation();
        test_reclaim_cursor_keeps_order_across_hits_and_transfers();
        test_reclaim_cursor_reconsiders_a_previously_excluded_source();
        test_reclaim_cursor_filters_retired_handles_and_defers_new_history();
        test_reclaim_cursor_can_demote_remaining_pages_of_the_same_history();
        test_reclaim_cursor_keeps_host_writeback_permissions();
        test_failed_demotion_keeps_scanning_independent_history();
        test_failed_demotion_reports_prior_host_release();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
