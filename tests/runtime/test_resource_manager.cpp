#include "runtime/engine/context_cache/resource_manager.h"
#include "models/common/context/recovery.h"
#include "models/common/context/release_inventory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct Usage {
    std::size_t device_bytes = 0, host_bytes = 0;
    std::vector<std::uint32_t> records;

    void merge(const Usage& other) {
        device_bytes += other.device_bytes;
        host_bytes += other.host_bytes;
        records.insert(records.end(), other.records.begin(), other.records.end());
    }

    void subtract(const Usage& other) {
        device_bytes -= other.device_bytes;
        host_bytes -= other.host_bytes;
        for (auto id : other.records) {
            const auto found = std::find(records.begin(), records.end(), id);
            require_record(found != records.end());
            records.erase(found);
        }
    }

    bool empty() const { return !device_bytes && !host_bytes && records.empty(); }
private:
    static void require_record(bool valid) {
        if (!valid) throw std::logic_error("fixture release inventory lost a record");
    }
};

struct Demand {
    std::size_t device_bytes = 0, host_bytes = 0;
};

struct Assessment {
    Demand demand;
    bool available = false, exceeds_capacity = false, pending_dependency = false;

    bool ready() const { return available; }

    bool feasible() const { return available; }

    bool requires_maintenance() const { return false; }

    bool waiting() const { return pending_dependency; }

    bool impossible() const { return exceeds_capacity; }
};

using Key  = std::vector<std::uint32_t>;
using Role = ninfer::runtime::CheckpointRole;

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
    using ContextDemand     = Demand;
    using ContextAssessment = Assessment;
    using ContextReleaseSet = Usage;
    using HostRequirement   = std::size_t;

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
        Role role                                = Role::SharedPrefix;
        ninfer::runtime::CheckpointOrigin origin = ninfer::runtime::CheckpointOrigin::Semantic;
    };

    struct HostBlock {
        std::size_t bytes;
        bool external_reference;
    };

    struct Metadata {
        std::uint32_t frontier;
        Role role;
        bool leased;
        ninfer::runtime::CheckpointOrigin origin = ninfer::runtime::CheckpointOrigin::Semantic;
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
    std::size_t demotion_submissions = 0;
    std::size_t device_capacity      = 4096;
    bool transfer_in_progress        = false;
    std::vector<Handle> placement_holders;
    std::uint64_t checkpoint_revision = 1;
    mutable ninfer::models::context::RecoveryIndex<Handle> recovery_index;
    using ReleaseInventory = ninfer::models::context::ReleaseInventory<Usage, Handle>;
    mutable std::shared_ptr<const ReleaseInventory> last_inventory;
    mutable std::uint64_t score_queries = 0, scored_handles = 0;
    mutable std::uint64_t host_plan_builds = 0;

    struct Recovery : ninfer::models::context::RecoveryEvaluation<Handle> {
        using Core = ninfer::models::context::RecoveryEvaluation<Handle>;
        const Program* program;

        Recovery(Core core, const Program& owner) : Core(std::move(core)), program(&owner) {}

        Recovery(Recovery&& other) noexcept
            : Core(std::move(other)), program(std::exchange(other.program, nullptr)) {}

        ~Recovery() {
            if (program) {
                program->score_queries += work().queries;
                program->scored_handles += work().removed_handles;
            }
        }
    };

    std::size_t host_block(std::size_t bytes, bool external_reference = false) {
        blocks.push_back({bytes, external_reference});
        return blocks.size() - 1;
    }

    Handle add(Key tokens, std::vector<std::size_t> host_blocks = {},
               Role role = Role::SharedPrefix) {
        const Handle handle{static_cast<std::uint32_t>(contents.size())};
        contents.push_back({std::move(tokens), std::move(host_blocks)});
        ++checkpoint_revision;
        contents.back().role = role;
        require(physical_usage().occupied.host_bytes <= host_capacity,
                "fixture exceeded its physical Host capacity");
        return handle;
    }

    [[nodiscard]] bool references(const Content& content, std::size_t block) const {
        return content.alive && std::find(content.host_blocks.begin(), content.host_blocks.end(),
                                          block) != content.host_blocks.end();
    }

    static std::size_t device_size(const Content& content) {
        return content.device_state + content.main_pages;
    }

    [[nodiscard]] Usage checkpoint_footprint(std::span<const Handle> handles) const {
        Usage result;
        std::vector<std::size_t> unique;
        for (const auto handle : handles) {
            const auto& content = contents.at(handle.index);
            require(content.alive, "cache queried a released checkpoint");
            result.device_bytes += device_size(content);
            for (const auto block : content.host_blocks) {
                if (std::find(unique.begin(), unique.end(), block) == unique.end()) {
                    unique.push_back(block);
                    result.host_bytes += blocks.at(block).bytes;
                }
            }
        }
        return result;
    }

    Assessment assess(Demand demand, const Usage& released = {}) const {
        const auto usage = physical_usage();
        return {demand,
                demand.device_bytes <=
                        device_capacity - usage.occupied.device_bytes + released.device_bytes &&
                    can_place_host(demand.host_bytes, released),
                demand.device_bytes > device_capacity || demand.host_bytes > host_capacity};
    }

    Assessment pressure(std::size_t device_bytes = 0, std::size_t host_bytes = 0) const {
        const auto usage = physical_usage();
        return assess(
            {device_bytes ? device_capacity - usage.occupied.device_bytes + device_bytes : 0,
             host_bytes ? host_capacity - usage.occupied.host_bytes + host_bytes : 0});
    }

    Demand host_demand(std::size_t bytes) const { return {.host_bytes = bytes}; }

    bool releases_metadata(const Usage&, const Assessment&) const { return false; }

    std::size_t released_bytes(const Usage& released, const Assessment& assessment) const {
        return (assessment.demand.device_bytes ? released.device_bytes : 0) +
               (assessment.demand.host_bytes ? released.host_bytes : 0);
    }

    std::size_t useful_release_bytes(const Usage& released, const Assessment& assessment) const {
        const auto usage    = physical_usage();
        const auto shortage = [](std::size_t requested, std::size_t free) {
            return requested > free ? requested - free : 0;
        };
        return std::min(released.device_bytes,
                        shortage(assessment.demand.device_bytes,
                                 device_capacity - usage.occupied.device_bytes)) +
               std::min(released.host_bytes, shortage(assessment.demand.host_bytes,
                                                      host_capacity - usage.occupied.host_bytes));
    }

    ninfer::runtime::ContextStart start_maintenance(const Assessment&) {
        throw std::logic_error("the policy fixture does not require relocation");
    }

    std::size_t checkpoint_capacity() const { return contents.size(); }

    std::vector<ninfer::runtime::ContextTransferRequirement>
    checkpoint_restore_requirements(Handle h) const {
        return inspect_source(Base{contents[h.index].tokens}, h)->transfers;
    }

    [[nodiscard]] Metadata checkpoint_metadata(Handle handle) const {
        const auto& content = contents.at(handle.index);
        require(content.alive, "cache queried a released checkpoint");
        return {static_cast<std::uint32_t>(content.tokens.size()), content.role, content.leased,
                content.origin};
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
        PhysicalUsage result{
            .capacity = {.device_bytes = device_capacity, .host_bytes = host_capacity}};
        for (const auto& content : contents)
            if (content.alive) result.occupied.device_bytes += device_size(content);
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
        ++checkpoint_revision;
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

    [[nodiscard]] bool has_context_operation() const { return transfer_in_progress; }

    bool release_redundant_host(std::span<const Handle>) { return false; }

    struct Demotion {
        Handle handle;
        std::size_t host_bytes;
        std::vector<Handle> sources;
        Usage released;
        std::vector<std::pair<Handle, std::uint32_t>> main_parts;
        std::vector<std::pair<Handle, std::size_t>> state_parts;
    };

    struct Release {
        std::vector<Handle> sources;
        Usage released;
    };

    struct ReclaimPlan {
        std::vector<Demotion> demotions;
        std::vector<Release> releases;
        std::shared_ptr<const ReleaseInventory> inventory;
    };

    [[nodiscard]] ReclaimPlan plan_reclaim(std::span<const Handle> allowed,
                                           std::span<const Handle> excluded, Demand demand) const {
        auto releases = plan_releases(allowed, excluded, demand);
        return {demotion_options(allowed, excluded, demand), std::move(releases.releases),
                std::move(releases.inventory)};
    }

    [[nodiscard]] std::vector<Demotion> demotion_options(std::span<const Handle> allowed,
                                                         std::span<const Handle> excluded,
                                                         Demand demand) const {
        std::vector<Demotion> result;
        if (!demand.device_bytes) return result;
        for (const auto handle : allowed) {
            const auto& content = contents.at(handle.index);
            if (!content.alive || !content.evictable || content.leased || !content.demotion_bytes ||
                std::find(excluded.begin(), excluded.end(), handle) != excluded.end())
                continue;
            const auto pages =
                content.device_state
                    ? 0
                    : std::min(content.main_pages,
                               content.demotion_page_limit.value_or(content.main_pages));
            const auto bytes = content.device_state ? 1 : pages;
            if (!bytes) continue;
            result.push_back({handle,
                              *content.demotion_bytes * bytes,
                              {handle},
                              {.device_bytes = bytes},
                              pages ? std::vector<std::pair<Handle, std::uint32_t>>{{handle, pages}}
                                    : std::vector<std::pair<Handle, std::uint32_t>>{}});
        }
        return result;
    }

    [[nodiscard]] ninfer::models::context::ReleasePlan<Release, Usage, Handle>
    plan_releases(std::span<const Handle> allowed, std::span<const Handle> excluded,
                  Demand demand) const {
        host_plan_builds += demand.host_bytes != 0;
        const auto eligible = [&](Handle handle) {
            const auto& content = contents.at(handle.index);
            return content.alive && content.evictable && !content.leased &&
                   !content.release_blocked &&
                   std::find(allowed.begin(), allowed.end(), handle) != allowed.end() &&
                   std::find(excluded.begin(), excluded.end(), handle) == excluded.end();
        };
        auto inventory = std::make_shared<ninfer::models::context::ReleaseInventory<Usage, Handle>>(
            contents.size());
        for (std::uint32_t i = 0; i < contents.size(); ++i)
            if (contents[i].alive) inventory->include({i});
        for (std::uint32_t i = 0; i < contents.size(); ++i) {
            const auto& content = contents[i];
            if (eligible({i}))
                inventory->add({Handle{i}}, {.device_bytes = device_size(content), .records = {i}});
        }
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (blocks[block].external_reference) continue;
            std::vector<Handle> holders;
            for (std::uint32_t i = 0; i < contents.size(); ++i)
                if (references(contents[i], block)) holders.push_back({i});
            if (!holders.empty() && std::all_of(holders.begin(), holders.end(), eligible))
                inventory->add(std::move(holders), {.host_bytes = blocks[block].bytes});
        }
        ninfer::models::context::ReleasePlan<Release, Usage, Handle> result{{}, inventory};
        const auto append = [&](std::vector<Handle> sources) {
            if (sources.empty() || !std::all_of(sources.begin(), sources.end(), eligible)) {
                return;
            }
            Usage released;
            for (const auto handle : sources) {
                const auto& content = contents.at(handle.index);
                released.device_bytes += device_size(content);
                released.records.push_back(handle.index);
            }
            released.host_bytes = host_bytes_released(sources);
            if (std::none_of(result.releases.begin(), result.releases.end(),
                             [&](const auto& action) { return action.sources == sources; })) {
                result.releases.push_back({std::move(sources), released});
            }
        };
        for (const auto handle : allowed) { append({handle}); }
        if (demand.host_bytes) {
            for (std::size_t block = 0; block < blocks.size(); ++block) {
                std::vector<Handle> sources;
                for (std::uint32_t i = 0; i < contents.size(); ++i) {
                    if (references(contents[i], block)) { sources.push_back({i}); }
                }
                append(std::move(sources));
            }
        }
        last_inventory = inventory;
        return result;
    }

    [[nodiscard]] bool prefix(Handle earlier, Handle later) const {
        const auto& a = contents.at(earlier.index).tokens;
        const auto& b = contents.at(later.index).tokens;
        return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
    }

    [[nodiscard]] auto evaluate_recovery(std::span<const Handle> advertised) const {
        struct View {
            const Program& program;

            const std::uint64_t& revision() const { return program.checkpoint_revision; }

            std::size_t capacity() const { return program.contents.size(); }

            Handle handle(std::uint32_t index) const { return {index}; }

            bool valid(Handle h) const { return program.valid_checkpoint(h); }

            std::uint32_t frontier(Handle h) const {
                return program.contents[h.index].tokens.size();
            }

            ninfer::models::context::RecoveryKey key(Handle h, std::uint32_t f) const {
                std::uint64_t hash = 0;
                for (std::uint32_t i = 0; i < f; ++i)
                    hash = hash * 131 + program.contents[h.index].tokens[i];
                return {{hash, 0}, 0};
            }

            bool prefix(Handle a, Handle b) const { return program.prefix(a, b); }
        };

        recovery_index.synchronize(View{*this});
        return Recovery(recovery_index.evaluate(advertised), *this);
    }

    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(Handle retained, const Base& base,
                                                             std::uint32_t target) const {
        const auto& tokens = contents.at(retained.index).tokens;
        return tokens.size() <= target && base.prefix_shortlist_key(tokens.size()) == tokens
                   ? static_cast<std::uint32_t>(tokens.size())
                   : 0;
    }

    std::size_t demotion_host_requirements(const Demotion& plan) const { return plan.host_bytes; }

    Demotion combine_demotions(std::span<const Demotion> quotes) const {
        Demotion result{};
        for (const auto& quote : quotes) {
            result.host_bytes += quote.host_bytes;
            result.released.merge(quote.released);
            for (const auto handle : quote.sources)
                if (std::find(result.sources.begin(), result.sources.end(), handle) ==
                    result.sources.end())
                    result.sources.push_back(handle);
            result.main_parts.insert(result.main_parts.end(), quote.main_parts.begin(),
                                     quote.main_parts.end());
            result.state_parts.insert(result.state_parts.end(), quote.state_parts.begin(),
                                      quote.state_parts.end());
            if (quote.main_parts.empty() && quote.state_parts.empty())
                result.state_parts.emplace_back(quote.handle, quote.host_bytes);
        }
        return result;
    }

    std::size_t host_write_bytes(std::size_t bytes) const { return bytes; }

    bool can_place_host(std::size_t bytes) const {
        const auto usage = physical_usage();
        if (bytes && std::any_of(placement_holders.begin(), placement_holders.end(),
                                 [&](auto h) { return valid_checkpoint(h); }))
            return false;
        return bytes <= usage.capacity.host_bytes - usage.occupied.host_bytes;
    }

    bool can_place_host(std::size_t bytes, const Usage& released) const {
        const auto usage = physical_usage();
        for (const auto h : placement_holders) {
            if (!bytes || !valid_checkpoint(h)) continue;
            if (std::find(released.records.begin(), released.records.end(), h.index) ==
                released.records.end())
                return false;
        }
        return bytes <= usage.capacity.host_bytes - usage.occupied.host_bytes + released.host_bytes;
    }

    ninfer::runtime::ContextStart start_demote(const Demotion& plan,
                                               std::span<const Handle> victims) {
        using ninfer::runtime::ContextStart;
        if (std::any_of(plan.sources.begin(), plan.sources.end(),
                        [&](auto handle) { return contents.at(handle.index).demotion_blocked; }))
            return ContextStart::Deferred;
        Usage released{.host_bytes = host_bytes_released(victims)};
        for (const auto victim : victims) released.records.push_back(victim.index);
        if (!can_place_host(plan.host_bytes, released)) return ContextStart::Stale;
        for (const auto victim : victims)
            require(release_checkpoint(victim), "prepared Host victim was not releasable");
        ++demotion_submissions;
        if (!plan.main_parts.empty() || !plan.state_parts.empty()) {
            for (const auto& [handle, pages] : plan.main_parts) {
                auto& content    = contents.at(handle.index);
                const auto bytes = *content.demotion_bytes * pages;
                if (bytes) { content.host_blocks.push_back(host_block(bytes)); }
                content.main_pages -= pages;
                demoted.push_back(handle);
            }
            for (const auto& [handle, bytes] : plan.state_parts) {
                auto& content = contents.at(handle.index);
                if (bytes) content.host_blocks.push_back(host_block(bytes));
                content.device_state = false;
                demoted.push_back(handle);
            }
        } else {
            auto& content = contents.at(plan.handle.index);
            if (plan.host_bytes) { content.host_blocks.push_back(host_block(plan.host_bytes)); }
            if (plan.released.device_bytes) { content.device_state = false; }
            demoted.push_back(plan.handle);
        }
        return ContextStart::Started;
    }
};

struct Model {
    using Session          = ::Program;
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
    const auto recomputed = f.program.add({1, 2, 3, 4, 5, 6}, {}, Role::Continuation);
    f.publish(shallow);
    f.publish(middle);
    f.publish(deep);
    f.publish(recomputed);
    const Base base{.tokens = {1, 2, 3, 4, 5, 6, 7}};
    const auto all = f.cache.candidates(f.program, base);
    require(all.size() == 5 && all[0].source.checkpoint == recomputed &&
                all[1].source.checkpoint == deep && all[2].source.checkpoint == middle &&
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

void test_shared_publication_deduplicates_complete_snapshots() {
    Fixture f(128);
    const auto first  = f.program.add({1, 2, 3});
    const auto second = f.program.add({1, 2, 3});
    f.publish(first);
    f.publish(second);
    require(f.program.valid_checkpoint(first) && !f.program.valid_checkpoint(second),
            "concurrent publication retained two identical public recovery points");
    const Base request{.tokens = {1, 2, 3, 4}};
    const auto semantic = ninfer::runtime::CheckpointOrigin::Semantic;
    require(f.cache.retain_shared_checkpoint(f.program, request, 3, semantic) &&
                !f.cache.retain_shared_checkpoint(f.program, request, 2, semantic),
            "shared capture lookup accepted a position without a complete state");
    require(f.cache.candidates(f.program, request).size() == 2,
            "duplicate shared publication left an extra source");
}

void test_intermediate_capture_preserves_semantic_history() {
    using Origin = ninfer::runtime::CheckpointOrigin;
    for (const auto origin : {Origin::Semantic, Origin::Intermediate}) {
        Fixture f(0);
        const auto semantic                       = f.program.add({1, 2, 3});
        f.program.contents[semantic.index].origin = origin;
        f.publish(semantic);
        if (origin == Origin::Intermediate)
            require(f.cache.retain_shared_checkpoint(f.program, Base{.tokens = {1, 2, 3, 4}}, 3,
                                                     Origin::Semantic),
                    "semantic capture could not retain an existing intermediate snapshot");
        const Base request{.tokens = {100, 101, 102, 103}};
        const auto admission =
            f.cache.capture_admission(f.program, 0, request, 4, false, Origin::Intermediate);
        auto cursor = f.cache.begin_reclaim(f.program, admission);
        require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                        ninfer::runtime::ReclaimProgress::Blocked &&
                    f.program.valid_checkpoint(semantic),
                "cold intermediate capture displaced a semantic shared prefix");
        require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed,
                "semantic retention prevented required execution");
    }
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
    const auto progress = f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor);
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
    require(f.cache.reclaim(f.program, f.program.pressure(0, 64), {}, fresh) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.released.empty() && f.cache.source_revocations(10) == 0 &&
                f.cache.source_revocations(20) == 0,
            "partial revocation fabricated Host capacity still owned by an older waiter");
    auto required = f.cache.begin_reclaim(f.program);
    require(f.cache.reclaim(f.program, f.program.pressure(0, 64), {}, required) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
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
    // Only the Device State is reclaimable; the history has no independently owned Device pages.
    f.program.contents[source.index].main_pages   = 0;
    const auto first                              = f.program.add({2}, {f.program.host_block(16)});
    const auto second                             = f.program.add({3}, {f.program.host_block(16)});
    f.program.contents[first.index].device_state  = false;
    f.program.contents[first.index].main_pages    = 0;
    f.program.contents[second.index].device_state = false;
    f.program.contents[second.index].main_pages   = 0;
    f.publish(source);
    f.publish(first);
    f.publish(second);
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    require(pinned.cache.reclaim(pinned.program, pinned.program.pressure(1), {&available, 1}) ==
                ninfer::runtime::ReclaimProgress::Blocked,
            "reclaim removed an excluded or pinned Native checkpoint");
    require(pinned.cache.reclaim(pinned.program, pinned.program.pressure(1)) ==
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
                f.cache.retention(dialogue_owner).reused,
            "late fanout completion displaced the recently adopted dialogue's soft protection");
    auto cursor                = f.cache.begin_reclaim(f.program);
    const auto released_before = f.program.released.size();
    require(
        f.cache.reclaim(f.program, f.program.pressure(0, 16), {}, cursor) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Transferring &&
                f.program.released.empty(),
            "an in-flight transfer restarted or consumed the fixed reclaim decision");
    f.program.transfer_in_progress = false;
    f.use_shared(first);
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {&source, 1}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{other},
            "reclaim deleted its current source");
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>({other, source}),
            "root fallback could not reclaim the preceding candidate's excluded source");
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>({retired, retained}),
            "fixed reclaim decision dereferenced a retired native handle");
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                f.program.valid_checkpoint(later),
            "fixed reclaim decision silently expanded to newly published history");
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(later),
            "a new necessary-execution decision omitted newly available history");
}

void test_reclaim_cursor_can_demote_remaining_pages_of_the_same_history() {
    Fixture f(128);
    const auto history          = f.program.add({1});
    auto& content               = f.program.contents[history.index];
    content.device_state        = false;
    content.main_pages          = 5;
    content.demotion_page_limit = 2;
    content.demotion_bytes      = 4;
    f.publish(history);
    auto cursor = f.cache.begin_reclaim(f.program);
    require(f.cache.reclaim(f.program, f.program.pressure(2), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                content.main_pages == 3,
            "first finite demotion did not release only its required pages");
    content.demotion_page_limit = 3;
    require(f.cache.reclaim(f.program, f.program.pressure(3), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                content.main_pages == 0 && f.program.physical_usage().occupied.host_bytes == 20,
            "later shortage could not demote the remaining pages of the same history");
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                content.alive && f.program.demoted.size() == 2,
            "already migrated physical content was copied or deleted again");
}

void test_reclaim_cursor_keeps_host_writeback_permissions() {
    Fixture f(64);
    (void)f.program.host_block(32, true);
    const auto incoming                               = f.program.add({1, 9});
    f.program.contents[incoming.index].demotion_bytes = 16;
    // Only the Device State is reclaimable; the history has no independently owned Device pages.
    f.program.contents[incoming.index].main_pages = 0;
    const auto donor                              = f.program.add({2}, {f.program.host_block(16)});
    const auto protected_host                     = f.program.add({3}, {f.program.host_block(16)});
    f.program.contents[donor.index].device_state  = false;
    f.program.contents[donor.index].main_pages    = 0;
    f.program.contents[protected_host.index].device_state = false;
    f.program.contents[protected_host.index].main_pages   = 0;
    f.publish(incoming);
    f.publish(donor);
    f.publish(protected_host);
    f.use_shared(protected_host);
    auto cursor = f.cache.begin_reclaim(f.program);
    f.use_shared(donor);
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
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
    require(cold.cache.reclaim(cold.program, cold.program.pressure(1), {}, cold_cursor) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                cold.program.demoted.empty() &&
                cold.program.released == std::vector<Handle>{source} &&
                cold.program.valid_checkpoint(hot),
            "mid-decision promotion expanded a cold writeback's Host eviction permission");
}

void test_deferred_demotion_keeps_sources() {
    Fixture f(64);
    const auto blocked                               = f.program.add({1});
    const auto next                                  = f.program.add({2});
    f.program.contents[blocked.index].demotion_bytes = 8;
    f.program.contents[next.index].demotion_bytes    = 8;
    f.publish(blocked);
    f.publish(next);
    f.program.contents[blocked.index].release_blocked  = true;
    f.program.contents[blocked.index].demotion_blocked = true;
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Deferred &&
                f.program.demoted.empty() && f.program.valid_checkpoint(next) &&
                f.program.contents[blocked.index].alive && f.program.released.empty(),
            "deferred demotion deleted a source or proceeded with an unreserved transfer");
}

void test_deferred_demotion_preserves_host_victims() {
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Deferred &&
                f.program.released.empty() && f.program.valid_checkpoint(victim) &&
                f.program.contents[incoming.index].alive && f.program.demoted.empty(),
            "deferred transfer discarded its planned Host victims");
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                ninfer::runtime::ReclaimProgress::Deferred,
            "deferred transfer was treated as a capacity change");
    f.program.contents[incoming.index].demotion_blocked = false;
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{victim} &&
                f.program.valid_checkpoint(incoming),
            "ready transfer could not commit its planned Host replacement");
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
        f.cache.reclaim(f.program, f.program.pressure(1)) ==
                ninfer::runtime::ReclaimProgress::Changed &&
            !f.program.valid_checkpoint(endpoint) && f.program.valid_checkpoint(input),
        "necessary execution discarded the long input instead of its one-token endpoint extension");
    for (std::uint32_t request = 0; request < 4; ++request) {
        require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                        ninfer::runtime::ReclaimProgress::Changed &&
                    !f.program.valid_checkpoint(short_history) && f.program.valid_checkpoint(input),
                "one-off short requests displaced the last long recovery point");
        short_history = f.program.add(tokens(3, 2000 + 10 * request));
        f.publish(short_history);
    }
    require(f.cache.erase(f.program, short_history), "could not retire final interference history");
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
                    ninfer::runtime::ReclaimProgress::Blocked &&
                !f.cache.host_victims(f.program, 64, {}, {}, {}, admission) &&
                f.program.valid_checkpoint(long_history),
            "optional short capture erased more recovery work than it added");
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    f.cache.observe_committed(incoming, 7, 0, 1, incoming.captures);
    require(!permission().priority.reused, "work before a finite candidate created demand");
    f.cache.observe_committed(incoming, 7, 1, 3, incoming.captures);
    f.cache.observe_committed(incoming, 7, 0, 3, incoming.captures);
    require(!permission().priority.reused, "same-request repeated observation created reuse");
    f.cache.observe_committed(incoming, 8, 0, 3, incoming.captures);
    const auto admission = permission();
    require(admission.priority.reused, "two completed requests failed to qualify new demand");
    auto cursor = f.cache.begin_reclaim(f.program, admission);
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cursor) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, cold) ==
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
    const auto loss = f.cache.release_cost(f.program, *host_victims);
    for (const auto victim : *host_victims)
        require(f.program.release_checkpoint(victim), "prepared capture victim was not released");
    f.cache.adopt_release(f.program, *host_victims, decision, loss);
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, decision) ==
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
    f.cache.observe_committed(incoming, 100, 32, 1000, incoming.captures);
    f.cache.observe_committed(incoming, 101, 32, 1000, incoming.captures);
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(device) && f.program.valid_checkpoint(host) &&
                f.program.valid_checkpoint(movable) && f.program.demoted.empty(),
            "zero-loss Device quote hid a more expensive Host history deletion");
}

void test_device_pressure_does_not_erase_host_only_sources() {
    Fixture f(32);
    const auto host = f.program.add(tokens(100), {f.program.host_block(16)});
    f.program.contents[host.index].device_state = false;
    f.program.contents[host.index].main_pages   = 0;
    f.publish(host);
    const auto device = f.program.add(tokens(1000, 10000));
    f.publish(device);
    f.use_shared(device);
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.valid_checkpoint(host) && !f.program.valid_checkpoint(device),
            "Device pressure erased a Host-only source that contributes no capacity");
}

void test_reclaim_does_not_reward_excess_release() {
    Fixture f(0);
    const auto large                           = f.program.add(tokens(1000));
    f.program.contents[large.index].main_pages = 31;
    const auto small                           = f.program.add(tokens(100, 10000));
    f.program.contents[small.index].main_pages = 0;
    f.publish(large);
    f.publish(small);
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.valid_checkpoint(large) && !f.program.valid_checkpoint(small),
            "reclaim overvalued capacity beyond the pending admission's shortage");
}

void test_device_candidates_share_host_inventory() {
    Fixture f(16);
    for (unsigned i = 0; i < 8; ++i) {
        const auto h = f.program.add(tokens(100 + i, 1000 + i * 1000));
        f.program.contents[h.index].demotion_bytes = 4;
        f.publish(h);
    }
    const auto host                             = f.program.add({1}, {f.program.host_block(16)});
    f.program.contents[host.index].device_state = false;
    f.publish(host);
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.demoted.size() == 1 && f.program.host_plan_builds == 1,
            "Device candidates rebuilt physical Host facts for each protected source or final "
            "preflight");
}

void test_optional_capture_cannot_borrow_a_demotion_sources_heat() {
    Fixture f(64);
    const auto host = f.program.add(tokens(100, 10000), {f.program.host_block(64)});
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
    require(f.cache.reclaim(f.program, f.program.pressure(1), {}, decision) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
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
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                !f.program.valid_checkpoint(endpoint) && f.program.valid_checkpoint(other),
            "an abandoned deep branch remained the proof after a consuming rewind");
}

std::uint64_t recovery_oracle(const Program& program, std::span<const Handle> removed,
                              std::span<const Handle> advertised) {
    std::uint64_t total = 0;
    for (const auto branch : removed) {
        if (std::any_of(removed.begin(), removed.end(), [&](auto other) {
                return other != branch && program.prefix(branch, other) &&
                       (program.contents[branch.index].tokens.size() <
                            program.contents[other.index].tokens.size() ||
                        other.index < branch.index);
            }))
            continue;
        std::uint64_t worst = 0;
        for (const auto h : removed) {
            if (!program.prefix(h, branch)) continue;
            std::uint64_t fallback = 0;
            for (const auto kept : advertised)
                if (std::find(removed.begin(), removed.end(), kept) == removed.end() &&
                    program.prefix(kept, h))
                    fallback = std::max<std::uint64_t>(fallback,
                                                       program.contents[kept.index].tokens.size());
            worst = std::max(worst, program.contents[h.index].tokens.size() - fallback);
        }
        total += worst;
    }
    return total;
}

void test_host_selection_against_full_rescoring() {
    std::mt19937 random(402);
    for (unsigned trial = 0; trial < 100; ++trial) {
        Fixture f(4096);
        for (unsigned block = 0; block < 10; ++block)
            f.program.host_block(8 + random() % 40, block == 9);
        for (unsigned i = 0; i < 12; ++i) {
            Key key;
            if (i && random() % 2) key = f.program.contents[random() % i].tokens;
            for (auto n = 1 + random() % 3; n; --n) key.push_back(random() % 6);
            std::vector<std::size_t> blocks{random() % 9};
            if (random() % 2) blocks.push_back(random() % 10);
            const auto h = f.program.add(std::move(key), std::move(blocks));
            f.publish(h);
        }
        f.program.host_capacity = f.program.physical_usage().occupied.host_bytes;
        auto cursor             = f.cache.begin_reclaim(f.program);
        std::vector<Handle> advertised;
        for (const auto& victim : cursor.victims) advertised.push_back(victim.handle);
        const auto required = f.program.host_capacity / 2;
        const auto plan     = f.program.plan_releases(advertised, {}, {.host_bytes = required});
        std::vector<Handle> expected;
        bool possible = true;
        while (f.program.host_bytes_released(expected) < required) {
            std::optional<std::vector<Handle>> best;
            std::uint64_t best_loss = 0, best_bytes = 0;
            std::size_t best_order = 0;
            for (const auto& action : plan.releases) {
                auto combined = expected;
                for (const auto h : action.sources)
                    if (std::find(combined.begin(), combined.end(), h) == combined.end())
                        combined.push_back(h);
                const auto bytes = f.program.host_bytes_released(combined);
                if (bytes <= f.program.host_bytes_released(expected)) continue;
                const auto units  = std::min<std::size_t>(bytes, required);
                const auto loss   = recovery_oracle(f.program, combined, advertised);
                std::size_t order = 0;
                for (std::size_t j = 0; j < cursor.victims.size(); ++j)
                    if (std::find(combined.begin(), combined.end(), cursor.victims[j].handle) !=
                        combined.end())
                        order = j;
                const auto left  = static_cast<unsigned __int128>(loss) * best_bytes;
                const auto right = static_cast<unsigned __int128>(best_loss) * units;
                if (!best || left < right || (left == right && order < best_order)) {
                    best       = std::move(combined);
                    best_loss  = loss;
                    best_bytes = units;
                    best_order = order;
                }
            }
            if (!best) {
                possible = false;
                break;
            }
            expected = std::move(*best);
        }
        const auto actual = f.cache.host_victims(f.program, required, {}, {}, {}, {}, &cursor);
        require(bool(actual) == possible && (!actual || *actual == expected),
                "incremental Host selection differs from full rescoring with shared blocks and "
                "branches");
    }
}

void test_reclaim_assesses_combined_device_and_host_demand() {
    Fixture f(2);
    const auto device                           = f.program.add({1});
    const auto host                             = f.program.add({2}, {f.program.host_block(2)});
    f.program.contents[host.index].device_state = false;
    f.program.contents[host.index].main_pages   = 0;
    f.program.device_capacity                   = 2;
    f.publish(device);
    f.publish(host);
    const Demand demand{1, 1};
    const auto assessment = f.program.assess(demand);
    require(!assessment.feasible(), "combined demand unexpectedly fits before reclaim");
    require(!f.program.assess(demand, {.device_bytes = 2}).feasible() &&
                !f.program.assess(demand, {.host_bytes = 2}).feasible(),
            "one capacity domain satisfied a demand in both domains");
    require(f.cache.reclaim(f.program, assessment) == ninfer::runtime::ReclaimProgress::Changed &&
                f.program.assess(demand).ready() && f.program.released.size() == 2,
            "reclaim did not combine the releases needed by the complete demand");
}

void test_host_placement_overrides_aggregate_free_bytes() {
    Fixture f(48);
    (void)f.program.host_block(16, true);
    const auto first                              = f.program.add({1}, {f.program.host_block(8)});
    const auto second                             = f.program.add({2}, {f.program.host_block(8)});
    const auto incoming                           = f.program.add(tokens(100, 1000));
    f.program.contents[first.index].device_state  = false;
    f.program.contents[first.index].main_pages    = 0;
    f.program.contents[second.index].device_state = false;
    f.program.contents[second.index].main_pages   = 0;
    f.program.contents[incoming.index].demotion_bytes = 8;
    f.program.placement_holders                       = {first, second};
    f.publish(first);
    f.publish(second);
    f.publish(incoming);
    const std::array excluded{second};
    require(!f.cache.host_victims(f.program, 8, incoming, {}, excluded) &&
                f.program.released.empty(),
            "Host placement accepted enough aggregate bytes without a realizable destination");
    require(f.cache.reclaim(f.program, f.program.pressure(1)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                f.program.released == std::vector<Handle>{first, second} &&
                f.program.demoted == std::vector<Handle>{incoming} &&
                f.program.valid_checkpoint(incoming),
            "Host replacement stopped selecting victims before placement became feasible");
}

void test_independent_host_candidates_reuse_scores() {
    for (const auto n : {16U, 32U, 64U, 128U}) {
        Fixture f(n);
        for (unsigned i = 0; i < n; ++i)
            f.publish(f.program.add({i + 1}, {f.program.host_block(1)}));
        (void)f.program.evaluate_recovery({});
        const auto comparisons = f.program.recovery_index.work().exact_comparisons;
        for (const auto count : {1U, n / 2}) {
            f.program.score_queries = f.program.scored_handles = 0;
            const auto victims = f.cache.host_victims(f.program, count, {});
            require(victims && victims->size() == count && f.program.score_queries == n &&
                        f.program.scored_handles == n &&
                        f.program.recovery_index.work().exact_comparisons == comparisons,
                    "Host trials rescored unrelated selected checkpoints or repeated exact "
                    "identity checks");
            require(f.program.last_inventory->work().owner_visits <= 2ULL * (n * count + count),
                    "Host trials rescanned the full physical union");
        }
    }
}

void test_execution_dependency_does_not_become_empty_payload_reclaim() {
    Fixture f(64);
    const auto retained = f.program.add({1, 2, 3});
    f.publish(retained);
    const Assessment waiting{.pending_dependency = true};
    require(f.cache.reclaim(f.program, waiting) == ninfer::runtime::ReclaimProgress::Deferred &&
                f.program.released.empty() && f.program.demoted.empty(),
            "execution dependency was reinterpreted as an immediately retryable empty demand");
}

void test_preservation_batches_only_until_complete_admission() {
    Fixture f(128);
    for (int i = 0; i < 3; ++i) {
        const auto h                           = f.program.add({static_cast<std::uint32_t>(i + 1)});
        f.program.contents[h.index].main_pages = 0;
        f.program.contents[h.index].demotion_bytes = 8;
        f.publish(h);
    }
    const auto demand = f.program.pressure(2).demand;
    require(
        f.cache.reclaim(f.program, f.program.assess(demand)) ==
                ninfer::runtime::ReclaimProgress::Changed &&
            f.program.assess(demand).feasible() && f.program.demotion_submissions == 1 &&
            f.program.demoted.size() == 2 && f.program.released.empty(),
        "preservation should admit the whole demand in one transfer without moving extra sources");

    Fixture limited(8);
    for (int i = 0; i < 2; ++i) {
        const auto h = limited.program.add({static_cast<std::uint32_t>(i + 1)});
        limited.program.contents[h.index].main_pages     = 0;
        limited.program.contents[h.index].demotion_bytes = 8;
        limited.publish(h);
    }
    require(limited.cache.reclaim(limited.program, limited.program.pressure(2)) ==
                    ninfer::runtime::ReclaimProgress::Changed &&
                limited.program.demoted.size() == 1 && limited.program.released.empty(),
            "a batch must reserve all Host destinations and must not introduce Host evictions");
}
} // namespace

int main() {
    try {
        test_shared_publication_deduplicates_complete_snapshots();
        test_execution_dependency_does_not_become_empty_payload_reclaim();
        test_preservation_batches_only_until_complete_admission();
        test_intermediate_capture_preserves_semantic_history();
        test_host_selection_against_full_rescoring();
        test_reclaim_assesses_combined_device_and_host_demand();
        test_host_placement_overrides_aggregate_free_bytes();
        test_independent_host_candidates_reuse_scores();
        test_demotion_competes_with_its_actual_host_recovery_loss();
        test_device_candidates_share_host_inventory();
        test_device_pressure_does_not_erase_host_only_sources();
        test_reclaim_does_not_reward_excess_release();
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
        test_one_off_fanout_competes_with_repeated_continuation();
        test_reclaim_cursor_keeps_order_across_hits_and_transfers();
        test_reclaim_cursor_reconsiders_a_previously_excluded_source();
        test_reclaim_cursor_filters_retired_handles_and_defers_new_history();
        test_reclaim_cursor_can_demote_remaining_pages_of_the_same_history();
        test_reclaim_cursor_keeps_host_writeback_permissions();
        test_deferred_demotion_keeps_sources();
        test_deferred_demotion_preserves_host_victims();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
