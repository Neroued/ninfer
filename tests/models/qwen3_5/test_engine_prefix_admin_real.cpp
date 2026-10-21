#include "ninfer/engine.h"
#include "runtime/engine/context_cache/prefix_table.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::PrefixCacheKey;
using ninfer::PrefixCacheSkip;
using ninfer::PrefixCacheTable;
using Plan   = ninfer::runtime::PrefixEvictionPlan;
using Record = ninfer::runtime::PrefixCheckpointRecord;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

PrefixCacheKey make_key(std::uint64_t digest0, std::uint64_t digest1, std::uint32_t frontier,
                        std::uint32_t tag = 0) noexcept {
    return PrefixCacheKey{digest0, digest1, frontier, tag};
}

Record make_record(PrefixCacheKey key, bool leased = false, std::uint32_t state_slots = 0,
                   std::uint32_t main_groups = 0, std::uint32_t mtp_groups = 0,
                   std::size_t host_bytes = 0, bool reused = false,
                   std::uint64_t last_demand = 0) noexcept {
    Record record;
    record.key              = key;
    record.leased           = leased;
    record.state_slots      = state_slots;
    record.kv_page_groups[0] = main_groups;
    record.kv_page_groups[1] = mtp_groups;
    record.host_bytes        = host_bytes;
    record.reused            = reused;
    record.last_demand       = last_demand;
    return record;
}

const std::size_t kMainStride      = 4096;
const std::size_t kMtpStride       = 8192;
const std::size_t kStateImageBytes = 196121600; // 187 MiB state image stride
const std::array<std::size_t, 2> kPoolBytes{kMainStride, kMtpStride};

// --- Prefix table and release policies (CPU; no device) ---

void test_table_groups_and_aggregates() {
    const auto a = make_key(1, 2, 128);
    const auto b = make_key(3, 4, 64);
    const std::vector<Record> records{
        make_record(a, false, 1, 100, 0, 10, false, 7),
        make_record(b, false, 0, 0, 5, 0),
        make_record(a, true, 0, 50, 0, 20, true, 12),
    };
    const auto table =
        ninfer::runtime::build_prefix_entries(records, kPoolBytes, kStateImageBytes);
    require(table.prefixes.size() == 2, "table did not group records by prefix identity");
    const auto& first = table.prefixes[0];
    require(first.key == a && first.tokens == 128, "table lost the prefix key or frontier");
    require(first.checkpoints == 2, "table did not count the fan-out checkpoints");
    require(first.leased, "table did not report a leased group member");
    require(first.state_slots == 1 && first.host_bytes == 30,
            "table did not aggregate the host quantities");
    require(first.kv_page_groups[0] == 150 && first.kv_page_groups[1] == 0,
            "table did not aggregate the per-pool page groups");
    require(first.reused && first.last_demand == 12,
            "table must OR the demand flag and take the max demand tick");
    require(first.evictable_bytes == 30 + 150 * kMainStride + kStateImageBytes,
            "table evictable bytes diverged from the pool strides and state images");
    const auto& second = table.prefixes[1];
    require(second.key == b && second.checkpoints == 1 && !second.leased,
            "table misgrouped the second prefix");
    require(second.kv_page_groups[1] == 5 && second.evictable_bytes == 5 * kMtpStride,
            "table misread the secondary pool");
    require(!second.reused && second.last_demand == 0,
            "a never-reused prefix must keep zero demand evidence");
}

void test_table_empty_and_zero_stride() {
    const auto empty =
        ninfer::runtime::build_prefix_entries({}, kPoolBytes, kStateImageBytes);
    require(empty.prefixes.empty(), "empty walk must yield an empty table");
    const std::vector<Record> records{make_record(make_key(9, 9, 16), false, 1, 0, 0, 0)};
    const auto table = ninfer::runtime::build_prefix_entries(records, {}, kStateImageBytes);
    require(table.prefixes.size() == 1 && table.prefixes[0].evictable_bytes == kStateImageBytes &&
                table.prefixes[0].state_slots == 1,
            "state-only prefix must report state_slots x image bytes");
}

void test_state_image_bytes() {
    const std::vector<Record> records{
        make_record(make_key(1, 1, 8), false, 3, 0, 0, 0),  // state images only
        make_record(make_key(2, 2, 8), false, 0, 0, 0, 0),  // fully shared: zero
    };
    const auto table =
        ninfer::runtime::build_prefix_entries(records, kPoolBytes, kStateImageBytes);
    require(table.prefixes.size() == 2, "state test lost a prefix");
    require(table.prefixes[0].evictable_bytes == 3 * kStateImageBytes &&
                table.prefixes[0].host_bytes == 0,
            "state-only bytes must be state_slots x image_bytes");
    require(table.prefixes[1].evictable_bytes == 0,
            "a fully shared prefix with no state slot must report zero");
}

void test_table_saturates() {
    const std::array<std::size_t, 2> huge{UINT64_MAX / 2, 0};
    const std::vector<Record> records{make_record(make_key(1, 1, 8), false, 2, 2, 0, 1)};
    const auto table =
        ninfer::runtime::build_prefix_entries(records, huge, kStateImageBytes);
    require(table.prefixes.size() == 1 && table.prefixes[0].evictable_bytes == UINT64_MAX,
            "evictable bytes must saturate instead of wrapping");
}

void test_idle_predicate() {
    using ninfer::runtime::prefix_cache_idle;
    require(prefix_cache_idle(false, true, false, false), "idle engine must pass the gate");
    require(!prefix_cache_idle(true, true, false, false),
            "a pending request must block the gate");
    require(!prefix_cache_idle(false, false, false, false),
            "resident work must block the gate");
    require(!prefix_cache_idle(false, true, true, false),
            "paused requests must block the gate");
    require(!prefix_cache_idle(false, true, false, true),
            "context operations must block the gate");
}

void test_plan_not_found() {
    const std::vector<Record> records{make_record(make_key(1, 1, 8))};
    const std::vector<PrefixCacheKey> selection{make_key(7, 7, 8)};
    const auto plan = ninfer::runtime::plan_prefix_evictions(
        records, selection, 0, kPoolBytes, kStateImageBytes);
    require(plan.keys.size() == 1 &&
                plan.keys[0].action == Plan::KeyPlan::Action::NotFound &&
                plan.keys[0].records.empty() && plan.expected_total_bytes == 0 &&
                !plan.stopped,
            "a missing prefix must plan a not-found skip");
}

void test_plan_leased_skipped() {
    const auto key = make_key(1, 1, 8);
    const std::vector<Record> records{
        make_record(key, false, 0, 10, 0, 0), make_record(key, true, 0, 10, 0, 0)};
    const std::vector<PrefixCacheKey> selection{key};
    const auto plan = ninfer::runtime::plan_prefix_evictions(
        records, selection, 0, kPoolBytes, kStateImageBytes);
    require(plan.keys.size() == 1 && plan.keys[0].action == Plan::KeyPlan::Action::Leased &&
                plan.keys[0].records.empty() && plan.expected_total_bytes == 0,
            "a leased group member must skip the whole prefix");
}

void test_plan_release_order_and_accounting() {
    const auto a = make_key(1, 1, 8);
    const auto b = make_key(2, 2, 16);
    const std::vector<Record> records{
        make_record(a, false, 1, 10, 0, 100),  // 100 + 10 * kMainStride + state image
        make_record(b, false, 0, 0, 5, 0),     // 5 * kMtpStride
        make_record(a, false, 0, 1, 0, 0),     // 1 * kMainStride
    };
    const std::vector<PrefixCacheKey> selection{b, a};
    const auto plan = ninfer::runtime::plan_prefix_evictions(
        records, selection, 0, kPoolBytes, kStateImageBytes);
    require(plan.keys.size() == 2, "plan must cover every requested prefix");
    const auto& first = plan.keys[0];
    require(first.key == b && first.action == Plan::KeyPlan::Action::Evict &&
                first.records.size() == 1 && first.records[0] == 1 &&
                first.expected_freed_bytes == 5 * kMtpStride,
            "plan must honor the request order");
    const auto& second = plan.keys[1];
    require(second.key == a && second.records.size() == 2 && second.records[0] == 0 &&
                second.records[1] == 2 &&
                second.expected_freed_bytes ==
                    100 + 11 * kMainStride + kStateImageBytes,
            "plan must release a group in walk order with exact bytes");
    require(plan.expected_total_bytes ==
                5 * kMtpStride + 100 + 11 * kMainStride + kStateImageBytes,
            "plan total must equal the per-prefix bytes");
}

void test_plan_max_bytes_cap() {
    const auto a = make_key(1, 1, 8);
    const auto b = make_key(2, 2, 16);
    const std::vector<Record> records{
        make_record(a, false, 0, 10, 0, 0),  // 40960
        make_record(b, false, 0, 10, 0, 0),  // 40960
        make_record(b, false, 0, 10, 0, 0),  // 40960
    };
    const std::vector<PrefixCacheKey> selection{a, b};
    const auto plan = ninfer::runtime::plan_prefix_evictions(
        records, selection, 50000, kPoolBytes, kStateImageBytes);
    require(plan.keys.size() == 2, "capped plan must still report the truncated prefix");
    require(plan.keys[0].records.size() == 1 && !plan.keys[0].truncated,
            "cap must consume the first prefix whole");
    require(plan.keys[1].action == Plan::KeyPlan::Action::Evict &&
                plan.keys[1].records.size() == 1 && plan.keys[1].truncated && plan.stopped,
            "cap must truncate the next prefix and stop the walk");
    require(plan.expected_total_bytes == 2 * 40960 && plan.expected_total_bytes >= 50000,
            "capped total must reach the cap without skipping records after it");
    // A zero cap is unbounded: every record is planned.
    const auto unbounded = ninfer::runtime::plan_prefix_evictions(
        records, selection, 0, kPoolBytes, kStateImageBytes);
    require(unbounded.expected_total_bytes == 3 * 40960 && !unbounded.stopped &&
                unbounded.keys[1].records.size() == 2,
            "zero max_bytes must not cap the release");
}

// --- Real-engine exercise (qwen3_5 lane) ---

ninfer::RequestOptions fixed_output(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::EngineOptions admin_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 1024;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 4;
    return options;
}

std::uint64_t table_bytes(const PrefixCacheTable& table) {
    std::uint64_t total = 0;
    for (const auto& entry : table.prefixes) { total += entry.evictable_bytes; }
    return total;
}

std::string table_fingerprint(const PrefixCacheTable& table) {
    std::string fingerprint;
    for (const auto& entry : table.prefixes) {
        fingerprint += std::to_string(entry.key.digest0) + ":" +
                       std::to_string(entry.key.digest1) + ":" +
                       std::to_string(entry.key.frontier) + ":" +
                       std::to_string(entry.key.identity_tag) + ":" +
                       std::to_string(entry.checkpoints) + ":" + std::to_string(entry.leased) + ":" +
                       std::to_string(entry.evictable_bytes) + ";";
    }
    return fingerprint;
}

std::vector<ninfer::TokenId> make_prefix(const ninfer::Engine& engine, const std::string& text,
                                         std::size_t length) {
    const auto seed = engine.tokenize_text(text);
    require(!seed.empty(), "artifact tokenizer produced no tokens");
    std::vector<ninfer::TokenId> prefix(length);
    for (std::size_t index = 0; index < length; ++index) {
        prefix[index] = seed[index % seed.size()];
    }
    return prefix;
}

int exercise_prefix_admin(const char* artifact) {
    ninfer::Engine engine(admin_engine_options(artifact));
    const auto prefix = make_prefix(engine, "Explain how a sequence continues from its stored prefix. ",
                                    1024);

    // Idle table: grouping, fan-out, honest bytes, no leases, state images counted.
    const auto primed = engine.generate(engine.prepare_tokens(prefix), fixed_output(8));
    (void)primed;
    auto table = engine.build_prefix_table();
    require(!table.prefixes.empty(), "prime request left no resident prefix checkpoints");
    bool state_seen = false;
    for (const auto& entry : table.prefixes) {
        require(entry.tokens == entry.key.frontier,
                "entry tokens diverged from the key frontier");
        require(entry.checkpoints >= 1, "entry lost its checkpoints");
        require(entry.evictable_bytes >= entry.host_bytes,
                "entry bytes lost its host contribution");
        require(!entry.leased, "idle table reported a leased prefix");
        require(!entry.reused && entry.last_demand == 0,
                "fresh prime reported demand evidence on a never-reused prefix");
        if (entry.state_slots > 0) {
            require(entry.evictable_bytes > entry.host_bytes,
                    "state-image bytes missing from the entry total");
            state_seen = true;
        }
    }
    require(state_seen, "prime request left no state-image checkpoint");
    const std::uint64_t resident_bytes = table_bytes(table);
    std::vector<PrefixCacheKey> keys;
    for (const auto& entry : table.prefixes) { keys.push_back(entry.key); }

    // Full release: exact freed-bytes accounting and an empty table.
    auto released = engine.evict_prefixes(keys, 0);
    require(!released.busy, "idle engine refused the release");
    require(released.skipped.empty(), "release reported a skip on a live prefix");
    require(released.evicted.size() == keys.size(),
            "release did not cover every requested prefix");
    require(released.total_freed_bytes == resident_bytes,
            "released bytes diverged from the table accounting");
    table = engine.build_prefix_table();
    require(table.prefixes.empty(), "release left resident checkpoints behind");

    // Idempotent: already-gone prefixes are skipped, not errors.
    const auto again = engine.evict_prefixes(keys, 0);
    require(!again.busy && again.total_freed_bytes == 0 && again.evicted.empty(),
            "releasing gone prefixes must be a no-op");
    require(again.skipped.size() == keys.size(), "gone prefixes must be reported");
    for (const auto& skip : again.skipped) {
        require(skip.reason == PrefixCacheSkip::Reason::NotFound,
                "gone prefix must be reported as not found");
    }

    // Missing prefix: no-op skip.
    const auto missing =
        engine.evict_prefixes(std::array<PrefixCacheKey, 1>{make_key(0xDEAD, 0xBEEF, 7)}, 0);
    require(!missing.busy && missing.total_freed_bytes == 0 && missing.skipped.size() == 1 &&
                missing.skipped[0].reason == PrefixCacheSkip::Reason::NotFound,
            "missing prefix must be a not-found skip");

    // Busy gate: an in-flight request refuses the release and leaves the table untouched.
    const auto reprimed = engine.generate(engine.prepare_tokens(prefix), fixed_output(8));
    (void)reprimed;
    table = engine.build_prefix_table();
    require(!table.prefixes.empty(), "re-prime left no resident checkpoints");
    const auto fingerprint = table_fingerprint(table);
    const auto extension   = make_prefix(
        engine, "Explain how a sequence continues from its stored prefix. ", 1056);
    auto handle = engine.submit(engine.prepare_tokens(extension), fixed_output(256));
    const auto no_op =
        std::array<PrefixCacheKey, 1>{make_key(0xCAFE, 0xF00D, 3)}; // Absent key: no cache change.
    // Queued (pending) and admitted requests both engage the idle gate: busy from the moment
    // the request is submitted, with no cache change.
    require(engine.evict_prefixes(no_op, 0).busy,
            "queued request did not engage the idle gate");
    bool busy = false;
    for (int attempt = 0; attempt < 10000 && !busy; ++attempt) {
        busy = engine.evict_prefixes(no_op, 0).busy;
    }
    require(busy, "in-flight request did not engage the idle gate");
    require(table_fingerprint(engine.build_prefix_table()) == fingerprint,
            "busy release changed the resident table");
    bool leased = false;
    for (int attempt = 0; attempt < 10000 && !leased; ++attempt) {
        leased = std::any_of(engine.build_prefix_table().prefixes.begin(),
                             engine.build_prefix_table().prefixes.end(),
                             [](const auto& entry) { return entry.leased; });
    }
    require(leased, "in-flight request did not pin a resident prefix");
    const auto in_flight = handle.wait();
    (void)in_flight;

    // Demand evidence: the extension request reused the resident prefix, so the table now
    // distinguishes an active prefix from one that never had repeat demand.
    table = engine.build_prefix_table();
    bool demand_seen = false;
    for (const auto& entry : table.prefixes) {
        if (entry.reused) {
            require(entry.last_demand > 0, "reused prefix reported no demand tick");
            demand_seen = true;
        } else {
            require(entry.last_demand == 0, "never-reused prefix reported a demand tick");
        }
    }
    require(demand_seen, "extension request did not mark the reused prefix as active");

    // max_bytes cap: a 1-byte cap stops after the first non-zero record, keeping the rest
    // resident. Fresh primes guarantee at least two non-zero checkpoints.
    const auto prefix_b = make_prefix(engine, "A second independent document for the cache. ", 512);
    engine.generate(engine.prepare_tokens(prefix), fixed_output(8));
    engine.generate(engine.prepare_tokens(prefix_b), fixed_output(8));
    table = engine.build_prefix_table();
    require(table.prefixes.size() >= 2, "cap scenario needs two resident prefixes");
    const std::uint64_t before = table_bytes(table);
    std::vector<PrefixCacheKey> capped_keys;
    for (const auto& entry : table.prefixes) { capped_keys.push_back(entry.key); }
    const auto capped = engine.evict_prefixes(capped_keys, 1);
    require(!capped.busy, "cap scenario ran while busy");
    std::uint64_t reported = 0;
    for (const auto& evicted : capped.evicted) { reported += evicted.freed_bytes; }
    require(capped.total_freed_bytes == reported && capped.total_freed_bytes > 0 &&
                capped.total_freed_bytes < before,
            "cap accounting diverged from the released entries");
    table = engine.build_prefix_table();
    require(!table.prefixes.empty(), "cap released the whole cache");
    require(table_bytes(table) + capped.total_freed_bytes == before,
            "cap left the byte ledger inconsistent");
    return 0;
}

} // namespace

int main() {
    try {
        test_table_groups_and_aggregates();
        test_table_empty_and_zero_stride();
        test_state_image_bytes();
        test_table_saturates();
        test_idle_predicate();
        test_plan_not_found();
        test_plan_leased_skipped();
        test_plan_release_order_and_accounting();
        test_plan_max_bytes_cap();
    } catch (const std::exception& error) {
        std::cerr << "prefix policy failed: " << error.what() << '\n';
        return 1;
    }
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        if (exercise_prefix_admin(artifact) == 0) {
            std::cout << "ok\n";
            return 0;
        }
    } catch (const std::exception& error) {
        std::cerr << "prefix admin failed: " << error.what() << '\n';
    }
    return 1;
}
