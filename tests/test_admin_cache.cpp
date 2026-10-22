#include "serve/admin_cache.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::serve::ServeOptions;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

int test_flag_gate() {
    int failures = 0;
    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(!defaults.enable_admin_cache,
                      "admin cache routes are enabled without --enable-admin-cache");
    failures += check(!ninfer::serve::admin_cache_routes_enabled(defaults),
                      "admin cache routes are registered while the flag is off");
    const ServeOptions enabled = parse({"ninfer-serve", "model.ninfer", "--enable-admin-cache"});
    failures += check(enabled.enable_admin_cache, "--enable-admin-cache did not parse");
    failures += check(ninfer::serve::admin_cache_routes_enabled(enabled),
                      "admin cache routes are not registered with --enable-admin-cache");
    return failures;
}

int test_key_round_trip() {
    int failures = 0;
    const ninfer::PrefixCacheKey key{
        .digest0      = std::numeric_limits<std::uint64_t>::max(),
        .digest1      = 0x0123456789abcdefULL,
        .frontier     = 12288,
        .identity_tag = 7,
    };
    const Json json = ninfer::serve::prefix_cache_key_json(key);
    failures += check(
        json.at("d0").is_string() && json.at("d0").get<std::string>() == "18446744073709551615" &&
            json.at("d1").get<std::string>() == "81985529216486895" && json.at("frontier") == 12288 &&
            json.at("tag") == 7,
        "prefix key digests are not exact decimal strings");
    failures += check(ninfer::serve::parse_prefix_cache_key(json) == key,
                      "prefix key JSON round trip changed the key");

    const std::vector<std::string> malformed = {
        R"({"d0":18446744073709551615,"d1":"1","frontier":0,"tag":0})",
        R"({"d0":"-1","d1":"1","frontier":0,"tag":0})",
        R"({"d0":"18446744073709551616","d1":"1","frontier":0,"tag":0})",
        R"({"d1":"1","frontier":0,"tag":0})",
        R"({"d0":"1","d1":"1","frontier":1.5,"tag":0})",
        R"({"d0":"1","d1":"1","frontier":0,"tag":4294967296})",
        R"([1,2,3])",
    };
    for (std::size_t i = 0; i < malformed.size(); ++i) {
        bool threw = false;
        try {
            (void)ninfer::serve::parse_prefix_cache_key(Json::parse(malformed[i]));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "malformed prefix key was accepted: " << malformed[i] << '\n';
            ++failures;
        }
    }
    return failures;
}

int test_table_render() {
    int failures = 0;
    ninfer::PrefixCacheTable table;
    ninfer::PrefixCacheEntry entry;
    entry.key             = {.digest0 = 1, .digest1 = 2, .frontier = 12288, .identity_tag = 0};
    entry.tokens          = 12288;
    entry.kv_page_groups[0] = 512;
    entry.kv_page_groups[1] = 3;
    entry.host_bytes      = 4096;
    entry.state_slots     = 1;
    entry.evictable_bytes = 83886080;
    entry.leased          = false;
    entry.checkpoints     = 2;
    entry.reused          = true;
    entry.last_demand     = 42;
    table.prefixes.push_back(entry);

    const ninfer::MemorySummary memory = {.host_context_capacity_bytes = 53687091200};
    ninfer::RuntimeStats stats;
    stats.host_context_occupied_bytes = 19197939712;
    stats.device_state_occupied_slots = 8;
    stats.kv_pools.push_back(
        {.id = 0, .label = "main", .occupied_page_groups = 5078, .capacity_page_groups = 40960});

    ninfer::EngineOptions engine_options;
    engine_options.max_concurrency = 2;
    engine_options.context_cache.device_state_slots = 3;
    const Json body =
        ninfer::serve::render_prefix_cache_table(table, memory, stats, engine_options);
    failures += check(
        body.at("capacity").at("host_context_bytes") == 53687091200 &&
            body.at("capacity").at("host_context_used_bytes") == 19197939712 &&
            body.at("capacity").at("device_kv_page_groups").at("0") == 5078 &&
            body.at("capacity").at("device_state_slots_used") == 8 &&
            body.at("capacity").at("device_state_slots_capacity") ==
                engine_options.max_concurrency +
                    engine_options.context_cache.device_state_slots.value(),
        "capacity block did not mirror the /metrics sources");
    const Json prefix = body.at("prefixes").at(0);
    failures += check(
        prefix.at("key").at("d0") == "1" && prefix.at("key").at("d1") == "2" &&
            prefix.at("tokens") == 12288 &&
            prefix.at("evictable").at("kv_page_groups").at("0") == 512 &&
            prefix.at("evictable").at("kv_page_groups").at("1") == 3 &&
            prefix.at("evictable").at("host_bytes") == 4096 &&
            prefix.at("evictable").at("state_slots") == 1 &&
            prefix.at("evictable").at("bytes") == 83886080 && prefix.at("leased") == false &&
            prefix.at("checkpoints") == 2 && prefix.at("reused") == true &&
            prefix.at("last_demand") == 42,
        "prefix entry did not render the frozen engine fields");
    return failures;
}

int test_evict_body() {
    int failures = 0;
    const std::string body =
        R"({"prefixes":[{"d0":"18374653149342844481","d1":"274452552429777362","frontier":12288,"tag":0}],"mode":"release","max_bytes":1073741824})";
    const auto selection = ninfer::serve::parse_prefix_evict_body(body);
    failures += check(
        selection.prefixes.size() == 1 &&
            selection.prefixes[0].digest0 == 18374653149342844481ULL &&
            selection.prefixes[0].digest1 == 274452552429777362ULL &&
            selection.prefixes[0].frontier == 12288 && selection.prefixes[0].identity_tag == 0 &&
            selection.max_bytes == 1073741824,
        "evict body did not parse the frozen request fields");

    const auto minimal = ninfer::serve::parse_prefix_evict_body(
        R"({"prefixes":[{"d0":"1","d1":"2","frontier":0,"tag":0}]})");
    failures += check(minimal.max_bytes == 0, "absent max_bytes did not default to unbounded");

    const std::vector<std::string> rejected = {
        R"({"prefixes":[{"d0":"1","d1":"2","frontier":0,"tag":0}],"mode":"demote"})",
        R"({"prefixes":[{"d0":"1","d1":"2","frontier":0,"tag":0}],"mode":123})",
        R"({"prefixes":[{"d0":"1","d1":"2","frontier":0,"tag":0}],"max_bytes":-1})",
        R"({"prefixes":[{"d0":"1","d1":"2","frontier":0,"tag":0}],"max_bytes":1.5})",
        R"({"prefixes":"nope"})",
        R"({"mode":"release"})",
        "not json",
    };
    for (std::size_t i = 0; i < rejected.size(); ++i) {
        bool rejected_400 = false;
        try {
            (void)ninfer::serve::parse_prefix_evict_body(rejected[i]);
        } catch (const ninfer::serve::ApiException& error) {
            rejected_400 = error.error().status == 400;
        }
        if (!rejected_400) {
            std::cerr << "malformed evict body was not rejected with 400: " << rejected[i]
                      << '\n';
            ++failures;
        }
    }
    return failures;
}

int test_evict_result_render() {
    int failures = 0;
    const ninfer::PrefixCacheEvictResult busy{.busy = true};
    const Json busy_body = ninfer::serve::render_prefix_evict_result(busy);
    failures += check(busy_body.at("status") == "busy" && busy_body.at("retry_after_s") == 5 &&
                          !busy_body.contains("evicted"),
                      "busy result did not render the retry contract");

    ninfer::PrefixCacheEvictResult ok;
    ok.total_freed_bytes = 83886080;
    ok.evicted.push_back(
        {.key = {.digest0 = 1, .digest1 = 2, .frontier = 4, .identity_tag = 5},
         .freed_bytes = 83886080});
    ok.skipped.push_back(
        {.key = {.digest0 = 3, .digest1 = 4, .frontier = 6, .identity_tag = 7},
         .reason = ninfer::PrefixCacheSkip::Reason::Leased});
    ok.skipped.push_back(
        {.key = {.digest0 = 5, .digest1 = 6, .frontier = 8, .identity_tag = 9},
         .reason = ninfer::PrefixCacheSkip::Reason::NotEvictable});
    const Json body = ninfer::serve::render_prefix_evict_result(ok);
    failures += check(
        body.at("status") == "ok" && body.at("evicted").at(0).at("key").at("d0") == "1" &&
            body.at("evicted").at(0).at("freed_bytes") == 83886080 &&
            body.at("skipped").at(0).at("reason") == "leased" &&
            body.at("skipped").at(1).at("reason") == "not_evictable" &&
            body.at("total_freed_bytes") == 83886080,
        "evict result did not render the frozen response fields");
    return failures;
}

} // namespace

int main() {
    const int failures = test_flag_gate() + test_key_round_trip() + test_table_render() +
                         test_evict_body() + test_evict_result_render();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
