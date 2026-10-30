#include "serve/admin_cache.h"

#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

[[nodiscard]] ApiError invalid_admin_cache_request(std::string message) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "invalid_admin_cache_request";
    error.message = std::move(message);
    return error;
}

// Digests travel as decimal strings: JSON numbers lose precision above 2^53.
[[nodiscard]] std::uint64_t parse_digest(const Json& value, std::string_view field) {
    const std::string label = std::string("prefix key '") + std::string(field) + "'";
    if (!value.is_string()) {
        throw std::invalid_argument(label + " must be a decimal string");
    }
    const std::string text = value.get<std::string>();
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(label + " must be a decimal string");
    }
    try {
        return std::stoull(text);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(label + " is out of range");
    }
}

[[nodiscard]] std::uint32_t parse_field_u32(const Json& value, std::string_view field) {
    const std::string label = std::string("prefix key '") + std::string(field) + "'";
    if (!value.is_number_unsigned() ||
        value.get<std::uint64_t>() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(label + " must be a nonnegative 32-bit integer");
    }
    return static_cast<std::uint32_t>(value.get<std::uint32_t>());
}

} // namespace

bool admin_cache_routes_enabled(const product::ServeOptions& options) noexcept {
    return options.enable_admin_cache;
}

nlohmann::json prefix_cache_key_json(const ninfer::PrefixCacheKey& key) {
    return Json{{"d0", std::to_string(key.digest0)},
                {"d1", std::to_string(key.digest1)},
                {"frontier", key.frontier},
                {"tag", key.identity_tag}};
}

ninfer::PrefixCacheKey parse_prefix_cache_key(const nlohmann::json& json) {
    if (!json.is_object()) {
        throw std::invalid_argument("prefix key must be an object");
    }
    const auto member = [&json](std::string_view field) -> const Json& {
        const auto it = json.find(std::string(field));
        if (it == json.end()) {
            throw std::invalid_argument(std::string("prefix key is missing '") +
                                        std::string(field) + "'");
        }
        return *it;
    };
    return ninfer::PrefixCacheKey{
        .digest0      = parse_digest(member("d0"), "d0"),
        .digest1      = parse_digest(member("d1"), "d1"),
        .frontier     = parse_field_u32(member("frontier"), "frontier"),
        .identity_tag = parse_field_u32(member("tag"), "tag"),
    };
}

nlohmann::json
render_prefix_cache_table(const ninfer::PrefixCacheTable& table,
                          const ninfer::EngineSnapshot& snapshot,
                          const ninfer::EngineDescription& description) {
    Json device_pools = Json::object();
    for (const auto& pool : snapshot.runtime.kv_pools) {
        device_pools[std::to_string(pool.id)] = pool.occupied_page_groups;
    }
    // Same sources as /metrics: host-context capacity and occupancy, occupied device KV page
    // groups per pool, and device StateImage slot usage and capacity.
    std::uint64_t host_capacity_bytes = 0, host_used_bytes = 0;
    for (const auto& pool : description.context_pools) {
        if (pool.domain == ninfer::ContextMemoryDomain::Host)
            host_capacity_bytes += pool.capacity_bytes;
    }
    for (const auto& pool : snapshot.runtime.context_pools) {
        if (pool.domain == ninfer::ContextMemoryDomain::Host)
            host_used_bytes += pool.live_bytes;
    }
    const Json capacity = Json{
        {"host_context_bytes", host_capacity_bytes},
        {"host_context_used_bytes", host_used_bytes},
        {"device_kv_page_groups", std::move(device_pools)},
        {"device_state_slots_used", snapshot.runtime.state_objects},
        {"device_state_slots_capacity", description.state_slots_capacity}};

    Json prefixes = Json::array();
    for (const auto& entry : table.prefixes) {
        Json pool_groups = Json::object();
        for (std::size_t pool = 0; pool < entry.kv_page_groups.size(); ++pool) {
            pool_groups[std::to_string(pool)] = entry.kv_page_groups[pool];
        }
        prefixes.push_back(Json{
            {"key", prefix_cache_key_json(entry.key)},
            {"tokens", entry.tokens},
            {"evictable",
             Json{{"kv_page_groups", std::move(pool_groups)},
                  {"host_bytes", entry.host_bytes},
                  {"state_slots", entry.state_slots},
                  {"bytes", entry.evictable_bytes}}},
            {"leased", entry.leased},
            {"checkpoints", entry.checkpoints},
            {"reused", entry.reused},
            {"last_demand", entry.last_demand},
        });
    }
    return Json{{"capacity", std::move(capacity)}, {"prefixes", std::move(prefixes)}};
}

PrefixEvictSelection parse_prefix_evict_body(const std::string& body) {
    Json parsed;
    try {
        parsed = Json::parse(body);
    } catch (const Json::parse_error& error) {
        throw ApiException(invalid_admin_cache_request(
            std::string("request body is not valid JSON: ") + error.what()));
    }
    if (!parsed.is_object()) {
        throw ApiException(invalid_admin_cache_request(
            "request body must be a JSON object with a 'prefixes' array"));
    }
    const auto prefixes = parsed.find("prefixes");
    if (prefixes == parsed.end() || !prefixes->is_array()) {
        throw ApiException(invalid_admin_cache_request("'prefixes' must be an array of keys"));
    }
    PrefixEvictSelection selection;
    selection.prefixes.reserve(prefixes->size());
    for (const auto& entry : *prefixes) {
        try {
            selection.prefixes.push_back(parse_prefix_cache_key(entry));
        } catch (const std::invalid_argument& error) {
            throw ApiException(invalid_admin_cache_request(error.what()));
        }
    }
    if (const auto mode = parsed.find("mode"); mode != parsed.end()) {
        if (!mode->is_string() || mode->get<std::string>() != "release") {
            throw ApiException(invalid_admin_cache_request(
                "'mode' must be \"release\"; v1 ships release only"));
        }
    }
    if (const auto max_bytes = parsed.find("max_bytes"); max_bytes != parsed.end()) {
        if (!max_bytes->is_number_unsigned()) {
            throw ApiException(invalid_admin_cache_request(
                "'max_bytes' must be a nonnegative integer"));
        }
        selection.max_bytes = max_bytes->get<std::uint64_t>();
    }
    return selection;
}

nlohmann::json render_prefix_evict_result(const ninfer::PrefixCacheEvictResult& result) {
    if (result.busy) {
        return Json{{"status", "busy"}, {"retry_after_s", 5}};
    }
    Json evicted = Json::array();
    for (const auto& eviction : result.evicted) {
        evicted.push_back(Json{{"key", prefix_cache_key_json(eviction.key)},
                               {"freed_bytes", eviction.freed_bytes}});
    }
    Json skipped = Json::array();
    for (const auto& skip : result.skipped) {
        const char* reason = "not_found";
        switch (skip.reason) {
        case ninfer::PrefixCacheSkip::Reason::Leased:
            reason = "leased";
            break;
        case ninfer::PrefixCacheSkip::Reason::NotEvictable:
            reason = "not_evictable";
            break;
        case ninfer::PrefixCacheSkip::Reason::NotFound:
            break;
        }
        skipped.push_back(Json{{"key", prefix_cache_key_json(skip.key)}, {"reason", reason}});
    }
    return Json{{"status", "ok"},
                {"evicted", std::move(evicted)},
                {"skipped", std::move(skipped)},
                {"total_freed_bytes", result.total_freed_bytes}};
}

} // namespace ninfer::serve
