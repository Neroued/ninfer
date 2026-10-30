#pragma once

// Flag-gated admin prefix-cache routes (GET /admin/cache/prefixes and
// POST /admin/cache/prefixes/evict). This layer only translates JSON to and from the frozen
// Engine prefix-cache API (Engine::build_prefix_table / Engine::evict_prefixes); the Engine
// enforces every safety property (idle gate, leased and erasable checks) while no execution is
// in flight.

#include "ninfer/observability/snapshot.h"
#include "ninfer/types.h"
#include "product/config/options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

// Registration gate for both admin routes. Off by default; while false the routes are not
// registered and requests to their paths receive the server's 404.
[[nodiscard]] bool admin_cache_routes_enabled(const product::ServeOptions& options) noexcept;

// Stable JSON identity of a resident prefix. The digests are decimal strings because JSON
// numbers lose precision above 2^53.
[[nodiscard]] nlohmann::json prefix_cache_key_json(const ninfer::PrefixCacheKey& key);
// Inverse of prefix_cache_key_json(). Throws std::invalid_argument on malformed input.
[[nodiscard]] ninfer::PrefixCacheKey parse_prefix_cache_key(const nlohmann::json& json);

// GET /admin/cache/prefixes response body. The capacity block is sourced from the same Engine
// data as /metrics: host-context capacity and occupancy, occupied device KV page groups per
// pool, and device StateImage slot usage and capacity.
[[nodiscard]] nlohmann::json
render_prefix_cache_table(const ninfer::PrefixCacheTable& table,
                          const ninfer::EngineSnapshot& snapshot,
                          const ninfer::EngineDescription& description);

// Parsed POST /admin/cache/prefixes/evict body.
struct PrefixEvictSelection {
    std::vector<ninfer::PrefixCacheKey> prefixes;
    std::uint64_t max_bytes = 0; // 0 means unbounded.
};

// Parses the POST body. Throws ApiException (HTTP 400) on malformed input.
[[nodiscard]] PrefixEvictSelection parse_prefix_evict_body(const std::string& body);

// POST /admin/cache/prefixes/evict response body.
[[nodiscard]] nlohmann::json render_prefix_evict_result(const ninfer::PrefixCacheEvictResult& result);

} // namespace ninfer::serve
