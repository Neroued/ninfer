#include "serve/response_store.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

std::size_t estimate_turn_bytes(const ChatTurn& turn) {
    std::size_t bytes = sizeof(ChatTurn) + turn.tool_call_id.size() + turn.reasoning_content.size();
    for (const ContentPart& part : turn.content) {
        bytes += sizeof(ContentPart) + part.text.size() + part.type_raw.size() +
                 part.source.value.size() + part.source.media_type.size() +
                 part.source.bytes.size();
    }
    for (const ToolCall& call : turn.tool_calls) {
        bytes += sizeof(ToolCall) + call.id.size() + call.name.size() + call.arguments_json.size();
    }
    return bytes;
}

std::size_t record_envelope_bytes(const StoredResponse& record) {
    std::size_t bytes = sizeof(StoredResponse) + record.id.size() + record.session_key.size() +
                        record.response.dump().size();
    for (const nlohmann::json& item : record.input_items) {
        bytes += sizeof(nlohmann::json) + item.dump().size();
    }
    return bytes;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

[[noreturn]] void throw_store_capacity() {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.code    = "response_store_capacity_exceeded";
    error.message = "response exceeds the configured local response store capacity";
    throw ApiException(std::move(error));
}

} // namespace

ResponseContext append_response_context(ResponseContext parent, std::vector<ChatTurn> turns) {
    auto node         = std::make_shared<ResponseContextNode>();
    node->parent      = std::move(parent);
    node->turns       = std::move(turns);
    node->owned_bytes = sizeof(ResponseContextNode);
    for (const ChatTurn& turn : node->turns) { node->owned_bytes += estimate_turn_bytes(turn); }
    node->cumulative_bytes =
        checked_add(node->owned_bytes, node->parent ? node->parent->cumulative_bytes : 0,
                    "response context byte accounting overflowed");
    node->cumulative_turns =
        checked_add(node->turns.size(), node->parent ? node->parent->cumulative_turns : 0,
                    "response context turn accounting overflowed");
    return node;
}

std::vector<ChatTurn> flatten_response_context(const ResponseContext& context) {
    std::vector<const ResponseContextNode*> nodes;
    for (ResponseContext node = context; node != nullptr; node = node->parent) {
        nodes.push_back(node.get());
    }
    std::vector<ChatTurn> turns;
    turns.reserve(context ? context->cumulative_turns : 0);
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        turns.insert(turns.end(), (*it)->turns.begin(), (*it)->turns.end());
    }
    return turns;
}

ResponseStore::ResponseStore(std::size_t max_records, std::size_t max_bytes)
    : max_records_(max_records), max_bytes_(max_bytes) {
    if (max_records_ == 0 || max_bytes_ == 0) {
        throw std::invalid_argument("response store limits must be positive");
    }
}

std::shared_ptr<const StoredResponse> ResponseStore::get(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto found = records_.find(id);
    if (found == records_.end()) { return {}; }
    lru_.splice(lru_.begin(), lru_, found->second.lru);
    return found->second.response;
}

void ResponseStore::put(StoredResponse response) {
    if (response.id.empty() || response.session_key.empty() ||
        response.session_key.size() > kMaximumContextCacheSessionKeyBytes ||
        !response.response.is_object()) {
        throw std::invalid_argument(
            "stored response must have an id, bounded session key and object body");
    }
    const std::size_t envelope_bytes = record_envelope_bytes(response);
    const std::size_t context_bytes  = response.context ? response.context->cumulative_bytes : 0;
    if (envelope_bytes > max_bytes_ || context_bytes > max_bytes_ - envelope_bytes) {
        throw_store_capacity();
    }

    auto owned = std::make_shared<const StoredResponse>(std::move(response));
    std::lock_guard lock(mutex_);
    if (records_.contains(owned->id)) {
        throw std::logic_error("duplicate response id in response store");
    }
    lru_.push_front(owned->id);
    try {
        records_.emplace(owned->id, Entry{owned, lru_.begin(), envelope_bytes});
    } catch (...) {
        lru_.pop_front();
        throw;
    }
    bool envelope_accounted = false;
    try {
        current_bytes_     = checked_add(current_bytes_, envelope_bytes,
                                         "response store byte accounting overflowed");
        envelope_accounted = true;
        retain_context_locked(owned->context);
    } catch (...) {
        if (envelope_accounted) { current_bytes_ -= envelope_bytes; }
        records_.erase(owned->id);
        lru_.pop_front();
        throw;
    }

    while (records_.size() > max_records_ || current_bytes_ > max_bytes_) {
        if (lru_.empty()) { throw std::logic_error("response store LRU is empty"); }
        auto victim = std::prev(lru_.end());
        if (*victim == owned->id) {
            if (victim == lru_.begin()) { throw_store_capacity(); }
            --victim;
        }
        const std::string victim_id = *victim;
        erase_locked(victim_id);
    }
}

bool ResponseStore::erase(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (!records_.contains(id)) { return false; }
    erase_locked(id);
    return true;
}

std::size_t ResponseStore::size() const {
    std::lock_guard lock(mutex_);
    return records_.size();
}

std::size_t ResponseStore::bytes() const {
    std::lock_guard lock(mutex_);
    return current_bytes_;
}

void ResponseStore::retain_context_locked(const ResponseContext& context) {
    if (!context) { return; }
    const auto root = live_context_references_.find(context.get());
    if (root != live_context_references_.end()) {
        if (root->second == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("response context reference count overflowed");
        }
        ++root->second;
        return;
    }

    std::vector<ResponseContext> new_nodes;
    ResponseContext cursor = context;
    while (cursor && !live_context_references_.contains(cursor.get())) {
        new_nodes.push_back(cursor);
        cursor = cursor->parent;
    }

    std::size_t inserted         = 0;
    const auto rollback_inserted = [&] {
        while (inserted != 0) {
            --inserted;
            current_bytes_ -= new_nodes[inserted]->owned_bytes;
            live_context_references_.erase(new_nodes[inserted].get());
        }
    };
    try {
        for (const ResponseContext& node : new_nodes) {
            const std::size_t next_bytes = checked_add(current_bytes_, node->owned_bytes,
                                                       "response store byte accounting overflowed");
            const auto [entry, added]    = live_context_references_.emplace(node.get(), 1);
            (void)entry;
            if (!added) { throw std::logic_error("response context activation raced itself"); }
            current_bytes_ = next_bytes;
            ++inserted;
        }
    } catch (...) {
        rollback_inserted();
        throw;
    }

    if (cursor) {
        const auto existing = live_context_references_.find(cursor.get());
        if (existing == live_context_references_.end()) {
            rollback_inserted();
            throw std::logic_error("response context activation lost its live parent");
        }
        if (existing->second == std::numeric_limits<std::size_t>::max()) {
            rollback_inserted();
            throw std::overflow_error("response context reference count overflowed");
        }
        ++existing->second;
    }
}

void ResponseStore::release_context_locked(const ResponseContext& context) {
    for (ResponseContext node = context; node != nullptr; node = node->parent) {
        const auto found = live_context_references_.find(node.get());
        if (found == live_context_references_.end() || found->second == 0) {
            throw std::logic_error("response context release lost its live node");
        }
        if (found->second != 1) {
            --found->second;
            return;
        }
        if (current_bytes_ < node->owned_bytes) {
            throw std::logic_error("response context byte accounting underflowed");
        }
        current_bytes_ -= node->owned_bytes;
        live_context_references_.erase(found);
    }
}

void ResponseStore::erase_locked(const std::string& id) {
    const auto found = records_.find(id);
    if (found == records_.end()) { return; }
    if (current_bytes_ < found->second.envelope_bytes) {
        throw std::logic_error("response envelope byte accounting underflowed");
    }
    current_bytes_ -= found->second.envelope_bytes;
    release_context_locked(found->second.response->context);
    lru_.erase(found->second.lru);
    records_.erase(found);
}

} // namespace ninfer::serve
