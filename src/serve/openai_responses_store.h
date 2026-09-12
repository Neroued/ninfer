#pragma once

// Process-local bounded storage for OpenAI Responses objects and their previous_response_id
// context DAG. Stored contexts are flattened when a continuation is submitted; the bounded session
// key is only an Engine lookup/retention hint and this store owns no Program capability.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

struct OpenAIResponseContextNode {
    std::shared_ptr<const OpenAIResponseContextNode> parent;
    std::vector<ChatTurn> turns;
    std::size_t owned_bytes      = 0;
    std::size_t cumulative_bytes = 0;
    std::size_t cumulative_turns = 0;
};

using OpenAIResponseContext = std::shared_ptr<const OpenAIResponseContextNode>;

OpenAIResponseContext append_openai_response_context(OpenAIResponseContext parent,
                                                     std::vector<ChatTurn> turns);
std::vector<ChatTurn> flatten_openai_response_context(const OpenAIResponseContext& context);

struct StoredOpenAIResponse {
    std::string id;
    std::string session_key;
    nlohmann::json response;
    std::vector<nlohmann::json> input_items;
    OpenAIResponseContext context;
    bool preserve_thinking = false;
};

class OpenAIResponsesStore {
public:
    OpenAIResponsesStore(std::size_t max_records, std::size_t max_bytes);

    // get() refreshes LRU recency. Returned immutable records remain valid if
    // another request evicts or deletes their public store entry.
    std::shared_ptr<const StoredOpenAIResponse> get(const std::string& id);
    void put(StoredOpenAIResponse response);
    bool erase(const std::string& id);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t bytes() const;

private:
    struct Entry {
        std::shared_ptr<const StoredOpenAIResponse> response;
        std::list<std::string>::iterator lru;
        std::size_t envelope_bytes = 0;
    };

    void retain_context_locked(const OpenAIResponseContext& context);
    void release_context_locked(const OpenAIResponseContext& context);
    void erase_locked(const std::string& id);

    std::size_t max_records_ = 0;
    std::size_t max_bytes_   = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> records_;
    std::list<std::string> lru_;
    std::unordered_map<const OpenAIResponseContextNode*, std::size_t> live_context_references_;
    std::size_t current_bytes_ = 0;
};

} // namespace ninfer::serve
