#include "serve/openai_responses_store.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ChatTurn text_turn(ninfer::ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

StoredOpenAIResponse record(std::string id, OpenAIResponseContext context) {
    StoredOpenAIResponse value;
    value.id          = std::move(id);
    value.session_key = "session-" + value.id;
    value.response =
        nlohmann::json{{"id", value.id}, {"object", "response"}, {"status", "completed"}};
    value.input_items.push_back(nlohmann::json{{"id", "msg_" + value.id}, {"type", "message"}});
    value.context = std::move(context);
    return value;
}

int test_context_dag() {
    const OpenAIResponseContext first =
        append_openai_response_context({}, {text_turn(ninfer::ChatRole::User, "one"),
                                            text_turn(ninfer::ChatRole::Assistant, "a")});
    const OpenAIResponseContext second =
        append_openai_response_context(first, {text_turn(ninfer::ChatRole::User, "two"),
                                               text_turn(ninfer::ChatRole::Assistant, "b")});
    const std::vector<ChatTurn> flattened = flatten_openai_response_context(second);
    int failures                          = 0;
    failures += check(flattened.size() == 4, "context chain flattened all turns");
    failures += check(flattened[0].content[0].text == "one" && flattened[3].content[0].text == "b",
                      "context chain preserves chronological order");
    failures += check(second->parent.get() == first.get(), "context nodes share their parent");
    return failures;
}

int test_lru_and_delete() {
    OpenAIResponsesStore store(2, 1ULL << 20);
    const OpenAIResponseContext root =
        append_openai_response_context({}, {text_turn(ninfer::ChatRole::User, "root")});
    store.put(record("resp_1", root));
    const OpenAIResponseContext child =
        append_openai_response_context(root, {text_turn(ninfer::ChatRole::Assistant, "child")});
    store.put(record("resp_2", child));
    (void)store.get("resp_1"); // resp_2 becomes the least-recently used entry.
    store.put(record("resp_3", append_openai_response_context(
                                   root, {text_turn(ninfer::ChatRole::User, "fork")})));

    int failures = 0;
    failures += check(store.get("resp_1") != nullptr, "get refreshes LRU recency");
    failures += check(store.get("resp_2") == nullptr, "least-recent response evicted");
    failures += check(store.get("resp_3") != nullptr, "new response retained");
    failures += check(store.size() == 2 && store.bytes() != 0, "store reports bounded usage");
    failures += check(store.erase("resp_1"), "stored response deleted");
    failures += check(!store.erase("resp_1") && store.get("resp_1") == nullptr,
                      "deleted response is no longer addressable");
    // The child/fork context owns a shared parent even when the parent's public
    // response entry is deleted.
    const std::shared_ptr<const StoredOpenAIResponse> fork = store.get("resp_3");
    failures += check(fork && flatten_openai_response_context(fork->context).size() == 2,
                      "descendant context survives parent response deletion");
    return failures;
}

int test_oversized_record() {
    OpenAIResponsesStore store(4, 256);
    StoredOpenAIResponse large =
        record("resp_large", append_openai_response_context(
                                 {}, {text_turn(ninfer::ChatRole::User, std::string(1024, 'x'))}));
    std::string code;
    try {
        store.put(std::move(large));
    } catch (const ApiException& exception) { code = exception.error().code; }
    int failures = 0;
    failures += check(code == "response_store_capacity_exceeded",
                      "oversized response fails deterministically");
    failures += check(store.size() == 0, "oversized insertion does not mutate store");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_context_dag();
    failures += test_lru_and_delete();
    failures += test_oversized_record();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
