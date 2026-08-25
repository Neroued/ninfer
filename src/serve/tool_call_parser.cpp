#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string_view>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

bool parse_function_open(std::string_view block, std::size_t& pos, bool tolerant,
                         std::size_t max_name_length, std::string& out_name) {
    skip_ws(block, pos);
    if (pos >= block.size() || block[pos] != '<') { return false; }

    constexpr std::string_view kStrictFn = "<function=";
    if (!tolerant) {
        if (!starts_with_at(block, pos, kStrictFn)) { return false; }
        const std::size_t name_begin = pos + kStrictFn.size();
        const std::size_t name_end   = block.find('>', name_begin);
        if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
        std::string name = std::string(block.substr(name_begin, name_end - name_begin));
        if (!valid_function_name(name, max_name_length)) { return false; }
        out_name = std::move(name);
        pos      = name_end + 1;
        return true;
    }

    std::size_t tag_len = 0;
    if (starts_with_at(block, pos, "<function")) {
        tag_len = 9;
    } else if (starts_with_at(block, pos, "<call")) {
        tag_len = 5;
    } else {
        return false;
    }

    std::size_t cur = pos + tag_len;
    skip_ws(block, cur);
    if (cur >= block.size()) { return false; }

    if (starts_with_at(block, cur, "name")) {
        cur += 4;
        skip_ws(block, cur);
    }
    if (cur < block.size() && (block[cur] == '=' || block[cur] == ':')) {
        ++cur;
        skip_ws(block, cur);
    }

    if (cur >= block.size()) { return false; }

    char quote = 0;
    if (block[cur] == '"' || block[cur] == '\'') {
        quote = block[cur];
        ++cur;
    }

    const std::size_t name_begin = cur;
    std::size_t name_end         = std::string_view::npos;
    if (quote != 0) {
        name_end = block.find(quote, name_begin);
        if (name_end == std::string_view::npos) { return false; }
        cur = name_end + 1;
        skip_ws(block, cur);
        const std::size_t gt = block.find('>', cur);
        if (gt == std::string_view::npos) { return false; }
        pos = gt + 1;
    } else {
        name_end = block.find('>', name_begin);
        if (name_end == std::string_view::npos) { return false; }
        pos = name_end + 1;
    }

    std::string name = trim_ascii(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    out_name = std::move(name);
    return true;
}

bool parse_parameter(std::string_view inner, std::size_t& pos, bool tolerant, Json& args) {
    skip_ws(inner, pos);
    if (pos >= inner.size()) { return false; }

    if (!tolerant) {
        constexpr std::string_view kParamOpen  = "<parameter=";
        constexpr std::string_view kParamClose = "</parameter>";
        if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
        const std::size_t name_begin = pos + kParamOpen.size();
        const std::size_t name_end   = inner.find('>', name_begin);
        if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
        const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
        pos                         = name_end + 1;
        const std::size_t value_end = inner.find(kParamClose, pos);
        if (value_end == std::string_view::npos) { return false; }
        const std::string raw_value = trim_ascii(inner.substr(pos, value_end - pos));
        Json parsed                 = Json::parse(raw_value, nullptr, false);
        args[key]                   = parsed.is_discarded() ? Json(raw_value) : parsed;
        pos                         = value_end + kParamClose.size();
        return true;
    }

    std::size_t tag_len = 0;
    if (starts_with_at(inner, pos, "<parameter")) {
        tag_len = 10;
    } else if (starts_with_at(inner, pos, "<param")) {
        tag_len = 6;
    } else {
        return false;
    }

    std::size_t cur = pos + tag_len;
    skip_ws(inner, cur);
    if (cur >= inner.size()) { return false; }

    if (starts_with_at(inner, cur, "name")) {
        cur += 4;
        skip_ws(inner, cur);
    }
    if (cur < inner.size() && (inner[cur] == '=' || inner[cur] == ':')) {
        ++cur;
        skip_ws(inner, cur);
    }
    if (cur >= inner.size()) { return false; }

    char quote = 0;
    if (inner[cur] == '"' || inner[cur] == '\'') {
        quote = inner[cur];
        ++cur;
    }

    const std::size_t name_begin = cur;
    std::size_t name_end         = std::string_view::npos;
    if (quote != 0) {
        name_end = inner.find(quote, name_begin);
        if (name_end == std::string_view::npos) { return false; }
        cur = name_end + 1;
        skip_ws(inner, cur);
        const std::size_t gt = inner.find('>', cur);
        if (gt == std::string_view::npos) { return false; }
        pos = gt + 1;
    } else {
        name_end = inner.find('>', name_begin);
        if (name_end == std::string_view::npos) { return false; }
        pos = name_end + 1;
    }

    std::string key = trim_ascii(inner.substr(name_begin, name_end - name_begin));
    if (key.empty()) { return false; }

    const std::size_t val_start = pos;
    std::size_t val_end         = std::string_view::npos;
    std::size_t next_pos        = std::string_view::npos;

    constexpr std::array<std::string_view, 2> kCloseTags = {"</parameter>", "</param>"};
    for (const auto& close_tag : kCloseTags) {
        const std::size_t found = inner.find(close_tag, val_start);
        if (found != std::string_view::npos &&
            (val_end == std::string_view::npos || found < val_end)) {
            val_end  = found;
            next_pos = found + close_tag.size();
        }
    }

    if (val_end == std::string_view::npos) {
        const std::size_t next_open = inner.find('<', val_start);
        if (next_open != std::string_view::npos) {
            val_end  = next_open;
            next_pos = next_open;
        } else {
            val_end  = inner.size();
            next_pos = inner.size();
        }
    }

    const std::string raw_value = trim_ascii(inner.substr(val_start, val_end - val_start));
    Json parsed                 = Json::parse(raw_value, nullptr, false);
    args[key]                   = parsed.is_discarded() ? Json(raw_value) : parsed;
    pos                         = next_pos;
    return true;
}

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length, bool tolerant,
                         ToolCall& out) {
    std::size_t pos = 0;
    std::string name;
    if (!parse_function_open(block, pos, tolerant, max_name_length, name)) { return false; }

    std::size_t function_end = std::string_view::npos;
    std::size_t close_len    = 0;

    if (!tolerant) {
        constexpr std::string_view kFunctionClose = "</function>";
        function_end                              = block.find(kFunctionClose, pos);
        if (function_end == std::string_view::npos) { return false; }
        close_len = kFunctionClose.size();
    } else {
        constexpr std::array<std::string_view, 4> kFnCloseTags = {
            "</function>", "</function_invocation>", "</call>", "</tool>"};
        for (const auto& close_tag : kFnCloseTags) {
            const std::size_t found = block.find(close_tag, pos);
            if (found != std::string_view::npos &&
                (function_end == std::string_view::npos || found < function_end)) {
                function_end = found;
                close_len    = close_tag.size();
            }
        }
        if (function_end == std::string_view::npos) {
            function_end = block.size();
            close_len    = 0;
        }
    }

    const std::string_view params = block.substr(pos, function_end - pos);
    Json args                     = Json::object();
    std::size_t param_pos         = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, tolerant, args)) {
            if (!tolerant) { return false; }
            const std::size_t next_tag = params.find('<', param_pos + 1);
            if (next_tag == std::string_view::npos) { break; }
            param_pos = next_tag;
        }
    }

    pos = function_end + close_len;
    skip_ws(block, pos);
    if (!tolerant && pos != block.size()) { return false; }

    out.id             = new_tool_call_id();
    out.name           = std::move(name);
    out.arguments_json = args.dump();
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 bool tolerant) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    std::size_t first        = text.find(kToolOpen);
    std::size_t open_tag_len = kToolOpen.size();

    if (first == std::string::npos && tolerant) {
        constexpr std::array<std::string_view, 3> kAltToolOpens = {
            "<tool_calls>", "<tool-call>", "<call>"};
        for (const auto& alt : kAltToolOpens) {
            const std::size_t found = text.find(alt);
            if (found != std::string::npos && (first == std::string::npos || found < first)) {
                first        = found;
                open_tag_len = alt.size();
            }
        }
        if (first == std::string::npos) {
            if (text.find("<function=") != std::string::npos ||
                text.find("<function name=") != std::string::npos ||
                text.find("<function:") != std::string::npos) {
                first        = text.find("<function");
                open_tag_len = 0;
            }
        }
    }

    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }

        std::size_t inner_begin   = pos;
        std::size_t close         = std::string::npos;
        std::size_t close_tag_len = 0;

        if (open_tag_len > 0 && starts_with_at(text, pos, kToolOpen)) {
            inner_begin   = pos + kToolOpen.size();
            close         = text.find(kToolClose, inner_begin);
            close_tag_len = kToolClose.size();
        } else if (tolerant && open_tag_len > 0) {
            bool matched_open = false;
            constexpr std::array<std::pair<std::string_view, std::string_view>, 3> kAltPairs = {{
                {"<tool_calls>", "</tool_calls>"},
                {"<tool-call>", "</tool-call>"},
                {"<call>", "</call>"},
            }};
            for (const auto& [open_tag, close_tag] : kAltPairs) {
                if (starts_with_at(text, pos, open_tag)) {
                    inner_begin   = pos + open_tag.size();
                    close         = text.find(close_tag, inner_begin);
                    close_tag_len = close_tag.size();
                    matched_open  = true;
                    break;
                }
            }
            if (!matched_open) {
                if (starts_with_at(text, pos, "<function")) {
                    inner_begin   = pos;
                    close         = std::string::npos;
                    close_tag_len = 0;
                } else {
                    if (!out.tool_calls.empty()) { break; }
                    return fallback(text);
                }
            }
        } else if (tolerant && open_tag_len == 0 && starts_with_at(text, pos, "<function")) {
            inner_begin   = pos;
            close         = std::string::npos;
            close_tag_len = 0;
        } else {
            if (tolerant && !out.tool_calls.empty()) { break; }
            return fallback(text);
        }

        if (close == std::string::npos && !tolerant) { return fallback(text); }
        const std::size_t block_end = close == std::string::npos ? text.size() : close;

        ToolCall call;
        if (!parse_one_tool_call(std::string_view(text).substr(inner_begin, block_end - inner_begin),
                                 max_tool_name_length, tolerant, call)) {
            if (tolerant && !out.tool_calls.empty()) { break; }
            return fallback(text);
        }
        out.tool_calls.push_back(std::move(call));
        if (close == std::string::npos) { break; }
        pos = close + close_tag_len;
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    pending_.append(text);
    const std::size_t marker = pending_.find(kToolOpen);
    if (marker != std::string::npos) {
        std::size_t safe_end = marker;
        while (safe_end != 0 &&
               std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
            --safe_end;
        }
        std::string visible = pending_.substr(0, safe_end);
        tool_region_        = pending_.substr(safe_end);
        pending_.clear();
        saw_tool_marker_ = true;
        emitted_bytes_ += visible.size();
        return visible;
    }

    const std::size_t prefix = longest_suffix_prefix(pending_, kToolOpen);
    std::size_t safe_end     = pending_.size() - prefix;
    while (safe_end != 0 && std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
        --safe_end;
    }
    std::string visible = pending_.substr(0, safe_end);
    pending_.erase(0, safe_end);
    emitted_bytes_ += visible.size();
    return visible;
}

std::string ToolCallStreamFilter::finish(bool is_tool_call_response) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    finished_ = true;
    if (is_tool_call_response) {
        pending_.clear();
        tool_region_.clear();
        return {};
    }
    std::string tail = std::move(pending_);
    tail += tool_region_;
    tool_region_.clear();
    emitted_bytes_ += tail.size();
    return tail;
}

} // namespace ninfer::serve
