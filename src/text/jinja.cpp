#include "text/jinja.h"

#include "text/unicode.h"

#include <jinja/parser.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <unordered_map>

namespace ninfer::text {
namespace {

// Interpreter error decoration must not change request cancellation/deadline errors.
struct Interrupted {
    std::exception_ptr error;
};

std::string pointer_component(std::string value) {
    jinja::string_replace_all(value, "~", "~0");
    jinja::string_replace_all(value, "/", "~1");
    return value;
}

jinja::value convert(const nlohmann::ordered_json& input, const std::string& pointer,
                     const std::unordered_map<std::string, std::uint32_t>& tags, bool literal) {
    using namespace jinja;
    value result;
    if (input.is_null())
        result = mk_val<value_none>();
    else if (input.is_boolean())
        result = mk_val<value_bool>(input.get<bool>());
    else if (input.is_number_float())
        result = mk_val<value_float>(input.get<double>());
    else if (input.is_number_unsigned()) {
        const auto number = input.get<std::uint64_t>();
        if (number > INT64_MAX) throw std::invalid_argument("Template integer exceeds int64");
        result = mk_val<value_int>(static_cast<std::int64_t>(number));
    } else if (input.is_number_integer())
        result = mk_val<value_int>(input.get<std::int64_t>());
    else if (input.is_string())
        result = mk_val<value_string>(string(input.get<std::string>(), 0, literal));
    else if (input.is_array()) {
        auto array = mk_val<value_array>();
        for (std::size_t i = 0; i < input.size(); ++i) {
            array->push_back(convert(input[i], pointer + "/" + std::to_string(i), tags, literal));
        }
        result = std::move(array);
    } else if (input.is_object()) {
        auto object = mk_val<value_object>();
        for (const auto& item : input.items()) {
            object->insert(mk_val<value_string>(string(item.key(), 0, literal)),
                           convert(item.value(), pointer + "/" + pointer_component(item.key()),
                                   tags, literal));
        }
        result = std::move(object);
    } else
        throw std::invalid_argument("Unsupported template JSON value");
    if (const auto it = tags.find(pointer); it != tags.end()) {
        result->origin = it->second;
        if (is_val<value_string>(result)) result->val_str.tag(it->second);
    }
    return result;
}

} // namespace

struct JinjaTemplate::Impl {
    std::string source_name;
    std::string source;
    jinja::program program;

    Impl(std::string text, std::string name) : source_name(std::move(name)) {
        try {
            unicode_internal::utf8_codepoints(text, "chat template");
            auto tokens   = jinja::lexer().tokenize(text);
            source        = std::move(tokens.source);
            tokens.source = source;
            program       = jinja::parse_from_tokens(tokens);
        } catch (const std::exception& error) {
            throw std::invalid_argument(source_name + ": " + error.what());
        }
    }
};

JinjaTemplate::JinjaTemplate(std::string source, std::string source_name)
    : impl_(std::make_shared<Impl>(std::move(source), std::move(source_name))) {}

TemplateOutput JinjaTemplate::render(const nlohmann::ordered_json& input,
                                     const TemplateRenderOptions& options) const {
    if (!input.is_object()) throw std::invalid_argument("Template context must be an object");
    if (options.checkpoint) options.checkpoint();
    try {
        jinja::context context(impl_->source);
        context.current_time = options.timestamp;
        std::size_t steps    = 0;
        if (options.checkpoint) {
            context.checkpoint = [&] {
                if ((++steps & 255) != 0) return;
                try {
                    options.checkpoint();
                } catch (...) { throw Interrupted{std::current_exception()}; }
            };
        }
        std::unordered_map<std::string, std::uint32_t> tags;
        for (const auto& region : options.regions) tags.emplace(region.pointer, region.tag);
        for (const auto& item : input.items()) {
            const bool literal =
                std::find(options.control_variables.begin(), options.control_variables.end(),
                          item.key()) == options.control_variables.end();
            context.set_val(item.key(), convert(item.value(), "/" + pointer_component(item.key()),
                                                tags, literal));
        }
        jinja::runtime runtime(context);
        const auto rendered = jinja::runtime::gather_string_parts(runtime.execute(impl_->program));
        TemplateOutput result;
        for (const auto& part : rendered->val_str.parts) {
            const auto begin = result.text.size();
            result.text += part.val;
            if (part.literal && !part.val.empty()) {
                if (!result.literal_spans.empty() && result.literal_spans.back().end == begin)
                    result.literal_spans.back().end = result.text.size();
                else
                    result.literal_spans.push_back({begin, result.text.size()});
            }
            if (part.origin)
                result.regions.push_back(
                    {part.origin, begin, result.text.size(), part.source_offset});
        }
        if (context.checkpoint) {
            try {
                options.checkpoint();
            } catch (...) { throw Interrupted{std::current_exception()}; }
        }
        return result;
    } catch (const Interrupted& interrupted) {
        std::rethrow_exception(interrupted.error);
    } catch (const std::exception& error) {
        throw std::invalid_argument(impl_->source_name + ": " + error.what());
    }
}

} // namespace ninfer::text
