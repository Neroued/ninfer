#include "runtime.h"
#include "unicode.h"
#include "value.h"

#include <sstream>
#include <string>
#include <cctype>
#include <vector>
#include <optional>
#include <algorithm>

#if defined(_WIN32)
// localtime_s swaps the argument order relative to POSIX localtime_r.
#include <time.h>
static bool jinja_localtime_r(const std::time_t* time, std::tm* out) {
    return ::localtime_s(out, time) == 0;
}
#else
#include <time.h>
static bool jinja_localtime_r(const std::time_t* time, std::tm* out) {
    return ::localtime_r(time, out) != nullptr;
}
#endif

namespace jinja {

// func_args method implementations

value func_args::get_kwarg(const std::string& key, value default_val) const {
    for (const auto& arg : args) {
        if (is_val<value_kwarg>(arg)) {
            auto* kwarg = cast_val<value_kwarg>(arg);
            if (kwarg->key == key) { return kwarg->val; }
        }
    }
    return default_val;
}

value func_args::get_kwarg_or_pos(const std::string& key, size_t pos) const {
    value val = get_kwarg(key, mk_val<value_undefined>());

    if (val->is_undefined() && pos < count() && !is_val<value_kwarg>(args[pos])) {
        return args[pos];
    }

    return val;
}

value func_args::get_pos(size_t pos) const {
    if (count() > pos) { return args[pos]; }
    throw raised_exception("Function '" + func_name + "' expected at least " +
                           std::to_string(pos + 1) + " arguments, got " + std::to_string(count()));
}

value func_args::get_pos(size_t pos, value default_val) const {
    if (count() > pos) { return args[pos]; }
    return default_val;
}

void func_args::push_back(const value& val) { args.push_back(val); }

void func_args::push_front(const value& val) { args.insert(args.begin(), val); }

const std::vector<value>& func_args::get_args() const { return args; }

static std::optional<int64_t> slice_bound(const value& val) {
    if (val->is_undefined() || val->is_none()) return std::nullopt;
    return val->as_int();
}

static value slice_value(const func_args& args) {
    args.ensure_count(4, 4);
    const auto input = args.get_pos(0);
    const auto start = slice_bound(args.get_pos(1));
    const auto stop  = slice_bound(args.get_pos(2));
    const auto step  = slice_bound(args.get_pos(3)).value_or(1);
    if (is_val<value_string>(input)) {
        return mk_val<value_string>(input->as_string().slice(start, stop, step));
    }
    const auto& array = input->as_array();
    const auto bounds = unicode::slice_indices(array.size(), start, stop, step);
    auto result       = mk_val<value_array>();
    for (int64_t i = bounds.start; step > 0 ? i < bounds.stop : i > bounds.stop;) {
        if (args.ctx.checkpoint) args.ctx.checkpoint();
        result->push_back(array[static_cast<size_t>(i)]);
        if ((step > 0 && i > INT64_MAX - step) || (step < 0 && i < INT64_MIN - step)) break;
        i += step;
    }
    return result;
}

static value split_string(const func_args& args, bool reverse) {
    args.ensure_count(1, 3);
    const auto separator = args.get_kwarg_or_pos("sep", 1);
    const auto count     = args.get_kwarg_or_pos("maxsplit", 2);
    const auto source    = args.get_pos(0)->as_string();
    std::optional<std::string> delimiter;
    if (!separator->is_undefined() && !separator->is_none())
        delimiter = separator->as_string().str();
    auto result = mk_val<value_array>();
    for (const auto& part :
         source.split(delimiter, count->is_undefined() ? -1 : count->as_int(), reverse)) {
        result->push_back(mk_val<value_string>(part));
    }
    return result;
}

template <typename T>
static value empty_value_fn(const func_args&) {
    if constexpr (std::is_same_v<T, value_int>) {
        return mk_val<T>(0);
    } else if constexpr (std::is_same_v<T, value_float>) {
        return mk_val<T>(0.0);
    } else if constexpr (std::is_same_v<T, value_bool>) {
        return mk_val<T>(false);
    } else {
        return mk_val<T>();
    }
}

template <typename T>
static value test_type_fn(const func_args& args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0));
    return mk_val<value_bool>(is_type);
}

template <typename T, typename U>
static value test_type_fn(const func_args& args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0)) || is_val<U>(args.get_pos(0));
    return mk_val<value_bool>(is_type);
}

template <typename T, typename U, typename V>
static value test_type_fn(const func_args& args) {
    args.ensure_count(1);
    bool is_type =
        is_val<T>(args.get_pos(0)) || is_val<U>(args.get_pos(0)) || is_val<V>(args.get_pos(0));
    return mk_val<value_bool>(is_type);
}

template <value_compare_op op>
static value test_compare_fn(const func_args& args) {
    args.ensure_count(2, 2);
    return mk_val<value_bool>(value_compare(args.get_pos(0), args.get_pos(1), op));
}

static value tojson(const func_args& args) {
    args.ensure_count(1, 5);
    value val_ascii      = args.get_kwarg_or_pos("ensure_ascii", 1);
    value val_indent     = args.get_kwarg_or_pos("indent", 2);
    value val_separators = args.get_kwarg_or_pos("separators", 3);
    value val_sort       = args.get_kwarg_or_pos("sort_keys", 4);
    int indent           = -1;

    if (is_val<value_int>(val_indent)) { indent = static_cast<int>(val_indent->as_int()); }
    const bool ensure_ascii = val_ascii->as_bool(); // undefined == false
    auto separators =
        (is_val<value_array>(val_separators) ? val_separators : mk_val<value_array>())->as_array();
    std::string item_sep =
        separators.size() > 0 ? separators[0]->as_string().str() : (indent < 0 ? ", " : ",");
    std::string key_sep  = separators.size() > 1 ? separators[1]->as_string().str() : ": ";
    std::string json_str = value_to_json(args.get_pos(0), indent, item_sep, key_sep, ensure_ascii,
                                         val_sort->as_bool());
    // Serialization produces data, including nested keys and values.
    auto result = mk_val<value_string>(string(json_str, 0, true));
    if (args.get_pos(0)->origin)
        result->val_str.tag(args.get_pos(0)->origin, false);
    else if (is_val<value_string>(args.get_pos(0))) {
        const auto& parts = args.get_pos(0)->val_str.parts;
        if (parts.size() == 1) result->val_str.tag(parts.front().origin, false);
    }
    return result;
}

template <bool is_reject>
static value selectattr(const func_args& args) {
    args.ensure_count(2, 4);
    args.ensure_vals<value_array, value_string, value_string, value_string>(true, true, false,
                                                                            false);

    auto arr          = args.get_pos(0)->as_array();
    auto attribute    = args.get_pos(1);
    auto out          = mk_val<value_array>();
    value val_default = mk_val<value_undefined>();

    if (args.count() == 2) {
        // example: array | selectattr("active")
        for (const auto& item : arr) {
            if (!is_val<value_object>(item)) {
                throw raised_exception("selectattr: item is not an object");
            }
            value attr_val   = item->at(attribute, val_default);
            bool is_selected = attr_val->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.count() == 3) {
        // example: array | selectattr("equalto", "text")
        // translated to: test_is_equalto(item, "text")
        std::string test_name = args.get_pos(1)->as_string().str();
        value test_val        = args.get_pos(2);
        auto& builtins        = global_builtins();
        auto it               = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto& item : arr) {
            func_args test_args(args.ctx);
            test_args.push_back(item);     // current object
            test_args.push_back(test_val); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected  = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.count() == 4) {
        // example: array | selectattr("status", "equalto", "active")
        // translated to: test_is_equalto(item.status, "active")
        std::string test_name = args.get_pos(2)->as_string().str();
        auto extra_arg        = args.get_pos(3);
        auto& builtins        = global_builtins();
        auto it               = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto& item : arr) {
            if (!is_val<value_object>(item)) {
                throw raised_exception("selectattr: item is not an object");
            }
            value attr_val = item->at(attribute, val_default);
            func_args test_args(args.ctx);
            test_args.push_back(attr_val);  // attribute value
            test_args.push_back(extra_arg); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected  = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;
    } else {
        throw raised_exception("selectattr: invalid number of arguments");
    }

    return out;
}

static value default_value(const func_args& args) {
    args.ensure_count(1, 3);
    const auto input       = args.get_pos(0);
    const bool use_default = input->is_undefined() ||
                             (args.get_kwarg_or_pos("boolean", 2)->as_bool() && !input->as_bool());
    return use_default ? args.get_pos(1, mk_val<value_string>("")) : input;
}

const func_builtins& global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             std::string msg = args.get_pos(0)->as_string().str();
             throw raised_exception("Jinja Exception: " + msg);
         }},
        {"namespace",
         [](const func_args& args) -> value {
             auto out = mk_val<value_object>();
             for (const auto& arg : args.get_args()) {
                 if (!is_val<value_kwarg>(arg)) {
                     throw raised_exception("namespace() arguments must be kwargs");
                 }
                 auto kwarg = cast_val<value_kwarg>(arg);
                 out->insert(kwarg->key, kwarg->val);
             }
             return out;
         }},
        {"strftime_now",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             std::string format = args.get_pos(0)->as_string().str();
            std::tm local{};
            if (!jinja_localtime_r(&args.ctx.current_time, &local)) {
                 throw raised_exception("strftime_now: invalid time");
             }
             if (format.empty()) return mk_val<value_string>("");
             for (size_t capacity = 128; capacity <= 65536; capacity *= 2) {
                 std::string buffer(capacity, '\0');
                 const auto size =
                     std::strftime(buffer.data(), buffer.size(), format.c_str(), &local);
                 if (size) {
                     buffer.resize(size);
                     return mk_val<value_string>(string(buffer, 0, true));
                 }
             }
             throw raised_exception("strftime_now: failed to format time");
         }},
        {"range",
         [](const func_args& args) -> value {
             args.ensure_count(1, 3);
             args.ensure_vals<value_int, value_int, value_int>(true, false, false);

             auto arg0 = args.get_pos(0);
             auto arg1 = args.get_pos(1, mk_val<value_undefined>());
             auto arg2 = args.get_pos(2, mk_val<value_undefined>());

             int64_t start, stop, step;
             if (args.count() == 1) {
                 start = 0;
                 stop  = arg0->as_int();
                 step  = 1;
             } else if (args.count() == 2) {
                 start = arg0->as_int();
                 stop  = arg1->as_int();
                 step  = 1;
             } else {
                 start = arg0->as_int();
                 stop  = arg1->as_int();
                 step  = arg2->as_int();
             }

             auto out = mk_val<value_array>();
             if (step == 0) { throw raised_exception("range() step argument must not be zero"); }
             for (int64_t i = start; step > 0 ? i < stop : i > stop;) {
                 if (args.ctx.checkpoint) args.ctx.checkpoint();
                 out->push_back(mk_val<value_int>(i));
                 if ((step > 0 && i > INT64_MAX - step) || (step < 0 && i < INT64_MIN - step))
                     break;
                 i += step;
             }
             return out;
         }},
        {"tojson", tojson},

        // tests
        {"test_is_boolean", test_type_fn<value_bool>},
        {"test_is_callable", test_type_fn<value_func>},
        {"test_is_odd",
         [](const func_args& args) -> value {
             args.ensure_vals<value_int>();
             int64_t val = args.get_pos(0)->as_int();
             return mk_val<value_bool>(val % 2 != 0);
         }},
        {"test_is_even",
         [](const func_args& args) -> value {
             args.ensure_vals<value_int>();
             int64_t val = args.get_pos(0)->as_int();
             return mk_val<value_bool>(val % 2 == 0);
         }},
        {"test_is_false",
         [](const func_args& args) -> value {
             args.ensure_count(1);
             bool val = is_val<value_bool>(args.get_pos(0)) && !args.get_pos(0)->as_bool();
             return mk_val<value_bool>(val);
         }},
        {"test_is_true",
         [](const func_args& args) -> value {
             args.ensure_count(1);
             bool val = is_val<value_bool>(args.get_pos(0)) && args.get_pos(0)->as_bool();
             return mk_val<value_bool>(val);
         }},
        {"test_is_divisibleby",
         [](const func_args& args) -> value {
             args.ensure_vals<value_int, value_int>();
             bool res = args.get_pos(0)->val_int % args.get_pos(1)->val_int == 0;
             return mk_val<value_bool>(res);
         }},
        {"test_is_string", test_type_fn<value_string>},
        {"test_is_integer", test_type_fn<value_int>},
        {"test_is_float", test_type_fn<value_float>},
        {"test_is_number", test_type_fn<value_int, value_float>},
        {"test_is_iterable", test_type_fn<value_array, value_string, value_undefined>},
        {"test_is_sequence", test_type_fn<value_array, value_string, value_undefined>},
        {"test_is_mapping", test_type_fn<value_object>},
        {"test_is_lower",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             return mk_val<value_bool>(args.get_pos(0)->val_str.is_lowercase());
         }},
        {"test_is_upper",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             return mk_val<value_bool>(args.get_pos(0)->val_str.is_uppercase());
         }},
        {"test_is_none", test_type_fn<value_none>},
        {"test_is_defined",
         [](const func_args& args) -> value {
             args.ensure_count(1);
             bool res = !args.get_pos(0)->is_undefined();
             return mk_val<value_bool>(res);
         }},
        {"test_is_undefined", test_type_fn<value_undefined>},
        {"test_is_eq", test_compare_fn<value_compare_op::eq>},
        {"test_is_equalto", test_compare_fn<value_compare_op::eq>},
        {"test_is_ge", test_compare_fn<value_compare_op::ge>},
        {"test_is_gt", test_compare_fn<value_compare_op::gt>},
        {"test_is_greaterthan", test_compare_fn<value_compare_op::gt>},
        {"test_is_lt", test_compare_fn<value_compare_op::lt>},
        {"test_is_lessthan", test_compare_fn<value_compare_op::lt>},
        {"test_is_ne", test_compare_fn<value_compare_op::ne>},
        {"test_is_in",
         [](const func_args& args) -> value {
             args.ensure_count(2);
             auto needle   = args.get_pos(0);
             auto haystack = args.get_pos(1);
             if (is_val<value_undefined>(haystack)) { return mk_val<value_bool>(false); }
             if (is_val<value_array>(haystack)) {
                 for (const auto& item : haystack->as_array()) {
                     if (*needle == *item) { return mk_val<value_bool>(true); }
                 }
                 return mk_val<value_bool>(false);
             }
             if (is_val<value_string>(haystack)) {
                 if (!is_val<value_string>(needle)) {
                     throw raised_exception("'in' test expects args[1] as string when args[0] is "
                                            "string, got args[1] as " +
                                            needle->type());
                 }
                 return mk_val<value_bool>(haystack->as_string().str().find(
                                               needle->as_string().str()) != std::string::npos);
             }
             if (is_val<value_object>(haystack)) {
                 return mk_val<value_bool>(haystack->has_key(needle));
             }
             throw raised_exception("'in' test expects iterable as first argument, got " +
                                    haystack->type());
         }},
        {"test_is_test",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             auto& builtins        = global_builtins();
             std::string test_name = args.get_pos(0)->val_str.str();
             auto it               = builtins.find("test_is_" + test_name);
             bool res              = it != builtins.end();
             return mk_val<value_bool>(res);
         }},
        {"test_is_sameas",
         [](const func_args& args) -> value {
             // Check if an object points to the same memory address as another object
             (void)args;
             throw not_implemented_exception("sameas test not implemented");
         }},
        {"test_is_escaped",
         [](const func_args& args) -> value {
             (void)args;
             throw not_implemented_exception("escaped test not implemented");
         }},
        {"test_is_filter",
         [](const func_args& args) -> value {
             (void)args;
             throw not_implemented_exception("filter test not implemented");
         }},
    };
    return builtins;
}

const func_builtins& value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs",
         [](const func_args& args) -> value {
             args.ensure_vals<value_int>();
             int64_t val = args.get_pos(0)->as_int();
             return mk_val<value_int>(val < 0 ? -val : val);
         }},
        {"int",
         [](const func_args& args) -> value {
             args.ensure_vals<value_int>();
             return mk_val<value_int>(args.get_pos(0)->as_int());
         }},
        {"float",
         [](const func_args& args) -> value {
             args.ensure_vals<value_int>();
             double val = static_cast<double>(args.get_pos(0)->as_int());
             return mk_val<value_float>(val);
         }},
        {"safe", tojson},
        {"string", tojson},
        {"tojson", tojson},
    };
    return builtins;
}

const func_builtins& value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs",
         [](const func_args& args) -> value {
             args.ensure_vals<value_float>();
             double val = args.get_pos(0)->as_float();
             return mk_val<value_float>(val < 0.0 ? -val : val);
         }},
        {"int",
         [](const func_args& args) -> value {
             args.ensure_vals<value_float>();
             int64_t val = static_cast<int64_t>(args.get_pos(0)->as_float());
             return mk_val<value_int>(val);
         }},
        {"float",
         [](const func_args& args) -> value {
             args.ensure_vals<value_float>();
             return mk_val<value_float>(args.get_pos(0)->as_float());
         }},
        {"safe", tojson},
        {"string", tojson},
        {"tojson", tojson},
    };
    return builtins;
}

static bool string_startswith(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) return false;
    return str.compare(0, prefix.length(), prefix) == 0;
}

static bool string_endswith(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

[[noreturn]] static value string_join_not_implemented(const func_args&) {
    throw not_implemented_exception("String join builtin not implemented");
}

const func_builtins& value_string_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"upper",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             jinja::string str = args.get_pos(0)->as_string().uppercase();
             return mk_val<value_string>(str);
         }},
        {"lower",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             jinja::string str = args.get_pos(0)->as_string().lowercase();
             return mk_val<value_string>(str);
         }},
        {"strip",
         [](const func_args& args) -> value {
             value val_input = args.get_pos(0);
             if (!is_val<value_string>(val_input)) {
                 throw raised_exception("strip() first argument must be a string");
             }
             value val_chars = args.get_kwarg_or_pos("chars", 1);
             if (val_chars->is_undefined()) {
                 return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true));
             } else {
                 return mk_val<value_string>(
                     args.get_pos(0)->as_string().strip(true, true, val_chars->as_string().str()));
             }
         }},
        {"rstrip",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             value val_chars = args.get_kwarg_or_pos("chars", 1);
             if (val_chars->is_undefined()) {
                 return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true));
             } else {
                 return mk_val<value_string>(
                     args.get_pos(0)->as_string().strip(false, true, val_chars->as_string().str()));
             }
         }},
        {"lstrip",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             value val_chars = args.get_kwarg_or_pos("chars", 1);
             if (val_chars->is_undefined()) {
                 return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false));
             } else {
                 return mk_val<value_string>(
                     args.get_pos(0)->as_string().strip(true, false, val_chars->as_string().str()));
             }
         }},
        {"title",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             jinja::string str = args.get_pos(0)->as_string().titlecase();
             return mk_val<value_string>(str);
         }},
        {"capitalize",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             jinja::string str = args.get_pos(0)->as_string().capitalize();
             return mk_val<value_string>(str);
         }},
        {"length",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             jinja::string str = args.get_pos(0)->as_string();
             return mk_val<value_int>(str.length());
         }},
        {"startswith",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string, value_string>();
             std::string str    = args.get_pos(0)->as_string().str();
             std::string prefix = args.get_pos(1)->as_string().str();
             return mk_val<value_bool>(string_startswith(str, prefix));
         }},
        {"endswith",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string, value_string>();
             std::string str    = args.get_pos(0)->as_string().str();
             std::string suffix = args.get_pos(1)->as_string().str();
             return mk_val<value_bool>(string_endswith(str, suffix));
         }},
        {"split", [](const func_args& args) { return split_string(args, false); }},
        {"rsplit", [](const func_args& args) { return split_string(args, true); }},
        {"replace",
         [](const func_args& args) -> value {
             args.ensure_count(3, 4);
             const auto source  = args.get_pos(0)->as_string();
             const auto old_str = args.get_pos(1)->as_string().str();
             const auto new_str = args.get_pos(2)->as_string();
             const auto limit   = args.get_kwarg_or_pos("count", 3);
             int64_t remaining  = limit->is_undefined() || limit->is_none() ? -1 : limit->as_int();
             if (!remaining) return mk_val<value_string>(source);
             const auto text = source.str();
             string result;
             size_t cursor = 0;
             if (old_str.empty()) {
                 result.append(new_str);
                 if (remaining > 0) --remaining;
                 for (const auto& ch : unicode::characters(text)) {
                     result.append(source.cut_bytes(ch.begin, ch.end));
                     cursor = ch.end;
                     if (remaining == 0) break;
                     result.append(new_str);
                     if (remaining > 0) --remaining;
                 }
             } else {
                 while (remaining != 0) {
                     if (args.ctx.checkpoint) args.ctx.checkpoint();
                     const auto pos = text.find(old_str, cursor);
                     if (pos == std::string::npos) break;
                     result.append(source.cut_bytes(cursor, pos)).append(new_str);
                     cursor = pos + old_str.size();
                     if (remaining > 0) --remaining;
                 }
             }
             result.append(source.cut_bytes(cursor, text.size()));
             return mk_val<value_string>(result);
         }},
        {"format",
         [](const func_args& args) -> value {
             value val_input = args.get_pos(0);
             if (!is_val<value_string>(val_input)) {
                 throw raised_exception("format() first argument must be a string");
             }
             const jinja::string& fmt = val_input->as_string();

             const std::string str = fmt.str();
             jinja::string result;
             size_t begin   = 0;
             size_t arg_idx = 1; // positional args follow the format string
             for (size_t i = 0; i < str.size(); ++i) {
                 if (str[i] != '{') continue;
                 if (i + 1 >= str.size() || str[i + 1] != '}') {
                     throw not_implemented_exception(
                         "format() only supports simple '{}' placeholders");
                 }
                 result.append(fmt.cut_bytes(begin, i));
                 result.append(args.get_pos(arg_idx++)->as_string());
                 begin = ++i + 1;
             }
             result.append(fmt.cut_bytes(begin, str.size()));
             return mk_val<value_string>(result);
         }},
        {"int",
         [](const func_args& args) -> value {
             value val_input   = args.get_pos(0);
             value val_default = args.get_kwarg_or_pos("default", 1);
             value val_base    = args.get_kwarg_or_pos("base", 2);
             const int base    = val_base->is_undefined() ? 10 : val_base->as_int();
             if (base != 0 && (base < 2 || base > 36)) {
                 // an out-of-range base makes std::stoi fail fast on the MSVC CRT instead of
                 // throwing
                 throw raised_exception("int() base must be 0 or between 2 and 36");
             }
             if (is_val<value_string>(val_input) == false) {
                 throw raised_exception("int() first argument must be a string");
             }
             std::string str = val_input->as_string().str();
             try {
                 return mk_val<value_int>(std::stoi(str, nullptr, base));
             } catch (...) {
                 return mk_val<value_int>(val_default->is_undefined() ? 0 : val_default->as_int());
             }
         }},
        {"float",
         [](const func_args& args) -> value {
             args.ensure_vals<value_string>();
             value val_default = args.get_kwarg_or_pos("default", 1);
             std::string str   = args.get_pos(0)->as_string().str();
             try {
                 return mk_val<value_float>(std::stod(str));
             } catch (...) {
                 return mk_val<value_float>(val_default->is_undefined() ? 0.0
                                                                        : val_default->as_float());
             }
         }},
        {"string",
         [](const func_args& args) -> value {
             // no-op
             args.ensure_vals<value_string>();
             return mk_val<value_string>(args.get_pos(0)->as_string());
         }},
        {"default", default_value},
        {"slice", slice_value},
        {"safe",
         [](const func_args& args) -> value {
             // no-op for now
             args.ensure_vals<value_string>();
             return args.get_pos(0);
         }},
        {"tojson", tojson},
        {"indent",
         [](const func_args& args) -> value {
             args.ensure_count(1, 4);
             value val_input  = args.get_pos(0);
             value val_width  = args.get_kwarg_or_pos("width", 1);
             const bool first = args.get_kwarg_or_pos("first", 2)->as_bool(); // undefined == false
             const bool blank = args.get_kwarg_or_pos("blank", 3)->as_bool(); // undefined == false
             if (!is_val<value_string>(val_input)) {
                 throw raised_exception("indent() first argument must be a string");
             }
             jinja::string indent;
             if (is_val<value_int>(val_width)) {
                 indent = string(std::string(val_width->as_int(), ' '));
             } else if (is_val<value_string>(val_width)) {
                 indent = val_width->as_string();
             } else {
                 indent = string("    ");
             }
             jinja::string result;
             const auto source = val_input->as_string();
             const auto input  = source.str();
             for (size_t begin = 0; begin < input.size();) {
                 const auto newline = input.find('\n', begin);
                 const auto end     = newline == std::string::npos ? input.size() : newline;
                 if (begin == 0 ? first : (begin != end || blank)) result.append(indent);
                 const auto next = newline == std::string::npos ? end : end + 1;
                 result.append(source.cut_bytes(begin, next));
                 begin = next;
             }
             if (!input.empty() && input.back() == '\n' && blank) result.append(indent);
             return mk_val<value_string>(result);
         }},
        {"join", string_join_not_implemented},
    };
    return builtins;
}

const func_builtins& value_bool_t::get_builtins() const {
    static const func_handler tostring = [](const func_args& args) -> value {
        args.ensure_vals<value_bool>();
        bool val = args.get_pos(0)->as_bool();
        return mk_val<value_string>(val ? "True" : "False");
    };
    static const func_builtins builtins = {
        {"default", default_value},
        {"int",
         [](const func_args& args) -> value {
             args.ensure_vals<value_bool>();
             bool val = args.get_pos(0)->as_bool();
             return mk_val<value_int>(val ? 1 : 0);
         }},
        {"float",
         [](const func_args& args) -> value {
             args.ensure_vals<value_bool>();
             bool val = args.get_pos(0)->as_bool();
             return mk_val<value_float>(val ? 1.0 : 0.0);
         }},
        {"safe", tostring},
        {"string", tostring},
        {"tojson", tojson},
    };
    return builtins;
}

[[noreturn]] static value array_unique_not_implemented(const func_args&) {
    throw not_implemented_exception("Array unique builtin not implemented");
}

const func_builtins& value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"list",
         [](const func_args& args) -> value {
             args.ensure_vals<value_array>();
             const auto& arr = args.get_pos(0)->as_array();
             auto result     = mk_val<value_array>();
             for (const auto& v : arr) { result->push_back(v); }
             return result;
         }},
        {"first",
         [](const func_args& args) -> value {
             args.ensure_vals<value_array>();
             const auto& arr = args.get_pos(0)->as_array();
             if (arr.empty()) { return mk_val<value_undefined>(); }
             return arr[0];
         }},
        {"last",
         [](const func_args& args) -> value {
             args.ensure_vals<value_array>();
             const auto& arr = args.get_pos(0)->as_array();
             if (arr.empty()) { return mk_val<value_undefined>(); }
             return arr[arr.size() - 1];
         }},
        {"length",
         [](const func_args& args) -> value {
             args.ensure_vals<value_array>();
             const auto& arr = args.get_pos(0)->as_array();
             return mk_val<value_int>(static_cast<int64_t>(arr.size()));
         }},
        {"slice", slice_value},
        {"selectattr", selectattr<false>},
        {"select", selectattr<false>},
        {"rejectattr", selectattr<true>},
        {"reject", selectattr<true>},
        {"join",
         [](const func_args& args) -> value {
             args.ensure_count(1, 3);
             if (!is_val<value_array>(args.get_pos(0))) {
                 throw raised_exception("join() first argument must be an array");
             }
             value val_delim        = args.get_kwarg_or_pos("d", 1);
             value attribute        = args.get_kwarg_or_pos("attribute", 2);
             const auto& arr        = args.get_pos(0)->as_array();
             const bool attr_is_int = is_val<value_int>(attribute);
             if (!attribute->is_undefined() && !is_val<value_string>(attribute) && !attr_is_int) {
                 throw raised_exception("join() attribute must be string or integer");
             }
             const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
             const auto delim       = val_delim->is_undefined() ? string() : val_delim->as_string();
             jinja::string result;
             for (size_t i = 0; i < arr.size(); ++i) {
                 value val_arr = arr[i];
                 if (!attribute->is_undefined()) {
                     if (attr_is_int && is_val<value_array>(val_arr)) {
                         val_arr = val_arr->at(attr_int);
                     } else if (!attr_is_int && is_val<value_object>(val_arr)) {
                         val_arr = val_arr->at(attribute);
                     }
                 }
                 if (!is_val<value_string>(val_arr) && !is_val<value_int>(val_arr) &&
                     !is_val<value_float>(val_arr)) {
                     throw raised_exception("join() can only join arrays of strings or numerics");
                 }
                 result.append(val_arr->as_string());
                 if (i < arr.size() - 1) result.append(delim);
             }
             return mk_val<value_string>(result);
         }},
        {"string",
         [](const func_args& args) -> value {
             args.ensure_vals<value_array>();

             return mk_val<value_string>(args.get_pos(0)->as_string());
         }},
        {"tojson", tojson},
        {"map",
         [](const func_args& args) -> value {
             args.ensure_count(2);
             if (!is_val<value_array>(args.get_pos(0))) {
                 throw raised_exception("map: first argument must be an array");
             }
             if (!is_val<value_kwarg>(args.get_args().at(1))) {
                 throw not_implemented_exception("map: filter-mapping not implemented");
             }
             value val              = args.get_pos(0);
             value attribute        = args.get_kwarg_or_pos("attribute", 1);
             const bool attr_is_int = is_val<value_int>(attribute);
             if (!is_val<value_string>(attribute) && !attr_is_int) {
                 throw raised_exception("map: attribute must be string or integer");
             }
             const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
             value default_val      = args.get_kwarg("default", mk_val<value_undefined>());
             auto out               = mk_val<value_array>();
             auto arr               = val->as_array();
             for (const auto& item : arr) {
                 value attr_val;
                 if (attr_is_int) {
                     attr_val =
                         is_val<value_array>(item) ? item->at(attr_int, default_val) : default_val;
                 } else {
                     attr_val = is_val<value_object>(item) ? item->at(attribute, default_val)
                                                           : default_val;
                 }
                 out->push_back(attr_val);
             }
             return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(out->as_array()))
                                             : out;
         }},
        {"append",
         [](const func_args& args) -> value {
             args.ensure_count(2);
             if (!is_val<value_array>(args.get_pos(0))) {
                 throw raised_exception("append: first argument must be an array");
             }
             const value_array_t* arr = cast_val<value_array>(args.get_pos(0));
             // need to use const_cast here to modify the array
             value_array_t* arr_editable = const_cast<value_array_t*>(arr);
             arr_editable->push_back(args.get_pos(1));
             return args.get_pos(0);
         }},
        {"pop",
         [](const func_args& args) -> value {
             args.ensure_count(1, 2);
             args.ensure_vals<value_array, value_int>(true, false);
             int64_t index            = args.count() == 2 ? args.get_pos(1)->as_int() : -1;
             const value_array_t* arr = cast_val<value_array>(args.get_pos(0));
             // need to use const_cast here to modify the array
             value_array_t* arr_editable = const_cast<value_array_t*>(arr);
             return arr_editable->pop_at(index);
         }},
        {"sort",
         [](const func_args& args) -> value {
             args.ensure_count(1, 4);
             if (!is_val<value_array>(args.get_pos(0))) {
                 throw raised_exception("sort: first argument must be an array");
             }
             value val         = args.get_pos(0);
             value val_reverse = args.get_kwarg_or_pos("reverse", 1);
             value val_case    = args.get_kwarg_or_pos("case_sensitive", 2);
             value attribute   = args.get_kwarg_or_pos("attribute", 3);
             // FIXME: sorting is currently always case sensitive
             // const bool case_sensitive = val_case->as_bool(); // undefined == false
             const bool reverse     = val_reverse->as_bool(); // undefined == false
             const bool attr_is_int = is_val<value_int>(attribute);
             const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
             std::vector<value> arr = val->as_array(); // copy
             std::sort(arr.begin(), arr.end(), [&](const value& a, const value& b) {
                 value val_a = a;
                 value val_b = b;
                 if (!attribute->is_undefined()) {
                     if (attr_is_int && is_val<value_array>(a) && is_val<value_array>(b)) {
                         val_a = a->at(attr_int);
                         val_b = b->at(attr_int);
                     } else if (!attr_is_int && is_val<value_object>(a) &&
                                is_val<value_object>(b)) {
                         val_a = a->at(attribute);
                         val_b = b->at(attribute);
                     } else {
                         throw raised_exception(
                             "sort: unsupported object attribute comparison between " + a->type() +
                             " and " + b->type());
                     }
                 }
                 return value_compare(val_a, val_b,
                                      reverse ? value_compare_op::gt : value_compare_op::lt);
             });
             return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr))
                                             : mk_val<value_array>(std::move(arr));
         }},
        {"reverse",
         [](const func_args& args) -> value {
             args.ensure_vals<value_array>();
             value val              = args.get_pos(0);
             std::vector<value> arr = val->as_array(); // copy
             std::reverse(arr.begin(), arr.end());
             return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr))
                                             : mk_val<value_array>(std::move(arr));
         }},
        {"min",
         [](const func_args& args) -> value {
             args.ensure_count(1, 4);
             args.ensure_vals<value_array>();
             value val_case  = args.get_kwarg_or_pos("case_sensitive", 1);
             value attribute = args.get_kwarg_or_pos("attribute", 2);
             if (!attribute->is_undefined()) {
                 throw not_implemented_exception("min: attribute not implemented");
             }
             // FIXME: min is currently always case sensitive
             (void)val_case;
             const auto& arr = args.get_pos(0)->as_array();
             if (arr.empty()) { return mk_val<value_undefined>(); }
             value result = arr[0];
             for (size_t i = 1; i < arr.size(); ++i) {
                 if (value_compare(arr[i], result, value_compare_op::lt)) { result = arr[i]; }
             }
             return result;
         }},
        {"max",
         [](const func_args& args) -> value {
             args.ensure_count(1, 4);
             args.ensure_vals<value_array>();
             value val_case  = args.get_kwarg_or_pos("case_sensitive", 1);
             value attribute = args.get_kwarg_or_pos("attribute", 2);
             if (!attribute->is_undefined()) {
                 throw not_implemented_exception("max: attribute not implemented");
             }
             // FIXME: max is currently always case sensitive
             (void)val_case;
             const auto& arr = args.get_pos(0)->as_array();
             if (arr.empty()) { return mk_val<value_undefined>(); }
             value result = arr[0];
             for (size_t i = 1; i < arr.size(); ++i) {
                 if (value_compare(arr[i], result, value_compare_op::gt)) { result = arr[i]; }
             }
             return result;
         }},
        {"unique", array_unique_not_implemented},
    };
    return builtins;
}

[[noreturn]] static value object_join_not_implemented(const func_args&) {
    throw not_implemented_exception("object join not implemented");
}

const func_builtins& value_object_t::get_builtins() const {
    if (!has_builtins) {
        static const func_builtins no_builtins = {};
        return no_builtins;
    }

    static const func_builtins builtins = {
        // {"default", default_value}, // cause issue with gpt-oss
        {"get",
         [](const func_args& args) -> value {
             args.ensure_count(2, 3);
             if (!is_val<value_object>(args.get_pos(0))) {
                 throw raised_exception("get: first argument must be an object");
             }
             if (!is_val<value_string>(args.get_pos(1))) {
                 throw raised_exception("get: second argument must be a string (key)");
             }
             value default_val = mk_val<value_none>();
             if (args.count() == 3) { default_val = args.get_pos(2); }
             const value obj = args.get_pos(0);
             const value key = args.get_pos(1);
             return obj->at(key, default_val);
         }},
        {"keys",
         [](const func_args& args) -> value {
             args.ensure_vals<value_object>();
             const auto& obj = args.get_pos(0)->as_ordered_object();
             auto result     = mk_val<value_array>();
             for (const auto& pair : obj) { result->push_back(pair.first); }
             return result;
         }},
        {"values",
         [](const func_args& args) -> value {
             args.ensure_vals<value_object>();
             const auto& obj = args.get_pos(0)->as_ordered_object();
             auto result     = mk_val<value_array>();
             for (const auto& pair : obj) { result->push_back(pair.second); }
             return result;
         }},
        {"items",
         [](const func_args& args) -> value {
             args.ensure_vals<value_object>();
             const auto& obj = args.get_pos(0)->as_ordered_object();
             auto result     = mk_val<value_array>();
             for (const auto& pair : obj) {
                 auto item = mk_val<value_tuple>(pair);
                 result->push_back(std::move(item));
             }
             return result;
         }},
        {"tojson", tojson},
        {"string",
         [](const func_args& args) -> value {
             args.ensure_vals<value_object>();

             return mk_val<value_string>(args.get_pos(0)->as_string());
         }},
        {"length",
         [](const func_args& args) -> value {
             args.ensure_vals<value_object>();
             const auto& obj = args.get_pos(0)->as_ordered_object();
             return mk_val<value_int>(static_cast<int64_t>(obj.size()));
         }},
        {"tojson",
         [](const func_args& args) -> value {
             args.ensure_vals<value_object>();
             // use global to_json
             return global_builtins().at("tojson")(args);
         }},
        {"dictsort",
         [](const func_args& args) -> value {
             value val_input   = args.get_pos(0);
             value val_case    = args.get_kwarg_or_pos("case_sensitive", 1);
             value val_by      = args.get_kwarg_or_pos("by", 2);
             value val_reverse = args.get_kwarg_or_pos("reverse", 3);
             // FIXME: sorting is currently always case sensitive
             // const bool case_sensitive = val_case->as_bool(); // undefined == false
             const bool reverse = val_reverse->as_bool(); // undefined == false
             const bool by_value =
                 is_val<value_string>(val_by) && val_by->as_string().str() == "value" ? true
                                                                                      : false;
             auto result = mk_val<value_object>(val_input); // copy
             std::sort(result->val_obj.begin(), result->val_obj.end(),
                       [&](const auto& a, const auto& b) {
                           if (by_value) {
                               return value_compare(a.second, b.second,
                                                    reverse ? value_compare_op::gt
                                                            : value_compare_op::lt);
                           } else {
                               return value_compare(a.first, b.first,
                                                    reverse ? value_compare_op::gt
                                                            : value_compare_op::lt);
                           }
                       });
             return result;
         }},
        {"join", object_join_not_implemented},
    };
    return builtins;
}

const func_builtins& value_none_t::get_builtins() const {
    static const func_handler tostring = [](const func_args&) -> value {
        return mk_val<value_string>("None");
    };
    static const func_builtins builtins = {
        {"default", default_value},
        {"tojson", tojson},
        {"string", tostring},
        {"safe", tostring},
        {"items", empty_value_fn<value_array>},
        {"map", empty_value_fn<value_array>},
        {"reject", empty_value_fn<value_array>},
        {"rejectattr", empty_value_fn<value_array>},
        {"select", empty_value_fn<value_array>},
        {"selectattr", empty_value_fn<value_array>},
        {"unique", empty_value_fn<value_array>},
    };
    return builtins;
}

const func_builtins& value_undefined_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"capitalize", empty_value_fn<value_string>},
        {"first", empty_value_fn<value_undefined>},
        {"items", empty_value_fn<value_array>},
        {"join", empty_value_fn<value_string>},
        {"last", empty_value_fn<value_undefined>},
        {"length", empty_value_fn<value_int>},
        {"list", empty_value_fn<value_array>},
        {"lower", empty_value_fn<value_string>},
        {"map", empty_value_fn<value_array>},
        {"max", empty_value_fn<value_undefined>},
        {"min", empty_value_fn<value_undefined>},
        {"reject", empty_value_fn<value_array>},
        {"rejectattr", empty_value_fn<value_array>},
        {"replace", empty_value_fn<value_string>},
        {"reverse", empty_value_fn<value_array>},
        {"safe", empty_value_fn<value_string>},
        {"select", empty_value_fn<value_array>},
        {"selectattr", empty_value_fn<value_array>},
        {"sort", empty_value_fn<value_array>},
        {"string", empty_value_fn<value_string>},
        {"strip", empty_value_fn<value_string>},
        {"sum", empty_value_fn<value_int>},
        {"title", empty_value_fn<value_string>},
        {"truncate", empty_value_fn<value_string>},
        {"unique", empty_value_fn<value_array>},
        {"upper", empty_value_fn<value_string>},
        {"wordcount", empty_value_fn<value_int>},
    };
    return builtins;
}

//////////////////////////////////


bool value_compare(const value& a, const value& b, value_compare_op op) {
    if (op == eq) return *a == *b;
    if (op == ne) return !(*a == *b);
    auto compare = [op](const auto& lhs, const auto& rhs) {
        if (op == ge) return lhs >= rhs;
        if (op == gt) return lhs > rhs;
        return lhs < rhs;
    };
    if (a->is_numeric() && b->is_numeric()) return compare(a->val_flt, b->val_flt);
    if (is_val<value_string>(a) && is_val<value_string>(b))
        return compare(a->as_string().str(), b->as_string().str());
    throw raised_exception("Cannot order " + a->type() + " and " + b->type());
}

// TODO: avoid circular references
std::string value_to_string_repr(const value& val) {
    if (is_val<value_string>(val)) {
        const std::string val_str = val->as_string().str();

        if (val_str.find('\'') != std::string::npos) {
            return value_to_json(val);
        } else {
            return "'" + val_str + "'";
        }
    } else {
        return val->as_repr();
    }
}

} // namespace jinja
