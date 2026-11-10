#include "lexer.h"
#include "runtime.h"
#include "value.h"
#include "utils.h"
#include "unicode.h"

#include <string>
#include <vector>
#include <memory>
#include <cmath>

namespace jinja {


static value_string exec_statements(const statements& stmts, context& ctx) {
    auto result = mk_val<value_array>();
    for (const auto& stmt : stmts) { result->push_back(stmt->execute(ctx)); }
    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(result, str);
    return str;
}

static std::string get_line_col(const std::string& source, size_t pos) {
    size_t line = 1;
    size_t col  = 1;
    for (size_t i = 0; i < pos && i < source.size(); i++) {
        if (source[i] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
    }
    return "line " + std::to_string(line) + ", column " + std::to_string(col);
}

static void ensure_key_type_allowed(const value& val) {
    if (!val->is_hashable()) {
        throw std::runtime_error("Type: " + val->type() + " is not allowed as object key");
    }
}

// execute with error handling
value statement::execute(context& ctx) {
    if (ctx.checkpoint) ctx.checkpoint();
    try {
        return execute_impl(ctx);
    } catch (const continue_statement::signal& /* ex */) {
        throw;
    } catch (const break_statement::signal& /* ex */) {
        throw;
    } catch (const rethrown_exception& /* ex */) {
        throw;
    } catch (const not_implemented_exception& /* ex */) { throw; } catch (const std::exception& e) {
        const std::string& source = *ctx.src;
        if (source.empty()) {
            std::ostringstream oss;
            oss << "\nError executing " << type() << " at position " << pos << ": " << e.what();
            throw rethrown_exception(oss.str());
        } else {
            std::ostringstream oss;
            oss << "\n------------\n";
            oss << "While executing " << type() << " at " << get_line_col(source, pos)
                << " in source:\n";
            oss << peak_source(source, pos) << "\n";
            oss << "Error: " << e.what();
            // throw as another exception to avoid repeated formatting
            throw rethrown_exception(oss.str());
        }
    }
}

value identifier::execute_impl(context& ctx) {
    auto it              = ctx.get_val(val);
    const auto& builtins = global_builtins();
    if (!it->is_undefined()) {

        return it;
    } else if (builtins.find(val) != builtins.end()) {
        return mk_val<value_func>(val, builtins.at(val));
    } else {
        return mk_val<value_undefined>(val);
    }
}

value object_literal::execute_impl(context& ctx) {
    auto obj = mk_val<value_object>();
    for (const auto& pair : val) {
        value key = pair.first->execute(ctx);
        value val = pair.second->execute(ctx);
        obj->insert(key, val);
    }
    return obj;
}

value binary_expression::execute_impl(context& ctx) {
    value left_val = left->execute(ctx);

    // Logical operators
    if (op.value == "and") {
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (op.value == "or") {
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    // Equality operators
    value right_val = right->execute(ctx);
    if (op.value == "==") {
        return mk_val<value_bool>(*left_val == *right_val);
    } else if (op.value == "!=") {
        return mk_val<value_bool>(!(*left_val == *right_val));
    }

    if (op.value == "~") {
        return mk_val<value_string>(left_val->as_string().append(right_val->as_string()));
    }

    auto test_is_in = [&]() -> bool {
        func_args args(ctx);
        args.push_back(left_val);
        args.push_back(right_val);
        return global_builtins().at("test_is_in")(args)->as_bool();
    };

    // Handle undefined and null values
    if (is_val<value_undefined>(left_val) || is_val<value_undefined>(right_val)) {
        if (is_val<value_undefined>(right_val) && (op.value == "in" || op.value == "not in")) {
            // Special case: `anything in undefined` is `false` and `anything not in undefined` is
            // `true`
            return mk_val<value_bool>(op.value == "not in");
        }
        throw std::runtime_error("Cannot perform operation " + op.value + " on undefined values");
    } else if (is_val<value_none>(left_val) || is_val<value_none>(right_val)) {
        if (!is_val<value_none>(right_val) && (op.value == "in" || op.value == "not in")) {
            // case: none in {'low': 1}
            // A null left operand is looked up like any other value.
            bool member = test_is_in();
            return mk_val<value_bool>(op.value == "in" ? member : !member);
        }
        throw std::runtime_error("Cannot perform operation on null values");
    }

    // Float operations
    if ((is_val<value_int>(left_val) || is_val<value_float>(left_val)) &&
        (is_val<value_int>(right_val) || is_val<value_float>(right_val))) {
        double a = left_val->as_float();
        double b = right_val->as_float();
        if (op.value == "+" || op.value == "-" || op.value == "*") {
            double res    = (op.value == "+") ? a + b : (op.value == "-") ? a - b : a * b;
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(res);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(res));
            }
        } else if (op.value == "/") {
            return mk_val<value_float>(a / b);
        } else if (op.value == "%") {
            double rem    = std::fmod(a, b);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(rem);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(rem));
            }
        } else if (op.value == "<") {
            return mk_val<value_bool>(a < b);
        } else if (op.value == ">") {
            return mk_val<value_bool>(a > b);
        } else if (op.value == ">=") {
            return mk_val<value_bool>(a >= b);
        } else if (op.value == "<=") {
            return mk_val<value_bool>(a <= b);
        }
    }

    // Array operations
    if (is_val<value_array>(left_val) && is_val<value_array>(right_val)) {
        if (op.value == "+") {
            auto& left_arr  = left_val->as_array();
            auto& right_arr = right_val->as_array();
            auto result     = mk_val<value_array>();
            for (const auto& item : left_arr) { result->push_back(item); }
            for (const auto& item : right_arr) { result->push_back(item); }
            return result;
        }
    } else if (is_val<value_array>(right_val) || is_val<value_range>(right_val)) {
        // case: 1 in [0, 1, 2]
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    // String concatenation with ~ and +
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val) && op.value == "+") {
        auto output  = left_val->as_string().append(right_val->as_string());
        auto res     = mk_val<value_string>();
        res->val_str = std::move(output);
        return res;
    }

    // Python-style string repetition
    // TODO: support array/tuple repetition (e.g., [1, 2] * 3 → [1, 2, 1, 2, 1, 2])
    if (op.value == "*" && ((is_val<value_string>(left_val) && is_val<value_int>(right_val)) ||
                            (is_val<value_int>(left_val) && is_val<value_string>(right_val)))) {
        const auto& str =
            is_val<value_string>(left_val) ? left_val->as_string() : right_val->as_string();
        const int64_t repeat =
            is_val<value_int>(right_val) ? right_val->as_int() : left_val->as_int();
        auto res = mk_val<value_string>();
        if (repeat <= 0) { return res; }
        for (int64_t i = 0; i < repeat; ++i) {
            if (ctx.checkpoint) ctx.checkpoint();
            res->val_str.append(str);
        }
        return res;
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        // case: "a" in "abc"
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    // Value key in object
    if (is_val<value_object>(right_val)) {
        // case: key in {key: value}
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    throw std::runtime_error("Unknown operator \"" + op.value + "\" between " + left_val->type() +
                             " and " + right_val->type());
}

static value try_builtin_func(context& ctx, const std::string& name, value& input,
                              bool undef_on_missing = false) {

    const auto& builtins = input->get_builtins();
    auto it       = builtins.find(name);
    if (it != builtins.end()) { return mk_val<value_func>(name, it->second, input); }
    if (undef_on_missing) { return mk_val<value_undefined>(name); }
    throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + input->type());
}

static value apply_filter(value input, const statement_ptr& filter, context& ctx) {


    auto set_filter_alias = [](auto& filter_id) {
        if (filter_id == "count") {
            filter_id = "length";
        } else if (filter_id == "d") {
            filter_id = "default";
        } else if (filter_id == "e") {
            filter_id = "escape";
        } else if (filter_id == "trim") {
            filter_id = "strip";
        }
    };

    if (is_stmt<identifier>(filter)) {
        auto filter_id = cast_stmt<identifier>(filter)->val;

        set_filter_alias(filter_id);
        // TODO: Refactor filters so this coercion can be done automatically
        if (!input->is_undefined() && !is_val<value_string>(input) &&
            (filter_id == "capitalize" || filter_id == "lower" || filter_id == "replace" ||
             filter_id == "strip" || filter_id == "title" || filter_id == "upper" ||
             filter_id == "wordcount")) {
            input = mk_val<value_string>(input->as_string());
        }
        return try_builtin_func(ctx, filter_id, input)->invoke(func_args(ctx));

    } else if (is_stmt<call_expression>(filter)) {
        auto call = cast_stmt<call_expression>(filter);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Filter callee must be an identifier");
        }
        auto filter_id = cast_stmt<identifier>(call->callee)->val;

        set_filter_alias(filter_id);
        func_args args(ctx);
        for (const auto& arg_expr : call->args) { args.push_back(arg_expr->execute(ctx)); }

        return try_builtin_func(ctx, filter_id, input)->invoke(args);

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value filter_expression::execute_impl(context& ctx) {
    return apply_filter(operand ? operand->execute(ctx) : val, filter, ctx);
}

value filter_statement::execute_impl(context& ctx) {
    return apply_filter(exec_statements(body, ctx), filter, ctx);
}

value test_expression::execute_impl(context& ctx) {
    // NOTE: "value is something" translates to function call "test_is_something(value)"
    const auto& builtins = global_builtins();

    std::string test_id;
    value input = operand->execute(ctx);

    func_args args(ctx);
    args.push_back(input);

    if (is_stmt<identifier>(test)) {
        test_id = cast_stmt<identifier>(test)->val;
    } else if (is_stmt<call_expression>(test)) {
        auto call = cast_stmt<call_expression>(test);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Test callee must be an identifier");
        }
        test_id = cast_stmt<identifier>(call->callee)->val;

        for (const auto& arg_expr : call->args) { args.push_back(arg_expr->execute(ctx)); }

    } else {
        throw std::runtime_error("Invalid test expression");
    }

    const std::string test_name = "test_is_" + test_id;
    auto it                     = builtins.find(test_name);
    if (it == builtins.end()) { throw std::runtime_error("Unknown test '" + test_id + "'"); }



    auto res = it->second(args);

    if (negate) {
        return mk_val<value_bool>(!res->as_bool());
    } else {
        return res;
    }
}

value unary_expression::execute_impl(context& ctx) {
    value operand_val = argument->execute(ctx);

    if (op.value == "not") {
        return mk_val<value_bool>(!operand_val->as_bool());
    } else if (op.value == "-") {
        if (is_val<value_int>(operand_val)) {
            return mk_val<value_int>(-operand_val->as_int());
        } else if (is_val<value_float>(operand_val)) {
            return mk_val<value_float>(-operand_val->as_float());
        } else {
            throw std::runtime_error("Unary - operator requires numeric operand");
        }
    }

    throw std::runtime_error("Unknown unary operator '" + op.value + "'");
}

value if_statement::execute_impl(context& ctx) {
    value test_val = test->execute(ctx);

    auto out = mk_val<value_array>();
    if (test_val->as_bool()) {
        for (auto& stmt : body) { out->push_back(stmt->execute(ctx)); }
    } else {
        for (auto& stmt : alternate) { out->push_back(stmt->execute(ctx)); }
    }
    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(out, str);
    return str;
}

namespace {
class IterableCursor {
public:
    explicit IterableCursor(value input) : input_(std::move(input)) {
        if (is_val<value_string>(input_)) {
            string_ = input_->as_string();
            bytes_  = string_.str();
        } else if (!is_val<value_array>(input_) && !is_val<value_object>(input_) &&
                   !is_val<value_range>(input_) && !input_->is_undefined()) {
            throw std::runtime_error("Expected iterable, got " + input_->type());
        }
    }

    uint64_t size() const {
        if (is_val<value_array>(input_)) return input_->as_array().size();
        if (is_val<value_object>(input_)) return input_->as_ordered_object().size();
        if (is_val<value_range>(input_)) return cast_val<value_range>(input_)->count;
        if (is_val<value_string>(input_)) return string_.length();
        return 0;
    }

    value next() {
        if (is_val<value_string>(input_)) {
            if (offset_ == bytes_.size()) return {};
            const auto begin = offset_++;
            while (offset_ < bytes_.size() &&
                   (static_cast<unsigned char>(bytes_[offset_]) & 0xc0) == 0x80)
                ++offset_;
            return mk_val<value_string>(string_.cut_bytes(begin, offset_));
        }
        if (offset_ >= size()) return {};
        if (is_val<value_array>(input_)) return input_->as_array()[offset_++];
        if (is_val<value_object>(input_)) return input_->as_ordered_object()[offset_++].first;
        return cast_val<value_range>(input_)->item(offset_++);
    }

private:
    value input_;
    string string_;
    std::string bytes_;
    uint64_t offset_ = 0;
};

class VariableBinding {
public:
    explicit VariableBinding(const statement_ptr& target) {
        if (const auto* id = cast_stmt<identifier>(target)) {
            names_.push_back(id->val);
        } else if (const auto* tuple = cast_stmt<tuple_literal>(target)) {
            unpack_ = true;
            for (const auto& item : tuple->val) {
                const auto* id = cast_stmt<identifier>(item);
                if (!id) throw std::runtime_error("Cannot unpack non-identifier type");
                names_.push_back(id->val);
            }
        } else {
            throw std::runtime_error("Invalid loop variable: " + target->type());
        }
    }

    void bind(context& scope, const value& item) const {
        if (!unpack_) {
            scope.set_val(names_.front(), item);
            return;
        }
        IterableCursor values(item);
        for (const auto& name : names_) {
            auto v = values.next();
            if (!v) throw std::runtime_error("Too few items to unpack");
            scope.set_val(name, v);
        }
        if (values.next()) throw std::runtime_error("Too many items to unpack");
    }

private:
    std::vector<std::string> names_;
    bool unpack_ = false;
};

class LoopValue final : public value_object_t {
public:
    LoopValue(value input, context& parent, VariableBinding binding, statement* filter)
        : source_(std::move(input)), filter_scope_(parent), binding_(std::move(binding)),
          filter_(filter) {
        has_builtins = false;
    }

    bool advance(context& body) {
        auto item = take();
        if (!item) return false;
        previous_ = std::move(current_);
        current_  = std::move(item);
        ++index_;
        binding_.bind(body, current_);
        return true;
    }

    bool as_bool() const override { return true; }

    value& at(const value& key, value& fallback) override {
        const auto name = key->as_string().str();
        if (name == "index")
            cached_ = number(index_);
        else if (name == "index0")
            cached_ = number(index_ - 1);
        else if (name == "first")
            cached_ = mk_val<value_bool>(index_ == 1);
        else if (name == "last")
            cached_ = mk_val<value_bool>(!peek());
        else if (name == "length")
            cached_ = number(length());
        else if (name == "revindex")
            cached_ = number(length() - index_ + 1);
        else if (name == "revindex0")
            cached_ = number(length() - index_);
        else if (name == "previtem")
            cached_ = previous_ ? previous_ : mk_val<value_undefined>("previtem");
        else if (name == "nextitem") {
            const auto item = peek();
            cached_         = item ? item : mk_val<value_undefined>("nextitem");
        } else
            return fallback;
        return cached_;
    }

private:
    static value number(uint64_t n) {
        if (n > INT64_MAX) throw std::runtime_error("loop length exceeds int64");
        return mk_val<value_int>(static_cast<int64_t>(n));
    }

    value read() {
        while (auto item = source_.next()) {
            if (filter_scope_.checkpoint) filter_scope_.checkpoint();
            if (filter_) {
                filter_scope_.clear_locals();
                binding_.bind(filter_scope_, item);
                if (!filter_->execute(filter_scope_)->as_bool()) continue;
            }
            return item;
        }
        return {};
    }

    value take() {
        if (pending_.empty()) return read();
        auto item = std::move(pending_.front());
        pending_.pop_front();
        return item;
    }

    value peek() {
        if (pending_.empty()) {
            auto item = read();
            if (item) pending_.push_back(std::move(item));
        }
        return pending_.empty() ? value{} : pending_.front();
    }

    uint64_t length() {
        if (!length_) {
            if (filter_) {
                while (auto item = read()) pending_.push_back(std::move(item));
                length_ = index_ + pending_.size();
            } else
                length_ = source_.size();
        }
        return *length_;
    }

    IterableCursor source_;
    context filter_scope_;
    VariableBinding binding_;
    statement* filter_;
    std::deque<value> pending_;
    std::optional<uint64_t> length_;
    uint64_t index_ = 0;
    value current_, previous_, cached_;
};
} // namespace

value for_statement::execute_impl(context& ctx) {
    const auto* selected = cast_stmt<select_expression>(iterable);
    auto input           = (selected ? selected->lhs : iterable)->execute(ctx);
    auto loop         = std::make_shared<LoopValue>(std::move(input), ctx, VariableBinding(loopvar),
                                            selected ? selected->test.get() : nullptr);
    auto result       = mk_val<value_array>();
    bool no_iteration = true;
    for (;;) {
        context body_scope(ctx);
        if (!loop->advance(body_scope)) break;
        body_scope.set_val("loop", loop);
        try {
            for (const auto& stmt : body) result->push_back(stmt->execute(body_scope));
        } catch (const continue_statement::signal&) {
            continue;
        } catch (const break_statement::signal&) { break; }
        no_iteration = false;
    }
    if (no_iteration)
        for (const auto& stmt : default_block) result->push_back(stmt->execute(ctx));
    auto output = mk_val<value_string>();
    gather_string_parts_recursive(result, output);
    return output;
}

value set_statement::execute_impl(context& ctx) {
    auto rhs = val ? val->execute(ctx) : exec_statements(body, ctx);

    if (is_stmt<identifier>(assignee)) {
        // case: {% set my_var = value %}
        auto var_name = cast_stmt<identifier>(assignee)->val;
        ctx.set_val(var_name, rhs);

    } else if (is_stmt<tuple_literal>(assignee)) {
        VariableBinding(assignee).bind(ctx, rhs);

    } else if (is_stmt<member_expression>(assignee)) {
        // case: {% set ns.my_var = value %}
        auto member = cast_stmt<member_expression>(assignee);
        if (member->computed) { throw std::runtime_error("Cannot assign to computed member"); }
        if (!is_stmt<identifier>(member->property)) {
            throw std::runtime_error("Cannot assign to member with non-identifier property");
        }
        auto prop_name = cast_stmt<identifier>(member->property)->val;

        value object = member->object->execute(ctx);
        if (!is_val<value_object>(object)) {
            throw std::runtime_error("Cannot assign to member of non-object");
        }
        auto obj_ptr = cast_val<value_object>(object);
        obj_ptr->insert(prop_name, rhs);

    } else {
        throw std::runtime_error("Invalid LHS inside assignment expression: " + assignee->type());
    }
    return mk_val<value_undefined>();
}

static void bind_parameters(const std::string& name, const statements& parameters,
                            const func_args& args, context& ctx) {
    std::vector<value> positional;
    std::map<std::string, value> named;
    for (const auto& input : args.get_args()) {
        if (is_val<value_kwarg>(input)) {
            const auto* keyword = cast_val<value_kwarg>(input);
            if (!named.emplace(keyword->key, keyword->val).second)
                throw std::runtime_error("duplicate macro argument: " + keyword->key);
        } else
            positional.push_back(input);
    }
    if (positional.size() > parameters.size())
        throw std::runtime_error("too many arguments to macro " + name);
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto* plain        = cast_stmt<identifier>(parameters[i]);
        const auto* with_default = cast_stmt<keyword_argument_expression>(parameters[i]);
        if (!plain && (!with_default || !is_stmt<identifier>(with_default->key))) {
            throw std::runtime_error("invalid parameter in macro " + name);
        }
        const auto key = plain ? plain->val : cast_stmt<identifier>(with_default->key)->val;
        value input;
        if (const auto it = named.find(key); it != named.end()) {
            if (i < positional.size()) throw std::runtime_error("duplicate macro argument: " + key);
            input = it->second;
            named.erase(it);
        } else if (i < positional.size())
            input = positional[i];
        else if (with_default)
            input = with_default->val->execute(ctx);
        else
            input = mk_val<value_undefined>(key);
        ctx.set_val(key, input);
    }
    if (!named.empty()) throw std::runtime_error("unknown macro argument: " + named.begin()->first);
}

value macro_statement::execute_impl(context& ctx) {
    if (!is_stmt<identifier>(this->name)) {
        throw std::runtime_error("Macro name must be an identifier");
    }
    std::string name = cast_stmt<identifier>(this->name)->val;

    const func_handler func = [this, name,
                               scope = ctx.capture_scope()](const func_args& args) -> value {
        context macro_ctx(args.ctx, scope);
        const auto caller = args.ctx.get_val("caller");
        if (!caller->is_undefined()) macro_ctx.set_val("caller", caller);

        bind_parameters(name, this->args, args, macro_ctx);

        // execute macro body
        auto res = exec_statements(this->body, macro_ctx);
        return res;
    };

    ctx.set_val(name, mk_val<value_func>(name, func));
    return mk_val<value_undefined>();
}

value call_statement::execute_impl(context& ctx) {
    auto call_expr = cast_stmt<call_expression>(this->call);
    if (!call_expr) { throw std::runtime_error("Call statement requires a valid call expression"); }

    value callee_val = call_expr->callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto* callee_func = cast_val<value_func>(callee_val);

    const func_handler func = [this, scope = ctx.current_scope()](const func_args& args) -> value {
        context block_ctx(args.ctx, scope);

        bind_parameters("caller", this->caller_args, args, block_ctx);

        auto res = exec_statements(this->body, block_ctx);
        return res;
    };

    context call_ctx(ctx);
    call_ctx.set_val("caller", mk_val<value_func>("caller", func));

    func_args args(call_ctx);

    for (const auto& arg_expr : call_expr->args) {
        auto arg_val = arg_expr->execute(ctx);
        args.push_back(arg_val);
    }

    return callee_func->invoke(args);
}

value member_expression::execute_impl(context& ctx) {
    value object = this->object->execute(ctx);

    value property;
    if (this->computed) {
        // syntax: obj[expr]

        if (is_stmt<slice_expression>(this->property)) {
            auto s         = cast_stmt<slice_expression>(this->property);
            value step_val = s->step_expr ? s->step_expr->execute(ctx) : mk_val<value_int>(1);
            value start_val =
                s->start_expr ? s->start_expr->execute(ctx) : mk_val<value_undefined>();
            value stop_val = s->stop_expr ? s->stop_expr->execute(ctx) : mk_val<value_undefined>();

            // translate to function call: obj.slice(start, stop, step)
            auto slice_func = try_builtin_func(ctx, "slice", object);
            func_args args(ctx);
            args.push_back(start_val);
            args.push_back(stop_val);
            args.push_back(step_val);
            return slice_func->invoke(args);
        } else {
            property = this->property->execute(ctx);
        }
    } else if (is_stmt<integer_literal>(this->property)) {
        // syntax: obj.index
        property = mk_val<value_int>(cast_stmt<integer_literal>(this->property)->val);
        if (property->as_int() < 0) {
            throw std::runtime_error("Static member property cannot be negative");
        }
    } else {
        // syntax: obj.prop
        if (!is_stmt<identifier>(this->property)) {
            throw std::runtime_error("Static member property must be an identifier");
        }
        property         = mk_val<value_string>(cast_stmt<identifier>(this->property)->val);
        std::string prop = property->as_string().str();

        // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
        // then obj.prop returns the built-in function, not the property value.
        // while obj['prop'] returns the property value.
        // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is
        // 123

        value val = try_builtin_func(ctx, prop, object, true);
        if (!is_val<value_undefined>(val)) { return val; }
        // else, fallthrough to normal property access below
    }

    value val = mk_val<value_undefined>("object_property");

    if (property->is_undefined()) { return val; }

    ensure_key_type_allowed(property);

    if (is_val<value_undefined>(object)) {
        return val;

    } else if (is_val<value_object>(object)) {
        auto key = property->as_string().str();
        val      = object->at(property, val);
        if (is_val<value_undefined>(val)) { val = try_builtin_func(ctx, key, object, true); }

    } else if (is_val<value_array>(object) || is_val<value_string>(object) ||
               is_val<value_range>(object)) {
        if (is_val<value_int>(property)) {
            int64_t index = property->as_int();
            if (is_val<value_range>(object)) {
                const auto* range       = cast_val<value_range>(object);
                const __int128 position = index < 0 ? __int128(range->count) + index : index;
                if (position >= 0 && position < range->count)
                    val = range->item(static_cast<uint64_t>(position));
            } else if (is_val<value_array>(object)) {
                auto& arr = object->as_array();
                if (index < 0) { index += static_cast<int64_t>(arr.size()); }
                if (index >= 0 && index < static_cast<int64_t>(arr.size())) { val = arr[index]; }
            } else { // value_string
                const auto str  = object->as_string();
                const auto size = static_cast<int64_t>(str.length());
                if (index < 0) index += size;
                if (index >= 0 && index < size) {
                    val = mk_val<value_string>(str.slice(index, index + 1));
                }
            }

        } else if (is_val<value_string>(property)) {
            auto key = property->as_string().str();
            val      = try_builtin_func(ctx, key, object, true);

        } else {
            throw std::runtime_error("Cannot access property with non-string/non-number: got " +
                                     property->type());
        }
    } else {
        if (!is_val<value_string>(property)) {
            throw std::runtime_error("Cannot access property with non-string: got " +
                                     property->type());
        }
        auto key = property->as_string().str();
        val      = try_builtin_func(ctx, key, object, true);
    }



    return val;
}

value call_expression::execute_impl(context& ctx) {
    // gather arguments
    func_args args(ctx);
    for (auto& arg_stmt : this->args) {
        auto arg_val = arg_stmt->execute(ctx);
        args.push_back(arg_val);
    }
    // execute callee
    value callee_val = callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto* callee_func = cast_val<value_func>(callee_val);
    return callee_func->invoke(args);
}

value keyword_argument_expression::execute_impl(context& ctx) {
    if (!is_stmt<identifier>(key)) {
        throw std::runtime_error("Keyword argument key must be identifiers");
    }

    std::string k = cast_stmt<identifier>(key)->val;

    value v = val->execute(ctx);

    return mk_val<value_kwarg>(k, v);
}

std::string runtime::debug_dump_program(const program& prog, const std::string& src) {
    std::ostringstream oss;
    size_t lvl = 0;
    context ctx;
    ctx.src.reset(new std::string(src));

    auto indent = [](size_t lvl) -> std::string { return std::string(lvl * 2, ' '); };

    ctx.visitor = [&](bool is_leaf, statement* node, std::vector<visitor_pair> children) {
        oss << indent(lvl) << node->type() << ":\n";
        lvl++;
        if (is_leaf) {
            const auto& pos = node->pos;
            oss << indent(lvl) << "(leaf) at " << get_line_col(src, pos) << " in source:\n";
            std::string snippet = peak_source(src, pos);
            string_replace_all(snippet, "\n", "\n" + indent(lvl));
            oss << indent(lvl) << snippet << "\n";
        } else {
            for (auto& [label, children_vec] : children) {
                oss << indent(lvl) << label << ":\n";
                lvl++;
                if (children_vec.empty()) {
                    oss << indent(lvl) << "<empty>\n\n";
                } else {
                    for (auto* child : children_vec) {
                        if (!child) { continue; }
                        child->visit(ctx);
                    }
                }
                lvl--;
            }
        }
        lvl--;
    };

    for (const auto& stmt : prog.body) { stmt->visit(ctx); }

    return oss.str();
}

} // namespace jinja
