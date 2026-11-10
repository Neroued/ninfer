#pragma once

#include "lexer.h"
#include "value.h"

#include <cassert>
#include <ctime>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jinja {

struct statement;
using statement_ptr = std::unique_ptr<statement>;
using statements    = std::vector<statement_ptr>;

// Helpers for dynamic casting and type checking
template <typename T>
struct extract_pointee_unique {
    using type = T;
};

template <typename U>
struct extract_pointee_unique<std::unique_ptr<U>> {
    using type = U;
};

template <typename T>
bool is_stmt(const statement_ptr& ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}

template <typename T>
T* cast_stmt(statement_ptr& ptr) {
    return dynamic_cast<T*>(ptr.get());
}

template <typename T>
const T* cast_stmt(const statement_ptr& ptr) {
    return dynamic_cast<const T*>(ptr.get());
}

// End Helpers



// for visiting AST nodes
// function signature: void(bool is_leaf, statement * node, pair of <label, children>)
using visitor_pair = std::pair<std::string, std::vector<statement*>>;
using visitor_fn   = std::function<void(bool, statement*, std::vector<visitor_pair>)>;

struct macro_statement;
struct closure_environment;
class render_scopes;
using scope_names = std::vector<std::string>;

struct binding_cell {
    value data;
    macro_statement* macro = nullptr;
    std::weak_ptr<closure_environment> owner;
    std::shared_ptr<binding_cell> inherited;

    value read() const;
    void assign(value input);
};

struct scope {
    scope* parent = nullptr;
    std::unordered_map<std::string, value> bindings;
    const scope_names* locals = nullptr;
    std::shared_ptr<closure_environment> imports;
    std::shared_ptr<closure_environment> environment;
    std::size_t references = 0;
};

enum class loop_flow { none, stop, next };

// Definitions use weak environment links; materialized callable values own their captures.
struct closure_environment : std::enable_shared_from_this<closure_environment> {
    explicit closure_environment(std::shared_ptr<render_scopes> owner) : owner(std::move(owner)) {}

    ~closure_environment();

    std::shared_ptr<render_scopes> owner;
    std::unordered_map<std::string, std::shared_ptr<binding_cell>> locals;
    std::unordered_map<std::string, std::shared_ptr<binding_cell>> bindings;
    std::vector<std::shared_ptr<closure_environment>> dependencies;
    closure_environment* previous = nullptr;
    closure_environment* next     = nullptr;
    bool registered               = false;
};

class render_scopes : public std::enable_shared_from_this<render_scopes> {
public:
    scope* acquire(scope* parent, const scope_names* locals,
                   std::shared_ptr<closure_environment> imports = {});
    void release(scope* frame);
    std::shared_ptr<closure_environment> environment(scope* frame);
    std::shared_ptr<closure_environment> capture(scope* frame, const scope_names& names);
    std::shared_ptr<binding_cell> local_cell(scope* frame, const std::string& name);
    void detach(closure_environment* environment);
    void close();
    void ensure_active() const;

    loop_flow flow = loop_flow::none;

private:
    std::pair<std::shared_ptr<binding_cell>, std::shared_ptr<closure_environment>>
    resolve_cell(scope* frame, const std::string& name);
    std::deque<scope> frames_;
    scope* free_                       = nullptr;
    closure_environment* environments_ = nullptr;
    bool closed_                       = false;
};

struct context {
    std::shared_ptr<std::string> src;
    std::time_t current_time;
    std::function<void()> checkpoint;
    visitor_fn visitor;

    explicit context(std::string source = "")
        : src(std::make_shared<std::string>(std::move(source))), current_time(std::time(nullptr)),
          scopes(std::make_shared<render_scopes>()), env(scopes->acquire(nullptr, nullptr)),
          root(true) {
        set_val("true", mk_val<value_bool>(true));
        set_val("True", mk_val<value_bool>(true));
        set_val("false", mk_val<value_bool>(false));
        set_val("False", mk_val<value_bool>(false));
        set_val("none", mk_val<value_none>());
        set_val("None", mk_val<value_none>());
    }

    ~context() {
        if (root)
            scopes->close();
        else
            scopes->release(env);
    }

    context(const context& parent, const scope_names* locals = nullptr)
        : src(parent.src), current_time(parent.current_time), checkpoint(parent.checkpoint),
          scopes(parent.scopes), env(scopes->acquire(parent.env, locals)) {}

    context(const context& caller, std::shared_ptr<closure_environment> captured,
            const scope_names* locals)
        : src(caller.src), current_time(caller.current_time), checkpoint(caller.checkpoint),
          scopes(caller.scopes), env(nullptr) {
        if (captured && captured->owner != scopes)
            throw std::runtime_error("Closure belongs to a different render");
        env = scopes->acquire(nullptr, locals, std::move(captured));
    }

    context& operator=(const context&) = delete;

    value find_val(const std::string& name) {
        for (auto* frame = env; frame; frame = frame->parent) {
            if (frame->environment) {
                const auto found = frame->environment->locals.find(name);
                if (found != frame->environment->locals.end()) return found->second->read();
            }
            if (const auto found = frame->bindings.find(name); found != frame->bindings.end())
                return found->second;
            if (frame->imports) {
                const auto found = frame->imports->bindings.find(name);
                if (found != frame->imports->bindings.end()) return found->second->read();
            }
        }
        return {};
    }

    value get_val(const std::string& name) {
        auto result = find_val(name);
        return result ? result : mk_val<value_undefined>(name);
    }

    void set_val(const std::string& name, const value& input) {
        if (env->environment) {
            const auto found = env->environment->locals.find(name);
            if (found != env->environment->locals.end()) {
                found->second->assign(input);
                return;
            }
        }
        env->bindings[name] = input;
    }

    void set_val(const value& name, const value& input) { set_val(name->as_string().str(), input); }

    void define_macro(const std::string& name, macro_statement* definition,
                      const scope_names& captures);

    std::shared_ptr<closure_environment> capture(const scope_names& names) const {
        return scopes->capture(env, names);
    }

    void set_locals(const scope_names& names) { env->locals = &names; }
    void clear_locals() { env->bindings.clear(); }

    std::shared_ptr<render_scopes> storage() const { return scopes; }
    loop_flow& flow() { return scopes->flow; }

private:
    std::shared_ptr<render_scopes> scopes;
    scope* env;
    bool root = false;
};

// utils for visiting AST nodes
static std::vector<statement*> stmts_to_ptr(const statements& stmts) {
    std::vector<statement*> children;
    for (const auto& stmt : stmts) { children.push_back(stmt.get()); }
    return children;
}

/**
 * Base class for all nodes in the AST.
 */
struct statement {
    size_t pos; // position in source, for debugging
    virtual ~statement() = default;

    virtual std::string type() const { return "Statement"; }

    virtual void visit(context& ctx) { ctx.visitor(true, this, {}); }

    // execute_impl must be overridden by derived classes
    virtual value execute_impl(context&) { throw_exec_error(); }

    virtual void render_impl(context&, string&);

    // execute is the public method to execute a statement with error handling
    value execute(context&);
    void render(context&, string&);

private:
    [[noreturn]] void throw_exec_error() const {
        throw std::runtime_error("cannot exec " + type());
    }
};

// Type Checking Utilities

template <typename T>
static void chk_type(const statement_ptr& ptr) {
    if (!ptr) return; // Allow null for optional fields
    assert(dynamic_cast<T*>(ptr.get()) != nullptr);
}

template <typename T, typename U>
static void chk_type(const statement_ptr& ptr) {
    if (!ptr) return;
    assert(dynamic_cast<T*>(ptr.get()) != nullptr || dynamic_cast<U*>(ptr.get()) != nullptr);
}

// Base Types

/**
 * Expressions will result in a value at runtime (unlike statements).
 */
struct expression : public statement {
    std::string type() const override { return "Expression"; }
};

struct output_statement : public statement {
    statement_ptr operand;

    explicit output_statement(statement_ptr&& operand) : operand(std::move(operand)) {}

    std::string type() const override { return "Output"; }

    value execute_impl(context& ctx) override {
        const auto result = operand->execute(ctx);
        if (is_val<value_string>(result)) return result;
        return mk_val<value_string>(result->as_string());
    }
};

// Statements

struct program : public statement {
    scope_names locals;
    statements body;

    program() = default;

    explicit program(statements&& body) : body(std::move(body)) {}

    std::string type() const override { return "Program"; }

    [[noreturn]] value execute_impl(context&) override {
        throw std::runtime_error("Cannot execute program directly, use jinja::runtime instead");
    }
};

struct if_statement : public statement {
    statement_ptr test;
    statements body;
    statements alternate;

    if_statement(statement_ptr&& test, statements&& body, statements&& alternate)
        : test(std::move(test)), body(std::move(body)), alternate(std::move(alternate)) {
        chk_type<expression>(this->test);
    }

    std::string type() const override { return "If"; }

    value execute_impl(context& ctx) override;
    void render_impl(context&, string&) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"test", {test.get()}},
                     {"body", stmts_to_ptr(body)},
                     {"alternate", stmts_to_ptr(alternate)}});
    }
};

struct identifier;
struct tuple_literal;

/**
 * Loop over each item in a sequence
 * https://jinja.palletsprojects.com/en/3.0.x/templates/#for
 */
struct for_statement : public statement {
    scope_names locals;
    scope_names filter_captures;
    statement_ptr loopvar; // Identifier | TupleLiteral
    statement_ptr iterable;
    statements body;
    statements default_block; // if no iteration took place

    for_statement(statement_ptr&& loopvar, statement_ptr&& iterable, statements&& body,
                  statements&& default_block)
        : loopvar(std::move(loopvar)), iterable(std::move(iterable)), body(std::move(body)),
          default_block(std::move(default_block)) {
        chk_type<identifier, tuple_literal>(this->loopvar);
        chk_type<expression>(this->iterable);
    }

    std::string type() const override { return "For"; }

    value execute_impl(context& ctx) override;
    void render_impl(context&, string&) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"loopvar", {loopvar.get()}},
                     {"iterable", {iterable.get()}},
                     {"body", stmts_to_ptr(body)},
                     {"default_block", stmts_to_ptr(default_block)}});
    }
};

struct break_statement : public statement {
    std::string type() const override { return "Break"; }

    value execute_impl(context& ctx) override {
        ctx.flow() = loop_flow::stop;
        return {};
    }
};

struct continue_statement : public statement {
    std::string type() const override { return "Continue"; }

    value execute_impl(context& ctx) override {
        ctx.flow() = loop_flow::next;
        return {};
    }
};

// do nothing
struct noop_statement : public statement {
    std::string type() const override { return "Noop"; }

    value execute_impl(context&) override { return {}; }
};

struct set_statement : public statement {
    statement_ptr assignee;
    statement_ptr val;
    statements body;

    set_statement(statement_ptr&& assignee, statement_ptr&& value, statements&& body)
        : assignee(std::move(assignee)), val(std::move(value)), body(std::move(body)) {
        chk_type<expression>(this->assignee);
        chk_type<expression>(this->val);
    }

    std::string type() const override { return "Set"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(
            false, this,
            {{"assignee", {assignee.get()}}, {"value", {val.get()}}, {"body", stmts_to_ptr(body)}});
    }
};

struct macro_statement : public statement {
    scope_names locals;
    scope_names captures;
    statement_ptr name;
    statements args;
    statements body;

    macro_statement(statement_ptr&& name, statements&& args, statements&& body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {
        chk_type<identifier>(this->name);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }

    std::string type() const override { return "Macro"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(
            false, this,
            {{"name", {name.get()}}, {"args", stmts_to_ptr(args)}, {"body", stmts_to_ptr(body)}});
    }
};

struct comment_statement : public statement {
    std::string val;

    explicit comment_statement(const std::string& v) : val(v) {}

    std::string type() const override { return "Comment"; }

    value execute_impl(context&) override { return {}; }
};

// Expressions

// Represents an omitted expression in a computed member, e.g. `a[]`.
struct blank_expression : public expression {
    std::string type() const override { return "BlankExpression"; }

    value execute_impl(context&) override { return mk_val<value_undefined>(); }
};

struct member_expression : public expression {
    statement_ptr object;
    statement_ptr property;
    bool computed; // true if obj[expr] and false if obj.prop
    value static_property;

    member_expression(statement_ptr&& object, statement_ptr&& property, bool computed);

    std::string type() const override { return "MemberExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"object", {object.get()}}, {"property", {property.get()}}});
    }
};

struct call_expression : public expression {
    statement_ptr callee;
    statements args;

    call_expression(statement_ptr&& callee, statements&& args)
        : callee(std::move(callee)), args(std::move(args)) {
        chk_type<expression>(this->callee);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }

    std::string type() const override { return "CallExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"callee", {callee.get()}}, {"args", stmts_to_ptr(args)}});
    }
};

/**
 * Represents a user-defined variable or symbol in the template.
 */
struct identifier : public expression {
    std::string val;

    explicit identifier(const std::string& val) : val(val) {}

    std::string type() const override { return "Identifier"; }

    value execute_impl(context& ctx) override;
};

// Literals

struct integer_literal : public expression {
    int64_t val;
    value constant;

    explicit integer_literal(int64_t val) : val(val), constant(mk_val<value_int>(val)) {}

    std::string type() const override { return "IntegerLiteral"; }

    value execute_impl(context&) override { return constant; }
};

struct float_literal : public expression {
    double val;
    value constant;

    explicit float_literal(double val) : val(val), constant(mk_val<value_float>(val)) {}

    std::string type() const override { return "FloatLiteral"; }

    value execute_impl(context&) override { return constant; }
};

struct string_literal : public expression {
    std::string val;
    value constant;

    explicit string_literal(const std::string& val)
        : val(val), constant(mk_val<value_string>(val)) {}

    std::string type() const override { return "StringLiteral"; }

    value execute_impl(context&) override { return constant; }
};

struct array_literal : public expression {
    statements val;

    explicit array_literal(statements&& val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }

    std::string type() const override { return "ArrayLiteral"; }

    value execute_impl(context& ctx) override {
        auto arr = mk_val<value_array>();
        for (const auto& item_stmt : val) { arr->push_back(item_stmt->execute(ctx)); }
        return arr;
    }
};

struct tuple_literal : public expression {
    statements val;

    explicit tuple_literal(statements&& val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }

    std::string type() const override { return "TupleLiteral"; }

    value execute_impl(context& ctx) override {
        auto arr = mk_val<value_array>();
        for (const auto& item_stmt : val) { arr->push_back(item_stmt->execute(ctx)); }
        return mk_val<value_tuple>(std::move(arr->as_array()));
    }
};

struct object_literal : public expression {
    std::vector<std::pair<statement_ptr, statement_ptr>> val;

    explicit object_literal(std::vector<std::pair<statement_ptr, statement_ptr>>&& val)
        : val(std::move(val)) {
        for (const auto& pair : this->val) {
            chk_type<expression>(pair.first);
            chk_type<expression>(pair.second);
        }
    }

    std::string type() const override { return "ObjectLiteral"; }

    value execute_impl(context& ctx) override;
};

// Complex Expressions

/**
 * An operation with two sides, separated by an operator.
 * Note: Either side can be a Complex Expression, with order
 * of operations being determined by the operator.
 */
struct binary_expression : public expression {
    token op;
    statement_ptr left;
    statement_ptr right;

    binary_expression(token op, statement_ptr&& left, statement_ptr&& right)
        : op(std::move(op)), left(std::move(left)), right(std::move(right)) {
        chk_type<expression>(this->left);
        chk_type<expression>(this->right);
    }

    std::string type() const override { return "BinaryExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"left", {left.get()}}, {"right", {right.get()}}});
    }
};

/**
 * An operation with two sides, separated by the | operator.
 * Operator precedence: https://github.com/pallets/jinja/issues/379#issuecomment-168076202
 */
struct filter_expression : public expression {
    // either an expression or a value is allowed
    statement_ptr operand;
    value_string val; // will be set by filter_statement

    statement_ptr filter;

    filter_expression(statement_ptr&& operand, statement_ptr&& filter)
        : operand(std::move(operand)), filter(std::move(filter)) {
        chk_type<expression>(this->operand);
        chk_type<identifier, call_expression>(this->filter);
    }

    filter_expression(value_string&& val, statement_ptr&& filter)
        : val(std::move(val)), filter(std::move(filter)) {
        chk_type<identifier, call_expression>(this->filter);
    }

    std::string type() const override { return "FilterExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"operand", {operand.get()}}, {"filter", {filter.get()}}});
    }
};

struct filter_statement : public statement {
    statement_ptr filter;
    statements body;

    filter_statement(statement_ptr&& filter, statements&& body)
        : filter(std::move(filter)), body(std::move(body)) {
        chk_type<identifier, call_expression>(this->filter);
    }

    std::string type() const override { return "FilterStatement"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"filter", {filter.get()}}, {"body", stmts_to_ptr(body)}});
    }
};

/**
 * An operation which filters a sequence of objects by applying a test to each object,
 * and only selecting the objects with the test succeeding.
 *
 * It may also be used as a shortcut for a ternary operator.
 */
struct select_expression : public expression {
    statement_ptr lhs;
    statement_ptr test;

    select_expression(statement_ptr&& lhs, statement_ptr&& test)
        : lhs(std::move(lhs)), test(std::move(test)) {
        chk_type<expression>(this->lhs);
        chk_type<expression>(this->test);
    }

    std::string type() const override { return "SelectExpression"; }

    value execute_impl(context& ctx) override {
        auto predicate = test->execute_impl(ctx);
        if (!predicate->as_bool()) { return mk_val<value_undefined>(); }
        return lhs->execute_impl(ctx);
    }

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"lhs", {lhs.get()}}, {"test", {test.get()}}});
    }
};

/**
 * An operation with two sides, separated by the "is" operator.
 * NOTE: "value is something" translates to function call "test_is_something(value)"
 */
struct test_expression : public expression {
    statement_ptr operand;
    bool negate;
    statement_ptr test;

    test_expression(statement_ptr&& operand, bool negate, statement_ptr&& test)
        : operand(std::move(operand)), negate(negate), test(std::move(test)) {
        chk_type<expression>(this->operand);
        chk_type<identifier, call_expression>(this->test);
    }

    std::string type() const override { return "TestExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"operand", {operand.get()}}, {"test", {test.get()}}});
    }
};

/**
 * An operation with one side (operator on the left).
 */
struct unary_expression : public expression {
    token op;
    statement_ptr argument;

    unary_expression(token op, statement_ptr&& argument)
        : op(std::move(op)), argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }

    std::string type() const override { return "UnaryExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"argument", {argument.get()}}});
    }
};

struct slice_expression : public expression {
    statement_ptr start_expr;
    statement_ptr stop_expr;
    statement_ptr step_expr;

    slice_expression(statement_ptr&& start_expr, statement_ptr&& stop_expr,
                     statement_ptr&& step_expr)
        : start_expr(std::move(start_expr)), stop_expr(std::move(stop_expr)),
          step_expr(std::move(step_expr)) {
        chk_type<expression>(this->start_expr);
        chk_type<expression>(this->stop_expr);
        chk_type<expression>(this->step_expr);
    }

    std::string type() const override { return "SliceExpression"; }

    [[noreturn]] value execute_impl(context&) override {
        throw std::runtime_error("must be handled by MemberExpression");
    }

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"start_expr", {start_expr.get()}},
                     {"stop_expr", {stop_expr.get()}},
                     {"step_expr", {step_expr.get()}}});
    }
};

struct keyword_argument_expression : public expression {
    statement_ptr key;
    statement_ptr val;

    keyword_argument_expression(statement_ptr&& key, statement_ptr&& val)
        : key(std::move(key)), val(std::move(val)) {
        chk_type<identifier>(this->key);
        chk_type<expression>(this->val);
    }

    std::string type() const override { return "KeywordArgumentExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"key", {key.get()}}, {"val", {val.get()}}});
    }
};

struct spread_expression : public expression {
    statement_ptr argument;

    explicit spread_expression(statement_ptr&& argument) : argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }

    std::string type() const override { return "SpreadExpression"; }

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"argument", {argument.get()}}});
    }
};

struct call_statement : public statement {
    scope_names locals;
    scope_names captures;
    statement_ptr call;
    statements caller_args;
    statements body;

    call_statement(statement_ptr&& call, statements&& caller_args, statements&& body)
        : call(std::move(call)), caller_args(std::move(caller_args)), body(std::move(body)) {
        chk_type<call_expression>(this->call);
        for (const auto& arg : this->caller_args) chk_type<expression>(arg);
    }

    std::string type() const override { return "CallStatement"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"call", {call.get()}},
                     {"caller_args", stmts_to_ptr(caller_args)},
                     {"body", stmts_to_ptr(body)}});
    }
};

struct ternary_expression : public expression {
    statement_ptr condition;
    statement_ptr true_expr;
    statement_ptr false_expr;

    ternary_expression(statement_ptr&& condition, statement_ptr&& true_expr,
                       statement_ptr&& false_expr)
        : condition(std::move(condition)), true_expr(std::move(true_expr)),
          false_expr(std::move(false_expr)) {
        chk_type<expression>(this->condition);
        chk_type<expression>(this->true_expr);
        chk_type<expression>(this->false_expr);
    }

    std::string type() const override { return "Ternary"; }

    value execute_impl(context& ctx) override {
        value cond_val = condition->execute(ctx);
        if (cond_val->as_bool()) {
            return true_expr->execute(ctx);
        } else {
            return false_expr->execute(ctx);
        }
    }

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"condition", {condition.get()}},
                     {"true_expr", {true_expr.get()}},
                     {"false_expr", {false_expr.get()}}});
    }
};

struct raised_exception : public std::exception {
    std::string message;

    raised_exception(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }
};

// Used to rethrow exceptions with modified messages
struct rethrown_exception : public std::exception {
    std::string message;

    rethrown_exception(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }
};

//////////////////////

static void gather_string_parts_recursive(const value& val, value_string& parts) {
    if (is_val<value_string>(val)) {
        const auto& str_val = cast_val<value_string>(val)->val_str;
        parts->val_str.append(str_val);
    } else if (is_val<value_int>(val) || is_val<value_float>(val) || is_val<value_bool>(val)) {
        std::string str_val = val->as_string().str();
        parts->val_str.append(str_val);
    } else if (is_val<value_array>(val)) {
        const auto& items = cast_val<value_array>(val)->as_array();
        for (const auto& item : items) { gather_string_parts_recursive(item, parts); }
    }
}

struct runtime {
    context& ctx;

    explicit runtime(context& ctx) : ctx(ctx) {}

    value_string execute(const program& prog);

    static value_string gather_string_parts(const value& val) {
        if (is_val<value_string>(val)) return std::static_pointer_cast<value_string_t>(val);
        value_string parts = mk_val<value_string>();
        gather_string_parts_recursive(val, parts);
        return parts;
    }

    static std::string debug_dump_program(const program& prog, const std::string& src);
};

} // namespace jinja
