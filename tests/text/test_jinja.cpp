#include "text/jinja.h"

#include <jinja/parser.h>

#include <algorithm>
#include <atomic>
#include <iterator>
#include <future>
#include <limits>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::ordered_json;
using ninfer::text::JinjaTemplate;

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << message << '\n';
    return 1;
}

int language_semantics() {
    struct Example {
        std::string source;
        Json input;
        std::string expected;
    };

    const Example cases[]{
        {"{{ missing is undefined }}|{{ value is none }}|{{ value|default('fallback') }}|{{ "
         "value|default('fallback', true) }}",
         {{"value", nullptr}},
         "True|True|None|fallback"},
        {"{{ missing }}|{{ [1, none, true, 'x'] }}|{{ 12 ~ none }}", Json::object(),
         "|[1, None, True, 'x']|12None"},
        {"{{ text|length }}|{{ text[-1] }}|{{ text[::-1] }}|{{ text[:-1:-1] }}|{{ text[99:] }}",
         {{"text", "北京🌏"}},
         "3|🌏|🌏京北||"},
        {"{% for c in text %}[{{ c }}]{% endfor %}", {{"text", "北京🌏"}}, "[北][京][🌏]"},
        {"{% for i in range(1000000000000) %}{{ i }}{% break %}{% endfor %}", Json::object(), "0"},
        {"{% for x,y in [[1,2],[3]] %}{{ x }}{% break %}{% endfor %}", Json::object(), "1"},
        {"{% for i in [1,2] %}A{% if true %}B{% break %}C{% endif %}D{% else %}E{% endfor %}",
         Json::object(), "ABE"},
        {"{% for i in [1,2] %}A{% if true %}B{% continue %}C{% endif %}D{% else %}E{% endfor %}",
         Json::object(), "ABABE"},
        {"{% for i in [1] %}{% set x %}A{% break %}{% endset %}B{% else %}E{% endfor %}|"
         "{% for i in [1] %}{% filter upper %}a{% break %}{% endfilter %}B{% else %}E{% endfor %}",
         Json::object(), "E|E"},
        {"{% set d={1:'one',2.0:'two',true:'bool'} %}{{ d[1] }}|{{ d[2] }}|{{ d[true] }}",
         Json::object(), "bool|two|bool"},
        {"{% set a,b = range(3,7,2) %}{{ a }}|{{ b }}", Json::object(), "3|5"},
        {"{% for i in [1] %}{% if loop %}{{ loop.index }}{% endif %}{% endfor %}", Json::object(),
         "1"},
        {"{% set x=0 %}{% for i in [1,2] %}{{ x }}{% set x=i %}{% endfor %}|{{ x }}",
         Json::object(), "00|0"},
        {"{% set n=namespace(v=0) %}{% for i in [1,2,3] if i>n.v %}{{ i }}"
         "{% set n.v=3 %}{% endfor %}",
         Json::object(), "1"},
        {"{% set n=namespace(v=0) %}{% for i in [1,2,3] if i>n.v %}{{ i }}/{{ loop.length }}"
         "{% set n.v=3 %}{% endfor %}",
         Json::object(), "1/32/33/3"},
        {"{% for i in range(5) if i%2==0 %}{{ loop.index }}:{{ i }}:"
         "{{ loop.previtem|default('-') }}:{{ loop.nextitem|default('-') }}:"
         "{{ loop.first }}:{{ loop.last }}:{{ loop.revindex }};{% endfor %}",
         Json::object(), "1:0:-:2:True:False:3;2:2:0:4:False:False:2;3:4:2:-:False:True:1;"},
        {"{{ range(8,1,-2)|list|tojson }}|{{ range(2,9,2)[1:]|list|tojson }}|"
         "{{ range(2,9,2)[-1] }}|{{ 6 in range(2,9,2) }}|{{ range(3)|length }}",
         Json::object(), "[8, 6, 4, 2]|[4, 6, 8]|8|True|3"},
        {"{{ text|upper }}|{{ text|lower }}|{{ text|capitalize }}",
         {{"text", "straße ΟΣ İ"}},
         "STRASSE ΟΣ İ|straße ος i̇|Straße ος i̇"},
        {"{{ text|trim }}|{{ text.split()|tojson }}|{{ text.rsplit(none,1)|tojson }}",
         {{"text", "\u0085北京\u3000two\tend\u001c"}},
         "北京　two\tend|[\"北京\", \"two\", \"end\"]|[\"\u0085北京　two\", \"end\"]"},
        {"{{ text.replace('', '-', 2) }}|{{ text.replace('京', 'x', 1) }}",
         {{"text", "北京🌏"}},
         "-北-京🌏|北x🌏"},
        {"{{ value|tojson(ensure_ascii=true,sort_keys=true,separators=(',',':')) }}",
         {{"value", {{"z", "🌏"}, {"a", false}}}},
         "{\"a\":false,\"z\":\"\\ud83c\\udf0f\"}"},
        {"{% macro show(x) %}{{ x|default('local') }}{% endmacro %}{% set ns=namespace(n=0) %}{% "
         "for x in [1,2] %}{% set ns.n=ns.n+x %}{% endfor %}{{ ns.n }}|{{ show() }}",
         Json::object(), "3|local"},
        {"{% set x='outer' %}{% macro show(a='A',b='B') %}{{ x }}:{{ a }}:{{ b }}{% endmacro %}{% "
         "for x in ['inner'] %}{{ show(b='Z') }}{% endfor %}",
         Json::object(), "outer:A:Z"},
        {"{% set n=namespace(f=none) %}{% for x in ['captured'] %}{% macro show() %}{{ x }}{% "
         "endmacro %}{% set n.f=show %}{% endfor %}{{ n.f() }}",
         Json::object(), "captured"},
        {"{% macro wrap() %}[{{ caller('value') }}]{% endmacro %}{% call(x) wrap() %}{{ x }}{% "
         "endcall %}",
         Json::object(), "[value]"},
        {"{{ number }}", {{"number", 1.23456789}}, "1.23456789"},
        {"{% filter upper %}{{ text }}{% endfilter %}", {{"text", "straße"}}, "STRASSE"},
        {"{% set ns=namespace(f=none) %}{% macro save() %}{% set ns.f=caller %}{% endmacro "
         "%}{% set x='old' %}{% call save() %}{{ x }}{% endcall %}{% set x='new' %}{{ "
         "ns.f() }}",
         Json::object(), "new"},
        {"{% set ns=namespace(f=none) %}{% for x in ['A'] %}{% macro first() %}{{ second() "
         "}}{% endmacro %}{% macro second() %}{{ x }}{% endmacro %}{% set ns.f=first %}{% "
         "endfor %}{{ ns.f() }}",
         Json::object(), "A"},
        {"{% set ns=namespace(f=none) %}{% for x in ['A'] %}{% macro a(n) %}{% if n %}{{ "
         "b(n-1) }}{% else %}{{ x }}{% endif %}{% endmacro %}{% macro b(n) %}{{ a(n) }}{% "
         "endmacro %}{% set ns.f=a %}{% endfor %}{{ ns.f(3) }}",
         Json::object(), "A"},
        {"{% set ns=namespace(f=none) %}{% for x in ['A'] %}{% macro helper() %}{{ x }}{% "
         "endmacro %}{% macro factory() %}{% macro nested() %}{{ helper() }}{% endmacro "
         "%}{% set ns.f=nested %}{% endmacro %}{{ factory() }}{% endfor %}{{ ns.f() }}",
         Json::object(), "A"},
        {"{% set x='outer' %}{% set ns=namespace(f=none) %}{% for i in [1] %}{% macro f() "
         "%}{{ x }}{% endmacro %}{{ f() }}|{% set x='inner' %}{% set ns.f=f %}{% endfor "
         "%}{{ ns.f() }}",
         Json::object(), "outer|inner"},
        {"{% set x='outer' %}{% macro f() %}{{ x }}|{% set x='inner' %}{{ x }}{% endmacro "
         "%}{{ f() }}|{{ x }}",
         Json::object(), "outer|inner|outer"},
        {"{% set x='X' %}{% macro f(a=x,b=a) %}{{ a }}{{ b }}{% endmacro %}{{ f() }}|{{ "
         "f('A') }}",
         Json::object(), "XX|AA"},
        {"{% set x='X' %}{% set k='key' %}{% macro f() %}{{ {'key':x}[k] }}|{{ "
         "[x]|join(',') }}|{{ x is string }}{% endmacro %}{{ f() }}",
         Json::object(), "X|X|True"},
        {"{% set ns=namespace(l=none) %}{% for x in [2] %}{% for i in [1,2,3] if i>=x %}{% "
         "set ns.l=loop %}{% break %}{% endfor %}{% endfor %}{{ ns.l.length }}|{{ "
         "ns.l.nextitem }}",
         Json::object(), "2|3"},
        {"{% set ns=namespace(f=none) %}{% for x in ['A'] %}{% macro a(n) %}{% if n %}{{ "
         "a(n-1) }}{% else %}{{ x }}{% endif %}{% endmacro %}{% set alias=a %}{% set "
         "ns.f=alias %}{% endfor %}{{ ns.f(2) }}",
         Json::object(), "A"},
        {"{% for i in a if i>0 %}{{ i }}/{{ loop.length }};{% if i==1 %}{% set "
         "_=a.append(3) %}{% endif %}{% endfor %}",
         {{"a", Json::array({1, 2})}},
         "1/2;2/2;"},
        {"{% for i in a %}{{ i }}:{{ loop.last }};{% if i==2 %}{% set _=a.append(3) %}{% "
         "endif %}{% endfor %}",
         {{"a", Json::array({1, 2})}},
         "1:False;2:True;"},
        {"{% for i in a %}{{ i }}/{{ loop.length }};{% if i==2 %}{% set _=a.append(3) %}{% "
         "endif %}{% endfor %}",
         {{"a", Json::array({1, 2})}},
         "1/2;2/2;3/2;"},
        {"{{ n is divisibleby(-1) }}|{{ n is divisibleby(2) }}|{{ 7 is divisibleby(3) }}",
         {{"n", std::numeric_limits<int64_t>::min()}},
         "True|True|False"},
        {"{% set x='old' %}{% set ns=namespace(f=none) %}{% for i in [1] %}"
         "{% macro f() %}{{ x }}{% endmacro %}{% if false %}{% set x='local' %}{% endif %}"
         "{% set ns.f=f %}{% endfor %}{% set x='new' %}{{ ns.f() }}",
         Json::object(), "new"},
    };
    int failures = 0;
    for (const auto& item : cases) {
        const auto actual = JinjaTemplate(item.source, "language-test").render(item.input).text;
        if (actual != item.expected) {
            std::cerr << "Jinja semantic mismatch in " << item.source
                      << "\nexpected: " << Json(item.expected).dump()
                      << "\nactual: " << Json(actual).dump() << '\n';
            ++failures;
        }
    }
    for (const char* source :
         {"{{ none + 'x' }}", "{{ 'abc'[::0] }}", "{{ missing|tojson }}",
          "{{ value|unknown_filter }}", "{{ 1 is divisibleby(0) }}",
          "{% for loop in [1] %}{{ loop }}{% endfor %}",
          "{% for loop,x in [[1,2]] %}{{ x }}{% endfor %}", "{% macro 1() %}x{% endmacro %}"}) {
        try {
            (void)JinjaTemplate(source, "error-test").render(Json::object());
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).starts_with("error-test:"),
                              "template error omitted its source name");
        }
    }
    return failures;
}

int closure_lifetimes() {
    using namespace jinja;
    const char* bodies[]{
        "{% call sink() %}{% endcall %}",
        "{% macro local() %}{{ token }}{% endmacro %}",
        "{% call save() %}{{ token }}{% endcall %}",
        "{% macro local() %}{{ token }}{% endmacro %}{% set ns.f=local %}",
        "{% macro local() %}{{ token }}{{ ns.f is defined }}{% endmacro %}{% set ns.f=local %}",
        "{% macro local(n) %}{% if n %}{{ alias(n-1) }}{% else %}{{ token }}{% endif %}"
        "{% endmacro %}{% set alias=local %}{{ local(2) }}"};
    const int expected_alive[]{0, 0, 1, 1, 1, 0};
    int failures = 0;
    for (std::size_t index = 0; index < std::size(bodies); ++index) {
        const std::string source = "{% set ns=namespace(f=none) %}"
                                   "{% macro sink() %}{{ caller() }}{% endmacro %}"
                                   "{% macro save() %}{% set ns.f=caller %}{% endmacro %}"
                                   "{% for i in range(32) %}{% set token=make_token() %}" +
                                   std::string(bodies[index]) + "{% endfor %}";
        auto program = parse_from_tokens(lexer().tokenize(source));
        std::vector<std::weak_ptr<value_t>> tokens;
        const auto alive = [&] {
            return std::count_if(tokens.begin(), tokens.end(),
                                 [](const auto& token) { return !token.expired(); });
        };
        {
            context ctx(source);
            ctx.set_val("make_token",
                        mk_val<value_func>("make_token", [&](const func_args&) -> value {
                            value token = mk_val<value_string>("value");
                            tokens.emplace_back(token);
                            return token;
                        }));
            runtime(ctx).execute(program);
            failures +=
                check(alive() == expected_alive[index], "closure retained unused iteration values");
        }
        failures += check(alive() == 0, "render teardown retained captured values");
    }

    for (bool cancel : {false, true}) {
        const std::string source =
            "{% set ns=namespace(f=none) %}{% for i in range(32) %}{% set token=make_token() %}"
            "{% macro local() %}{{ token }}{{ ns.f is defined }}{% endmacro %}{% set ns.f=local %}"
            "{% if i==4 %}{{ raise_exception('stop') }}{% endif %}{% endfor %}";
        auto program = parse_from_tokens(lexer().tokenize(source));
        std::vector<std::weak_ptr<value_t>> tokens;
        bool interrupted = false;
        try {
            context ctx(source);
            ctx.set_val("make_token",
                        mk_val<value_func>("make_token", [&](const func_args&) -> value {
                            value token = mk_val<value_string>("value");
                            tokens.emplace_back(token);
                            return token;
                        }));
            if (cancel)
                ctx.checkpoint = [&] {
                    if (tokens.size() >= 3) throw std::runtime_error("cancelled");
                };
            runtime(ctx).execute(program);
        } catch (const std::exception&) { interrupted = true; }
        failures += check(interrupted && !tokens.empty(),
                          "render did not exercise interrupted capture cleanup");
        for (const auto& token : tokens)
            failures += check(token.expired(), "interrupted render retained a value");
    }

    const std::string source =
        "{% set ns=namespace(l=none,f=none,c=none) %}"
        "{% macro save() %}{% set ns.c=caller %}{% endmacro %}"
        "{% call save() %}literal{% endcall %}{% for x in [1] %}"
        "{% for i in [1,2] if i>=x %}{% set ns.l=loop %}{% break %}{% endfor %}"
        "{% macro f() %}{{ x }}{% endmacro %}{% set ns.f=f %}{% endfor %}";
    auto program = parse_from_tokens(lexer().tokenize(source));
    value saved_loop, saved_macro, saved_caller;
    {
        context ctx(source);
        runtime(ctx).execute(program);
        saved_loop   = ctx.get_val("ns")->at("l");
        saved_macro  = ctx.get_val("ns")->at("f");
        saved_caller = ctx.get_val("ns")->at("c");
    }
    bool rejected = false;
    try {
        saved_loop->find(mk_val<value_string>("length"));
    } catch (const std::runtime_error&) { rejected = true; }
    failures += check(rejected, "loop used an ended render");
    rejected = false;
    try {
        context another;
        func_args args(another);
        saved_macro->invoke(args);
    } catch (const std::runtime_error&) { rejected = true; }
    failures += check(rejected, "macro used an ended render");
    rejected = false;
    try {
        context another;
        func_args args(another);
        saved_caller->invoke(args);
    } catch (const std::runtime_error&) { rejected = true; }
    failures += check(rejected, "caller used an ended render");
    saved_loop.reset();
    saved_macro.reset();
    saved_caller.reset();
    return failures;
}

int origins_and_requests() {
    const JinjaTemplate compiled(
        "{% filter upper %}{{ text|trim }}{% endfilter %}|{{ tool|tojson }}", "shared-test");
    const Json context{{"text", " hello "}, {"tool", {{"name", "f"}}}};
    const std::vector<ninfer::text::TemplateInputRegion> regions{{"/text", 7}, {"/tool", 9}};
    ninfer::text::TemplateRenderOptions options{.timestamp = 0, .regions = regions};
    const auto result = compiled.render(context, options);
    int failures = check(result.text == "HELLO|{\"name\": \"f\"}" && result.regions.size() == 2 &&
                             result.regions[0].tag == 7 && result.regions[0].begin == 0 &&
                             result.regions[0].end == 5 && !result.regions[0].source_offset &&
                             result.regions[1].tag == 9 && result.regions[1].begin == 6 &&
                             result.regions[1].end == result.text.size(),
                         "transformed/JSON input regions were lost or incorrectly mapped");
    failures += check(compiled.render(context).regions.empty(),
                      "unannotated render retained request origins");
    const auto clipped =
        JinjaTemplate("{{ text[1:3] }}", "slice-test").render({{"text", "a北京c"}}, options);
    failures += check(clipped.regions.size() == 1 && clipped.regions[0].source_offset == 1 &&
                          clipped.regions[0].end == 6,
                      "Unicode slice lost byte origin mapping");
    const auto indexed =
        JinjaTemplate("{{ ('a' ~ text ~ 'z')[1] }}|{{ ('a' ~ text ~ 'z')[-2] }}", "index-origin")
            .render({{"text", "北京🌏"}}, options);
    failures +=
        check(indexed.text == "北|🌏" && indexed.regions.size() == 2 &&
                  indexed.regions[0].source_offset == 0 && indexed.regions[1].source_offset == 6,
              "Unicode indexing lost source coordinates across string parts");
    const std::vector<ninfer::text::TemplateInputRegion> nested_regions{{"/obj", 10},
                                                                        {"/obj/b", 11}};
    const auto nested =
        JinjaTemplate("{% for k,v in obj.items() %}{{ k }}={{ v }};{% endfor %}", "object-origins")
            .render({{"obj", {{"a", "x"}, {"b", "é"}}}}, {.regions = nested_regions});
    bool inherited = false, exact_child = false;
    for (const auto& region : nested.regions) {
        if (region.tag == 10) {
            inherited = true;
            failures +=
                check(!region.source_offset, "object origin invented exact child byte coordinates");
        }
        if (region.tag == 11)
            exact_child = region.source_offset == 0 &&
                          nested.text.substr(region.begin, region.end - region.begin) == "é";
    }
    failures += check(nested.text == "a=x;b=é;" && inherited && exact_child,
                      "object origin inheritance or explicit child precedence failed");
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.push_back(std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i)
                if (compiled.render(context, options).text != result.text) return false;
            return true;
        }));
    }
    for (auto& worker : workers)
        failures += check(worker.get(), "shared template leaked mutable request state");

    struct Cancelled : std::runtime_error {
        Cancelled() : std::runtime_error("cancelled") {}
    };

    int calls          = 0;
    options.checkpoint = [&] {
        if (++calls > 2) throw Cancelled();
    };
    try {
        (void)JinjaTemplate("{% for i in range(1000000) %}{{ i }}{% endfor %}", "cancel-test")
            .render(Json::object(), options);
        failures += check(false, "template ignored cancellation");
    } catch (const Cancelled&) {}
    options.checkpoint = {};
    const JinjaTemplate clock("{{ strftime_now('%Y') }}", "clock-test");
    failures += check(clock.render(Json::object(), options).text ==
                          clock.render(Json::object(), options).text,
                      "time snapshot was not reused");
    return failures;
}

int trim_semantics() {
    const JinjaTemplate trims(
        "[{{ text.strip(chars) }}]|[{{ text.lstrip(chars) }}]|[{{ text.rstrip(chars) }}]",
        "trim-test");
    int failures = check(trims.render({{"text", "北🌏x🌏北"}, {"chars", "🌏北"}}).text ==
                             "[x]|[x🌏北]|[北🌏x]",
                         "selected Unicode trim changed left/right semantics");
    failures +=
        check(trims.render({{"text", "北🌏"}, {"chars", "🌏北"}}).text == "[]|[]|[]" &&
                  trims.render({{"text", "北🌏"}, {"chars", ""}}).text == "[北🌏]|[北🌏]|[北🌏]" &&
                  trims.render({{"text", ""}, {"chars", "🌏北"}}).text == "[]|[]|[]",
              "empty or fully trimmed Unicode input changed");
    failures +=
        check(JinjaTemplate("[{{ text.strip() }}]|[{{ text.lstrip() }}]|[{{ text.rstrip() }}]",
                            "whitespace-trim")
                      .render({{"text", "\u0085\u001c\u3000\t"}})
                      .text == "[]|[]|[]",
              "default Unicode whitespace trim changed");

    const std::vector<ninfer::text::TemplateInputRegion> regions{{"/text", 7}};
    const auto clipped = JinjaTemplate("{{ ('　' ~ text ~ '　').strip() }}", "trim-origin")
                             .render({{"text", "　<|image_pad|>　"}}, {.regions = regions});
    failures +=
        check(clipped.text == "<|image_pad|>" && clipped.regions.size() == 1 &&
                  clipped.regions[0].tag == 7 && clipped.regions[0].begin == 0 &&
                  clipped.regions[0].end == clipped.text.size() &&
                  clipped.regions[0].source_offset == 3 && clipped.literal_spans.size() == 1 &&
                  clipped.literal_spans[0].begin == 0 &&
                  clipped.literal_spans[0].end == clipped.text.size(),
              "trim across string parts lost literal status or byte origin");
    failures += check(
        clipped.boundary_mappings.size() == 2 && clipped.boundary_mappings[0].tag == 7 &&
            clipped.boundary_mappings[0].source_begin == 0 &&
            clipped.boundary_mappings[0].source_end == 3 &&
            clipped.boundary_mappings[0].offset == 0 && clipped.boundary_mappings[1].tag == 7 &&
            clipped.boundary_mappings[1].source_begin == 16 &&
            clipped.boundary_mappings[1].source_end == 19 &&
            clipped.boundary_mappings[1].offset == 13,
        "trim did not map deleted Unicode source boundaries to the surviving content edges");

    const auto empty = JinjaTemplate("before{{ text|trim }}after", "empty-trim-origin")
                           .render({{"text", " \t"}}, {.regions = regions});
    failures += check(
        empty.text == "beforeafter" && empty.regions.empty() && empty.literal_spans.empty() &&
            empty.boundary_mappings.size() == 1 && empty.boundary_mappings[0].tag == 7 &&
            empty.boundary_mappings[0].source_begin == 0 &&
            empty.boundary_mappings[0].source_end == 2 && empty.boundary_mappings[0].offset == 6,
        "fully trimmed input lost its boundary or created literal output bytes");

    const auto quoted = JinjaTemplate("{{ text|trim|tojson }}", "trim-json-origin")
                            .render({{"text", " value\n"}}, {.regions = regions});
    failures += check(
        quoted.text == "\"value\"" && quoted.regions.size() == 1 && quoted.regions[0].tag == 7 &&
            quoted.regions[0].begin == 0 && quoted.regions[0].end == quoted.text.size() &&
            !quoted.regions[0].source_offset && quoted.boundary_mappings.empty(),
        "JSON encoding of trimmed input lost its origin or retained exact boundaries");

    const std::string invalid = std::string("left") + '\x80' + "right";
    for (const char* source : {"{{ text.strip() }}", "{{ text.lstrip() }}", "{{ text.rstrip() }}",
                               "{{ text.strip('') }}"}) {
        try {
            (void)JinjaTemplate(source, "invalid-trim").render({{"text", invalid}});
            failures += check(false, "trim skipped invalid UTF-8 in the string interior");
        } catch (const std::invalid_argument&) {}
    }
    try {
        (void)trims.render({{"text", "valid"}, {"chars", invalid}});
        failures += check(false, "trim accepted an invalid UTF-8 selected character set");
    } catch (const std::invalid_argument&) {}
    return failures;
}

int literal_content() {
    struct Example {
        const char* source;
        Json context;
        const char* expected;
    };

    const Example cases[]{
        {"{{ data|lower|trim }}", {{"data", " <|IMAGE_PAD|> "}}, "<|image_pad|>"},
        {"{{ data[1:-1][::-1][::-1] }}", {{"data", "x<|image_pad|>x"}}, "<|image_pad|>"},
        {"{{ data.split('|x|')|join('') }}", {{"data", "<|image|x|_pad|>"}}, "<|image_pad|>"},
        {"{{ data|replace('x', '') }}", {{"data", "<|image_xpad|>"}}, "<|image_pad|>"},
        {"{{ data.format('') }}", {{"data", "<|image_{}pad|>"}}, "<|image_pad|>"},
        {"{{ '{}'.format(data) }}", {{"data", "<|image_pad|>"}}, "<|image_pad|>"},
        {"{{ data|indent(2, true)|trim|safe }}", {{"data", "<|image_pad|>"}}, "<|image_pad|>"},
        {"{{ data|indent(2) }}", {{"data", "\n<|image_pad|>\n"}}, "\n  <|image_pad|>\n"},
        {"{% macro show(x) %}{{ x }}{% endmacro %}{% set ns=namespace(x=show(data)) %}{{ ns.x }}",
         {{"data", "<|image_pad|>"}},
         "<|image_pad|>"},
        {"{% set captured %}{{ data }}{% endset %}{{ captured }}",
         {{"data", "<|image_pad|>"}},
         "<|image_pad|>"},
        {"{{ data*2 }}", {{"data", "<|image_pad|>"}}, "<|image_pad|><|image_pad|>"},
        {"{{ data|tojson }}",
         {{"data", {{"<|image_pad|>", "<|image_pad|>"}}}},
         "{\"<|image_pad|>\": \"<|image_pad|>\"}"},
        {"{{ data|string }}", {{"data", Json::array({"<|image_pad|>"})}}, "['<|image_pad|>']"},
        {"{% for k, v in data.items() %}{{ k }}{{ v }}{% endfor %}",
         {{"data", {{"<|image_pad|>", "<|image_pad|>"}}}},
         "<|image_pad|><|image_pad|>"},
        {"{{ data|upper|lower }}", {{"data", "ß<|IMAGE_PAD|>İ"}}, "ss<|image_pad|>i̇"},
    };
    int failures                     = 0;
    constexpr std::string_view start = "<|im_start|>", end = "<|im_end|>", pad = "<|image_pad|>";
    for (const auto& item : cases) {
        const auto output =
            JinjaTemplate(std::string(start) + item.source + std::string(end), "literal-content")
                .render(item.context);
        if (output.text != std::string(start) + item.expected + std::string(end)) {
            std::cerr << "literal-content text mismatch: " << item.source << '\n';
            ++failures;
        }
        failures +=
            check(!ninfer::text::overlaps(output.literal_spans, 0, start.size()) &&
                      !ninfer::text::overlaps(output.literal_spans, output.text.size() - end.size(),
                                              output.text.size()),
                  "input content shielded surrounding template controls");
        for (auto pos = output.text.find(pad); pos != std::string::npos;
             pos      = output.text.find(pad, pos + pad.size())) {
            if (!ninfer::text::overlaps(output.literal_spans, pos, pos + pad.size())) {
                std::cerr << "literal-content marker lost through: " << item.source << '\n';
                ++failures;
            }
        }
    }
    const auto mixed =
        JinjaTemplate("{{ ('<|IM_START|>' ~ data ~ '<|IM_END|>')|lower }}", "mixed-case")
            .render({{"data", "ß<|IMAGE_PAD|>İ"}});
    failures +=
        check(mixed.text == "<|im_start|>ß<|image_pad|>i̇<|im_end|>" &&
                  mixed.literal_spans.size() == 1 && mixed.literal_spans[0].begin == start.size() &&
                  mixed.literal_spans[0].end == mixed.text.size() - end.size(),
              "mixed Unicode case conversion lost content/control boundaries");
    const auto sigma =
        JinjaTemplate("{{ ('Ο' ~ data)|lower }}", "mixed-sigma").render({{"data", "Σ"}});
    failures += check(sigma.text == "ος" && sigma.literal_spans.size() == 1 &&
                          sigma.literal_spans[0].begin == 2 && sigma.literal_spans[0].end == 4,
                      "case conversion lost context across content/control boundaries");
    const auto explicit_control =
        JinjaTemplate("{{ data.replace('x', '<|image_pad|>') }}", "replace")
            .render({{"data", "x"}});
    failures += check(explicit_control.text == pad && explicit_control.literal_spans.empty(),
                      "template-authored replacement lost its control meaning");
    const std::vector<std::string> tokens{"bos_token"};
    const auto variables = JinjaTemplate("{{ bos_token }}{{ data }}", "token-variable")
                               .render({{"bos_token", "<|im_start|>"}, {"data", "<|im_start|>"}},
                                       {.control_variables = tokens});
    failures +=
        check(!ninfer::text::overlaps(variables.literal_spans, 0, start.size()) &&
                  ninfer::text::overlaps(variables.literal_spans, start.size(), 2 * start.size()),
              "engine token variables and ordinary kwargs were not distinguished");
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--render") {
        for (std::string line; std::getline(std::cin, line);) {
            Json result;
            try {
                const auto input = Json::parse(line);
                result["text"]   = JinjaTemplate(input.at("source"), "reference-case")
                                     .render(input.at("context"))
                                     .text;
                result["ok"] = true;
            } catch (const std::exception& error) {
                result["ok"]    = false;
                result["error"] = error.what();
            }
            std::cout << result.dump() << '\n';
        }
        return 0;
    }
    try {
        const int failures = language_semantics() + closure_lifetimes() + origins_and_requests() +
                             trim_semantics() + literal_content();
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
