#include "text/jinja.h"

#include <atomic>
#include <future>
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
    for (const char* source : {"{{ none + 'x' }}", "{{ 'abc'[::0] }}", "{{ missing|tojson }}",
                               "{{ value|unknown_filter }}"}) {
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
        const int failures =
            language_semantics() + origins_and_requests() + trim_semantics() + literal_content();
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
