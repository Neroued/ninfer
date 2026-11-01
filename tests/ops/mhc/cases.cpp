#include "support.h"
#include "core/decode_graph.h"

namespace ninfer::test::mhc {
int verify(const Fixture& f, const std::string& label) {
    int failed = 0;
    std::vector<double> residual;
    if (f.form == Form::Post || f.form == Form::PostPre || f.form == Form::ExpandPre) {
        residual = from_device_bf16(f.tr.data, f.x.size());
        if (f.form == Form::ExpandPre) {
            const auto actual = from_device<std::uint16_t>(f.tr.data, f.x.size());
            for (int m = 0; m < f.m; ++m)
                for (int i = 0; i < S; ++i)
                    for (int d = 0; d < D; ++d)
                        if (actual[std::size_t(m) * K + i * D + d] !=
                            f32_to_bf16(f.y[std::size_t(m) * D + d])) {
                            std::cerr << label << ": expansion changed BF16 words\n";
                            return 1;
                        }
        } else
            for (int m = 0; m < f.m; ++m) {
                const auto expected = post_reference(f, m);
                failed += verify_reduction(label + " residual",
                                           std::span(residual.data() + std::size_t(m) * K, K),
                                           expected, kBf16);
            }
    }
    if (f.form == Form::Post) return failed;
    const auto actual = from_device_bf16(f.outputs.block_input.data, f.y.size());
    for (auto v : actual)
        if (!std::isfinite(v)) return failed + 1;
    std::vector<double> q, c;
    if (has_pre(f.form)) {
        const auto qf = from_device<float>(f.outputs.coefficients.post.data, f.q.size());
        const auto cf = from_device<float>(f.outputs.coefficients.mix.data, f.c.size());
        q.assign(qf.begin(), qf.end());
        c.assign(cf.begin(), cf.end());
        for (const auto* values : {&q, &c})
            for (auto v : *values)
                if (!std::isfinite(v)) return failed + 1;
    }
    for (int m = 0; m < f.m; ++m) {
        if (f.m > 128 && m >= 8 && m != f.m / 2 && m != f.m - 1) continue;
        std::vector<double> x(K);
        if (f.form == Form::Pre)
            std::copy_n(f.x.data() + std::size_t(m) * K, K, x.data());
        else if (f.form == Form::ExpandPre)
            for (int i = 0; i < S; ++i)
                std::copy_n(f.y.data() + std::size_t(m) * D, D, x.data() + i * D);
        else {
            x = post_reference(f, m);
            for (auto& v : x) v = round_bf16(v);
        }
        if (f.form == Form::PostMeanNorm) {
            std::vector<double> expected(D);
            double ss = 0;
            for (int d = 0; d < D; ++d) {
                for (int j = 0; j < S; ++j) expected[d] += x[j * D + d] / S;
                ss += expected[d] * expected[d];
            }
            const double inverse = 1 / std::sqrt(ss / D + f.final_epsilon);
            for (int d = 0; d < D; ++d) expected[d] *= inverse * double(f.gamma[d]);
            failed += verify_reduction(label + " hidden",
                                       std::span(actual.data() + std::size_t(m) * D, D), expected,
                                       kFinalHidden);
            continue;
        }
        const auto check_pre = [&](const PreReference& expected, const char* suffix,
                                   const ReductionCriterion& coefficient_criterion) {
            int errors = verify_reduction(label + suffix + " block",
                                          std::span(actual.data() + std::size_t(m) * D, D),
                                          expected.u, kBf16);
            errors += verify_reduction(label + suffix + " q", std::span(q.data() + m * S, S),
                                       expected.q, coefficient_criterion);
            errors +=
                verify_reduction(label + suffix + " C", std::span(c.data() + m * S * S, S * S),
                                 expected.c, coefficient_criterion);
            return errors;
        };
        failed += check_pre(pre_reference(f, x), " formula",
                            f.form == Form::PostPre ? kFusedCoefficients : kCoefficients);
        if (f.form == Form::PostPre)
            failed +=
                check_pre(pre_reference(f, std::span(residual.data() + std::size_t(m) * K, K)),
                          " published BF16 seam", kCoefficients);
    }
    return failed;
}

namespace {
int run(Form form, int m, Pattern pattern, bool alias, bool captured, CudaExecutionView e,
        std::size_t tensor_guard_bytes = 256) {
    Fixture f(form, m, alias, tensor_guard_bytes);
    f.set(pattern);
    f.upload(e.stream());
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (captured) {
        definition.capture(e.stream(), [&] { f.invoke(e); });
        graph.instantiate(definition);
    }
    int failed = 0;
    for (int phase = 0; phase < (captured ? 2 : 1); ++phase) {
        if (phase) {
            f.set(pattern, phase);
            f.upload(e.stream());
        }
        if (captured)
            graph.launch(e.stream());
        else
            f.invoke(e);
        cuda_synchronize(e.stream());
        if (f.arena.used() != 256 || f.arena.peak_used() != 256 + f.capacity)
            throw std::runtime_error("mHC workspace cursor/capacity mismatch");
        const auto label = std::string(name(form)) + " M=" + std::to_string(m);
        failed += verify(f, label) + f.preserved(label);
    }
    return failed;
}
} // namespace

void add_cases(OpSuite& suite) {
    for (auto form : {Form::Pre, Form::Post, Form::ExpandPre, Form::PostPre, Form::PostMeanNorm}) {
        auto add = [&](int m, Pattern pattern, bool alias = false, bool graph = false) {
            const auto id = std::string(name(form)) + "/m" + std::to_string(m) + "/pattern" +
                            std::to_string(int(pattern)) + (alias ? "/inplace" : "/disjoint") +
                            (graph ? "/graph" : "/eager");
            suite.add(id, [=] {
                return run(form, m, pattern, alias, graph,
                           cuda_environment().context.execution_view());
            });
        };
        for (int m : {1, 7, 17, 128, 129, 513, 4097})
            if (has_pre(form) || m == 1 || m == 17 || m == 128 || m == 4097)
                add(m, Pattern::General);
        if (has_pre(form))
            suite.add(std::string(name(form)) + "/alignment16", [=] {
                return run(form, 9, Pattern::General, false, false,
                           cuda_environment().context.execution_view(), 272);
            });
        if (has_pre(form))
            for (int m : {15, 16}) add(m, Pattern::General);
        if (form == Form::Pre)
            for (int m : {256, 257, 607, 608, 609}) add(m, Pattern::General);
        if (form == Form::ExpandPre) {
            for (int m : {511, 512}) add(m, Pattern::General);
            for (auto p : {Pattern::Small, Pattern::Saturated, Pattern::Unbalanced}) add(513, p);
        }
        if (form == Form::Post)
            for (int m : {1023, 1024, 1025}) add(m, Pattern::General);
        for (auto p : {Pattern::Zero, Pattern::Saturated, Pattern::Unbalanced, Pattern::Midpoint,
                       Pattern::Small})
            if (has_pre(form) || (p != Pattern::Saturated && p != Pattern::Unbalanced)) add(3, p);
        add(7, Pattern::General, false, true);
        if (form == Form::Pre || form == Form::ExpandPre) add(1, Pattern::General, false, true);
        if (form == Form::Pre) add(609, Pattern::General, false, true);
        if (form == Form::ExpandPre) add(513, Pattern::General, false, true);
        if (form == Form::PostMeanNorm)
            for (int m : {8, 9}) add(m, Pattern::General);
        if (form == Form::PostMeanNorm) add(12, Pattern::General, false, true);
        if (form == Form::Post || form == Form::PostPre) {
            add(1, Pattern::Midpoint, true);
            add(17, Pattern::General, true, true);
            add(form == Form::Post ? 1025 : 129, Pattern::General, true, true);
            if (form == Form::PostPre) add(513, Pattern::General, true, true);
        }
        suite.add(std::string(name(form)) + "/contract",
                  [=] { return validate(form, cuda_environment().context.execution_view()); });
    }
}
} // namespace ninfer::test::mhc
