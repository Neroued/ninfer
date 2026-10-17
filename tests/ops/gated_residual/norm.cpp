#include "support.h"
#include "ops/suite.h"
#include "core/decode_graph.h"

namespace ninfer::test::hc {
int verify(const Fixture& f, const std::string& label, bool exact_residual) {
    const auto ref            = reference(f);
    int failed                = 0;
    const auto residual_words = f.form == Form::GroupedNorm
                                    ? std::vector<std::uint16_t>{}
                                    : from_device<std::uint16_t>(f.tr.data, f.h.size());
    std::vector<double> residual(residual_words.size());
    std::transform(residual_words.begin(), residual_words.end(), residual.begin(), bf16_to_f32);
    if (f.form == Form::ExpandNorm)
        failed += verify_exact((label + " expansion").c_str(), residual_words, ref.residual_words);
    else if (f.form != Form::GroupedNorm)
        failed += verify_pointwise(label + " residual", residual, ref.residual, kResidual);
    if (exact_residual)
        failed += verify_exact((label + " midpoint residual").c_str(), residual_words,
                               ref.residual_words);
    if (f.form != Form::Combine) {
        const auto actual         = from_device_bf16(f.tz.data, f.h.size());
        const bool combined       = f.form == Form::CombineNorm || f.form == Form::AddNorm;
        const auto seam_reference = combined
                                        ? grouped_norm(residual, f.weight, f.columns, f.epsilon)
                                        : std::vector<double>{};
        for (int m = 0; m < f.columns; ++m)
            for (int s = 0; s < kStreams; ++s) {
                const auto base = (std::size_t(m) * kStreams + s) * kWidth;
                const auto part = [&](const std::vector<double>& values) {
                    return std::span<const double>(values.data() + base, kWidth);
                };
                const auto tag =
                    label + " column=" + std::to_string(m) + " stream=" + std::to_string(s);
                failed += verify_reduction(tag, part(actual), part(ref.normalized), kNormalized);
                if (combined)
                    failed += verify_reduction(tag + " actual seam", part(actual),
                                               part(seam_reference), kNormalized);
            }
    }
    return failed;
}

namespace {
int run(Form form, int m, Pattern pattern, bool alias, bool captured, CudaExecutionView e) {
    Fixture f(form, m, alias);
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
        const auto label = std::string(name(form)) + " M=" + std::to_string(m) +
                           " pattern=" + std::to_string(int(pattern)) +
                           " alias=" + std::to_string(alias) +
                           " graph=" + std::to_string(captured) + " phase=" + std::to_string(phase);
        const bool exact_residual =
            pattern == Pattern::Midpoint && (form == Form::Combine || form == Form::CombineNorm);
        failed += verify(f, label, exact_residual) + f.check(label);
    }
    return failed;
}
} // namespace

void add_norm_cases(OpSuite& suite) {
    for (Form form :
         {Form::ExpandNorm, Form::GroupedNorm, Form::Combine, Form::CombineNorm, Form::AddNorm}) {
        const auto add = [&](int m, Pattern pattern, bool alias, bool graph) {
            const auto id = std::string(name(form)) + "/m" + std::to_string(m) + "/pattern" +
                            std::to_string(int(pattern)) + (alias ? "/alias" : "/disjoint") +
                            (graph ? "/graph" : "/eager");
            suite.add(id,
                      [=] {
                          return run(form, m, pattern, alias, graph,
                                     cuda_environment().context.execution_view());
                      },
                      {{"execution", graph ? "graph" : "eager"}});
        };
        for (int m : {1, 3, 8, 16, 31, 32, 33, 127, 128, 129, 512, 2048})
            add(m, Pattern::Random, false, false);
        for (auto pattern : {Pattern::Affine, Pattern::Zero, Pattern::Constant,
                             Pattern::Cancellation, Pattern::Midpoint, Pattern::Small})
            add(3, pattern, false, false);
        add(7 * 8, Pattern::Random, false, true);
        if (form == Form::ExpandNorm)
            for (int m : {2047, 2048, 2049})
                add(m, m == 2049 ? Pattern::Constant : Pattern::Random, false, m == 2048);
        if (form == Form::GroupedNorm)
            for (int m : {2815, 2816, 2817})
                add(m, m == 2817 ? Pattern::Constant : Pattern::Random, false, m == 2816);
        const bool combine =
            form == Form::Combine || form == Form::CombineNorm || form == Form::AddNorm;
        if (combine) {
            add(1, Pattern::Affine, true, false);
            add(129, Pattern::Random, true, false);
            add(7 * 8, Pattern::Midpoint, true, true);
        }
        if (form == Form::CombineNorm || form == Form::AddNorm) {
            for (int m : {1535, 1536, 1537, 1840, 1841, 1842, 3071, 3072, 3073})
                add(m, Pattern::Random, false, false);
            add(1664, Pattern::Midpoint, false, true);
            for (int m : {1823, 1824, 1825, 3679, 3680, 3681}) add(m, Pattern::Random, true, false);
            add(2048, Pattern::Random, false, true);
            add(4096, Pattern::Random, true, false);
        }
        suite.add(std::string(name(form)) + "/contract",
                  [=] { return validate(form, cuda_environment().context.execution_view()); });
    }
}
} // namespace ninfer::test::hc
