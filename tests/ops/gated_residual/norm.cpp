#include "support.h"
#include "core/platform.h"
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
        const auto actual   = from_device_bf16(f.tz.data, f.h.size());
        const bool combined = f.form == Form::CombineNorm || f.form == Form::AddNorm;
        const auto seam_reference =
            combined ? grouped_norm(residual, f.weight, f.columns, f.epsilon, f.width)
                     : std::vector<double>{};
        for (int m = 0; m < f.columns; ++m)
            for (int s = 0; s < kStreams; ++s) {
                const auto base = (std::size_t(m) * kStreams + s) * f.width;
                const auto part = [&](const std::vector<double>& values) {
                    return std::span<const double>(values.data() + base, f.width);
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
int run(Form form, int m, Pattern pattern, bool alias, bool captured, CudaExecutionView e,
        int width = kWidth) {
    Fixture f(form, m, alias, width);
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
            pattern == Pattern::Midpoint &&
            (form == Form::Combine || form == Form::CombineNorm || form == Form::AddNorm);
        failed += verify(f, label, exact_residual) + f.check(label);
    }
    return failed;
}
} // namespace

void add_norm_cases(OpSuite& suite) {
#if NINFER_TARGET_RTX5090
    for (Form form :
         {Form::ExpandNorm, Form::GroupedNorm, Form::Combine, Form::CombineNorm, Form::AddNorm}) {
        const auto add = [&](int m, Pattern pattern, bool alias = false, bool graph = false) {
            const auto id = std::string(name(form)) + "/h5120/m" + std::to_string(m) + "/pattern" +
                            std::to_string(int(pattern)) + (alias ? "/alias" : "/disjoint") +
                            (graph ? "/graph" : "/eager");
            suite.add(
                id,
                [=] {
                    return run(form, m, pattern, alias, graph,
                               cuda_environment().context.execution_view(), 5120);
                },
                {{"execution", graph ? "graph" : "eager"}, {"format", "bf16"}, {"width", "5120"}});
        };
        for (int m : {1, 128, 512, 2048}) add(m, Pattern::Random);
        for (auto pattern : {Pattern::Affine, Pattern::Zero, Pattern::Constant,
                             Pattern::Cancellation, Pattern::Midpoint, Pattern::Small})
            add(3, pattern);
        add(6 * 8, Pattern::Random, false, true);
        if (form != Form::Combine) add(1, Pattern::Gain);
        if (form == Form::Combine || form == Form::CombineNorm || form == Form::AddNorm) {
            add(129, Pattern::Random, true);
            add(3 * 2, Pattern::Midpoint, true, true);
        }
        if (form == Form::CombineNorm || form == Form::AddNorm) {
            // Each case covers both sides of one dispatch boundary, without
            // multiplying the numerical-pattern and extent dimensions.
            for (const auto [boundary, alias] :
                 {std::pair{256, true}, std::pair{640, false}, std::pair{832, true}}) {
                suite.add(std::string(name(form)) + "/h5120/boundary" + std::to_string(boundary) +
                              (alias ? "/alias" : "/disjoint"),
                          [=] {
                              int failed = 0;
                              for (int m : {boundary - 1, boundary, boundary + 1})
                                  failed += run(form, m, Pattern::Random, alias, false,
                                                cuda_environment().context.execution_view(), 5120);
                              return failed;
                          },
                          {{"width", "5120"}, {"format", "bf16"}, {"execution", "eager"}});
            }
            add(1024, Pattern::Midpoint, true, true);
        }
        suite.add(std::string(name(form)) + "/h5120/contract",
                  [=] { return validate(form, cuda_environment().context.execution_view(), 5120); },
                  {{"width", "5120"}, {"format", "bf16"}});
    }
#endif
    for (Form form :
         {Form::ExpandNorm, Form::GroupedNorm, Form::Combine, Form::CombineNorm, Form::AddNorm}) {
        const auto add = [&](int m, Pattern pattern, bool alias, bool graph) {
            const auto id = std::string(name(form)) + "/h2560/m" + std::to_string(m) + "/pattern" +
                            std::to_string(int(pattern)) + (alias ? "/alias" : "/disjoint") +
                            (graph ? "/graph" : "/eager");
            suite.add(id,
                      [=] {
                          return run(form, m, pattern, alias, graph,
                                     cuda_environment().context.execution_view());
                      },
                      {{"execution", graph ? "graph" : "eager"}, {"width", "2560"}});
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
            for (int m : {1023, 1024, 1025}) {
                add(m, Pattern::Random, false, false);
                add(m, Pattern::Random, true, false);
            }
            for (int m : {1535, 1536, 1537, 1840, 1841, 1842, 3071, 3072, 3073})
                add(m, Pattern::Random, false, false);
            add(1664, Pattern::Midpoint, false, true);
            for (int m : {1823, 1824, 1825, 3679, 3680, 3681}) add(m, Pattern::Random, true, false);
            add(2048, Pattern::Random, false, true);
            add(4096, Pattern::Random, true, false);
        }
        suite.add(std::string(name(form)) + "/h2560/contract",
                  [=] { return validate(form, cuda_environment().context.execution_view()); },
                  {{"width", "2560"}});
    }
}
} // namespace ninfer::test::hc
