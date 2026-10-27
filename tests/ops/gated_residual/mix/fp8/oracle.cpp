#include "support.h"

namespace ninfer::test::hc::mix::fp8 {
Ref reference(const Weights& w, std::span<const float> z) {
    Ref ref{std::vector<double>(Catalog * H), std::vector<double>(Catalog * 4)};
    parallel_rows(Catalog, std::uint64_t(Catalog) * R * K, [&](int first, int last) {
        for (int c = first; c < last; ++c) {
            std::vector<double> u(R);
            for (int r = 0; r < R; ++r) {
                double a = 0;
                for (int k = 0; k < K; ++k)
                    a += double(w.wd[std::size_t(r) * K + k]) * z[c * K + k];
                a *= .25;
                u[r] = a / (1. + std::exp(-a));
            }
            for (int d = 0; d < H; ++d) {
                double sum = 0;
                for (int s = 0; s < 4; ++s) {
                    double g = 0;
                    for (int r = 0; r < R; ++r)
                        g += double(w.wu[std::size_t(s * H + d) * R + r]) * u[r];
                    sum += z[c * K + s * H + d] / (1. + std::exp(-g));
                }
                ref.output[c * H + d] = sum * .25;
            }
            for (int s = 0; s < 4; ++s) {
                double b = 0;
                for (int k = 0; k < K; ++k)
                    b += double(bf16_to_f32(w.inject.bits[s * K + k])) * z[c * K + k];
                ref.injection[c * 4 + s] = 2. / (1. + std::exp(-b * .25));
            }
        }
    });
    return ref;
}

int Fixture::check(std::span<const float>, const Ref& ref, const std::string& label) const {
    const auto actual = from_device_bf16(x.data, std::size_t(H) * m);
    // Two permitted A8 activation encodings each have E4M3 unit roundoff 1/16.
    // A16 staging uses only BF16; the criteria are selected by public policy.
    const ReductionCriterion criterion =
        policy == ops::LinearPolicy::A16Only && up_policy == ops::LinearPolicy::A16Only
            ? ReductionCriterion{.008, 1.e-5, .016}
            : ReductionCriterion{.025, 1.e-5, .08};
    int failed = 0;
    for (int c = 0; c < m; ++c)
        failed += verify_reduction(label + " column=" + std::to_string(c),
                                   std::span(actual).subspan(c * H, H),
                                   std::span(ref.output).subspan((c % Catalog) * H, H), criterion);
    if (!final) {
        auto actual_j = from_device<float>(j.data, std::size_t(4) * m);
        std::vector<double> jr(actual_j.begin(), actual_j.end()), expected(jr.size());
        for (int c = 0; c < m; ++c)
            for (int s = 0; s < 4; ++s) expected[c * 4 + s] = ref.injection[(c % Catalog) * 4 + s];
        failed += verify_pointwise(label + " injection", jr, expected, kMixInjection);
    }
    failed += verify_exact((label + " z preserved").c_str(),
                           from_device<std::uint16_t>(dz.data(), words.size()), words);
    failed += dz.verify_guards(label) + dx.verify_guards(label) + dj.verify_guards(label) +
              scratch.verify_guards(label);
    if (arena.used() != 0) {
        std::cerr << label << " arena cursor leaked\n";
        ++failed;
    }
    return failed;
}
} // namespace ninfer::test::hc::mix::fp8
