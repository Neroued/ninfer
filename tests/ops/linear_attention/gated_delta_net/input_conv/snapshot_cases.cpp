#include "support.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gdn_input_conv;

namespace {
// Capacity-only checks need weight metadata, without allocating device payloads.
Weight capacity_weight(QType format, std::int32_t rows, std::int32_t columns) {
    Weight weight;
    weight.qtype = format;
    weight.n     = rows;
    weight.k     = columns;
    return weight;
}

std::vector<SnapshotCase> cases(Profile p) {
    using Policy = ops::LinearPolicy;
    std::vector<SnapshotCase> result;
    const auto batch = [&](int width, int count, std::vector<int> valid, unsigned seed,
                           Policy policy = Policy::A16Only) {
        result.push_back({.width     = width,
                          .policy    = policy,
                          .batch     = count,
                          .valid     = std::move(valid),
                          .conv_seed = seed});
    };
    if (p.qk_rows == 1024) {
        for (int t = 1; t <= 16; ++t)
            result.push_back({.width          = t,
                              .graph          = t == 1 || t == 6 || t == 16,
                              .full_reference = t == 1 || t == 6,
                              .batch          = 1});
        for (int t : {17,  31,  32,  33,  63,  64,  65,  95,   96,   97,   127,  128, 129,
                      191, 192, 193, 256, 511, 512, 513, 1023, 1024, 1025, 2048, 4096})
            result.push_back({.width = t, .graph = t == 33 || t == 512 || t == 4096, .batch = 1});
        for (int b = 2; b <= 8; ++b)
            for (int t = 1; t <= 16; ++t) {
                std::vector<int> valid(b);
                for (int i = 0; i < b; ++i) valid[i] = t - i % t;
                batch(t, b, std::move(valid), 739);
                result.back().graph = (b == 2 && t == 6) || (b == 8 && t == 16);
            }
        return result;
    }
    if (p.format == QType::Q4_G64_FP16) {
        for (int t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 33, 64})
            result.push_back(
                {.width = t, .initial = t == 5 ? 0 : t + 1, .full_reference = t >= 7 && t <= 10});
        batch(1, 8, {}, 631);
        batch(5, 2, {}, 631);
        batch(4, 2, {4, 2}, 631);
        batch(16, 3, {16, 9, 4}, 631);
    } else if (p.format == QType::Q8_G32_FP16) {
        const auto widths = p.hidden == 2048 ? std::vector{1, 2, 15, 16, 17}
                                             : std::vector{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                           11, 12, 13, 14, 15, 16, 17, 33, 129};
        for (int t : widths) result.push_back({.width = t, .initial = t == 2 ? 0 : t + 1});
        if (p.hidden == 2560)
            for (int t : {1, 8, 16, 17, 65, 192, 1024, 1025, 1152})
                result.push_back({.width = t, .graph = true});
        if (p.hidden == 2560)
            for (int t : {95, 96, 97, 127, 128, 191, 193, 1023, 1151, 1153, 4096})
                result.push_back({.width = t});
        for (int b : p.hidden == 2048 ? std::vector{2} : std::vector{2, 3, 4, 5, 6, 7, 8})
            for (int t : p.hidden == 2048
                             ? std::vector{16}
                             : std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
                std::vector<int> valid(b);
                for (int i = 0; i < b; ++i) valid[i] = t - i % t;
                batch(t, b, valid, 733);
            }
    } else if (p.format == QType::NVFP4) {
        result = {{1},
                  {3, Policy::AllowA4},
                  {4, Policy::AllowA4},
                  {17, Policy::AllowA4, 0},
                  {1024, Policy::AllowA4}};
        batch(6, 3, {6, 3, 1}, 829, Policy::AllowA4);
    } else {
        result = {{.width = 1, .convenience = true, .shared_selectors = true},
                  {3},
                  {4},
                  {6},
                  {7},
                  {9, Policy::AllowA8},
                  {10, Policy::AllowA8},
                  {10},
                  {11},
                  {16, Policy::AllowA8},
                  {17, Policy::AllowA8, 1},
                  {257, Policy::AllowA8},
                  {512, Policy::AllowA8},
                  {1025, Policy::AllowA8}};
        batch(4, 2, {4, 2}, 937, Policy::AllowA8);
        batch(16, 8, {16, 13, 11, 7, 5, 3, 2, 1}, 941, Policy::AllowA8);
    }
    return result;
}
} // namespace

namespace ninfer::test::gdn_input_conv {
int snapshot_capacity_contract() {
    const ops::PairedProjectionWeights q4_q5_capacity_weights{
        capacity_weight(QType::Q4_G64_FP16, 4096, 5120),
        capacity_weight(QType::Q5_G64_FP16, 12288, 5120)};
    const ops::SingleProjectionWeight q8_capacity_weights{
        capacity_weight(QType::Q8_G32_FP16, 12288, 2048)};

    int failures = 0;
    const std::size_t q4_interval =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(q4_q5_capacity_weights, 1, 1, 6);
    const std::size_t q4_witness =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(q4_q5_capacity_weights, 1, 4, 4);
    const std::size_t q4_right_endpoint =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(q4_q5_capacity_weights, 1, 6, 6);
    if (q4_interval != q4_witness || q4_witness == 0 || q4_right_endpoint != 0) {
        std::cerr << "Q4/Q5 snapshot interval did not retain its non-monotonic T=4 route\n";
        ++failures;
    }
    if (ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(q8_capacity_weights, 1, 1, 16) !=
            0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(q8_capacity_weights, 1, 1, 17) !=
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(q8_capacity_weights, 1, 17,
                                                                       17)) {
        std::cerr << "Q8 snapshot interval did not preserve its zero/nonzero route boundary\n";
        ++failures;
    }
    for (int batch = 1; batch <= 8; ++batch) {
        for (int first = 1; first <= 16; ++first) {
            std::size_t peak = 0;
            for (int last = first; last <= 16; ++last) {
                const auto capacity = [&](int begin, int end) {
                    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                        ops::SingleProjectionWeight{
                            capacity_weight(QType::Q8_G32_FP16, 16384, 2560),
                            ops::LinearPolicy::A16Only},
                        batch, begin, end);
                };
                peak = std::max(peak, capacity(last, last));
                if (capacity(first, last) != peak) {
                    std::cerr << "Q8 snapshot interval missed its interior high-water\n";
                    ++failures;
                }
            }
        }
    }
#if NINFER_TARGET_SPARK
    for (int b = 1; b <= 8; ++b)
        for (int first = 1; first <= 16; ++first) {
            std::size_t peak = 0;
            for (int last = first; last <= 16; ++last) {
                const auto capacity = [&](int lo, int hi) {
                    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                        ops::SingleProjectionWeight{
                            capacity_weight(QType::Q8_G32_FP16, 8192, 2560)},
                        b, lo, hi);
                };
                peak = std::max(peak, capacity(last, last));
                if (peak != capacity(first, last)) ++failures;
            }
        }
#endif
    const std::size_t nvfp4_a4_4 = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        ops::SingleProjectionWeight{capacity_weight(QType::NVFP4, 16384, 5120),
                                    ops::LinearPolicy::AllowA4},
        1, 4, 4);
    if (ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            ops::SingleProjectionWeight{capacity_weight(QType::NVFP4, 16384, 5120),
                                        ops::LinearPolicy::A16Only},
            1, 1, 16) != 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            ops::SingleProjectionWeight{capacity_weight(QType::NVFP4, 16384, 5120),
                                        ops::LinearPolicy::AllowA4},
            1, 1, 3) != 0 ||
        nvfp4_a4_4 == 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            ops::SingleProjectionWeight{capacity_weight(QType::NVFP4, 16384, 5120),
                                        ops::LinearPolicy::AllowA4},
            1, 1, 4) != nvfp4_a4_4) {
        std::cerr << "NVFP4 snapshot interval did not preserve its A16/A4 route boundary\n";
        ++failures;
    }
    const auto fp8_snapshot_capacity = [](ops::LinearPolicy policy, std::int32_t batch,
                                          std::int32_t min_width, std::int32_t max_width) {
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            ops::SingleProjectionWeight{capacity_weight(QType::FP8_E4M3FN_ROW_BF16, 16384, 5120),
                                        policy},
            batch, min_width, max_width);
    };
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        for (int batch : {1, 2, 8}) {
            std::size_t peak = 0;
            for (int width = 1; width <= 16; ++width) {
                peak = std::max(peak, fp8_snapshot_capacity(policy, batch, width, width));
                if (fp8_snapshot_capacity(policy, batch, 1, width) != peak) {
                    std::cerr << "FP8 snapshot interval misses an execution capacity\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

} // namespace ninfer::test::gdn_input_conv

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("gdn_input_snapshot/capacity", snapshot_capacity_contract);
    std::vector<Profile> selected_profiles(std::begin(profiles), std::end(profiles));
#if NINFER_TARGET_SPARK
    selected_profiles.push_back(local_profile);
#endif
    for (auto profile : selected_profiles) {
        const auto tests = cases(profile);
        std::vector<OpCase> catalogue;
        for (auto c : tests) {
            std::string id = std::string("gdn_input_snapshot/") + profile.name + "/k" +
                             std::to_string(profile.hidden) + "/t" + std::to_string(c.width) +
                             (c.batch ? "/batch" + std::to_string(c.batch) : "/single") + "/" +
                             policy_name(c.policy) + (c.graph ? "/graph" : "/eager");
            catalogue.push_back({id,
                                 {{"format", profile.name},
                                  {"policy", policy_name(c.policy)},
                                  {"execution", c.graph ? "graph" : "eager"}}});
        }
        suite.add(std::move(catalogue), [=](auto selected) {
            Weights weights(profile, false);
            int failures = 0;
            for (auto i : selected) failures += run_snapshot(weights, tests[i]);
            return failures + weights.preserved();
        });
    }
    return suite.main(argc, argv);
}
