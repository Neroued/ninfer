#include "support.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gdn_input_conv;

namespace {
std::vector<int> ragged(int width, int batch) {
    std::vector<int> valid(batch);
    for (int b = 0; b < batch; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
    return valid;
}

std::vector<RecordCase> cases(Profile p) {
    using Policy = ops::LinearPolicy;
    std::vector<RecordCase> result;
    if (p.qk_rows == 1024) {
        for (int b = 1; b <= 8; ++b)
            for (int t = 2; t <= 16; ++t) {
                result.push_back({t, b, {}, 1801U + t + b});
                auto valid = ragged(t, b);
                if (b == 1) valid[0] = t - 1;
                result.push_back({t, b, std::move(valid), 1901U + t + b});
            }
        return result;
    }
    if (p.format == QType::Q4_G64_FP16) {
        for (int t = 2; t <= 16; ++t) {
            result.push_back({t, 1, {}, 1400U + t});
            result.push_back({t, 8, ragged(t, 8), 1450U + t});
        }
        result.push_back({5, 3, {5, 3, 1}, 1491});
        result.push_back({4, 4, {4, 3, 2, 1}, 1492});
        result.push_back({4, 2, {}, 1493});
        result.push_back({4, 2, {4, 2}, 1494});
    } else if (p.format == QType::Q8_G32_FP16) {
        result = {{2, 1, {1}, 1511}, {16, 1, {}, 1521}, {16, 8, {16, 13, 9, 7, 5, 3, 2, 1}, 1531}};
        if (p.hidden == 2560)
            for (int b = 1; b <= 8; ++b)
                for (int t = 2; t <= 16; ++t) {
                    std::vector<int> valid(b);
                    for (int row = 0; row < b; ++row) valid[row] = t - row % t;
                    result.push_back({t, b, valid, 1541U + t + b});
                }
    } else {
        const bool fp8 = p.format == QType::FP8_E4M3FN_ROW_BF16;
        for (auto policy : {Policy::A16Only, fp8 ? Policy::AllowA8 : Policy::AllowA4}) {
            for (int t = 2; t <= 16; ++t) {
                result.push_back({t, 1, {}, (fp8 ? 1700U : 1600U) + t, policy});
                result.push_back({t, 8, ragged(t, 8), (fp8 ? 1750U : 1650U) + t, policy});
            }
            if (fp8)
                for (int b : {2, 3, 4}) result.push_back({4, b, ragged(4, b), 1810U + b, policy});
        }
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
#if NINFER_TARGET_SPARK
    suite.add("gdn_input_record/local_capacity", [] {
        Weight metadata;
        metadata.qtype = QType::Q8_G32_FP16;
        metadata.n     = 8192;
        metadata.k     = 2560;
        const ops::SingleProjectionWeight weights{metadata};
        int failures = 0;
        for (int b = 1; b <= 8; ++b) {
            for (int first = 2; first <= 16; ++first)
                for (int last = first; last <= 16; ++last)
                    if (ops::gdn_input_proj_conv_record_workspace_capacity_bytes(weights, b, first,
                                                                                 last) != 0)
                        ++failures;
            try {
                (void)ops::gdn_input_proj_conv_record_workspace_capacity_bytes(weights, b, 1, 16);
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
        return failures;
    });
#endif
    std::vector<Profile> selected_profiles(std::begin(profiles), std::end(profiles));
#if NINFER_TARGET_SPARK
    selected_profiles.push_back(local_profile);
#endif
    for (auto profile : selected_profiles) {
        const auto tests = cases(profile);
        std::vector<OpCase> catalogue;
        for (auto c : tests) {
            std::string id = std::string("gdn_input_record/") + profile.name + "/k" +
                             std::to_string(profile.hidden) + "/w" + std::to_string(c.width) +
                             "/b" + std::to_string(c.batch) + "/" + policy_name(c.policy) +
                             "/valid";
            for (int v : c.valid) id += "_" + std::to_string(v);
            const bool graph = c.width == 2 || c.width == 9 || c.width == 10 || c.width == 16;
            catalogue.push_back({id,
                                 {{"format", profile.name},
                                  {"policy", policy_name(c.policy)},
                                  {"execution", graph ? "graph" : "eager"}}});
        }
        suite.add(std::move(catalogue), [=](auto selected) {
            Weights weights(profile, true);
            int failures = 0;
            for (auto i : selected) failures += run_record(weights, tests[i]);
            return failures + weights.preserved();
        });
    }
    return suite.main(argc, argv);
}
