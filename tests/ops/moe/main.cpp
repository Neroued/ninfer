#include "fixture.h"

#include <tuple>

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test;
    OpSuite suite;
    suite.add("moe/weight_contract", moe::test_weights);
    for (const auto [gu, down, format] :
         {std::tuple{QType::Q4_G64_FP16, QType::Q5_G64_FP16, "q4_q5"},
          std::tuple{QType::Q4_G64_FP16, QType::Q6_G64_FP16, "q4_q6"},
          std::tuple{QType::Q8_G32_FP16, QType::Q8_G32_FP16, "q8"}}) {
        std::vector<moe::FixtureCase> cases;
        moe::add_complete_cases(cases, gu, down);
        if (down == QType::Q5_G64_FP16) moe::add_route_cases(cases);
        moe::add_routed_cases(cases, gu);
        std::vector<OpCase> catalogue;
        for (auto& c : cases) {
            c.metadata.id = "moe/" + std::string(format) + "/" + c.metadata.id;
            c.metadata.tags.emplace_back("format", format);
            catalogue.push_back(c.metadata);
        }
        suite.add(std::move(catalogue), [=](auto selected) {
            DeviceContext device;
            moe::Fixture fixture(gu, down);
            int failures = 0;
            for (auto i : selected) failures += cases[i].run(fixture, device.execution_view());
            return failures + fixture.verify_preserved();
        });
    }
    return suite.main(argc, argv);
}
