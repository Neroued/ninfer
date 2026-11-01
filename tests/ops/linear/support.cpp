#include "ops/linear/support.h"

namespace ninfer::test::linear_suite {

unsigned policy_bit(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return 1;
    case ops::LinearPolicy::AllowA8:
        return 2;
    case ops::LinearPolicy::AllowA4:
        return 4;
    }
    throw std::invalid_argument("unknown Linear policy");
}

std::string_view policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return "a16";
    case ops::LinearPolicy::AllowA8:
        return "a8";
    case ops::LinearPolicy::AllowA4:
        return "a4";
    }
    throw std::invalid_argument("unknown Linear policy");
}

std::string_view format_name(QType type) {
    switch (type) {
    case QType::BF16:
        return "bf16";
    case QType::Q4_G64_FP16:
        return "q4";
    case QType::Q5_G64_FP16:
        return "q5";
    case QType::Q6_G64_FP16:
        return "q6";
    case QType::Q8_G32_FP16:
        return "q8";
    case QType::FP8_E4M3FN_ROW_BF16:
        return "fp8";
    case QType::NVFP4:
        return "nvfp4";
    default:
        throw std::invalid_argument("unknown Linear fixture format");
    }
}

Case make_case(std::string_view format, int n, int k, int t, ops::LinearPolicy policy,
               Execution execution, std::string_view recipe) {
    std::string id = std::string(format) + "/" + std::to_string(n) + "x" + std::to_string(k) + "/" +
                     std::string(recipe) + "/" + std::string(policy_name(policy)) + "/t" +
                     std::to_string(t) + (execution == Execution::Graph ? "/graph" : "/eager");
    return {std::move(id), std::string(format), n, k, t, policy_bit(policy), execution};
}

void Suite::add(std::vector<Case> cases, Run run) {
    if (cases.empty()) throw std::invalid_argument("empty Linear case group");
    groups_.push_back({std::move(cases), std::move(run)});
}

void Suite::add(Case test, std::function<int()> run) {
    add(std::vector<Case>{std::move(test)}, [run = std::move(run)](auto, auto) { return run(); });
}

int Suite::main(int argc, char** argv) const {
    OpSuite suite;
    suite.filter("shape", 2);
    suite.filter("t");
    for (const auto& group : groups_) {
        std::vector<OpCase> cases;
        for (const auto& test : group.cases) {
            OpCase entry{test.id,
                         {{"format", test.format},
                          {"shape", std::to_string(test.n) + " " + std::to_string(test.k)},
                          {"t", std::to_string(test.t)},
                          {"execution", test.execution == Execution::Graph ? "graph" : "eager"}}};
            for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                ops::LinearPolicy::AllowA4})
                if (test.policies & policy_bit(policy))
                    entry.tags.emplace_back("policy", policy_name(policy));
            cases.push_back(std::move(entry));
        }
        suite.add(std::move(cases),
                  [&group](auto selected) { return group.run(selected, group.cases); });
    }
    return suite.main(argc, argv);
}

} // namespace ninfer::test::linear_suite
