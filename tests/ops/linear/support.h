#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear_common.h"
#include "ops/suite.h"

#include <functional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::test::linear_suite {

enum class Execution { Eager, Graph };

unsigned policy_bit(ops::LinearPolicy policy);
std::string_view policy_name(ops::LinearPolicy policy);
std::string_view format_name(QType type);

struct Case {
    std::string id;
    std::string format;
    int n, k, t;
    unsigned policies;
    Execution execution;
};

Case make_case(std::string_view format, int n, int k, int t, ops::LinearPolicy policy,
               Execution execution, std::string_view recipe);

// A group owns its fixture recipe. Only selected indices reach its runner; constructing a
// catalogue must not allocate tensors, generate weights, query Ops, or initialize CUDA.
class Suite {
public:
    using Run = std::function<int(std::span<const std::size_t>, std::span<const Case>)>;
    void add(std::vector<Case> cases, Run run);
    void add(Case test, std::function<int()> run);
    int main(int argc, char** argv) const;

private:
    struct Group {
        std::vector<Case> cases;
        Run run;
    };

    std::vector<Group> groups_;
};

} // namespace ninfer::test::linear_suite
