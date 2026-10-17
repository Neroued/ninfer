#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::test {

class OpTestSkip : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct OpCase {
    std::string id;
    std::vector<std::pair<std::string, std::string>> tags;
};

// The catalogue contains metadata only. Fixtures and device queries belong to runners;
// grouped runners receive selected indices so immutable fixtures can be reused.
class OpSuite {
public:
    using Run = std::function<int(std::span<const std::size_t>)>;

    explicit OpSuite(bool needs_cuda = true) : needs_cuda_(needs_cuda) {}

    void add(std::string id, std::function<int()> run,
             std::vector<std::pair<std::string, std::string>> tags = {});
    void add(OpCase test, std::function<int()> run);
    void add(std::vector<OpCase> cases, Run run);
    void filter(std::string name, int arguments = 1);
    int main(int argc, char** argv) const;

private:
    struct Group {
        std::vector<OpCase> cases;
        Run run;
    };

    std::vector<Group> groups_;
    std::vector<std::pair<std::string, int>> filters_{
        {"format", 1}, {"policy", 1}, {"execution", 1}, {"suite", 1}};
    bool needs_cuda_;
};

// Parallelize independent outputs, retaining each output's accumulation order. Nested calls
// execute serially so oracles can compose without oversubscription or waiting on their own pool.
void parallel_rows(int rows, std::uint64_t work, const std::function<void(int, int)>& run);

} // namespace ninfer::test
