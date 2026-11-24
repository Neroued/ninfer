#include "ops/suite.h"

#include "core/cpu/task_pool.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>

namespace ninfer::test {

void OpSuite::add(std::string id, std::function<int()> run,
                  std::vector<std::pair<std::string, std::string>> tags) {
    add(OpCase{std::move(id), std::move(tags)}, std::move(run));
}

void OpSuite::add(OpCase test, std::function<int()> run) {
    add(std::vector<OpCase>{std::move(test)}, [run = std::move(run)](auto) { return run(); });
}

void OpSuite::add(std::vector<OpCase> cases, Run run) {
    if (cases.empty()) throw std::invalid_argument("empty Op case group");
    for (auto& test : cases) {
        test.tags.emplace_back("suite", test.id.substr(0, test.id.find('/')));
        if (std::none_of(test.tags.begin(), test.tags.end(),
                         [](const auto& tag) { return tag.first == "execution"; }))
            test.tags.emplace_back("execution", "eager");
    }
    groups_.push_back({std::move(cases), std::move(run)});
}

void OpSuite::filter(std::string name, int arguments) {
    if (arguments < 1) throw std::invalid_argument("filter needs at least one argument");
    filters_.emplace_back(std::move(name), arguments);
}

int OpSuite::main(int argc, char** argv) const {
    try {
        bool list = false;
        std::map<std::string, std::string> filters;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "Usage: " << argv[0] << " [--list] [--case ID]";
                for (const auto& [name, count] : filters_) {
                    std::cout << " [--" << name;
                    for (int j = 0; j < count; ++j) std::cout << " VALUE";
                    std::cout << ']';
                }
                std::cout
                    << "\nFilters intersect declared cases. Graph sequences are indivisible.\n";
                return 0;
            }
            if (option == "--list") {
                list = true;
                continue;
            }
            const auto found = std::find_if(filters_.begin(), filters_.end(), [&](const auto& f) {
                return option == "--" + f.first;
            });
            if (option != "--case" && found == filters_.end())
                throw std::invalid_argument("unknown option: " + option);
            const int count = option == "--case" ? 1 : found->second;
            std::string value;
            for (int j = 0; j < count; ++j) {
                if (++i == argc) throw std::invalid_argument("missing value for " + option);
                if (j) value += ' ';
                value += argv[i];
            }
            if (!filters.emplace(option.substr(2), std::move(value)).second)
                throw std::invalid_argument("repeated filter: " + option);
        }

        std::set<std::string> ids;
        std::vector<std::vector<std::size_t>> selected(groups_.size());
        std::size_t count = 0;
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            for (std::size_t c = 0; c < groups_[g].cases.size(); ++c) {
                const auto& test = groups_[g].cases[c];
                if (test.id.empty() || !ids.insert(test.id).second)
                    throw std::logic_error("empty or duplicate case ID: " + test.id);
                const bool matches =
                    std::all_of(filters.begin(), filters.end(), [&](const auto& f) {
                        if (f.first == "case") return f.second == test.id;
                        return std::any_of(
                            test.tags.begin(), test.tags.end(), [&](const auto& tag) {
                                return tag.first == f.first && tag.second == f.second;
                            });
                    });
                if (!matches) continue;
                selected[g].push_back(c);
                ++count;
                if (list) std::cout << test.id << '\n';
            }
        }
        if (!count) throw std::invalid_argument("no declared cases match the filters");
        if (list) return 0;
        if (needs_cuda_ && cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }

        const auto start    = std::chrono::steady_clock::now();
        int failures        = 0;
        std::size_t skipped = 0;
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            if (selected[g].empty()) continue;
            const auto& id = groups_[g].cases[selected[g].front()].id;
            try {
                const int result = groups_[g].run(selected[g]);
                if (result) std::cerr << "FAIL group starting at " << id << '\n';
                failures += result != 0;
            } catch (const OpTestSkip& reason) {
                skipped += selected[g].size();
                std::cout << "SKIP " << id << ": " << reason.what() << '\n';
            } catch (const std::exception& error) {
                std::cerr << "FAIL group starting at " << id << ": " << error.what() << '\n';
                ++failures;
            }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << (failures           ? "FAIL "
                      : skipped == count ? "SKIP "
                                         : "OK ")
                  << count - skipped << " cases, " << skipped << " skipped, " << seconds << " s\n";
        return failures ? 1 : skipped == count ? 77 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

void parallel_rows(int rows, std::uint64_t work, const std::function<void(int, int)>& run) {
    static thread_local bool in_worker = false;
    if (rows < 2 || work < 262144 || in_worker) {
        run(0, rows);
        return;
    }
    const unsigned threads = std::min(8U, std::max(1U, std::thread::hardware_concurrency()));
    static TaskPool pool({.threads = threads, .queue_capacity = threads});
    const int parts = std::min(rows, static_cast<int>(threads));
    std::vector<std::future<void>> results;
    results.reserve(parts);
    try {
        for (int part = 0; part < parts; ++part) {
            auto task = std::make_shared<std::packaged_task<void()>>([&, part] {
                struct WorkerScope {
                    WorkerScope() { in_worker = true; }
                    ~WorkerScope() { in_worker = false; }
                } scope;
                run(static_cast<std::int64_t>(rows) * part / parts,
                    static_cast<std::int64_t>(rows) * (part + 1) / parts);
            });
            results.push_back(task->get_future());
            pool.enqueue([task] { (*task)(); });
        }
    } catch (...) {
        for (auto& result : results) result.wait();
        throw;
    }
    for (auto& result : results) result.wait();
    for (auto& result : results) result.get();
}

} // namespace ninfer::test
