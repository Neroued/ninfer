#include "ops/linear/support.h"

#include "core/cpu/task_pool.h"

#include <charconv>
#include <future>
#include <map>
#include <optional>
#include <thread>

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
    // Preserve repeated invocations in the original tables without conflating their identity.
    std::map<std::string, unsigned> occurrences;
    for (auto& test : cases) {
        const auto occurrence = ++occurrences[test.id];
        if (occurrence > 1) test.id += "/repeat" + std::to_string(occurrence);
    }
    groups_.push_back({std::move(cases), std::move(run)});
}

void Suite::add(Case test, std::function<int()> run) {
    add(std::vector<Case>{std::move(test)}, [run = std::move(run)](auto, auto) { return run(); });
}

int Suite::main(int argc, char** argv) const {
    try {
        bool list = false;
        std::optional<std::string_view> format, id;
        std::optional<unsigned> policy;
        std::optional<int> n, k, t;
        std::optional<Execution> execution;
        const auto positive = [](std::string_view value) {
            int result = 0;
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), result);
            if (error != std::errc{} || end != value.data() + value.size() || result <= 0)
                throw std::invalid_argument("expected a positive integer: " + std::string(value));
            return result;
        };
        for (int i = 1; i < argc; ++i) {
            const std::string_view option(argv[i]);
            const auto value = [&]() -> std::string_view {
                if (++i == argc)
                    throw std::invalid_argument("missing value for " + std::string(option));
                return argv[i];
            };
            if (option == "--help") {
                std::cout << "Usage: " << argv[0]
                          << " [--list] [--format FORMAT] [--policy a16|a8|a4]"
                             " [--shape N K] [--t T] [--case ID] [--execution eager|graph]\n"
                             "Filters select declared cases. Graph sequences are indivisible.\n";
                return 0;
            } else if (option == "--list")
                list = true;
            else if (option == "--format")
                format = value();
            else if (option == "--case")
                id = value();
            else if (option == "--shape") {
                n = positive(value());
                k = positive(value());
            } else if (option == "--t")
                t = positive(value());
            else if (option == "--policy") {
                const auto name = value();
                if (name == "a16")
                    policy = 1;
                else if (name == "a8")
                    policy = 2;
                else if (name == "a4")
                    policy = 4;
                else
                    throw std::invalid_argument("policy must be a16, a8, or a4");
            } else if (option == "--execution") {
                const auto name = value();
                if (name == "eager")
                    execution = Execution::Eager;
                else if (name == "graph")
                    execution = Execution::Graph;
                else
                    throw std::invalid_argument("execution must be eager or graph");
            } else
                throw std::invalid_argument("unknown option: " + std::string(option));
        }
        std::vector<std::vector<std::size_t>> selected(groups_.size());
        std::map<std::string, bool> ids;
        std::size_t count = 0;
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            for (std::size_t c = 0; c < groups_[g].cases.size(); ++c) {
                const auto& test = groups_[g].cases[c];
                if (!ids.emplace(test.id, true).second)
                    throw std::logic_error("duplicate case ID: " + test.id);
                if ((format && *format != test.format) || (id && *id != test.id) ||
                    (policy && !(test.policies & *policy)) || (n && *n != test.n) ||
                    (k && *k != test.k) || (t && *t != test.t) ||
                    (execution && *execution != test.execution))
                    continue;
                selected[g].push_back(c);
                ++count;
                if (list) std::cout << test.id << '\n';
            }
        }
        if (count == 0) throw std::invalid_argument("no declared cases match the filters");
        if (list) return 0;
        return linear_common::run_test_main(argv[0], [&] {
            int failures = 0;
            for (std::size_t g = 0; g < groups_.size(); ++g) {
                if (selected[g].empty()) continue;
                try {
                    failures += groups_[g].run(selected[g], groups_[g].cases);
                } catch (const std::exception& error) {
                    std::cerr << groups_[g].cases[selected[g].front()].id << ": " << error.what()
                              << '\n';
                    ++failures;
                }
            }
            std::cout << count << " cases selected\n";
            return failures;
        });
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

void parallel_rows(int rows, std::uint64_t work, const std::function<void(int, int)>& run) {
    if (rows < 2 || work < 262144) {
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
    // Wait for all borrowers of run before propagating an exception.
    for (auto& result : results) result.wait();
    for (auto& result : results) result.get();
}

} // namespace ninfer::test::linear_suite
