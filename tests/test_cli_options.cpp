#include "options.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ninfer::cli::Options defaults = parse({"ninfer", "model.ninfer", "--prompt", "hello"});
    failures += check(!defaults.enable_vision, "CLI Vision is not disabled by default");

    const ninfer::cli::Options enabled =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision"});
    failures += check(enabled.enable_vision, "--vision did not enable CLI Vision");

    failures += check(ninfer::cli::usage_text("ninfer").find("--vision") != std::string::npos,
                      "CLI help omits --vision");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
