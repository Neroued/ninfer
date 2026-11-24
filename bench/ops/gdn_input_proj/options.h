#pragma once

#include "op_measurement.cuh"
#include "ninfer/ops/gdn_input_proj.h"

namespace ninfer::bench::gdn_input {
struct Options {
    measurement::Options timing;
    std::string op = "snapshot", format = "q8", policy = "a16";
    int n = 16384, k = 2560, query_rows = 2048, batch = 1;
    std::vector<int> tokens{1}, valid;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "GDN input Op benchmark\n"
                         "  --op snapshot|record --format q4q5|q8|fp8|nvfp4\n"
                         "  --n N --k K --query-rows Q --policy a16|allow-a8|allow-a4\n"
                         "  --tokens T[,T...] or --sweep START:END[:STEP]\n"
                         "  --batch B --valid-columns T[,T...]\n"
                         "Uses physical T and B. Defaults: Q8 "
                         "[16384,2560], T=1.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--format")
            o.format = value;
        else if (flag == "--n")
            o.n = measurement::integer(value);
        else if (flag == "--k")
            o.k = measurement::integer(value);
        else if (flag == "--query-rows")
            o.query_rows = measurement::integer(value);
        else if (flag == "--policy")
            o.policy = value;
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--tokens") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::integers(value);
            explicit_tokens = true;
        } else if (flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::sweep(value);
            explicit_tokens = true;
        } else if (flag == "--valid-columns")
            o.valid = measurement::integers(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.op != "snapshot" && o.op != "record") throw std::invalid_argument("unknown input Op");
    if (o.policy != "a16" && o.policy != "allow-a8" && o.policy != "allow-a4")
        throw std::invalid_argument("unknown activation policy");
    if (o.format == "q4q5" && o.policy != "a16")
        throw std::invalid_argument("paired Q4/Q5 weights use A16");
    if (!o.valid.empty() && (o.tokens.size() != 1 || o.valid.size() != std::size_t(o.batch) ||
                             *std::max_element(o.valid.begin(), o.valid.end()) > o.tokens.front()))
        throw std::invalid_argument("valid columns must match exact T/B");
    if (o.format != "q4q5" && o.format != "q8" && o.format != "fp8" && o.format != "nvfp4")
        throw std::invalid_argument("unknown weight format");
    if (o.n <= 2LL * o.query_rows || (o.n - 2LL * o.query_rows) % 2 != 0)
        throw std::invalid_argument("fixture requires N=2*Q+2*V with positive V");
    if (o.format == "q4q5" && (o.n != 16384 || o.query_rows != 2048))
        throw std::invalid_argument("paired fixture uses Q4 rows [0,4096), Q5 rows [4096,16384)");
    return o;
}

inline ops::LinearPolicy policy(const Options& o) {
    return o.policy == "a16"        ? ops::LinearPolicy::A16Only
           : o.policy == "allow-a8" ? ops::LinearPolicy::AllowA8
                                    : ops::LinearPolicy::AllowA4;
}
} // namespace ninfer::bench::gdn_input
