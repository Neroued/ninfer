#pragma once

#include "op_measurement.cuh"

#include <limits>

namespace ninfer::bench::ple {

struct Options {
    measurement::Options timing;
    std::string op = "conv-add-norm", valid = "dense", accepted = "dense";
    std::string lookup_shards = "lower";
    std::vector<int> tokens{1};
    int batch = 1, offset = 0;
    bool inplace = false;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false, lookup_option = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "PLE lookup, gate, convolution and state extraction benchmark\n"
                         "  --op lookup|gate|conv-add|conv-add-norm|state-extract\n"
                         "  --lookup-shards lower|upper|alternating|all|none (lookup only)\n"
                         "  --tokens T[,T...] or --sweep START:END[:STEP] --batch B\n"
                         "  --valid dense|mixed|N[,N...] --offset N (gate/convolution)\n"
                         "  --accepted dense|mixed|N[,N...] (state extraction) --inplace\n"
                         "H=2560, streams=4, history=9, B=1..8. Defaults: conv-add-norm, T=B=1.\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--inplace") {
            o.inplace = true;
            continue;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--tokens" || flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            explicit_tokens = true;
            o.tokens =
                flag == "--tokens" ? measurement::integers(value) : measurement::sweep(value);
        } else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--offset")
            o.offset = measurement::integer(value, 0);
        else if (flag == "--valid")
            o.valid = value;
        else if (flag == "--accepted")
            o.accepted = value;
        else if (flag == "--lookup-shards") {
            o.lookup_shards = value;
            lookup_option   = true;
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.batch > 8 || (o.op != "gate" && o.op != "conv-add" && o.op != "conv-add-norm" &&
                        o.op != "state-extract" && o.op != "lookup"))
        throw std::invalid_argument("unsupported Op/batch");
    if (o.op == "gate" && o.inplace) throw std::invalid_argument("gate has no in-place output");
    if ((lookup_option && o.op != "lookup") ||
        (o.op == "lookup" && (o.inplace || o.offset || o.accepted != "dense")))
        throw std::invalid_argument("options do not belong to lookup");
    if (o.lookup_shards != "lower" && o.lookup_shards != "upper" &&
        o.lookup_shards != "alternating" && o.lookup_shards != "all" && o.lookup_shards != "none")
        throw std::invalid_argument("invalid lookup shard placement");
    if ((o.op == "state-extract" && o.offset != 0) ||
        (o.op != "state-extract" && o.accepted != "dense"))
        throw std::invalid_argument(
            "offset applies to gate/convolution; accepted applies to extraction");
    if (*std::max_element(o.tokens.begin(), o.tokens.end()) >
        std::numeric_limits<int>::max() - o.offset)
        throw std::invalid_argument("record capacity exceeds the Tensor extent domain");
    return o;
}

inline std::vector<int> extents(const std::string& mode, int t, int b) {
    std::vector<int> result;
    if (mode == "dense")
        result.assign(b, t);
    else if (mode == "mixed") {
        for (int i = 0; i < b; ++i) result.push_back(i == 0 ? 0 : std::max(1, t - i % t));
    } else
        result = measurement::integers(mode, 0);
    if (result.size() != std::size_t(b) || *std::max_element(result.begin(), result.end()) > t)
        throw std::invalid_argument("prefix lengths must fit T/B");
    return result;
}

inline std::string csv_list(const std::vector<int>& values) {
    std::string result;
    for (int x : values) {
        if (!result.empty()) result += ':';
        result += std::to_string(x);
    }
    return result;
}

} // namespace ninfer::bench::ple
