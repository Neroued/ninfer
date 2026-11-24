#pragma once
#include "op_measurement.cuh"
#include <limits>

namespace ninfer::bench::gdn_control {
enum class Op { Control, Projection, NormProjection };
enum class Storage { Parent, Pair };

struct Options {
    measurement::Options timing;
    Op op           = Op::Projection;
    Storage storage = Storage::Parent;
    int heads = 48, input_rows = 2560, batch = 1;
    std::vector<int> tokens{1};
};

inline const char* name(Op op) {
    switch (op) {
    case Op::Control:
        return "control";
    case Op::Projection:
        return "projection";
    case Op::NormProjection:
        return "norm-projection";
    }
    throw std::logic_error("invalid control Op");
}

inline int columns(int tokens, int batch) {
    const auto m = std::int64_t(tokens) * batch;
    if (m > std::numeric_limits<int>::max()) throw std::invalid_argument("matrix columns overflow");
    return int(m);
}

inline Options parse(int argc, char** argv) {
    Options o;
    bool have_tokens = false, have_sweep = false, have_weights = false, have_rows = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "GDN control Op benchmark\n  --op control|projection|norm-projection\n"
                   "  --heads H --input-rows K --weights parent|pair --batch B\n"
                   "  --tokens M[,M...] or --sweep START:END[:STEP]\n"
                   "Projection [2H,K]: [96,5120], [64,2048], [96,2560]. Norm excludes K=2560.\n"
                   "Spark projection also admits [48,2560] (H=24).\n"
                   "Paired weights: [48,5120]. Direct control: H=48. Matrix M=tokens*batch.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op") {
            if (value == "control")
                o.op = Op::Control;
            else if (value == "projection")
                o.op = Op::Projection;
            else if (value == "norm-projection")
                o.op = Op::NormProjection;
            else
                throw std::invalid_argument("unknown control Op");
        } else if (flag == "--heads")
            o.heads = measurement::integer(value);
        else if (flag == "--input-rows") {
            o.input_rows = measurement::integer(value);
            have_rows    = true;
        } else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--weights") {
            have_weights = true;
            if (value == "parent")
                o.storage = Storage::Parent;
            else if (value == "pair")
                o.storage = Storage::Pair;
            else
                throw std::invalid_argument("weights: parent|pair");
        } else if (flag == "--tokens") {
            o.tokens    = measurement::integers(value);
            have_tokens = true;
        } else if (flag == "--sweep") {
            o.tokens   = measurement::sweep(value);
            have_sweep = true;
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (have_tokens && have_sweep)
        throw std::invalid_argument("tokens and sweep are mutually exclusive");
    if (o.op == Op::Control && (have_weights || have_rows))
        throw std::invalid_argument("direct control has no projection weights or input rows");
    for (int m : o.tokens) (void)columns(m, o.batch);
    return o;
}
} // namespace ninfer::bench::gdn_control
