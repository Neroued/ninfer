#pragma once
#include "op_measurement.cuh"

namespace ninfer::bench::rmsnorm {
enum class Op { Plain, Offset, Silu, Sigmoid };

struct Options {
    measurement::Options timing;
    Op op         = Op::Plain;
    int dimension = 128, heads = 48, batch = 1;
    int reserve_tokens = 0, condition_us = 0;
    float epsilon = 1.0e-6F;
    std::vector<int> tokens{1};
};

inline const char* name(Op op) {
    switch (op) {
    case Op::Plain:
        return "rmsnorm";
    case Op::Offset:
        return "offset";
    case Op::Silu:
        return "silu-gated";
    case Op::Sigmoid:
        return "sigmoid-gated";
    }
    throw std::logic_error("invalid RMSNorm Op");
}

inline Options parse(int argc, char** argv) {
    Options o;
    bool have_tokens = false, have_sweep = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "RMSNorm Op benchmark\n  --op rmsnorm|offset|silu-gated|sigmoid-gated\n"
                   "  --dim D --heads H --batch B --tokens M[,M...] or --sweep START:END[:STEP]\n"
                   "  --reserve-tokens M --condition-us N --epsilon E\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op") {
            if (value == "rmsnorm")
                o.op = Op::Plain;
            else if (value == "offset")
                o.op = Op::Offset;
            else if (value == "silu-gated")
                o.op = Op::Silu;
            else if (value == "sigmoid-gated")
                o.op = Op::Sigmoid;
            else
                throw std::invalid_argument("unknown RMSNorm Op");
        } else if (flag == "--dim")
            o.dimension = measurement::integer(value);
        else if (flag == "--heads")
            o.heads = measurement::integer(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--reserve-tokens")
            o.reserve_tokens = measurement::integer(value);
        else if (flag == "--condition-us")
            o.condition_us = measurement::integer(value, 0);
        else if (flag == "--epsilon") {
            std::size_t end;
            o.epsilon = std::stof(std::string(value), &end);
            if (end != value.size() || !(o.epsilon > 0) || !std::isfinite(o.epsilon))
                throw std::invalid_argument("epsilon must be positive and finite");
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
    for (int m : o.tokens)
        (void)Tensor::metadata(DType::BF16, {o.dimension, o.heads, m, o.batch}).numel();
    return o;
}
} // namespace ninfer::bench::rmsnorm
