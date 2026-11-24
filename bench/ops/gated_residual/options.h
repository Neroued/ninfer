#pragma once
#include "op_measurement.cuh"
#include <limits>

namespace ninfer::bench::hc {
inline constexpr int kWidth   = 2560;
inline constexpr int kStreams = 4;
inline constexpr int kRank    = 320;

enum class Op {
    ExpandNorm,
    GroupedNorm,
    Combine,
    CombineNorm,
    AddNorm,
    CombineThenNorm,
    AddThenNorm,
    BlockMix,
    FinalMix
};

inline const char* name(Op op) {
    switch (op) {
    case Op::ExpandNorm:
        return "expand-norm";
    case Op::GroupedNorm:
        return "grouped-rmsnorm";
    case Op::Combine:
        return "combine";
    case Op::CombineNorm:
        return "combine-norm";
    case Op::AddNorm:
        return "add-norm";
    case Op::CombineThenNorm:
        return "combine-then-norm";
    case Op::AddThenNorm:
        return "add-then-norm";
    case Op::BlockMix:
        return "block-mix";
    case Op::FinalMix:
        return "final-mix";
    }
    throw std::logic_error("invalid HC benchmark Op");
}

inline bool mixes(Op op) { return op == Op::BlockMix || op == Op::FinalMix; }

inline bool combines(Op op) {
    return op == Op::Combine || op == Op::CombineNorm || op == Op::AddNorm ||
           op == Op::CombineThenNorm || op == Op::AddThenNorm;
}

struct Options {
    measurement::Options timing;
    Op op = Op::CombineNorm;
    std::vector<int> tokens{1};
    int batch        = 1;
    bool in_place    = false;
    int condition_us = 200;
};

inline int columns(int tokens, int batch) {
    const auto result = std::int64_t(tokens) * batch;
    if (result > std::numeric_limits<int>::max())
        throw std::invalid_argument("matrix columns overflow");
    return int(result);
}

inline Options parse(int argc, char** argv) {
    Options o;
    bool have_tokens = false, have_sweep = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "HC gated residual Op benchmark\n"
                   "  --op expand-norm|grouped-rmsnorm|combine|combine-norm|add-norm\n"
                   "       |combine-then-norm|add-then-norm|block-mix|final-mix\n"
                   "  --alias disjoint|inplace --batch B --condition-us N\n"
                   "  --tokens T[,T...] or --sweep START:END[:STEP]\n"
                   "H=2560, streams=4, M=tokens*batch. Staged comparisons time both public Ops.\n"
                   "GPU activity prelude: 200 us by default; 0 disables it. Outside Op timing.\n"
                   "In-place input restoration precedes timing and cold-cache eviction.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op") {
            bool found = false;
            for (auto op :
                 {Op::ExpandNorm, Op::GroupedNorm, Op::Combine, Op::CombineNorm, Op::AddNorm,
                  Op::CombineThenNorm, Op::AddThenNorm, Op::BlockMix, Op::FinalMix})
                if (value == name(op)) {
                    o.op  = op;
                    found = true;
                }
            if (!found) throw std::invalid_argument("unknown HC Op");
        } else if (flag == "--alias") {
            if (value != "disjoint" && value != "inplace")
                throw std::invalid_argument("alias: disjoint|inplace");
            o.in_place = value == "inplace";
        } else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--condition-us")
            o.condition_us = measurement::integer(value, 0);
        else if (flag == "--tokens") {
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
    if (o.in_place && !combines(o.op))
        throw std::invalid_argument("this Op has no in-place residual form");
    for (int t : o.tokens) (void)columns(t, o.batch);
    return o;
}
} // namespace ninfer::bench::hc
