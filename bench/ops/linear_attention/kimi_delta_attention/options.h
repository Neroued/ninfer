#pragma once
#include "op_measurement.cuh"

namespace ninfer::bench::kda {
struct Options {
    measurement::Options timing;
    std::string op = "record", valid = "dense", commits = "dense";
    std::vector<int> tokens{4};
    int heads = 32, qk_heads = 32, batch = 1, layers = 34;
    bool alias_records = true;
};

inline Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout << "KDA public Ops benchmark\n"
                         "  --op core|update|record|fold|input-update|input-record|chain\n"
                         "  --tokens T[,T...] --batch B --layers L --qk-heads H --value-heads H\n"
                         "  --valid dense|N[,N...] --commits dense|N[,N...]\n"
                         "  --kv-records alias|copy (record only; default alias)\n"
                         "core is single sequence; update is T=1; fold measures all layers.\n"
                         "chain measures one input-record, recurrence-record and one-layer fold.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value, 1, 8);
        else if (flag == "--layers")
            o.layers = measurement::integer(value);
        else if (flag == "--qk-heads")
            o.qk_heads = measurement::integer(value);
        else if (flag == "--value-heads")
            o.heads = measurement::integer(value);
        else if (flag == "--valid")
            o.valid = value;
        else if (flag == "--commits")
            o.commits = value;
        else if (flag == "--kv-records") {
            if (value != "alias" && value != "copy")
                throw std::invalid_argument("kv-records: alias|copy");
            o.alias_records = value == "alias";
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.op != "core" && o.op != "update" && o.op != "record" && o.op != "fold" &&
        o.op != "input-update" && o.op != "input-record" && o.op != "chain")
        throw std::invalid_argument("invalid op");
    if (o.heads < o.qk_heads || o.heads % o.qk_heads || (o.op == "core" && o.batch != 1))
        throw std::invalid_argument("invalid geometry");
    return o;
}

inline std::vector<int> counts(const std::string& value, int width, int batch, bool commit) {
    auto out = value == "dense" ? std::vector<int>(batch, width)
                                : measurement::integers(value, commit ? 0 : 1, width);
    if (out.size() == 1) out.resize(batch, out[0]);
    if (out.size() != std::size_t(batch)) throw std::invalid_argument("counts must match batch");
    return out;
}
} // namespace ninfer::bench::kda
