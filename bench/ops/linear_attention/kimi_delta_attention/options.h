#pragma once
#include "op_measurement.cuh"

namespace ninfer::bench::kda {
struct Options {
    measurement::Options timing;
    std::string op = "record", valid = "dense", commits = "dense";
    std::string format = "q8", policy = "default";
    std::vector<int> tokens{4};
    int heads = 32, qk_heads = 32, batch = 1, layers = 34;
    bool alias_records = true;
};

inline Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout
                << "KDA public Ops benchmark\n"
                   "  --op core|update|record|fold|input-update|input-record|chain\n"
                   "       layer-update|layer-record|layer-prefill|layer-chain\n"
                   "  --tokens T[,T...] --batch B --layers L --qk-heads H --value-heads H\n"
                   "  --valid dense|N[,N...] --commits dense|N[,N...]\n"
                   "  --kv-records alias|copy (record only; default alias)\n"
                   "  --format q8|mxfp8 --policy a16|a8 (input projection; default follows "
                   "format)\n"
                   "core is single sequence; update is T=1; fold measures all layers.\n"
                   "MXFP8/H32 fold uses actual projection/recurrence records, prepared untimed.\n"
                   "chain measures one input-record, recurrence-record and one-layer fold.\n"
                   "layer-* additionally includes BF16 control split/pair, gated RMSNorm,\n"
                   "and the MXFP8 O projection; layer-chain also folds its actual records.\n"
                   "layer-update requires T=1; layer-prefill requires B=1 and dense validity.\n";
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
        else if (flag == "--format") {
            if (value != "q8" && value != "mxfp8") throw std::invalid_argument("format: q8|mxfp8");
            o.format = value;
        } else if (flag == "--policy") {
            if (value != "a16" && value != "a8") throw std::invalid_argument("policy: a16|a8");
            o.policy = value;
        } else if (flag == "--kv-records") {
            if (value != "alias" && value != "copy")
                throw std::invalid_argument("kv-records: alias|copy");
            o.alias_records = value == "alias";
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.op != "core" && o.op != "update" && o.op != "record" && o.op != "fold" &&
        o.op != "input-update" && o.op != "input-record" && o.op != "chain" &&
        o.op != "layer-update" && o.op != "layer-record" && o.op != "layer-prefill" &&
        o.op != "layer-chain")
        throw std::invalid_argument("invalid op");
    if (o.heads < o.qk_heads || o.heads % o.qk_heads ||
        ((o.op == "core" || o.op == "layer-prefill") && o.batch != 1))
        throw std::invalid_argument("invalid geometry");
    if (o.op.starts_with("layer-") && (o.format != "mxfp8" || o.heads != 32 || o.qk_heads != 32))
        throw std::invalid_argument("layer benchmark requires MXFP8 and Hq=Hv=32");
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
