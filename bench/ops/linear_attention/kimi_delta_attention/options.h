#pragma once
#include "core/dtype.h"
#include "op_measurement.cuh"
#include "ragged_workload.h"

#include <optional>

namespace ninfer::bench::kda {
inline const char* state_dtype_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "fp32"; }

struct Options {
    measurement::Options timing;
    std::string op = "record", valid = "dense", commits = "dense";
    std::string format = "q8", policy = "default";
    std::vector<int> tokens{4};
    std::vector<ragged::SegmentSpec> ragged_specs;
    ragged::StateOffsetForm state_form = ragged::StateOffsetForm::Distinct;
    DType state_dtype                  = DType::FP32;
    int heads = 32, qk_heads = 32, batch = 1, layers = 34;
};

inline Options parse(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false, explicit_state = false;
    std::optional<std::string_view> ragged_text;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::cout
                << "KDA public Ops benchmark\n"
                   "  --op core|update|record|fold|input-update|input-record|chain\n"
                   "       layer-update|layer-record|layer-prefill|layer-chain\n"
                   "  --tokens T[,T...] or --sweep START:END[:STEP]\n"
                   "  --batch B --layers L --qk-heads H --value-heads H\n"
                   "  --valid dense|N[,N...] --commits dense|N[,N...]\n"
                   "  --ragged-segments SPEC (core or input-update)\n"
                   "  --state-form distinct|alias|mixed|permuted|shared-source (ragged)\n"
                   "  --state-dtype fp32|bf16 (default fp32)\n"
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
        else if (flag == "--tokens" || flag == "--sweep") {
            o.tokens =
                flag == "--tokens" ? measurement::integers(value) : measurement::sweep(value);
            explicit_tokens = true;
        } else if (flag == "--ragged-segments")
            ragged_text = value;
        else if (flag == "--state-form") {
            o.state_form   = ragged::parse_state_form(value);
            explicit_state = true;
        } else if (flag == "--state-dtype") {
            if (value == "fp32")
                o.state_dtype = DType::FP32;
            else if (value == "bf16")
                o.state_dtype = DType::BF16;
            else
                throw std::invalid_argument("state-dtype: fp32|bf16");
        } else if (flag == "--batch")
            o.batch = measurement::integer(value);
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
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.op != "core" && o.op != "update" && o.op != "record" && o.op != "fold" &&
        o.op != "input-update" && o.op != "input-record" && o.op != "chain" &&
        o.op != "layer-update" && o.op != "layer-record" && o.op != "layer-prefill" &&
        o.op != "layer-chain")
        throw std::invalid_argument("invalid op");
    if (ragged_text) {
        if (explicit_tokens)
            throw std::invalid_argument(
                "--ragged-segments is mutually exclusive with --tokens and --sweep");
        if ((o.op != "input-update" && o.op != "core") || o.batch != 1 || o.valid != "dense")
            throw std::invalid_argument(
                "ragged segments require input-update or core, batch 1 and dense validity");
        o.ragged_specs = ragged::parse_ragged_specs(*ragged_text);
    } else if (explicit_state)
        throw std::invalid_argument("state-form requires ragged segments");
    if (o.heads < o.qk_heads || o.heads % o.qk_heads ||
        ((o.op == "core" || o.op == "layer-prefill") && o.batch != 1))
        throw std::invalid_argument("invalid geometry");
    if (o.op.starts_with("layer-") && (o.format != "mxfp8" || o.heads != 32 || o.qk_heads != 32))
        throw std::invalid_argument("layer benchmark requires MXFP8 and Hq=Hv=32");
    return o;
}

inline std::vector<int> counts(const std::string& value, int width, int batch, bool commit) {
    auto out =
        value == "dense" ? std::vector<int>(batch, width) : measurement::integers(value, 0, width);
    if (out.size() == 1) out.resize(batch, out[0]);
    if (out.size() != std::size_t(batch)) throw std::invalid_argument("counts must match batch");
    return out;
}
} // namespace ninfer::bench::kda
