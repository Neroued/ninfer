#pragma once

#include "core/dtype.h"
#include "op_measurement.cuh"
#include "ragged_workload.h"

#include <optional>

namespace ninfer::bench::gdn_recurrence {
enum class Mode { Running, BatchUpdate, ChunkedRaw, ChunkedNormalized, Recurrent };

inline const char* state_dtype_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "fp32"; }

struct Options {
    measurement::Options timing;
    Mode mode    = Mode::Running;
    int qk_heads = 16, value_heads = 48, batch = 1;
    DType state_dtype                  = DType::FP32;
    ragged::StateOffsetForm state_form = ragged::StateOffsetForm::Distinct;
    std::vector<int> tokens{1024};
    std::vector<ragged::SegmentSpec> ragged_specs;
    bool breakdown      = false;
    std::string qk_norm = "fused";
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false, explicit_state = false;
    std::optional<std::string_view> ragged_text;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "Gated DeltaNet recurrence benchmark\n"
                   "  --mode running|batch-update|chunked-raw|chunked-normalized|recurrent\n"
                   "  --tokens T[,T...] or --sweep START:END[:STEP] --qk-heads H "
                   "--value-heads H --batch B\n"
                   "  --qk-norm fused|composed (composed requires B=1)\n"
                   "  --state-dtype fp32|bf16 (default fp32)\n"
                   "  --breakdown (chunked stages)\n"
                   "  --ragged-segments SPEC (running only)\n"
                   "  --state-form distinct|alias|mixed|permuted|shared-source\n"
                   "  Non-ragged modes accept distinct and alias (batch-update default alias).\n"
                   "Running and batch-update call public Ops; other modes and breakdown are "
                   "private stage analysis.\n"
                   "chunked-raw normalizes raw Q/K; chunked-normalized consumes "
                   "pre-normalized BF16 Q/K.\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--breakdown") {
            o.breakdown = true;
            continue;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--mode") {
            if (value == "running")
                o.mode = Mode::Running;
            else if (value == "batch-update")
                o.mode = Mode::BatchUpdate;
            else if (value == "chunked-raw")
                o.mode = Mode::ChunkedRaw;
            else if (value == "chunked-normalized")
                o.mode = Mode::ChunkedNormalized;
            else if (value == "recurrent")
                o.mode = Mode::Recurrent;
            else
                throw std::invalid_argument("unknown recurrence mode");
        } else if (flag == "--tokens") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::integers(value);
            explicit_tokens = true;
        } else if (flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::sweep(value);
            explicit_tokens = true;
        } else if (flag == "--qk-heads")
            o.qk_heads = measurement::integer(value);
        else if (flag == "--value-heads")
            o.value_heads = measurement::integer(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--ragged-segments")
            ragged_text = value;
        else if (flag == "--state-form") {
            o.state_form   = ragged::parse_state_form(value);
            explicit_state = true;
        } else if (flag == "--qk-norm")
            o.qk_norm = value;
        else if (flag == "--state-dtype") {
            if (value == "fp32")
                o.state_dtype = DType::FP32;
            else if (value == "bf16")
                o.state_dtype = DType::BF16;
            else
                throw std::invalid_argument("state-dtype: fp32|bf16");
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (ragged_text) {
        if (explicit_tokens)
            throw std::invalid_argument(
                "--ragged-segments is mutually exclusive with --tokens and --sweep");
        if (o.mode != Mode::Running || o.breakdown || o.batch != 1)
            throw std::invalid_argument(
                "ragged segments require running mode, batch 1 and no breakdown");
        o.ragged_specs = ragged::parse_ragged_specs(*ragged_text);
    } else if (o.state_form != ragged::StateOffsetForm::Distinct &&
               o.state_form != ragged::StateOffsetForm::Alias)
        throw std::invalid_argument("non-ragged modes accept distinct and alias state forms");
    if (!explicit_state && o.mode == Mode::BatchUpdate)
        o.state_form = ragged::StateOffsetForm::Alias;
    if (o.mode == Mode::BatchUpdate && !explicit_tokens) o.tokens = {1};
    if (o.value_heads < o.qk_heads || o.value_heads % o.qk_heads ||
        (o.mode != Mode::BatchUpdate && o.batch != 1))
        throw std::invalid_argument("invalid head/batch geometry");
    if (o.qk_norm != "fused" && o.qk_norm != "composed")
        throw std::invalid_argument("unknown normalization");
    if (o.qk_norm == "composed" && (o.mode != Mode::BatchUpdate || o.batch != 1))
        throw std::invalid_argument("composed normalization requires batch-update with B=1");
    if (o.breakdown && o.mode == Mode::BatchUpdate)
        throw std::invalid_argument("breakdown requires a prefill mode");
    for (int t : o.tokens) {
        if (o.mode == Mode::BatchUpdate && t != 1)
            throw std::invalid_argument("batch-update requires T=1");
        if ((o.mode == Mode::ChunkedRaw || o.mode == Mode::ChunkedNormalized) && t < 16)
            throw std::invalid_argument("chunked mode requires T>=16");
    }
    return o;
}
} // namespace ninfer::bench::gdn_recurrence
