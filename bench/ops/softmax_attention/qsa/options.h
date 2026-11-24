#pragma once

#include "op_measurement.cuh"
#include "context_bucket_workload.h"
#include "ragged_workload.h"
#include "ninfer/types.h"

namespace ninfer::bench::qsa {

enum class Projection { QueryGate, KeyValueIndex, Joint, KeyValueIndexKey };

inline const char* name(Projection p) {
    switch (p) {
    case Projection::QueryGate:
        return "qg";
    case Projection::KeyValueIndex:
        return "kv-index";
    case Projection::Joint:
        return "joint";
    case Projection::KeyValueIndexKey:
        return "kv-index-key";
    }
    throw std::invalid_argument("invalid QSA projection");
}

// Canonical fixture geometry; Op admission remains in the corresponding entry.
inline int prepare_query_heads(int kv_heads) { return kv_heads == 1 ? 12 : 24; }

inline std::vector<int> segments(Projection p, int kv_heads) {
    switch (p) {
    case Projection::QueryGate:
        return {3072 * kv_heads, 3072 * kv_heads};
    case Projection::KeyValueIndex:
        return {256 * kv_heads, 256 * kv_heads, 512, 128};
    case Projection::Joint:
        return {256 * kv_heads, 256 * kv_heads, 128, 3072 * kv_heads, 3072 * kv_heads, 512};
    case Projection::KeyValueIndexKey:
        return {256 * kv_heads, 256 * kv_heads, 128};
    }
    throw std::invalid_argument("invalid QSA projection");
}

inline const char* name(KvCacheStorage storage) { return kv_cache_profile_info(storage).name; }

enum class StateAlias { Disjoint, Inplace, Mixed };

struct RaggedSpec {
    ragged::SegmentSpec lengths;
    std::vector<int> histories;
};

inline std::vector<int> parse_histories(std::string_view spec, std::size_t segments) {
    if (spec.find_first_of(",x") == spec.npos)
        return std::vector<int>(segments, measurement::integer(spec, 0));
    std::vector<int> histories;
    for (;;) {
        const auto comma  = spec.find(',');
        const auto item   = spec.substr(0, comma);
        const auto repeat = item.find('x');
        const int count   = repeat == item.npos ? 1 : measurement::integer(item.substr(0, repeat));
        const int prefix =
            measurement::integer(repeat == item.npos ? item : item.substr(repeat + 1), 0);
        if (std::size_t(count) > segments - histories.size())
            throw std::invalid_argument("too many ragged histories for the partition");
        histories.insert(histories.end(), count, prefix);
        if (comma == spec.npos) break;
        spec.remove_prefix(comma + 1);
    }
    if (histories.size() != segments)
        throw std::invalid_argument("ragged history count must match the segment count");
    return histories;
}

inline std::vector<RaggedSpec> ragged_specs(std::vector<ragged::SegmentSpec> lengths,
                                            std::optional<std::string_view> text) {
    std::vector<std::string_view> histories;
    if (text) {
        auto remaining = *text;
        for (;;) {
            const auto end = remaining.find(';');
            histories.push_back(remaining.substr(0, end));
            if (end == remaining.npos) break;
            remaining.remove_prefix(end + 1);
        }
        if (histories.size() != lengths.size() &&
            (histories.size() != 1 || histories.front().find_first_of(",x") != text->npos))
            throw std::invalid_argument("use one scalar history or one spec per ragged partition");
    }
    std::vector<RaggedSpec> specs;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        const auto history = histories.empty() ? "0" : histories[histories.size() == 1 ? 0 : i];
        auto prefixes      = parse_histories(history, lengths[i].size());
        for (std::size_t s = 0; s < prefixes.size(); ++s)
            if (prefixes[s] > std::numeric_limits<int>::max() - lengths[i][s])
                throw std::invalid_argument("ragged visible end exceeds INT32_MAX");
        specs.push_back({std::move(lengths[i]), std::move(prefixes)});
    }
    return specs;
}

struct Options {
    measurement::Options timing;
    std::string op    = "input-proj";
    std::string entry = "both";
    std::vector<Projection> projections{Projection::QueryGate, Projection::KeyValueIndex,
                                        Projection::Joint};
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    int rounds             = 1;
    int graph_calls        = 1;
    int query_heads = 24, kv_heads = 2;
    std::vector<int> capacities{16}, bases{6};
    std::vector<int> contexts{8192}, prefixes;
    bool explicit_contexts         = false;
    int context_capacity           = 0;
    bool context_capacity_explicit = false;
    std::optional<ops::ContextBucket> bucket;
    bool explicit_bucket             = false;
    bool explicit_valid              = false;
    workload::LengthDist length_dist = workload::LengthDist::Uniform;
    workload::ValidDist valid_dist   = workload::ValidDist::Full;
    int selected_blocks              = 512;
    std::string selection            = "random";
    std::string score_pattern        = "random";
    std::string valid = "dense", accepted = "dense", layout = "contiguous";
    int batch        = 1;
    int axes         = 1;
    bool inplace     = false;
    bool graph_stats = false;
    std::vector<RaggedSpec> packed_specs;
    int shared_prefix_tokens = 0;
    StateAlias state_alias   = StateAlias::Disjoint;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool token_options = false, state_options = false;
    bool projection_only = false, tail_only = false, prepare_only = false, head_options = false;
    bool indexer_only = false, base_options = false, sparse_only = false, cache_option = false;
    bool context_workload_options = false, score_option = false, layout_option = false;
    bool padded_shape = false, shared_prefix_option = false, state_alias_option = false;
    std::vector<ragged::SegmentSpec> partitions;
    std::optional<std::string_view> histories;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "QSA Op benchmark\n"
                   "  --jobs-file PATH (standalone mode, one tab-separated argument list per "
                   "line)\n"
                   "  --op input-proj|tail-extract|prepare|prepare-kv|indexer|sparse-gqa (default "
                   "input-proj)\n"
                   "  --projection qg|kv-index|joint|kv-index-key|all\n"
                   "  --kv-heads 1|2|4 (4: RTX 5090 prepare/sparse GQA, default 2)\n"
                   "  --entry split|linear|both\n"
                   "  --tokens M[,M...] or --sweep START:END[:STEP] --rounds N\n"
                   "Prepare: --tokens T[,T...] --batch B --axes 1|3 --inplace\n"
                   "  --kv-cache bf16|int8|fp8|nvfp4|k8v4 (default bf16)\n"
                   "  --base-positions F[,F...] --valid dense|N[,N...] --layout "
                   "contiguous|layered|unaligned\n"
                   "Indexer: --tokens T[,T...] --batch B --contexts L[,L...]\n"
                   "  --bucket auto|2k|8k|16k|32k|64k|128k|256k (default auto)\n"
                   "  --context-capacity N (legacy bucket selection if --bucket is omitted)\n"
                   "  --length-dist uniform|skewed-1long|staggered|shared-prefix (single L)\n"
                   "  --valid dense|N[,N...] or --valid-dist full|ragged|with-inert\n"
                   "  --score-pattern random|ties\n"
                   "Sparse GQA: --tokens T[,T...] --batch B --contexts L[,L...]\n  "
                   "--query-heads Hq --kv-heads Hkv (24/2 or 12/1; RTX 5090 prepare/sparse GQA "
                   "also accept 24/4)\n  --prefix-tokens "
                   "P[,P...] (derive each context as prefix + valid length)\n"
                   "  --selection random|shared|contiguous --selected-blocks 0..512\n"
                   "  --bucket auto|2k|8k|16k|32k|64k|128k|256k (default auto)\n"
                   "  --context-capacity N (auto bucket selection, default maximum live L)\n"
                   "  --length-dist uniform|skewed-1long|staggered|shared-prefix (single L)\n"
                   "  --kv-cache bf16|int8|fp8|nvfp4|k8v4 (default bf16)\n"
                   "  --valid dense|N[,N...] or --valid-dist full|ragged|with-inert\n"
                   "  --graph-calls N (prepare/prepare-kv/sparse-gqa, warm Graph only)\n"
                   "Packed prepare/prepare-kv/indexer/sparse-gqa:\n"
                   "  --ragged-segments 'L,KxL;...' --ragged-histories 'P,KxP;...' (default 0)\n"
                   "  --shared-prefix-tokens N (indexer/sparse-gqa, multiple of 64, default 0)\n"
                   "  --state-alias disjoint|inplace|mixed (prepare/prepare-kv, default disjoint)\n"
                   "  --graph-stats (also emit graph node/kernel counts for padded QSA rows)\n"
                   "Tail extraction:\n"
                   "  --capacity R[,R...] --batch B --base-positions F0[,F0...]\n"
                   "  --valid dense|N[,N...] --accepted dense|mixed|N[,N...]\n"
                   "  --layout contiguous|layered|unaligned --inplace\n"
                   "A single F0/count broadcasts to all B rows. B>=1.\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--inplace") {
            o.inplace     = true;
            state_options = true;
            continue;
        }
        if (flag == "--graph-stats") {
            o.graph_stats = true;
            continue;
        }
        state_options |= flag == "--batch" || flag == "--base-positions" || flag == "--valid" ||
                         flag == "--layout";
        projection_only |= flag == "--entry" || flag == "--projection";
        head_options |= flag == "--kv-heads";
        tail_only |= flag == "--capacity" || flag == "--accepted";
        prepare_only |= flag == "--axes";
        cache_option |= flag == "--kv-cache";
        indexer_only |= flag == "--contexts" || flag == "--context-capacity";
        base_options |= flag == "--base-positions";
        sparse_only |= flag == "--selection" || flag == "--selected-blocks" ||
                       flag == "--query-heads" || flag == "--prefix-tokens";
        score_option |= flag == "--score-pattern";
        context_workload_options |=
            flag == "--bucket" || flag == "--length-dist" || flag == "--valid-dist";
        layout_option |= flag == "--layout";
        padded_shape |= flag == "--batch" || flag == "--valid" || flag == "--valid-dist" ||
                        flag == "--length-dist" || flag == "--contexts" ||
                        flag == "--prefix-tokens" || flag == "--base-positions";
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--query-heads")
            o.query_heads = measurement::integer(value);
        else if (flag == "--kv-heads")
            o.kv_heads = measurement::integer(value);
        else if (flag == "--ragged-segments")
            partitions = ragged::parse_ragged_specs(value);
        else if (flag == "--ragged-histories")
            histories = value;
        else if (flag == "--shared-prefix-tokens") {
            shared_prefix_option   = true;
            o.shared_prefix_tokens = measurement::integer(value, 0);
            if (o.shared_prefix_tokens % 64)
                throw std::invalid_argument("shared-prefix-tokens must be a multiple of 64");
        } else if (flag == "--state-alias") {
            state_alias_option = true;
            if (value == "inplace")
                o.state_alias = StateAlias::Inplace;
            else if (value == "mixed")
                o.state_alias = StateAlias::Mixed;
            else if (value != "disjoint")
                throw std::invalid_argument("state-alias: disjoint|inplace|mixed");
            else
                o.state_alias = StateAlias::Disjoint;
        } else if (flag == "--contexts") {
            o.contexts          = measurement::integers(value, 0);
            o.explicit_contexts = true;
        } else if (flag == "--prefix-tokens")
            o.prefixes = measurement::integers(value, 0);
        else if (flag == "--bucket") {
            o.bucket          = workload::parse_bucket_override(value);
            o.explicit_bucket = true;
        } else if (flag == "--length-dist")
            o.length_dist = workload::parse_length_dist(value);
        else if (flag == "--valid-dist")
            o.valid_dist = workload::parse_valid_dist(value);
        else if (flag == "--score-pattern")
            o.score_pattern = value;
        else if (flag == "--selection")
            o.selection = value;
        else if (flag == "--selected-blocks")
            o.selected_blocks = measurement::integer(value, 0);
        else if (flag == "--context-capacity") {
            o.context_capacity          = measurement::integer(value, 0);
            o.context_capacity_explicit = true;
        } else if (flag == "--kv-cache") {
            bool matched = false;
            for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                 KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                                 KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value})
                if (value == name(storage)) {
                    o.storage = storage;
                    matched   = true;
                }
            if (!matched) throw std::invalid_argument("kv-cache: bf16|int8|fp8|nvfp4|k8v4");
        } else if (flag == "--axes")
            o.axes = measurement::integer(value);
        else if (flag == "--capacity")
            o.capacities = measurement::integers(value);
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--base-positions")
            o.bases = measurement::integers(value, 0);
        else if (flag == "--valid") {
            o.valid          = value;
            o.explicit_valid = true;
        } else if (flag == "--accepted")
            o.accepted = value;
        else if (flag == "--layout")
            o.layout = value;
        else if (flag == "--entry") {
            if (value != "split" && value != "linear" && value != "both")
                throw std::invalid_argument("entry: split|linear|both");
            o.entry = value;
        } else if (flag == "--projection") {
            if (value == "all")
                o.projections = {Projection::QueryGate, Projection::KeyValueIndex,
                                 Projection::Joint, Projection::KeyValueIndexKey};
            else if (value == "qg")
                o.projections = {Projection::QueryGate};
            else if (value == "kv-index")
                o.projections = {Projection::KeyValueIndex};
            else if (value == "joint")
                o.projections = {Projection::Joint};
            else if (value == "kv-index-key")
                o.projections = {Projection::KeyValueIndexKey};
            else
                throw std::invalid_argument("projection: qg|kv-index|joint|kv-index-key|all");
        } else if (flag == "--tokens" || flag == "--sweep") {
            if (token_options) throw std::invalid_argument("specify one token list or sweep");
            token_options = true;
            o.tokens =
                flag == "--tokens" ? measurement::integers(value) : measurement::sweep(value);
        } else if (flag == "--graph-calls")
            o.graph_calls = measurement::integer(value);
        else if (flag == "--rounds")
            o.rounds = measurement::integer(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.op != "input-proj" && o.op != "tail-extract" && o.op != "prepare" &&
        o.op != "prepare-kv" && o.op != "indexer" && o.op != "sparse-gqa")
        throw std::invalid_argument(
            "op: input-proj|tail-extract|prepare|prepare-kv|indexer|sparse-gqa");
    if (((o.op == "indexer" || o.op == "sparse-gqa") && layout_option) ||
        (o.op != "input-proj" && o.op != "prepare" && o.op != "prepare-kv" &&
         o.op != "sparse-gqa" && head_options) ||
        (o.op != "indexer" && score_option) ||
        (o.op != "indexer" && o.op != "sparse-gqa" && context_workload_options) ||
        (o.op != "prepare" && o.op != "prepare-kv" && o.op != "sparse-gqa" && cache_option) ||
        (o.op != "sparse-gqa" && sparse_only) ||
        (o.op == "sparse-gqa" &&
         (tail_only || projection_only || prepare_only || base_options || o.inplace)) ||
        (o.op == "input-proj" && (state_options || tail_only || prepare_only || indexer_only)) ||
        (o.op == "tail-extract" &&
         (token_options || projection_only || prepare_only || indexer_only)) ||
        ((o.op == "prepare" || o.op == "prepare-kv") &&
         (tail_only || projection_only || indexer_only)) ||
        (o.op == "indexer" &&
         (tail_only || projection_only || prepare_only || base_options || o.inplace)))
        throw std::invalid_argument("options must belong to the selected QSA Op");
    if (o.op == "prepare-kv" && o.inplace)
        throw std::invalid_argument("prepare-kv has no query output to alias");
    const bool prepare = o.op == "prepare" || o.op == "prepare-kv";
    const bool cached  = o.op == "indexer" || o.op == "sparse-gqa";
    if (o.graph_stats && !prepare && !cached)
        throw std::invalid_argument(
            "graph-stats requires prepare, prepare-kv, indexer or sparse-gqa");
    if (!partitions.empty()) {
        if ((!prepare && !cached) || padded_shape || token_options)
            throw std::invalid_argument("ragged-segments requires a packed QSA Op and shape");
        o.packed_specs = ragged_specs(std::move(partitions), histories);
    } else if (histories || shared_prefix_option || state_alias_option) {
        throw std::invalid_argument(
            "ragged history, prefix and state options require ragged-segments");
    }
    if ((shared_prefix_option && !cached) || (state_alias_option && !prepare))
        throw std::invalid_argument(
            "shared prefix belongs to cached Ops, state alias to prepare Ops");
    if (!o.prefixes.empty() && o.explicit_contexts)
        throw std::invalid_argument("prefix-tokens and contexts are alternatives");
    if (o.score_pattern != "random" && o.score_pattern != "ties")
        throw std::invalid_argument("score-pattern: random|ties");
    if (o.selected_blocks > 512 ||
        (o.selection != "random" && o.selection != "shared" && o.selection != "contiguous"))
        throw std::invalid_argument("invalid sparse selection/count");
    if (o.batch < 1 || (o.axes != 1 && o.axes != 3) ||
        (o.layout != "contiguous" && o.layout != "layered" && o.layout != "unaligned"))
        throw std::invalid_argument("invalid QSA batch/axes/layout");
    if (o.graph_calls > 1 && (o.timing.execution != "graph" || o.timing.cache != "warm" ||
                              o.inplace || o.state_alias != StateAlias::Disjoint ||
                              (o.op != "prepare" && o.op != "prepare-kv" && o.op != "sparse-gqa")))
        throw std::invalid_argument(
            "graph-calls > 1 requires non-aliasing prepare/prepare-kv/sparse-gqa and warm Graph");
    return o;
}

} // namespace ninfer::bench::qsa
