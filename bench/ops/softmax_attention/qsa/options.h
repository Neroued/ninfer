#pragma once

#include "op_measurement.cuh"
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

inline const char* name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::invalid_argument("invalid QSA KV storage");
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
    int query_heads = 24, kv_heads = 2;
    std::vector<int> capacities{16}, bases{6};
    std::vector<int> contexts{8192}, prefixes;
    bool explicit_contexts         = false;
    int context_capacity           = 0;
    bool context_capacity_explicit = false;
    int selected_blocks            = 512;
    std::string selection          = "random";
    std::string score_pattern      = "random";
    std::string valid = "dense", accepted = "dense", layout = "contiguous";
    int batch    = 1;
    int axes     = 1;
    bool inplace = false;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool token_options = false, state_options = false;
    bool projection_only = false, tail_only = false, prepare_only = false, head_options = false;
    bool indexer_only = false, base_options = false, sparse_only = false, cache_option = false;
    bool score_option = false, layout_option = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout
                << "QSA Op benchmark\n"
                   "  --op input-proj|tail-extract|prepare|prepare-kv|indexer|sparse-gqa (default "
                   "input-proj)\n"
                   "  --projection qg|kv-index|joint|kv-index-key|all\n"
                   "  --kv-heads 1|2 (input projection/prepare/sparse GQA, default 2)\n"
                   "  --entry split|linear|both\n"
                   "  --tokens M[,M...] or --sweep START:END[:STEP] --rounds N\n"
                   "Prepare: --tokens T[,T...] --batch B --axes 1|3 --inplace\n"
                   "  --kv-cache bf16|int8|fp8|nvfp4|k8v4 (default bf16)\n"
                   "  --base-positions F[,F...] --valid dense|N[,N...] --layout "
                   "contiguous|layered|unaligned\n"
                   "Indexer: --tokens T[,T...] --batch B --contexts L[,L...]\n"
                   "  --context-capacity N (default maximum L) --score-pattern random|ties\n"
                   "  --valid dense|N[,N...]\n"
                   "Sparse GQA: --tokens T[,T...] --batch B --contexts L[,L...]\n  "
                   "--query-heads Hq --kv-heads Hkv (24/2 or 12/1)\n  --prefix-tokens "
                   "P[,P...] (derive each context as prefix + valid length)\n"
                   "  --selection random|shared|contiguous --selected-blocks 0..512\n"
                   "  --context-capacity N (default maximum L)\n"
                   "  --kv-cache bf16|int8|fp8|nvfp4|k8v4 (default bf16)\n"
                   "  --valid dense|N[,N...]\n"
                   "Tail extraction:\n"
                   "  --capacity R[,R...] --batch B --base-positions F0[,F0...]\n"
                   "  --valid dense|N[,N...] --accepted dense|mixed|N[,N...]\n"
                   "  --layout contiguous|layered|unaligned --inplace\n"
                   "A single F0/count broadcasts to all B rows. B=1..8.\n";
            measurement::help();
            std::exit(0);
        }
        if (flag == "--inplace") {
            o.inplace     = true;
            state_options = true;
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
        layout_option |= flag == "--layout";
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--query-heads")
            o.query_heads = measurement::integer(value);
        else if (flag == "--kv-heads")
            o.kv_heads = measurement::integer(value);
        else if (flag == "--contexts") {
            o.contexts          = measurement::integers(value, 0);
            o.explicit_contexts = true;
        } else if (flag == "--prefix-tokens")
            o.prefixes = measurement::integers(value, 0);
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
                                 KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                 KvCacheStorage::Fp8KeyNvfp4Value})
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
        else if (flag == "--valid")
            o.valid = value;
        else if (flag == "--accepted")
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
        } else if (flag == "--rounds")
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
    if (!o.prefixes.empty() && o.explicit_contexts)
        throw std::invalid_argument("prefix-tokens and contexts are alternatives");
    if (o.score_pattern != "random" && o.score_pattern != "ties")
        throw std::invalid_argument("score-pattern: random|ties");
    if (o.selected_blocks > 512 ||
        (o.selection != "random" && o.selection != "shared" && o.selection != "contiguous"))
        throw std::invalid_argument("invalid sparse selection/count");
    if (o.batch > 8 || (o.axes != 1 && o.axes != 3) ||
        (o.layout != "contiguous" && o.layout != "layered" && o.layout != "unaligned"))
        throw std::invalid_argument("invalid QSA batch/axes/layout");
    return o;
}

} // namespace ninfer::bench::qsa
