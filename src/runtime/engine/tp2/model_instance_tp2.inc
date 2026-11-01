// Translation-unit fragment of src/runtime/engine/model_instance.cpp, included once at its end
// (after its closing namespace): the Engine option checks of the two-device mode. Not a header.
// Our file; see docs/maintainer/upstream-merge.md.

namespace ninfer::runtime {
namespace {

void validate_tensor_parallel_options(const EngineOptions& options) {
    // Rejected before the artifact is read. The full output head is split by vocabulary across
    // the ranks, and the DFlash2 candidate ranking has a vocabulary-split form only for the
    // optimized head (ops::linear_topk_split).
    if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
        options.speculative.proposal_head != ProposalHead::Optimized) {
        throw std::invalid_argument(
            "Engine tp 2 DFlash2 requires the optimized proposal head (--lm-head-draft)");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Int8Group64 &&
        options.kv_cache != KvCacheStorage::Nvfp4Group16) {
        throw std::invalid_argument("Engine tp 2 supports only bf16, int8 and nvfp4 KV caches");
    }
}

} // namespace
} // namespace ninfer::runtime
