#pragma once

#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::nvtx {

enum class Category : std::uint32_t {
    Runtime = 1,
    Prefill,
    Decode,
    Mtp,
    DFlash,
    Attention,
    Gdn,
    PostMixer,
    Moe,
    Control,
    Graph,
    Vision,
    Scoring,
};

enum class Name : std::size_t {
    Generate,
    Prefill,
    Decode,
    EngineBoundary,
    ProgramSubmit,
    DeviceWait,
    ProgramPost,
    EngineCommitOutput,
    EngineMaintenance,
    AdmissionPolicy,
    ContextProgress,
    StatsPublication,
    DecodeMtpRound,
    DecodeOrdinaryRound,
    DecodeMtpSubmit,
    DecodeMtpWait,
    DecodeDFlashRound,
    DecodeDFlashSubmit,
    DecodeDFlashWait,
    DecodeOrdinarySubmit,
    DecodeOrdinaryWait,
    PrefillMtpChunk,
    PrefillLayerFull,
    VerifyLayerFull,
    PrefillAttention,
    VerifyAttention,
    PrefillPostMixer,
    VerifyPostMixer,
    PrefillLayerGdn,
    VerifyLayerGdn,
    PrefillGdn,
    VerifyGdn,
    PrefillChunk,
    SparseMoePrefill,
    SparseMoeSmallT,
    SparseMoeDecode,
    EngineLoad,
    FrontendPrepare,
    Score,
    ControlBatch,
    CudaGraphPrepare,
    CudaGraphCapture,
    CudaGraphInstantiate,
    CudaGraphUpdate,
    CudaGraphUpload,
    CudaGraphLaunch,
    DecodeEager,
    VisionEncode,
    VisionPatchEmbedding,
    VisionLayer,
    VisionAttention,
    VisionMlp,
    VisionMerge,
    MtpForward,
    MtpProposal,
    DecodeMtpTarget,
    DecodeMtpDraft,
    DFlashContextAppend,
    DFlashProposal,
    DFlashLayer,
    DFlashAttention,
    DFlashMlp,
    DecodeDFlashTarget,
    Count,
};

[[nodiscard]] inline std::uint32_t color(Category category) noexcept {
    switch (category) {
    case Category::Runtime:
        return 0xff4c78a8u;
    case Category::Prefill:
        return 0xff59a14fu;
    case Category::Decode:
        return 0xfff28e2bu;
    case Category::Mtp:
        return 0xffb279a2u;
    case Category::DFlash:
        return 0xffaf7aa1u;
    case Category::Attention:
        return 0xff76b7b2u;
    case Category::Gdn:
        return 0xffe15759u;
    case Category::PostMixer:
        return 0xffedc948u;
    case Category::Moe:
        return 0xffb07aa1u;
    case Category::Control:
        return 0xff9c9c9cu;
    case Category::Graph:
        return 0xff79706eu;
    case Category::Vision:
        return 0xff86bcb6u;
    case Category::Scoring:
        return 0xffff9da7u;
    }
    return 0xff9c9c9cu;
}

[[nodiscard]] inline nvtxDomainHandle_t domain() noexcept {
    static nvtxDomainHandle_t handle = [] {
        nvtxDomainHandle_t out = nvtxDomainCreateA("ninfer");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Runtime), "runtime");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Prefill), "prefill");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Decode), "decode");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Mtp), "mtp");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::DFlash), "dflash");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Attention), "attention");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Gdn), "gdn");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::PostMixer), "post-mixer");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Moe), "moe");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Control), "control");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Graph), "cuda-graph");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Vision), "vision");
        nvtxDomainNameCategoryA(out, static_cast<std::uint32_t>(Category::Scoring), "scoring");
        return out;
    }();
    return handle;
}

[[nodiscard]] inline nvtxStringHandle_t registered_message(Name name) noexcept {
    static constexpr auto names = std::to_array<const char*>({
        "generate",
        "prefill",
        "decode",
        "engine.boundary",
        "program.submit",
        "device.wait",
        "program.post",
        "engine.commit_output",
        "engine.maintenance",
        "engine.admission_policy",
        "engine.context_progress",
        "engine.stats_publication",
        "decode.mtp_round",
        "decode.ordinary_round",
        "decode.mtp.submit",
        "decode.mtp.wait",
        "decode.dflash_round",
        "decode.dflash.submit",
        "decode.dflash.wait",
        "decode.ordinary.submit",
        "decode.ordinary.wait",
        "prefill.mtp_chunk",
        "prefill.layer.full",
        "verify.layer.full",
        "prefill.attention",
        "verify.attention",
        "prefill.post_mixer",
        "verify.post_mixer",
        "prefill.layer.gdn",
        "verify.layer.gdn",
        "prefill.gdn",
        "verify.gdn",
        "prefill.chunk",
        "sparse_moe.prefill",
        "sparse_moe.small_t",
        "sparse_moe.decode",
        "engine.load",
        "frontend.prepare",
        "score",
        "engine.control_batch",
        "cuda_graph.prepare",
        "cuda_graph.capture",
        "cuda_graph.instantiate",
        "cuda_graph.update",
        "cuda_graph.upload",
        "cuda_graph.launch",
        "decode.eager",
        "vision.encode",
        "vision.patch_embedding",
        "vision.layer",
        "vision.attention",
        "vision.mlp",
        "vision.merge",
        "mtp.forward",
        "mtp.proposal",
        "decode.mtp.target",
        "decode.mtp.draft",
        "dflash.context_append",
        "dflash.proposal",
        "dflash.layer",
        "dflash.attention",
        "dflash.mlp",
        "decode.dflash.target",
    });
    static_assert(names.size() == static_cast<std::size_t>(Name::Count));
    static const auto handles = [] {
        std::array<nvtxStringHandle_t, names.size()> out{};
        for (std::size_t i = 0; i < names.size(); ++i) {
            out[i] = nvtxDomainRegisterStringA(domain(), names[i]);
        }
        return out;
    }();
    return handles[static_cast<std::size_t>(name)];
}

[[nodiscard]] inline nvtxEventAttributes_t attributes(Name name, Category category,
                                                      std::uint64_t payload) noexcept {
    nvtxEventAttributes_t result{};
    result.version            = NVTX_VERSION;
    result.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    result.category           = static_cast<std::uint32_t>(category);
    result.colorType          = NVTX_COLOR_ARGB;
    result.color              = color(category);
    result.payloadType        = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    result.payload.ullValue   = payload;
    result.messageType        = NVTX_MESSAGE_TYPE_REGISTERED;
    result.message.registered = registered_message(name);
    return result;
}

class ScopedRange {
public:
    explicit ScopedRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : domain_(domain()) {
        const nvtxEventAttributes_t event = attributes(name, category, payload);
        nvtxDomainRangePushEx(domain_, &event);
    }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&)                 = delete;
    ScopedRange& operator=(ScopedRange&&)      = delete;

    ~ScopedRange() noexcept { nvtxDomainRangePop(domain_); }

private:
    nvtxDomainHandle_t domain_;
};

// Process ranges can begin and end on different threads. Use this only for ownership lifetimes;
// nested execution phases belong on ScopedRange so their stack hierarchy remains explicit.
class ScopedAsyncRange {
public:
    explicit ScopedAsyncRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : domain_(domain()) {
        const nvtxEventAttributes_t event = attributes(name, category, payload);
        id_                               = nvtxDomainRangeStartEx(domain_, &event);
    }

    ScopedAsyncRange(const ScopedAsyncRange&)            = delete;
    ScopedAsyncRange& operator=(const ScopedAsyncRange&) = delete;
    ScopedAsyncRange(ScopedAsyncRange&&)                 = delete;
    ScopedAsyncRange& operator=(ScopedAsyncRange&&)      = delete;

    ~ScopedAsyncRange() noexcept { nvtxDomainRangeEnd(domain_, id_); }

private:
    nvtxDomainHandle_t domain_;
    nvtxRangeId_t id_{};
};

} // namespace ninfer::nvtx
