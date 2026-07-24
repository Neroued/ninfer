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
};

enum class Name : std::size_t {
    Generate,
    Prefill,
    Decode,
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
        return out;
    }();
    return handle;
}

[[nodiscard]] inline nvtxStringHandle_t registered_message(Name name) noexcept {
    static constexpr std::array<const char*, static_cast<std::size_t>(Name::Count)> names{
        "generate",
        "prefill",
        "decode",
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
    };
    static const auto handles = [] {
        std::array<nvtxStringHandle_t, names.size()> out{};
        for (std::size_t i = 0; i < names.size(); ++i) {
            out[i] = nvtxDomainRegisterStringA(domain(), names[i]);
        }
        return out;
    }();
    return handles[static_cast<std::size_t>(name)];
}

class ScopedRange {
public:
    explicit ScopedRange(Name name, Category category, std::uint64_t payload = 0) noexcept
        : domain_(domain()) {
        nvtxEventAttributes_t attributes{};
        attributes.version            = NVTX_VERSION;
        attributes.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.category           = static_cast<std::uint32_t>(category);
        attributes.colorType          = NVTX_COLOR_ARGB;
        attributes.color              = color(category);
        attributes.payloadType        = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        attributes.payload.ullValue   = payload;
        attributes.messageType        = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = registered_message(name);
        nvtxDomainRangePushEx(domain_, &attributes);
    }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&)                 = delete;
    ScopedRange& operator=(ScopedRange&&)      = delete;

    ~ScopedRange() noexcept { nvtxDomainRangePop(domain_); }

private:
    nvtxDomainHandle_t domain_;
};

} // namespace ninfer::nvtx
