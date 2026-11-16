#pragma once

#include "models/common/context/directory/metadata.h"
#include "models/common/context/request_ledger.h"
#include "models/common/frontend/media.h"
#include "runtime/contract/recovery.h"

#include <array>
#include <cstdint>
#include <memory>
#include <map>
#include <span>
#include <vector>

namespace ninfer::models::context {

using PrefixId  = runtime::PrefixId;
using PrefixKey = runtime::PrefixKey;

struct MediaIdentityView {
    const frontend::VisionItem* item                       = nullptr;
    const std::array<std::uint8_t, 32>* preparation_digest = nullptr;
};

// Input coordinates are borrowed from RequestPlan. Missing axes mean token ordinals;
// generated coordinates follow ordinal + generated_delta on all axes.
struct InputIdentityView {
    TokenView tokens;
    std::span<const std::uint8_t> types;
    std::array<std::span<const std::int32_t>, 3> positions;
    std::int32_t generated_delta = 0;
    std::span<const MediaIdentityView> media;

    [[nodiscard]] std::size_t size() const noexcept { return tokens.size(); }

    [[nodiscard]] std::array<std::int32_t, 3> position(std::uint32_t) const;
    [[nodiscard]] std::uint8_t type(std::uint32_t) const;
    [[nodiscard]] const MediaIdentityView* media_at(std::uint32_t) const;
};

// Sparse, request-owned demand evidence. Computing a later key resumes at the nearest
// already requested frontier; no digest array grows with every token.
class PrefixFingerprints {
public:
    [[nodiscard]] PrefixKey key(std::uint32_t domain, const InputIdentityView&,
                                std::uint32_t frontier) const;
private:
    mutable std::map<std::uint32_t, std::array<std::uint64_t, 2>> keys_;
};

// Reconstruction metadata is shared independently of token storage. Policy quotes may
// retain this spine after eviction without keeping any token/coordinate blocks alive.
class ReplayMedia {
public:
    ReplayMedia(MetadataBudget&, std::shared_ptr<const ReplayMedia> previous,
                std::shared_ptr<const frontend::VisionItem> item, std::size_t& accounted_bytes);
    ~ReplayMedia();

    [[nodiscard]] const auto& previous() const noexcept { return previous_; }

    [[nodiscard]] const auto& item() const noexcept { return *item_; }
private:
    MetadataAccount memory_;
    mutable std::shared_ptr<const ReplayMedia> previous_;
    std::shared_ptr<const frontend::VisionItem> item_;
    std::size_t& accounted_bytes_;
};

class PrefixStore;

// Pins precisely one semantic endpoint and the path needed to identify it. It does not
// own a request, media payload, State or KV. The store outlives all owning references.
class PrefixRef {
public:
    PrefixRef() = default;
    ~PrefixRef();
    PrefixRef(const PrefixRef&);
    PrefixRef& operator=(const PrefixRef&);
    PrefixRef(PrefixRef&&) noexcept;
    PrefixRef& operator=(PrefixRef&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept { return store_ != nullptr; }

    [[nodiscard]] PrefixId id() const noexcept { return id_; }

    [[nodiscard]] std::uint32_t frontier() const;
    [[nodiscard]] PrefixKey key() const;
    void reset() noexcept;

private:
    PrefixRef(PrefixStore&, PrefixId);
    PrefixStore* store_ = nullptr;
    PrefixId id_;
    friend class PrefixStore;
};

// The matched frontier may lie inside an edge. It is an input-identity proof, not a
// promise that a complete checkpoint exists there. Mutation invalidates this cursor.
struct PrefixMatch {
    PrefixId edge;
    std::uint32_t frontier = 0, domain = 0;
    std::uint64_t revision = 0;
};

struct PrefixUsage {
    std::size_t nodes = 0, stored_tokens = 0, owned_bytes = 0;
    std::uint64_t compared_tokens = 0;
};

// Bounds supplied by the input producer. Zero media stride means text-only input;
// nonzero strides are minimum token distances between items and between frame spans.
struct PrefixMetadataLayout {
    bool explicit_positions    = false;
    std::uint32_t media_stride = 0, span_stride = 0;
};

class PrefixStore {
public:
    static constexpr std::uint32_t block_tokens = 256;

    explicit PrefixStore(MetadataBudget&);
    ~PrefixStore();
    PrefixStore(const PrefixStore&)            = delete;
    PrefixStore& operator=(const PrefixStore&) = delete;

    [[nodiscard]] PrefixMatch match(std::uint32_t domain, const InputIdentityView&,
                                    std::uint32_t limit, PrefixId proven_base = {}) const;
    [[nodiscard]] bool matches(PrefixId, const InputIdentityView&) const;
    [[nodiscard]] bool is_prefix(PrefixId earlier, PrefixId later) const;
    [[nodiscard]] bool valid(PrefixId) const noexcept;
    [[nodiscard]] std::uint32_t frontier(PrefixId) const;
    [[nodiscard]] std::uint32_t domain(PrefixId) const;
    [[nodiscard]] PrefixKey key(PrefixId) const;
    [[nodiscard]] PrefixId parent(PrefixId) const;
    [[nodiscard]] PrefixId ancestor(PrefixId, std::uint32_t frontier) const;
    [[nodiscard]] std::vector<PrefixId> ancestors(PrefixMatch) const;
    [[nodiscard]] std::shared_ptr<const ReplayMedia> replay_media(PrefixId) const;
    [[nodiscard]] std::vector<runtime::PrefixRelation> relations(std::span<const PrefixId>) const;

    // Preparation interns only the requested prefix. The optional base is an already
    // proved prefix of the same immutable request input. The result is not published to
    // the checkpoint index; dropping it cancels its ownership without retaining suffixes.
    [[nodiscard]] PrefixRef prepare(std::uint32_t domain, const InputIdentityView&,
                                    std::uint32_t frontier, const PrefixRef* base = nullptr);
    [[nodiscard]] PrefixRef retain(PrefixId);
    [[nodiscard]] PrefixUsage usage() const noexcept;
    [[nodiscard]] std::uint64_t revision() const noexcept;

    // Storage accounting is based on actual token coverage and endpoint bounds, not
    // the number of old requests. Includes allocation growth and one bounded split.
    [[nodiscard]] static std::size_t metadata_bytes(std::size_t tokens, std::size_t endpoints,
                                                    PrefixMetadataLayout = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void release(PrefixId) noexcept;
    friend class PrefixRef;
};

// One immutable request input. A prior exact match remains a proof for its surviving
// ancestors; catalog mutations only require a new traversal when that proof is insufficient.
class PrefixQuery {
public:
    PrefixQuery() = default;

    PrefixQuery(const PrefixStore& store, std::uint32_t domain, InputIdentityView input)
        : store_(&store), domain_(domain), input_(input) {}

    [[nodiscard]] bool matches(PrefixId) const;
    [[nodiscard]] std::vector<PrefixId> ancestors() const;
    [[nodiscard]] PrefixId endpoint(std::uint32_t frontier) const;

    [[nodiscard]] PrefixKey key(std::uint32_t f) const {
        return fingerprints_.key(domain_, input_, f);
    }
private:
    [[nodiscard]] PrefixMatch match() const;
    const PrefixStore* store_ = nullptr;
    std::uint32_t domain_     = 0;
    InputIdentityView input_;
    mutable PrefixMatch matched_;
    mutable std::vector<PrefixId> proof_;
    PrefixFingerprints fingerprints_;
};

struct PrefixUpdate {
    PrefixRef prefix;
    std::uint32_t frontier = 0;
};

// The unsealed tail is a view into RequestLedger, never another full prompt copy.
// Updates are prepared before irreversible commit and adopted without allocation.
class PrefixBuilder {
public:
    PrefixBuilder() = default;
    PrefixBuilder(PrefixStore&, std::uint32_t domain, PrefixRef prefix = {});

    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }

    [[nodiscard]] const PrefixRef& prefix() const noexcept { return prefix_; }

    [[nodiscard]] PrefixUpdate prepare(const InputIdentityView&, std::uint32_t frontier,
                                       bool seal = false) const;
    void commit(PrefixUpdate&&) noexcept;
    [[nodiscard]] PrefixRef try_checkpoint(const InputIdentityView&, std::uint32_t frontier) const;
private:
    PrefixStore* store_   = nullptr;
    std::uint32_t domain_ = 0, frontier_ = 0;
    PrefixRef prefix_;
};

} // namespace ninfer::models::context
