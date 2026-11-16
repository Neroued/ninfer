#include "models/common/context/prefix.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::models::context {
namespace {
using Digest = std::array<std::uint64_t, 2>;
constexpr Digest kInitial{1469598103934665603ULL, 7809847782465536322ULL};
constexpr Digest kPrime{1099511628211ULL, 14029467366897019727ULL};
constexpr auto kNone = UINT32_MAX;

void mix(Digest& digest, std::uint64_t value) {
    digest[0] = std::rotl(digest[0] ^ value, 17) * kPrime[0];
    digest[1] = std::rotl(digest[1] + (value ^ 0x9e3779b97f4a7c15ULL), 29) * kPrime[1];
}

struct Attribute {
    std::array<std::int32_t, 3> position{};
    std::uint8_t type                                          = 0;
    friend bool operator==(const Attribute&, const Attribute&) = default;
};

Attribute attribute(const InputIdentityView& input, std::uint32_t i) {
    return {input.position(i), input.type(i)};
}

void mix_media(Digest& digest, const MediaIdentityView& media) {
    const auto& item = *media.item;
    mix(digest, 0x6e696e6665722d76ULL);
    mix(digest, static_cast<std::uint8_t>(item.modality));
    mix(digest, item.grid.temporal);
    mix(digest, item.grid.height);
    mix(digest, item.grid.width);
    mix(digest, item.patch_begin);
    mix(digest, item.patch_count);
    for (auto byte : item.content_digest) mix(digest, byte);
    mix(digest, item.timestamps.size());
    for (auto stamp : item.timestamps)
        mix(digest, std::bit_cast<std::uint64_t>(stamp == 0 ? 0.0 : stamp));
    mix(digest, item.token_spans.size());
    for (auto span : item.token_spans) {
        mix(digest, span.begin);
        mix(digest, span.count);
    }
    mix(digest, media.preparation_digest != nullptr);
    if (media.preparation_digest)
        for (auto byte : *media.preparation_digest) mix(digest, byte);
}

void mix_token(Digest& digest, TokenId token, const Attribute& a) {
    mix(digest, 0x6e696e6665722d74ULL);
    mix(digest, static_cast<std::uint32_t>(token));
    mix(digest, a.type);
    for (auto p : a.position) mix(digest, static_cast<std::uint32_t>(p));
}

Digest first_key(TokenId token, const Attribute& a, const MediaIdentityView* media) {
    auto result = kInitial;
    if (media) mix_media(result, *media);
    mix_token(result, token, a);
    return result;
}

bool equal_media(const MediaIdentityView* a, const MediaIdentityView* b) {
    if (!a || !b) return a == b;
    if (bool(a->preparation_digest) != bool(b->preparation_digest)) return false;
    if (a->preparation_digest && *a->preparation_digest != *b->preparation_digest) return false;
    const auto& x = *a->item;
    const auto& y = *b->item;
    return x.modality == y.modality && x.grid.temporal == y.grid.temporal &&
           x.grid.height == y.grid.height && x.grid.width == y.grid.width &&
           x.patch_begin == y.patch_begin && x.patch_count == y.patch_count &&
           x.content_digest == y.content_digest && x.timestamps == y.timestamps &&
           x.token_spans.size() == y.token_spans.size() &&
           std::equal(x.token_spans.begin(), x.token_spans.end(), y.token_spans.begin(),
                      [](auto l, auto r) { return l.begin == r.begin && l.count == r.count; });
}

auto media_from(std::span<const MediaIdentityView> media, std::uint32_t frontier) {
    return std::lower_bound(media.begin(), media.end(), frontier, [](auto entry, auto f) {
        return entry.item->token_spans.front().begin < f;
    });
}

class MediaCursor {
public:
    MediaCursor(const InputIdentityView& input, std::uint32_t begin)
        : remaining_(media_from(input.media, begin), input.media.end()) {}

    const MediaIdentityView* take(std::uint32_t i) {
        if (remaining_.empty() || remaining_.front().item->token_spans.front().begin != i)
            return nullptr;
        const auto* result = &remaining_.front();
        remaining_         = remaining_.subspan(1);
        return result;
    }
private:
    std::span<const MediaIdentityView> remaining_;
};
} // namespace

std::array<std::int32_t, 3> InputIdentityView::position(std::uint32_t i) const {
    std::array<std::int32_t, 3> result;
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (i < positions[axis].size()) {
            result[axis] = positions[axis][i];
        } else {
            const auto value = std::int64_t(i) + generated_delta;
            if (value < INT32_MIN || value > INT32_MAX)
                throw std::out_of_range("prefix input position exceeds int32");
            result[axis] = static_cast<std::int32_t>(value);
        }
    }
    return result;
}

std::uint8_t InputIdentityView::type(std::uint32_t i) const {
    return i < types.size() ? types[i] : 0;
}

const MediaIdentityView* InputIdentityView::media_at(std::uint32_t i) const {
    const auto found = media_from(media, i);
    return found != media.end() && found->item->token_spans.front().begin == i ? &*found : nullptr;
}

PrefixKey PrefixFingerprints::key(std::uint32_t domain, const InputIdentityView& input,
                                  std::uint32_t frontier) const {
    if (frontier > input.size()) throw std::out_of_range("prefix key exceeds input");
    auto next = keys_.lower_bound(frontier);
    if (next != keys_.end() && next->first == frontier) return {next->second, frontier, domain};
    auto digest         = kInitial;
    std::uint32_t begin = 0;
    if (next != keys_.begin()) {
        const auto previous = std::prev(next);
        begin               = previous->first;
        digest              = previous->second;
    }
    MediaCursor media(input, begin);
    for (auto i = begin; i < frontier; ++i) {
        if (const auto* item = media.take(i)) mix_media(digest, *item);
        mix_token(digest, input.tokens[i], attribute(input, i));
    }
    keys_.emplace_hint(next, frontier, digest);
    return {digest, frontier, domain};
}

ReplayMedia::ReplayMedia(MetadataBudget& budget, std::shared_ptr<const ReplayMedia> previous,
                         std::shared_ptr<const frontend::VisionItem> item,
                         std::size_t& accounted_bytes)
    : memory_(budget), previous_(std::move(previous)), item_(std::move(item)),
      accounted_bytes_(accounted_bytes) {
    memory_.acquire(sizeof(*this));
    accounted_bytes_ += sizeof(*this);
}

ReplayMedia::~ReplayMedia() {
    accounted_bytes_ -= sizeof(*this);
    auto previous = std::move(previous_);
    while (previous && previous.use_count() == 1) {
        auto next = std::move(previous->previous_);
        previous.reset();
        previous = std::move(next);
    }
}

struct PrefixStore::Impl {
    struct Media {
        MetadataAccount memory;
        frontend::VisionItem item;
        std::optional<std::array<std::uint8_t, 32>> preparation;
        std::size_t* total;

        Media(MetadataBudget& budget, MediaIdentityView source, std::size_t& bytes)
            : memory(budget), item(*source.item), total(&bytes) {
            if (source.preparation_digest) preparation = *source.preparation_digest;
            memory.acquire(sizeof(*this) + item.timestamps.capacity() * sizeof(double) +
                           item.token_spans.capacity() * sizeof(frontend::TokenSpan));
            *total += memory.bytes();
        }

        ~Media() { *total -= memory.bytes(); }

        MediaIdentityView view() const { return {&item, preparation ? &*preparation : nullptr}; }
    };

    struct MediaEvent {
        std::uint32_t offset;
        std::shared_ptr<const Media> value;
    };

    struct Node {
        MetadataAccount memory;
        std::uint32_t parent = kNone, first_child = kNone;
        std::uint32_t previous = kNone, next = kNone, hash_next = kNone;
        std::uint32_t domain = 0, depth = 0, length = 0, references = 0;
        std::array<TokenId, block_tokens> tokens{};
        Attribute origin;
        std::array<std::int64_t, 3> stride{1, 1, 1};
        std::vector<Attribute> explicit_attributes;
        std::vector<MediaEvent> media;
        std::shared_ptr<const ReplayMedia> replay;
        Digest digest = kInitial, edge_key{};

        explicit Node(MetadataBudget& budget) : memory(budget) { memory.acquire(sizeof(*this)); }

        Attribute at(std::uint32_t i) const {
            if (!explicit_attributes.empty()) return explicit_attributes[i];
            auto result = origin;
            for (unsigned axis = 0; axis < 3; ++axis)
                result.position[axis] =
                    static_cast<std::int32_t>(origin.position[axis] + stride[axis] * i);
            return result;
        }
    };

    struct Slot {
        std::unique_ptr<Node> node;
        std::uint64_t generation = 1;
        std::uint32_t next_free  = kNone;
    };

    struct RelationMark {
        std::uint64_t epoch    = 0;
        std::uint32_t endpoint = kNone, nearest = kNone;
        bool resolved = false;
    };

    std::vector<RelationMark> relation_marks;
    std::vector<std::uint32_t> relation_path;
    std::uint64_t relation_epoch = 0;
    PrefixStore* owner;
    MetadataBudget& budget;
    MetadataAccount memory;
    std::vector<Slot> slots;
    std::vector<std::uint32_t> buckets;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> roots;
    std::uint32_t first_free = kNone;
    std::uint64_t revision   = 0;
    std::size_t live_nodes = 0, stored_tokens = 0, node_bytes = 0, media_bytes = 0;
    mutable std::uint64_t compared_tokens = 0;

    Impl(PrefixStore& owner, MetadataBudget& budget)
        : owner(&owner), budget(budget), memory(budget) {
        memory.acquire(sizeof(*this));
    }

    ~Impl() {
        for (const auto& slot : slots)
            if (slot.node && slot.node->references) std::terminate();
        slots.clear();
    }

    PrefixId id(std::uint32_t i) const { return {owner, i, slots[i].generation}; }

    bool valid(PrefixId h) const {
        return h.owner == owner && h.index < slots.size() && slots[h.index].node &&
               slots[h.index].generation == h.generation;
    }

    Node& require(PrefixId h) const {
        if (!valid(h)) throw std::logic_error("prefix handle is stale or belongs to another store");
        return *slots[h.index].node;
    }

    std::size_t bucket(std::uint32_t parent, Digest key) const {
        auto h = key[0] ^ std::rotl(key[1], 23) ^ (std::uint64_t(parent) * 0x9e3779b97f4a7c15ULL);
        h ^= h >> 31;
        return h & (buckets.size() - 1);
    }

    void hash_insert(std::uint32_t i) {
        auto& n     = *slots[i].node;
        auto& head  = buckets[bucket(n.parent, n.edge_key)];
        n.hash_next = head;
        head        = i;
    }

    void hash_remove(std::uint32_t i) {
        auto& n    = *slots[i].node;
        auto* next = &buckets[bucket(n.parent, n.edge_key)];
        while (*next != i) {
            if (*next == kNone) std::terminate();
            next = &slots[*next].node->hash_next;
        }
        *next       = n.hash_next;
        n.hash_next = kNone;
    }

    void prepare_slot() {
        if (first_free != kNone) return;
        if (slots.size() == UINT32_MAX) throw std::length_error("prefix store exceeds uint32");
        const auto capacity     = slots.size() == slots.capacity()
                                      ? std::max<std::size_t>(16, slots.size() * 2)
                                      : slots.capacity();
        const auto bucket_count = std::bit_ceil(capacity * 2);
        // A failed reserve may leave any earlier vector enlarged. Complete every buffer
        // before publishing its logical size or rebuilding the live hash chains.
        memory.reserve(relation_marks, capacity);
        memory.reserve(relation_path, capacity);
        memory.reserve(slots, capacity);
        memory.reserve(buckets, bucket_count);
        relation_marks.resize(capacity);
        if (buckets.size() != bucket_count) {
            buckets.assign(bucket_count, kNone);
            for (std::uint32_t i = 0; i < slots.size(); ++i)
                if (slots[i].node && slots[i].node->parent != kNone) hash_insert(i);
        }
    }

    std::uint32_t allocate() {
        prepare_slot();
        auto node = std::make_unique<Node>(budget);
        std::uint32_t i;
        if (first_free == kNone) {
            i = static_cast<std::uint32_t>(slots.size());
            slots.emplace_back();
        } else {
            i          = first_free;
            first_free = slots[i].next_free;
        }
        slots[i].node = std::move(node);
        ++live_nodes;
        node_bytes += sizeof(Node);
        return i;
    }

    void free(std::uint32_t i) {
        auto& slot = slots[i];
        node_bytes -= slot.node->memory.bytes();
        stored_tokens -= slot.node->length;
        slot.node.reset();
        ++slot.generation;
        slot.next_free = first_free;
        first_free     = i;
        --live_nodes;
        ++revision;
    }

    void dense(Node& n) {
        if (!n.explicit_attributes.empty()) return;
        const auto before = n.memory.bytes();
        n.memory.reserve(n.explicit_attributes, block_tokens);
        node_bytes += n.memory.bytes() - before;
        n.explicit_attributes.resize(n.length);
        for (std::uint32_t i = 0; i < n.length; ++i) {
            auto a = n.origin;
            for (unsigned axis = 0; axis < 3; ++axis)
                a.position[axis] =
                    static_cast<std::int32_t>(n.origin.position[axis] + n.stride[axis] * i);
            n.explicit_attributes[i] = a;
        }
    }

    void reserve_media(Node& n, std::size_t count) {
        const auto before = n.memory.bytes();
        n.memory.reserve(n.media, count);
        node_bytes += n.memory.bytes() - before;
    }

    void prepare_merge(Node& child, const Node& parent) {
        std::array<Attribute, block_tokens> values;
        auto count  = child.length;
        auto events = child.media.size();
        for (std::uint32_t i = 0; i < child.length; ++i)
            values[block_tokens - count + i] = child.at(i);
        const Node* previous = &parent;
        while (previous->length && count + previous->length <= block_tokens) {
            count += previous->length;
            events += previous->media.size();
            for (std::uint32_t i = 0; i < previous->length; ++i)
                values[block_tokens - count + i] = previous->at(i);
            previous = slots[previous->parent].node.get();
        }
        if (!child.media.empty()) reserve_media(child, events);
        if (count < 2 || !child.explicit_attributes.empty()) return;
        const auto offset = block_tokens - count;
        const auto origin = values[offset];
        std::array<std::int64_t, 3> stride;
        for (unsigned a = 0; a < 3; ++a)
            stride[a] = std::int64_t(values[offset + 1].position[a]) - origin.position[a];
        for (std::uint32_t i = 1; i < count; ++i) {
            bool affine = values[offset + i].type == origin.type;
            for (unsigned a = 0; a < 3; ++a)
                affine &= values[offset + i].position[a] == origin.position[a] + stride[a] * i;
            if (!affine) {
                dense(child);
                return;
            }
        }
    }

    void attributes(Node& n, std::span<const Attribute> values) {
        n.origin = values.front();
        n.stride = {1, 1, 1};
        if (values.size() > 1)
            for (unsigned a = 0; a < 3; ++a)
                n.stride[a] = std::int64_t(values[1].position[a]) - values[0].position[a];
        bool affine = true;
        for (std::size_t i = 0; i < values.size() && affine; ++i) {
            affine = values[i].type == n.origin.type;
            for (unsigned a = 0; a < 3; ++a)
                affine &=
                    std::int64_t(values[i].position[a]) == n.origin.position[a] + n.stride[a] * i;
        }
        if (!affine) {
            dense(n);
            std::copy(values.begin(), values.end(), n.explicit_attributes.begin());
        }
    }

    void link(std::uint32_t i, std::uint32_t parent) {
        auto& n    = *slots[i].node;
        n.parent   = parent;
        n.previous = kNone;
        n.next     = slots[parent].node->first_child;
        if (n.next != kNone) slots[n.next].node->previous = i;
        slots[parent].node->first_child = i;
        hash_insert(i);
    }

    void unlink(std::uint32_t i) {
        auto& n = *slots[i].node;
        hash_remove(i);
        if (n.previous == kNone)
            slots[n.parent].node->first_child = n.next;
        else
            slots[n.previous].node->next = n.next;
        if (n.next != kNone) slots[n.next].node->previous = n.previous;
        n.previous = n.next = kNone;
    }

    std::uint32_t find_root(std::uint32_t domain) const {
        for (auto [key, index] : roots)
            if (key == domain) return index;
        return kNone;
    }

    std::uint32_t ensure_root(std::uint32_t domain) {
        if (const auto found = find_root(domain); found != kNone) return found;
        memory.reserve(roots, roots.size() + 1);
        const auto i          = allocate();
        slots[i].node->domain = domain;
        roots.emplace_back(domain, i);
        ++revision;
        return i;
    }

    Digest digest_to(const Node& n, std::uint32_t count) const {
        auto digest       = slots[n.parent].node->digest;
        std::size_t event = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            if (event < n.media.size() && n.media[event].offset == i)
                mix_media(digest, n.media[event++].value->view());
            mix_token(digest, n.tokens[i], n.at(i));
        }
        return digest;
    }

    std::uint32_t common(const Node& n, const InputIdentityView& view, std::uint32_t begin,
                         std::uint32_t end) const {
        const auto count = std::min(n.length, end - begin);
        MediaCursor input_media(view, begin);
        std::size_t event = 0;
        std::uint32_t i   = 0;
        for (; i < count; ++i) {
            ++compared_tokens;
            if (n.tokens[i] != view.tokens[begin + i] || n.at(i) != attribute(view, begin + i))
                break;
            const auto media = event < n.media.size() && n.media[event].offset == i
                                   ? std::optional(n.media[event++].value->view())
                                   : std::nullopt;
            if (!equal_media(media ? &*media : nullptr, input_media.take(begin + i))) break;
        }
        return i;
    }

    std::pair<std::uint32_t, std::uint32_t> child(std::uint32_t parent,
                                                  const InputIdentityView& view,
                                                  std::uint32_t begin, std::uint32_t end) const {
        const auto key =
            first_key(view.tokens[begin], attribute(view, begin), view.media_at(begin));
        auto i = buckets[bucket(parent, key)];
        while (i != kNone) {
            const auto& n = *slots[i].node;
            if (n.parent == parent && n.edge_key == key) {
                const auto count = common(n, view, begin, end);
                if (count) return {i, count};
            }
            i = n.hash_next;
        }
        return {kNone, 0};
    }

    std::uint32_t append(std::uint32_t parent, const InputIdentityView& view, std::uint32_t begin,
                         std::uint32_t end) {
        const auto i = allocate();
        try {
            auto& n  = *slots[i].node;
            n.domain = slots[parent].node->domain;
            n.length = std::min(block_tokens, end - begin);
            stored_tokens += n.length;
            n.depth = begin + n.length;
            view.tokens.copy(begin, std::span(n.tokens).first(n.length));
            std::array<Attribute, block_tokens> values;
            for (std::uint32_t j = 0; j < n.length; ++j) values[j] = attribute(view, begin + j);
            attributes(n, std::span(values).first(n.length));
            n.replay         = slots[parent].node->replay;
            const auto first = media_from(view.media, begin);
            const auto last  = media_from(view.media, begin + n.length);
            reserve_media(n, last - first);
            for (auto item = first; item != last; ++item) {
                const auto media = std::make_shared<Media>(budget, *item, media_bytes);
                n.media.push_back(
                    {std::uint32_t(item->item->token_spans.front().begin - begin), media});
                n.replay = std::make_shared<ReplayMedia>(
                    budget, n.replay,
                    std::shared_ptr<const frontend::VisionItem>(media, &media->item), media_bytes);
            }
            const auto m = !n.media.empty() && n.media[0].offset == 0
                               ? std::optional(n.media[0].value->view())
                               : std::nullopt;
            n.edge_key   = first_key(n.tokens[0], n.at(0), m ? &*m : nullptr);
            prepare_merge(n, *slots[parent].node);
            n.parent = parent;
            n.digest = digest_to(n, n.length);
            link(i, parent);
            ++revision;
            return i;
        } catch (...) {
            free(i);
            throw;
        }
    }

    std::uint32_t split(std::uint32_t child, std::uint32_t count) {
        const auto i = allocate();
        auto& c      = *slots[child].node;
        auto& n      = *slots[i].node;
        try {
            n.domain = c.domain;
            n.length = count;
            stored_tokens += count;
            n.depth = c.depth - c.length + count;
            std::copy_n(c.tokens.begin(), count, n.tokens.begin());
            std::array<Attribute, block_tokens> values;
            for (std::uint32_t j = 0; j < count; ++j) values[j] = c.at(j);
            attributes(n, std::span(values).first(count));
            reserve_media(n, std::count_if(c.media.begin(), c.media.end(), [&](const auto& event) {
                              return event.offset < count;
                          }));
            for (const auto& event : c.media) {
                if (event.offset >= count) break;
                n.media.push_back(event);
            }
            prepare_merge(n, *slots[c.parent].node);
            n.replay = c.replay;
            while (n.replay && n.replay->item().token_spans.front().begin >= n.depth)
                n.replay = n.replay->previous();
            n.edge_key = c.edge_key;
            n.parent   = c.parent;
            n.digest   = digest_to(c, count);
        } catch (...) {
            free(i);
            throw;
        }
        const auto parent = c.parent;
        unlink(child);
        link(i, parent);
        const auto origin = c.at(count);
        std::move(c.tokens.begin() + count, c.tokens.begin() + c.length, c.tokens.begin());
        if (!c.explicit_attributes.empty()) {
            std::move(c.explicit_attributes.begin() + count, c.explicit_attributes.end(),
                      c.explicit_attributes.begin());
            c.explicit_attributes.resize(c.length - count);
        }
        c.origin = origin;
        c.length -= count;
        stored_tokens -= count;
        std::erase_if(c.media, [&](const auto& event) { return event.offset < count; });
        for (auto& event : c.media) event.offset -= count;
        const auto m = !c.media.empty() && c.media[0].offset == 0
                           ? std::optional(c.media[0].value->view())
                           : std::nullopt;
        c.edge_key   = first_key(c.tokens[0], c.at(0), m ? &*m : nullptr);
        link(child, i);
        ++revision;
        return i;
    }

    bool merge(std::uint32_t parent) {
        auto& p = *slots[parent].node;
        if (p.parent == kNone || p.references || p.first_child == kNone) return false;
        const auto child = p.first_child;
        auto& c          = *slots[child].node;
        if (c.next != kNone || p.length + c.length > block_tokens) return false;
        std::array<Attribute, block_tokens> values;
        const auto length = p.length + c.length;
        for (std::uint32_t i = 0; i < p.length; ++i) values[i] = p.at(i);
        for (std::uint32_t i = 0; i < c.length; ++i) values[p.length + i] = c.at(i);
        auto stride = p.stride;
        if (p.length == 1)
            for (unsigned axis = 0; axis < 3; ++axis)
                stride[axis] = std::int64_t(values[1].position[axis]) - values[0].position[axis];
        bool affine = true;
        for (std::uint32_t i = 0; i < length && affine; ++i) {
            affine = values[i].type == values[0].type;
            for (unsigned axis = 0; axis < 3; ++axis)
                affine &= std::int64_t(values[i].position[axis]) ==
                          values[0].position[axis] + stride[axis] * i;
        }
        if (!affine && c.explicit_attributes.empty() && p.explicit_attributes.empty())
            std::terminate();
        const auto grandparent = p.parent;
        unlink(child);
        unlink(parent);
        if (!affine) {
            if (c.explicit_attributes.empty()) {
                const auto bytes = p.explicit_attributes.capacity() * sizeof(Attribute);
                p.memory.transfer(bytes, c.memory);
                c.explicit_attributes = std::move(p.explicit_attributes);
            }
            c.explicit_attributes.resize(length);
            std::copy_n(values.begin(), length, c.explicit_attributes.begin());
        } else if (!c.explicit_attributes.empty()) {
            c.explicit_attributes.resize(length);
            std::copy_n(values.begin(), length, c.explicit_attributes.begin());
        }
        std::move_backward(c.tokens.begin(), c.tokens.begin() + c.length,
                           c.tokens.begin() + length);
        std::copy_n(p.tokens.begin(), p.length, c.tokens.begin());
        c.origin = values[0];
        c.stride = stride;
        c.length = length;
        stored_tokens += p.length;
        if (!p.media.empty() || !c.media.empty()) {
            if (c.media.empty()) {
                const auto bytes = p.media.capacity() * sizeof(MediaEvent);
                // A split may leave an empty, preallocated vector in the child.
                const auto prior = c.media.capacity() * sizeof(MediaEvent);
                std::vector<MediaEvent>().swap(c.media);
                c.memory.release(prior);
                node_bytes -= prior;
                p.memory.transfer(bytes, c.memory);
                c.media = std::move(p.media);
            } else {
                for (auto& event : c.media) event.offset += p.length;
                c.media.insert(c.media.begin(), p.media.begin(), p.media.end());
            }
        }
        c.edge_key = p.edge_key;
        link(child, grandparent);
        p.first_child = kNone;
        free(parent);
        return true;
    }

    void collect(std::uint32_t i) noexcept {
        while (slots[i].node && slots[i].node->parent != kNone && !slots[i].node->references) {
            auto& n           = *slots[i].node;
            const auto parent = n.parent;
            if (n.first_child != kNone) {
                if (!merge(i)) break;
                i = parent;
                continue;
            }
            unlink(i);
            free(i);
            i = parent;
        }
    }
};

PrefixStore::PrefixStore(MetadataBudget& budget) : impl_(std::make_unique<Impl>(*this, budget)) {}

PrefixStore::~PrefixStore() = default;

bool PrefixStore::valid(PrefixId h) const noexcept { return impl_->valid(h); }

std::uint32_t PrefixStore::frontier(PrefixId h) const { return impl_->require(h).depth; }

std::uint32_t PrefixStore::domain(PrefixId h) const { return impl_->require(h).domain; }

std::uint64_t PrefixStore::revision() const noexcept { return impl_->revision; }

PrefixId PrefixStore::parent(PrefixId h) const {
    const auto p = impl_->require(h).parent;
    return p == kNone ? PrefixId{} : impl_->id(p);
}

PrefixMatch PrefixStore::match(std::uint32_t domain, const InputIdentityView& view,
                               std::uint32_t limit, PrefixId proven_base) const {
    if (limit > view.size()) throw std::out_of_range("prefix lookup exceeds its input");
    auto current = impl_->find_root(domain);
    if (proven_base.owner) {
        const auto& base = impl_->require(proven_base);
        if (base.domain != domain || base.depth > limit)
            throw std::invalid_argument("prefix lookup has an invalid proof base");
        current = proven_base.index;
    }
    PrefixMatch result{
        {}, current == kNone ? 0U : impl_->slots[current].node->depth, domain, impl_->revision};
    if (current == kNone) return result;
    result.edge = impl_->id(current);
    while (result.frontier < limit) {
        const auto [child, count] = impl_->child(current, view, result.frontier, limit);
        if (child == kNone) break;
        result.edge = impl_->id(child);
        result.frontier += count;
        if (count != impl_->slots[child].node->length) break;
        current = child;
    }
    return result;
}

bool PrefixStore::matches(PrefixId h, const InputIdentityView& view) const {
    const auto& n = impl_->require(h);
    if (n.depth > view.size()) return false;
    const auto found = match(n.domain, view, n.depth);
    return found.frontier == n.depth && found.edge == h;
}

bool PrefixStore::is_prefix(PrefixId earlier, PrefixId later) const {
    const auto& a = impl_->require(earlier);
    auto i        = later.index;
    const auto& b = impl_->require(later);
    if (a.domain != b.domain || a.depth > b.depth) return false;
    while (impl_->slots[i].node->depth > a.depth) i = impl_->slots[i].node->parent;
    return impl_->id(i) == earlier;
}

PrefixKey PrefixStore::key(PrefixId h) const {
    const auto& n = impl_->require(h);
    return {n.digest, n.depth, n.domain};
}

std::vector<PrefixId> PrefixStore::ancestors(PrefixMatch match) const {
    if (match.revision != impl_->revision) throw std::logic_error("prefix lookup cursor is stale");
    std::vector<PrefixId> result;
    if (!match.edge.owner) return result;
    auto i = match.edge.index;
    (void)impl_->require(match.edge);
    while (impl_->slots[i].node->parent != kNone) {
        const auto& n = *impl_->slots[i].node;
        if (n.depth <= match.frontier) result.push_back(impl_->id(i));
        i = n.parent;
    }
    return result;
}

std::shared_ptr<const ReplayMedia> PrefixStore::replay_media(PrefixId h) const {
    return impl_->require(h).replay;
}

PrefixRef PrefixStore::prepare(std::uint32_t domain, const InputIdentityView& view, std::uint32_t f,
                               const PrefixRef* base) {
    if (f > view.size()) throw std::out_of_range("prepared prefix exceeds its input");
    auto current        = impl_->ensure_root(domain);
    std::uint32_t begin = 0;
    if (base && *base) {
        const auto& n = impl_->require(base->id());
        if (n.domain != domain || n.depth > f)
            throw std::invalid_argument("invalid prefix extension base");
        current = base->id().index;
        begin   = n.depth;
    }
    try {
        while (begin < f) {
            const auto [child, count] = impl_->child(current, view, begin, f);
            if (child == kNone)
                current = impl_->append(current, view, begin, f);
            else if (count < impl_->slots[child].node->length)
                current = impl_->split(child, count);
            else
                current = child;
            begin = impl_->slots[current].node->depth;
        }
        return PrefixRef(*this, impl_->id(current));
    } catch (...) {
        impl_->collect(current);
        throw;
    }
}

std::vector<runtime::PrefixRelation> PrefixStore::relations(std::span<const PrefixId> ids) const {
    std::vector<runtime::PrefixRelation> out(ids.size());
    auto& p = *impl_;
    if (!++p.relation_epoch) {
        std::fill(p.relation_marks.begin(), p.relation_marks.end(), Impl::RelationMark{});
        ++p.relation_epoch;
    }
    const auto mark = [&](std::uint32_t index) -> Impl::RelationMark& {
        auto& entry = p.relation_marks[index];
        if (entry.epoch != p.relation_epoch) entry = {.epoch = p.relation_epoch};
        return entry;
    };
    for (std::uint32_t i = 0; i < ids.size(); ++i) {
        if (!valid(ids[i])) continue;
        auto& entry = mark(ids[i].index);
        if (entry.endpoint == kNone) entry.endpoint = i;
        out[i].representative = entry.endpoint;
    }
    for (std::uint32_t i = 0; i < ids.size(); ++i) {
        if (out[i].representative == kNone) continue;
        p.relation_path.clear();
        auto node     = p.require(ids[i]).parent;
        auto ancestor = kNone;
        while (node != kNone) {
            auto& entry = mark(node);
            if (entry.endpoint != kNone) {
                ancestor = entry.endpoint;
                break;
            }
            if (entry.resolved) {
                ancestor = entry.nearest;
                break;
            }
            p.relation_path.push_back(node);
            node = p.slots[node].node->parent;
        }
        for (auto index : p.relation_path) {
            auto& entry    = mark(index);
            entry.resolved = true;
            entry.nearest  = ancestor;
        }
        out[i].parent = ancestor;
    }
    return out;
}

PrefixId PrefixStore::ancestor(PrefixId id, std::uint32_t f) const {
    if (!valid(id)) return {};
    while (frontier(id) > f) id = parent(id);
    return id;
}

PrefixMatch PrefixQuery::match() const {
    if (!store_) throw std::logic_error("unbound request prefix query");
    if (!store_->valid(matched_.edge) || matched_.revision != store_->revision()) {
        const auto base = store_->ancestor(matched_.edge, matched_.frontier);
        matched_        = store_->match(domain_, input_, input_.size(), base);
        proof_          = store_->ancestors(matched_);
        std::sort(proof_.begin(), proof_.end(), [](auto a, auto b) { return a.index < b.index; });
    }
    return matched_;
}

bool PrefixQuery::matches(PrefixId id) const {
    if (!store_ || !store_->valid(id) || store_->domain(id) != domain_) return false;
    const auto proven = [&] {
        const auto found =
            std::lower_bound(proof_.begin(), proof_.end(), id.index,
                             [](auto value, auto index) { return value.index < index; });
        return found != proof_.end() && *found == id;
    };
    if (proven()) return true;
    (void)match();
    return proven();
}

std::vector<PrefixId> PrefixQuery::ancestors() const { return store_->ancestors(match()); }

PrefixId PrefixQuery::endpoint(std::uint32_t f) const {
    const auto found = match();
    return store_->ancestor(found.edge, std::min(f, found.frontier));
}

PrefixRef PrefixStore::retain(PrefixId id) { return PrefixRef(*this, id); }

void PrefixStore::release(PrefixId id) noexcept {
    if (!impl_->valid(id) || !impl_->slots[id.index].node->references) std::terminate();
    --impl_->slots[id.index].node->references;
    impl_->collect(id.index);
}

PrefixUsage PrefixStore::usage() const noexcept {
    return {impl_->live_nodes, impl_->stored_tokens,
            impl_->memory.bytes() + impl_->node_bytes + impl_->media_bytes, impl_->compared_tokens};
}

std::size_t PrefixStore::metadata_bytes(std::size_t tokens, std::size_t endpoints,
                                        PrefixMetadataLayout layout) {
    // Unreferenced unary edges coalesce up to block_tokens. Every shorter boundary is
    // accounted by an endpoint/branch; slot/hash growth and relation projection are included.
    const auto node_bytes =
        sizeof(Impl::Node) + 2 * sizeof(Impl::Slot) + 8 * sizeof(std::uint32_t) + 64;
    const auto ceil   = [](std::size_t a, std::size_t b) { return a / b + (a % b != 0); };
    auto per_token    = ceil(2 * node_bytes, block_tokens);
    auto per_endpoint = 2 * node_bytes;
    if (layout.explicit_positions) {
        per_token += 2 * sizeof(Attribute);
        per_endpoint += 2 * block_tokens * sizeof(Attribute);
    }
    if (layout.media_stride) {
        if (!layout.span_stride)
            throw std::invalid_argument("media identity requires a span stride");
        const auto item_bytes = sizeof(Impl::Media) + sizeof(ReplayMedia) + 64;
        per_token += ceil(item_bytes + 2 * sizeof(Impl::MediaEvent), layout.media_stride) +
                     ceil(sizeof(double) + sizeof(frontend::TokenSpan), layout.span_stride);
        // Short edges and a prepared coalescence can each round up one media item.
        per_endpoint +=
            2 * item_bytes + 2 * ceil(block_tokens, layout.media_stride) * sizeof(Impl::MediaEvent);
    }
    return sizeof(Impl) + 16 * node_bytes + tokens * per_token + endpoints * per_endpoint;
}

PrefixRef::PrefixRef(PrefixStore& store, PrefixId id) : store_(&store), id_(id) {
    ++store.impl_->require(id).references;
}

PrefixRef::~PrefixRef() { reset(); }

PrefixRef::PrefixRef(const PrefixRef& other) : store_(other.store_), id_(other.id_) {
    if (store_) ++store_->impl_->require(id_).references;
}

PrefixRef& PrefixRef::operator=(const PrefixRef& other) {
    if (this != &other) {
        PrefixRef copy(other);
        *this = std::move(copy);
    }
    return *this;
}

PrefixRef::PrefixRef(PrefixRef&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), id_(other.id_) {}

PrefixRef& PrefixRef::operator=(PrefixRef&& other) noexcept {
    if (this != &other) {
        reset();
        store_ = std::exchange(other.store_, nullptr);
        id_    = other.id_;
    }
    return *this;
}

void PrefixRef::reset() noexcept {
    if (auto* store = std::exchange(store_, nullptr)) store->release(id_);
    id_ = {};
}

std::uint32_t PrefixRef::frontier() const { return store_ ? store_->frontier(id_) : 0; }

PrefixKey PrefixRef::key() const {
    if (!store_) throw std::logic_error("empty prefix reference has no key");
    return store_->key(id_);
}

PrefixBuilder::PrefixBuilder(PrefixStore& store, std::uint32_t domain, PrefixRef prefix)
    : store_(&store), domain_(domain), frontier_(prefix.frontier()), prefix_(std::move(prefix)) {
    if (prefix_ && store.domain(prefix_.id()) != domain)
        throw std::invalid_argument("prefix builder has a foreign identity domain");
}

PrefixUpdate PrefixBuilder::prepare(const InputIdentityView& view, std::uint32_t f,
                                    bool seal) const {
    if (!store_ || f < frontier_ || f > view.size())
        throw std::invalid_argument("invalid committed prefix update");
    const auto end = seal ? f
                          : std::max(prefix_.frontier(),
                                     f / PrefixStore::block_tokens * PrefixStore::block_tokens);
    return {end == prefix_.frontier() ? prefix_ : store_->prepare(domain_, view, end, &prefix_), f};
}

void PrefixBuilder::commit(PrefixUpdate&& update) noexcept {
    prefix_   = std::move(update.prefix);
    frontier_ = update.frontier;
}

PrefixRef PrefixBuilder::try_checkpoint(const InputIdentityView& view, std::uint32_t f) const {
    if (!store_ || f > view.size()) throw std::invalid_argument("invalid checkpoint prefix");
    try {
        return store_->prepare(domain_, view, f, prefix_.frontier() <= f ? &prefix_ : nullptr);
    } catch (const MetadataExhausted&) { return {}; }
}

} // namespace ninfer::models::context
