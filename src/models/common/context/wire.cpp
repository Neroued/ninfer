#include "models/common/context/wire.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::models::context {
namespace {

using distributed::wire::Encoder;
using distributed::wire::Decoder;

void range(Encoder& out, ContextRange value) {
    out.u8(static_cast<std::uint8_t>(value.memory));
    out.i64(value.offset);
    out.u64(value.bytes);
}

ContextRange range(Decoder& in) {
    return {static_cast<ContextMemory>(in.u8()), in.i64(), in.u64()};
}

void resource(Encoder& out, const runtime::ContextTransferResource& value) {
    out.boolean(value.pool.has_value());
    if (value.pool) out.u16(value.pool->index);
}

runtime::ContextTransferResource resource(Decoder& in) {
    runtime::ContextTransferResource value;
    if (in.boolean()) value.pool = runtime::KvPoolId{in.u16()};
    return value;
}

} // namespace

void encode_actions(Encoder& out, const ContextActions& actions) {
    out.u64(actions.operations.size());
    for (const auto& operation : actions.operations)
        std::visit(
            [&](const auto& action) {
                using T = std::decay_t<decltype(action)>;
                if constexpr (std::is_same_v<T, RecordClear>) {
                    out.u8(1);
                    range(out, action.destination);
                } else if constexpr (std::is_same_v<T, RecordCopy>) {
                    out.u8(2);
                    range(out, action.source);
                    range(out, action.destination);
                    resource(out, action.resource);
                    out.u32(action.image_count);
                    out.u32(action.page_groups);
                } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                    out.u8(3);
                    out.u32(action.pool_id);
                    out.i32(action.row);
                    out.u32(action.logical_begin);
                    out.u64(action.offsets.size());
                    for (auto offset : action.offsets) out.i64(offset);
                } else {
                    static_assert(std::is_same_v<T, PeerCopy>);
                    out.u8(4);
                    out.u32(action.peer);
                    out.boolean(action.send);
                    out.u64(action.ranges.size());
                    for (auto value : action.ranges) range(out, value);
                    resource(out, action.resource);
                    out.u32(action.image_count);
                    out.u32(action.page_groups);
                }
            },
            operation);
}

ContextActions decode_actions(Decoder& in) {
    ContextActions actions;
    const auto count = in.count(1);
    actions.operations.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        switch (in.u8()) {
        case 1:
            actions.operations.emplace_back(RecordClear{range(in)});
            break;
        case 2: {
            RecordCopy copy;
            copy.source      = range(in);
            copy.destination = range(in);
            copy.resource    = resource(in);
            copy.image_count = in.u32();
            copy.page_groups = in.u32();
            actions.operations.emplace_back(copy);
            break;
        }
        case 3: {
            MappingUpdate mapping;
            mapping.pool_id       = in.u32();
            mapping.row           = in.i32();
            mapping.logical_begin = in.u32();
            mapping.offsets.resize(in.count(8));
            for (auto& offset : mapping.offsets) offset = in.i64();
            actions.operations.emplace_back(std::move(mapping));
            break;
        }
        case 4: {
            PeerCopy copy;
            copy.peer = in.u32();
            copy.send = in.boolean();
            copy.ranges.resize(in.count(17));
            for (auto& value : copy.ranges) value = range(in);
            copy.resource    = resource(in);
            copy.image_count = in.u32();
            copy.page_groups = in.u32();
            actions.operations.emplace_back(std::move(copy));
            break;
        }
        default:
            throw std::invalid_argument("unknown Context action in message");
        }
    }
    return actions;
}

void encode_directory(distributed::wire::Encoder& out, const ContextDirectorySpec& c) {
    out.u64(c.ranks.size());
    for (const auto& rank : c.ranks) {
        out.u32(rank.rank);
        out.u64(rank.device_bytes);
        out.u64(rank.host_bytes);
    }
    out.u32(c.state.logical_capacity);
    out.u64(c.state.layouts.size());
    for (const auto& layout : c.state.layouts) {
        out.u32(layout.variant);
        out.u64(layout.parts.size());
        for (const auto& part : layout.parts) {
            out.u32(part.rank);
            out.u64(part.image_bytes);
            out.u64(part.alignment);
        }
    }
    out.u64(c.metadata_budget_bytes);
    out.u64(c.pools.size());
    for (const auto& pool : c.pools) {
        for (auto n : {pool.pool_id, pool.logical_pages, pool.histories, pool.page_capacity,
                       pool.table_rows})
            out.u32(n);
        out.u64(pool.parts.size());
        for (const auto& part : pool.parts) {
            out.u32(part.rank);
            out.u64(part.record.bytes);
            out.u64(part.record.alignment);
        }
    }
}

ContextDirectorySpec decode_directory(distributed::wire::Decoder& in) {
    ContextDirectorySpec c;
    c.ranks.resize(in.count(20));
    for (auto& rank : c.ranks) rank = {in.u32(), in.u64(), in.u64()};
    c.state.logical_capacity = in.u32();
    c.state.layouts.resize(in.count(12));
    for (auto& layout : c.state.layouts) {
        layout.variant = in.u32();
        layout.parts.resize(in.count(20));
        for (auto& part : layout.parts) part = {in.u32(), in.u64(), in.u64()};
    }
    c.metadata_budget_bytes = in.u64();
    c.pools.resize(in.count(28));
    for (auto& pool : c.pools) {
        pool.pool_id       = in.u32();
        pool.logical_pages = in.u32();
        pool.histories     = in.u32();
        pool.page_capacity = in.u32();
        pool.table_rows    = in.u32();
        pool.parts.resize(in.count(20));
        for (auto& part : pool.parts) part = {in.u32(), {in.u64(), in.u64()}};
    }
    return c;
}

} // namespace ninfer::models::context
