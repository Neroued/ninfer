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

} // namespace ninfer::models::context
