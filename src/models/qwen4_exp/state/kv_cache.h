#pragma once

#include "models/common/context/storage.h"
#include "ninfer/ops/qsa.h"
#include "models/qwen4_exp/config.h"

namespace ninfer::models::qwen4_exp {
struct QsaCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout tables;
    PagedKVStorageLayout storage;
    std::uint32_t layers;
    std::int32_t head_dim, kv_heads;
};

[[nodiscard]] QsaCacheLayout plan_qsa_cache(LayoutBuilder&, const QsaConfig&, std::uint32_t layers,
                                            std::uint32_t max_context, std::uint32_t pages,
                                            std::int32_t table_rows, KvCacheStorage);

class QsaCache {
public:
    QsaCache(MemorySpan, const QsaCacheLayout&);
    [[nodiscard]] ops::QsaCacheView layer(std::uint32_t) const;

    [[nodiscard]] context::DevicePageStorage& pages() { return pages_; }

    [[nodiscard]] const context::DevicePageStorage& pages() const { return pages_; }

    [[nodiscard]] context::ExecutionTableStorage& tables() { return tables_; }

    [[nodiscard]] const context::ExecutionTableStorage& tables() const { return tables_; }
private:
    QsaCacheLayout layout_;
    context::DevicePageStorage pages_;
    context::ExecutionTableStorage tables_;
};
} // namespace ninfer::models::qwen4_exp
