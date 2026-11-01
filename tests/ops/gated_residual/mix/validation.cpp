#include "support.h"
#include <array>

namespace ninfer::test::hc::mix {
namespace {
template <class Call>
int rejects(const char* label, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::overflow_error&) {
        return 0;
    }
    std::cerr << "Mix " << label << ": invalid arguments were accepted\n";
    return 1;
}
} // namespace

int validate(Weights& w, std::span<const Column> ref, CudaExecutionView e) {
    Fixture f(w, 2);
    const auto catalog = input_catalog(w.pattern);
    f.upload(catalog, e.stream());
    int failed = 0;
    std::vector<Tensor*> tensors{&f.z, &f.x};
    if (w.kind == Kind::Block) tensors.push_back(&f.j);
    for (auto* t : tensors) {
        const auto saved = *t;
        t->dtype         = t->dtype == DType::BF16 ? DType::FP32 : DType::BF16;
        failed += rejects("dtype", [&] { f.invoke(e); });
        *t = saved;
        --t->ne[0];
        failed += rejects("shape", [&] { f.invoke(e); });
        *t      = saved;
        t->data = nullptr;
        failed += rejects("null", [&] { f.invoke(e); });
        *t      = saved;
        t->data = static_cast<std::byte*>(t->data) + 2;
        failed += rejects("alignment", [&] { f.invoke(e); });
        *t = saved;
        t->nb[1] += 16;
        failed += rejects("stride", [&] { f.invoke(e); });
        *t       = saved;
        t->ne[3] = 2;
        failed += rejects("rank", [&] { f.invoke(e); });
        *t = saved;
    }
    for (int m : {0, -1}) {
        f.z.ne[2] = m;
        failed += rejects("nonpositive M", [&] { f.invoke(e); });
    }
    f.z.ne[2]  = f.columns;
    f.z.device = Device::cpu();
    failed += rejects("operand device", [&] { f.invoke(e); });
    f.z.device = e.device();
    for (auto bounds : {std::pair{0, 1}, std::pair{-1, 1}, std::pair{2, 1}})
        failed += rejects("query interval", [&] { (void)f.query(bounds.first, bounds.second); });

    for (Weight* weight : {&w.down, &w.up}) {
        const auto saved   = *weight;
        const auto invalid = [&] {
            failed += rejects("weight execution", [&] { f.invoke(e); });
            failed += rejects("weight query", [&] { (void)f.query(1, 2); });
        };
        weight->qtype = QType::FP32;
        invalid();
        *weight        = saved;
        weight->layout = QuantLayout::RowSplit;
        invalid();
        *weight = saved;
        ++weight->n;
        invalid();
        *weight = saved;
        ++weight->k;
        invalid();
        *weight = saved;
        ++weight->padded_shape[0];
        invalid();
        *weight               = saved;
        weight->payload_bytes = 0;
        invalid();
        *weight       = saved;
        weight->qdata = nullptr;
        failed += rejects("weight backing", [&] { f.invoke(e); });
        // A capacity query reads host metadata and does not need device backing.
        if (f.query(1, 2) == 0) ++failed;
        *weight       = saved;
        weight->qdata = static_cast<const std::byte*>(weight->qdata) + 2;
        failed += rejects("weight alignment", [&] { f.invoke(e); });
        *weight = saved;
    }

    const auto output = f.x, injection = f.j;
    for (const void* input :
         {f.z.data, const_cast<void*>(w.down.qdata), const_cast<void*>(w.up.qdata)}) {
        f.x.data = const_cast<void*>(input);
        failed += rejects("output/input overlap", [&] { f.invoke(e); });
        f.x.data = static_cast<std::byte*>(const_cast<void*>(input)) + 16;
        failed += rejects("partial output overlap", [&] { f.invoke(e); });
        f.x = output;
        if (w.kind == Kind::Block) {
            f.j.data = const_cast<void*>(input);
            failed += rejects("injection/input overlap", [&] { f.invoke(e); });
            f.j = injection;
        }
    }
    if (w.kind == Kind::Block) {
        f.j.data = f.x.data;
        failed += rejects("outputs overlap", [&] { f.invoke(e); });
        f.j = injection;
    }

    MemoryArena too_small(f.scratch.span().subspan(0, f.capacity - 1));
    try {
        if (w.kind == Kind::Block)
            ops::hc_mix(f.z, ops::HcMixWeights{w.down, w.up}, too_small, f.x, f.j, e);
        else
            ops::hc_mix(f.z, ops::HcFinalMixWeights{w.down, w.up}, too_small, f.x, e);
        std::cerr << "Mix accepted undersized workspace\n";
        ++failed;
    } catch (const std::bad_alloc&) {}
    MemoryArena overlapping(MemorySpan{const_cast<void*>(w.down.qdata), f.capacity, w.down.device});
    failed += rejects("workspace overlap", [&] {
        if (w.kind == Kind::Block)
            ops::hc_mix(f.z, ops::HcMixWeights{w.down, w.up}, overlapping, f.x, f.j, e);
        else
            ops::hc_mix(f.z, ops::HcFinalMixWeights{w.down, w.up}, overlapping, f.x, e);
    });

    for (auto [first, last] :
         {std::pair{1, 2048}, std::pair{128, 129}, std::pair{160, 161}, std::pair{192, 193},
          std::pair{224, 225}, std::pair{320, 321}, std::pair{384, 385}, std::pair{448, 449},
          std::pair{512, 513}, std::pair{672, 673}, std::pair{768, 769}, std::pair{1200, 1201},
          std::pair{1360, 1361}, std::pair{1632, 1633}}) {
        std::size_t maximum = 0;
        for (int m = first; m <= last; ++m) maximum = std::max(maximum, f.query(m, m));
        if (f.query(first, last) != maximum) {
            std::cerr << "Mix interval capacity does not equal its maximum singleton capacity\n";
            ++failed;
        }
    }

    // An enclosing caller may already own allocations. The Op restores that cursor and
    // consumes exactly its queried call-local capacity from the aligned starting position.
    constexpr std::size_t prefix_bytes = 768;
    GuardedDeviceBuffer nested_storage(prefix_bytes + f.capacity);
    // Guard initialization uses the default stream; e.stream() is nonblocking.
    cuda_synchronize(nullptr);
    MemoryArena nested(nested_storage.span());
    const auto prefix = nested.alloc_bytes(prefix_bytes, 256);
    std::vector<std::uint16_t> prefix_words(prefix_bytes / 2, 0x4261);
    cuda_check(cudaMemcpyAsync(prefix.data, prefix_words.data(), prefix_bytes,
                               cudaMemcpyHostToDevice, e.stream()),
               "Mix enclosing allocation");
    if (w.kind == Kind::Block)
        ops::hc_mix(f.z, ops::HcMixWeights{w.down, w.up}, nested, f.x, f.j, e);
    else
        ops::hc_mix(f.z, ops::HcFinalMixWeights{w.down, w.up}, nested, f.x, e);
    cuda_synchronize(e.stream());
    failed +=
        verify(f, ref, "Mix enclosing arena") + nested_storage.verify_guards("Mix nested arena");
    failed +=
        verify_exact("Mix enclosing allocation preserved",
                     from_device<std::uint16_t>(prefix.data, prefix_words.size()), prefix_words);
    if (nested.used() != prefix_bytes || nested.peak_used() != prefix_bytes + f.capacity) {
        std::cerr << "Mix enclosing arena cursor/high-water mismatch\n";
        ++failed;
    }

    // Read-only sharing is legal. Qualify the real computation with z bound to a weight row.
    Fixture shared(w, 1);
    // This case skips upload(), which normally completes default-stream initialization.
    cuda_synchronize(nullptr);
    shared.z.data = const_cast<void*>(w.down.qdata);
    shared.invoke(e);
    cuda_synchronize(e.stream());
    const auto expected = reference_column(w, std::span{w.down_values}.first(kTotal));
    const std::array<Column, 1> shared_ref{expected};
    failed += verify(shared, shared_ref, "Mix shared read-only input/weight");
    failed += shared.scratch.verify_guards("Mix shared scratch");
    return failed;
}
} // namespace ninfer::test::hc::mix
