#include "support.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <limits>
#include <new>

namespace ninfer::test::gdn_control {
namespace {
template <class Exception = std::invalid_argument, class Call>
int rejects(std::string_view name, Call&& call) {
    try {
        call();
    } catch (const Exception&) { return 0; }
    std::cerr << name << ": rejection expected\n";
    return 1;
}
} // namespace

int validate(Profile p, bool normalize_input, CudaExecutionView execution) {
    Fixture f(p, {512}, normalize_input);
    f.repaint(execution.stream());
    cuda_synchronize(execution.stream());
    const auto name  = label(p, {512}, normalize_input);
    const auto old_g = from_device<std::uint32_t>(f.g.data(), std::size_t(p.heads) * 512);
    const auto old_b = from_device<std::uint32_t>(f.beta.data(), std::size_t(p.heads) * 512);
    int failed       = 0;
    auto launch      = [&] { f.invoke(f.arena, execution); };
    std::vector<Tensor*> tensors{&f.input, &f.a_log_tensor, &f.dt_bias_tensor, &f.outputs.g,
                                 &f.outputs.beta};
    if (normalize_input) {
        tensors.push_back(&f.norm_weight);
        tensors.push_back(&f.normalized_tensor);
    }
    for (auto* t : tensors) {
        const auto saved = *t;
        t->dtype         = t->dtype == DType::BF16 ? DType::FP32 : DType::BF16;
        failed += rejects(name + " dtype", launch);
        *t = saved;
        --t->ne[0];
        failed += rejects(name + " shape", launch);
        *t = saved;
        t->nb[0] += 2;
        failed += rejects(name + " stride", launch);
        *t      = saved;
        t->data = nullptr;
        failed += rejects(name + " backing", launch);
        *t = saved;
    }
    for (int columns : {0, -1}) {
        const auto saved = f.input;
        f.input.ne[1]    = columns;
        failed += rejects(name + " extent", launch);
        f.input = saved;
    }
    const auto saved_g = f.outputs.g;
    for (const void* data :
         {f.input.data, f.a_log_tensor.data, f.dt_bias_tensor.data, f.outputs.beta.data}) {
        f.outputs.g.data = const_cast<void*>(data);
        failed += rejects(name + " output overlap", launch);
        f.outputs.g = saved_g;
    }
    f.outputs.g.data = static_cast<std::byte*>(f.outputs.beta.data) + 4;
    failed += rejects(name + " partial overlap", launch);
    f.outputs.g         = saved_g;
    const auto saved_a  = f.a_log_tensor;
    f.a_log_tensor.data = f.dt_bias_tensor.data;
    failed += rejects(name + " input overlap", launch);
    f.a_log_tensor     = saved_a;
    const auto saved_x = f.input;
    f.input.data       = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 15);
    failed += rejects<std::overflow_error>(name + " range overflow", launch);
    f.input                  = saved_x;
    const auto saved_weights = f.weights.prepared;
    auto& w = std::holds_alternative<ops::SingleProjectionWeight>(f.weights.prepared)
                  ? std::get<ops::SingleProjectionWeight>(f.weights.prepared).weight
                  : std::get<ops::PairedProjectionWeights>(f.weights.prepared).first;
    w.qtype = QType::Q8_G32_FP16;
    failed += rejects(name + " weight format", launch);
    f.weights.prepared = saved_weights;
    auto mutate        = [&](auto change) {
        auto& weight = std::holds_alternative<ops::SingleProjectionWeight>(f.weights.prepared)
                                  ? std::get<ops::SingleProjectionWeight>(f.weights.prepared).weight
                                  : std::get<ops::PairedProjectionWeights>(f.weights.prepared).first;
        change(weight);
        failed += rejects(name + " weight metadata", launch);
        f.weights.prepared = saved_weights;
    };
    mutate([](Weight& a) { --a.payload_bytes; });
    mutate([](Weight& a) { a.scales = a.qdata; });
    mutate([](Weight& a) { ++a.padded_shape[1]; });
    mutate([](Weight& a) { a.ndim = 3; });
    mutate([](Weight& a) { a.layout = QuantLayout::RowSplit; });
    mutate([](Weight& a) { a.qdata = nullptr; });
    mutate([](Weight& a) { a.qdata = static_cast<const std::byte*>(a.qdata) + 2; });
    mutate([](Weight& weight) { weight.device = Device::cpu(); });
    mutate([&](Weight& weight) { weight.device = Device::cuda(execution.device().index() + 1); });
    if (normalize_input)
        for (float eps : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()})
            failed += rejects(name + " epsilon", [&] {
                ops::gdn_norm_control_proj(f.input, f.norm_weight, eps, f.weights.prepared,
                                           f.a_log_tensor, f.dt_bias_tensor,
                                           {f.normalized_tensor, f.outputs}, f.arena, execution);
            });
    if (f.capacity) {
        MemoryArena small(f.scratch.span().subspan(0, f.capacity - 1));
        failed += rejects<std::bad_alloc>(name + " insufficient arena",
                                          [&] { f.invoke(small, execution); });
        if (small.used()) ++failed;
    }
    MemoryArena overlapping(MemorySpan{f.device_x.data(), f.capacity, f.device_x.device()});
    failed += rejects(name + " scratch overlap", [&] { f.invoke(overlapping, execution); });
    failed += verify_exact((name + " rejected g untouched").c_str(),
                           from_device<std::uint32_t>(f.g.data(), old_g.size()), old_g);
    failed += verify_exact((name + " rejected beta untouched").c_str(),
                           from_device<std::uint32_t>(f.beta.data(), old_b.size()), old_b);
    failed += f.check(name);
    if (f.arena.used() != f.prefix) ++failed;
    if (p.input_rows == 2560) {
        failed += rejects(name + " norm domain",
                          [&] { (void)workspace_capacity(f.weights.prepared, true, 1, 8); });
        failed += rejects(name + " norm execution domain", [&] {
            ops::gdn_norm_control_proj(f.input, f.norm_weight, kEps, f.weights.prepared,
                                       f.a_log_tensor, f.dt_bias_tensor,
                                       {f.normalized_tensor, f.outputs}, f.arena, execution);
        });
        const auto first = std::get<ops::SingleProjectionWeight>(f.weights.prepared).weight;
        Weight a = first, b = first;
        a.n = b.n  = 48;
        a.shape[0] = b.shape[0] = 48;
        a.padded_shape[0] = b.padded_shape[0] = 48;
        b.qdata                     = static_cast<const std::byte*>(a.qdata) + 48 * 2560 * 2;
        ops::ProjectionWeights pair = ops::PairedProjectionWeights{a, b};
        failed +=
            rejects(name + " pair domain", [&] { (void)workspace_capacity(pair, false, 1, 8); });
    }
    return failed;
}

int validate_capacity(Profile p, bool normalize_input) {
    Weights weights(p, Pattern::Random);
    auto metadata = weights.prepared;
    std::visit(
        [](auto& value) {
            if constexpr (requires { value.weight; }) {
                value.weight.qdata   = nullptr;
                value.weight.payload = nullptr;
            } else {
                value.first.qdata = value.second.qdata = nullptr;
                value.first.payload = value.second.payload = nullptr;
            }
        },
        metadata);
    int failed = 0;
    for (auto [first, last] :
         std::array<std::pair<int, int>, 4>{{{1, 128}, {2, 512}, {129, 1025}, {513, 8193}}}) {
        std::size_t expected = 0;
        for (int m = first; m <= last; ++m)
            expected = std::max(expected, workspace_capacity(metadata, normalize_input, m, m));
        if (workspace_capacity(metadata, normalize_input, first, last) != expected) {
            std::cerr << "control interval capacity omitted a peak\n";
            ++failed;
        }
    }
    for (auto [first, last] : std::array<std::pair<int, int>, 3>{{{0, 1}, {1, 0}, {2, 1}}})
        failed += rejects("invalid capacity interval", [&] {
            (void)workspace_capacity(metadata, normalize_input, first, last);
        });
    const int last = std::numeric_limits<std::int32_t>::max();
    if (workspace_capacity(metadata, normalize_input, last, last) != 0 ||
        workspace_capacity(metadata, normalize_input, 1, last) <
            workspace_capacity(metadata, normalize_input, 1, 8193))
        ++failed;
    return failed;
}
} // namespace ninfer::test::gdn_control
