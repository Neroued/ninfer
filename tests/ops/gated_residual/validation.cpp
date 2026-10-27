#include "support.h"

namespace ninfer::test::hc {
namespace {
template <class Call>
int rejects(const char* label, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::overflow_error&) {
        return 0;
    }
    std::cerr << label << ": did not reject invalid operands\n";
    return 1;
}
} // namespace

int validate(Form form, CudaExecutionView e, int width) {
    Fixture f(form, 2, false, width);
    f.set(Pattern::Random);
    f.upload(e.stream());
    int failed = 0;
    std::vector<Tensor*> used{form == Form::ExpandNorm ? &f.tb : &f.th};
    if (form == Form::Combine || form == Form::CombineNorm || form == Form::AddNorm)
        used.push_back(&f.tb);
    if (form == Form::Combine || form == Form::CombineNorm) used.push_back(&f.tj);
    if (form != Form::Combine) used.push_back(&f.tw);
    if (form != Form::GroupedNorm) used.push_back(&f.tr);
    if (form != Form::Combine) used.push_back(&f.tz);
    for (auto* t : used) {
        const auto saved = *t;
        t->dtype         = t->dtype == DType::BF16 ? DType::FP32 : DType::BF16;
        failed += rejects("dtype", [&] { f.invoke(e); });
        *t = saved;
        t->ne[0] -= 1;
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
    Tensor& activation          = form == Form::ExpandNorm ? f.tb : f.th;
    const auto saved_activation = activation;
    for (int m : {0, -1}) {
        activation.ne[form == Form::ExpandNorm ? 1 : 2] = m;
        failed += rejects("nonpositive columns", [&] { f.invoke(e); });
        activation = saved_activation;
    }
    activation.device = Device::cpu();
    failed += rejects("operand device", [&] { f.invoke(e); });
    activation = saved_activation;
    if (form != Form::Combine) {
        for (float eps : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
            f.epsilon = eps;
            failed += rejects("epsilon", [&] { f.invoke(e); });
        }
        f.epsilon = kEpsilon;
    }
    const auto residual = f.tr, normalized = f.tz;
    for (auto* input : used) {
        if (input == &f.tr || input == &f.tz) continue;
        if (form != Form::GroupedNorm) {
            f.tr.data = input->data;
            if (input != &f.th || form == Form::ExpandNorm)
                failed += rejects("residual/input overlap", [&] { f.invoke(e); });
            f.tr.data = static_cast<std::byte*>(input->data) + 16;
            failed += rejects("partial overlap", [&] { f.invoke(e); });
            f.tr = residual;
        }
        if (form != Form::Combine) {
            f.tz.data = input->data;
            failed += rejects("normalized/input overlap", [&] { f.invoke(e); });
            f.tz = normalized;
        }
    }
    if (form != Form::Combine && form != Form::GroupedNorm) {
        f.tz.data = f.tr.data;
        failed += rejects("outputs overlap", [&] { f.invoke(e); });
        f.tz = normalized;
    }
    if (form == Form::Combine || form == Form::CombineNorm || form == Form::AddNorm) {
        f.tr = f.th;
        for (Tensor* input : {&f.tb, &f.tj, &f.tw}) {
            if (std::find(used.begin(), used.end(), input) == used.end()) continue;
            const auto saved = *input;
            input->data      = f.th.data;
            failed += rejects("in-place/input overlap", [&] { f.invoke(e); });
            *input = saved;
        }
        f.tr = residual;
    }
    // Read-only sharing is legal and tested through real device backing.
    if (form == Form::GroupedNorm) {
        f.tw.data = f.th.data;
        std::copy_n(f.h.begin(), f.weight.size(), f.weight.begin());
        f.invoke(e);
        cuda_synchronize(e.stream());
        failed += verify(f, "shared read-only input/weight", false);
    }
    return failed;
}
} // namespace ninfer::test::hc
