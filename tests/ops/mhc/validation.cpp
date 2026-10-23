#include "support.h"

namespace ninfer::test::mhc {
int validate(Form form, CudaExecutionView e) {
    Fixture f(form, 3, false);
    int failed         = 0;
    const auto rejects = [&](auto&& call) {
        try {
            call();
            ++failed;
            std::cerr << "mHC accepted invalid operands\n";
        } catch (const std::exception&) {}
        if (f.arena.used() != 256) {
            ++failed;
            std::cerr << "mHC failed to restore arena\n";
        }
    };
    auto& input      = form == Form::ExpandPre ? f.ty : f.tx;
    const auto saved = input;
    input.ne[0]      = 2048;
    rejects([&] { f.invoke(e); });
    input       = saved;
    input.dtype = DType::FP32;
    rejects([&] { f.invoke(e); });
    input       = saved;
    input.ne[2] = 0;
    rejects([&] { f.invoke(e); });
    input = saved;
    input.nb[1] += 16;
    rejects([&] { f.invoke(e); });
    input      = saved;
    input.data = static_cast<char*>(saved.data) + 2;
    rejects([&] { f.invoke(e); });
    input = saved;
    if (has_pre(form)) {
        const auto out             = f.outputs;
        f.outputs.block_input.data = input.data;
        rejects([&] { f.invoke(e); });
        f.outputs                       = out;
        f.outputs.coefficients.mix.data = f.outputs.coefficients.post.data;
        rejects([&] { f.invoke(e); });
        f.outputs                        = out;
        auto p                           = f.parameters;
        f.parameters.sinkhorn_iterations = 0;
        rejects([&] { f.invoke(e); });
        f.parameters                = p;
        f.parameters.branch_epsilon = std::numeric_limits<float>::infinity();
        rejects([&] { f.invoke(e); });
        f.parameters                = p;
        const auto w                = f.weights;
        f.weights.projection.layout = QuantLayout::RowSplit;
        rejects([&] { f.invoke(e); });
        f.weights = w;
    }
    if (f.capacity > 0) {
        MemoryArena tiny({f.scratch.data(), f.capacity - 1, e.device()});
        rejects([&] { f.invoke(e, &tiny); });
        if (tiny.used() != 0) ++failed;
        MemoryArena overlapping(form == Form::ExpandPre ? f.dy.span() : f.dx.span());
        rejects([&] { f.invoke(e, &overlapping); });
        if (overlapping.used() != 0) ++failed;
    }
    if (form == Form::Post || form == Form::PostPre) {
        auto out  = f.tr;
        f.tr.data = static_cast<char*>(f.tx.data) + 16;
        rejects([&] { f.invoke(e); });
        f.tr       = out;
        f.tr.data  = f.tx.data;
        auto block = f.ty;
        f.ty.data  = f.tx.data;
        rejects([&] { f.invoke(e); });
        f.ty                    = block;
        auto c                  = f.coefficients;
        f.coefficients.mix.data = f.tx.data;
        rejects([&] { f.invoke(e); });
        f.coefficients = c;
        f.tr           = out;
    }
    if (form == Form::PostPre) {
        const auto out              = f.outputs;
        f.outputs.coefficients.post = f.coefficients.post;
        rejects([&] { f.invoke(e); });
        f.outputs                  = out;
        f.outputs.coefficients.mix = f.coefficients.mix;
        rejects([&] { f.invoke(e); });
        f.outputs = out;
    }
    if (form == Form::PostMeanNorm) {
        const auto output          = f.outputs.block_input;
        f.outputs.block_input.data = f.tx.data;
        rejects([&] { f.invoke(e); });
        f.outputs.block_input = output;
        f.final_epsilon       = 0;
        rejects([&] { f.invoke(e); });
        f.final_epsilon       = 1.e-5F;
        const auto gamma      = f.weights.gamma;
        f.weights.gamma.ne[0] = D - 1;
        rejects([&] { f.invoke(e); });
        f.weights.gamma = gamma;
    }
    if (auto query = capacity_query(form)) {
        const int sm = e.multiprocessor_count();
        for (const auto interval :
             {std::pair{1, 17}, std::pair{7, 128}, std::pair{15, 17}, std::pair{127, 129},
              std::pair{511, 513}, std::pair{607, 609}, std::pair{1023, 1025}}) {
            std::size_t maximum = 0;
            for (int m = interval.first; m <= interval.second; ++m)
                maximum = std::max(maximum, query({D, S}, sm, m, m));
            if (maximum != query({D, S}, sm, interval.first, interval.second)) ++failed;
        }
        rejects([&] { (void)query({D, S}, 0, 1, 1); });
        rejects([&] { (void)query({D, S}, sm, 2, 1); });
        rejects([&] { (void)query({2048, S}, sm, 1, 1); });
    }
    return failed;
}
} // namespace ninfer::test::mhc
