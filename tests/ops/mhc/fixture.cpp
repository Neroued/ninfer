#include "support.h"
#include "cuda_fixture.h"

namespace ninfer::test::mhc {
const char* name(Form f) {
    switch (f) {
    case Form::Pre:
        return "pre";
    case Form::Post:
        return "post";
    case Form::ExpandPre:
        return "expand-pre";
    case Form::PostPre:
        return "post-pre";
    case Form::PostMeanNorm:
        return "post-mean-norm";
    }
    throw std::logic_error("unknown mHC form");
}

bool has_pre(Form f) { return f == Form::Pre || f == Form::ExpandPre || f == Form::PostPre; }

CapacityQuery capacity_query(Form f) {
    switch (f) {
    case Form::Pre:
        return ops::mhc_pre_workspace_capacity_bytes;
    case Form::ExpandPre:
        return ops::mhc_expand_pre_workspace_capacity_bytes;
    case Form::PostPre:
        return ops::mhc_post_pre_workspace_capacity_bytes;
    case Form::PostMeanNorm:
        return ops::mhc_post_mean_norm_workspace_capacity_bytes;
    case Form::Post:
        return nullptr;
    }
    throw std::logic_error("unknown mHC form");
}

Fixture::Fixture(Form form, int m, bool alias, std::size_t guard)
    : form(form), m(m), alias(alias), x(std::size_t(K) * m), y(std::size_t(D) * m), f(N * K),
      base(N), scale(3), gamma(D), q(S * m), c(S * S * m), dx(x.size() * 2, guard),
      dy(y.size() * 2, guard), df(f.size() * 2, guard), db(N * 2, guard), ds(3 * 2, guard),
      dg(D * 2, guard), dq(q.size() * 4, guard), dc(c.size() * 4, guard), dr(x.size() * 2, guard),
      du(y.size() * 2, guard), dnq(q.size() * 4, guard), dnc(c.size() * 4, guard),
      tx(dx.span(), DType::BF16, {D, S, m}), ty(dy.span(), DType::BF16, {D, m}),
      tr(alias ? dx.span() : dr.span(), DType::BF16, {D, S, m}),
      capacity(form == Form::Post
                   ? 0
                   : capacity_query(form)({D, S}, cuda_environment().context.multiprocessor_count(),
                                          m, m)),
      scratch(capacity + 256), arena(scratch.span()) {
    direct_bf16_weight::HostWeight metadata{N, K, std::vector<std::uint16_t>(N * K)};
    weights      = {metadata.device_weight(df.span()), Tensor(db.span(), DType::BF16, {N}),
                    Tensor(ds.span(), DType::BF16, {3}), Tensor(dg.span(), DType::BF16, {D})};
    coefficients = {Tensor(dq.span(), DType::FP32, {S, m}),
                    Tensor(dc.span(), DType::FP32, {S, S, m})};
    outputs      = {
        Tensor(du.span(), DType::BF16, {D, m}),
        {Tensor(dnq.span(), DType::FP32, {S, m}), Tensor(dnc.span(), DType::FP32, {S, S, m})}};
    (void)arena.alloc_bytes(256);
    CUDA_CHECK(cudaMemset(scratch.data(), 0xa5, 256));
}

void Fixture::set(Pattern pattern, int phase) {
    fill_uniform(x, 4101 + phase, -1.F, 1.F);
    fill_uniform(y, 4103 + phase, -0.75F, 0.75F);
    fill_uniform(f, 4107 + phase, -0.015625F, 0.015625F);
    fill_uniform(base, 4111 + phase, -0.8F, 0.8F);
    fill_uniform(gamma, 4117 + phase, -1.25F, 1.25F);
    scale = {0.75F, 1.25F, 0.5F};
    for (int t = 0; t < m; ++t) {
        for (int i = 0; i < S; ++i) {
            q[t * S + i] = 0.2F + 0.4F * ((i + phase) % S);
            for (int j = 0; j < S; ++j)
                c[t * S * S + i * S + j] = float(1 + ((7 * i + 3 * j + phase) % 11)) / 16;
            for (int d = 0; d < D; ++d) x[std::size_t(t) * K + i * D + d] *= std::ldexp(1.F, i - 1);
        }
    }
    if (pattern == Pattern::Zero) {
        std::fill(x.begin(), x.end(), 0.F);
        std::fill(y.begin(), y.end(), 0.F);
        for (std::size_t i = 0; i < y.size(); i += 2) y[i] = -0.F;
    }
    if (pattern == Pattern::Saturated) {
        for (int i = 0; i < N; ++i) base[i] = i % 2 ? 100.F : -100.F;
    }
    if (pattern == Pattern::Unbalanced) {
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) base[8 + 4 * i + j] = j == 0 ? 30.F : float(-10 * (i + j));
        parameters.sinkhorn_epsilon    = 0.015625F;
        parameters.pre_epsilon         = 0.125F;
        parameters.sinkhorn_iterations = 3;
        parameters.post_multiplier     = 1.375F;
        parameters.coefficient_epsilon = 0.1F;
        parameters.branch_epsilon      = 0.3F;
    }
    if (pattern == Pattern::Midpoint) {
        std::fill(x.begin(), x.end(), 1.F);
        std::fill(y.begin(), y.end(), 1.F);
        std::fill(c.begin(), c.end(), 0.F);
        for (int t = 0; t < m; ++t)
            for (int i = 0; i < S; ++i) {
                c[t * 16 + i * 4 + i] = 1.F;
                q[t * 4 + i]          = std::ldexp(1.F, -8) * (i % 2 ? 3 : 1);
            }
    }
    if (pattern == Pattern::Small) {
        for (auto& v : x) v *= 1.e-4F;
        for (auto& v : y) v *= 1.e-4F;
    }
    for (auto* v : {&x, &y, &f, &base, &scale, &gamma}) round_to_bf16(*v);
}

namespace {
void upload_bf16(GuardedDeviceBuffer& b, const std::vector<float>& values) {
    std::vector<std::uint16_t> words(values.size());
    std::transform(values.begin(), values.end(), words.begin(), f32_to_bf16);
    b.copy_from_host(words.data(), words.size() * 2);
}

int unchanged(const std::string& label, const GuardedDeviceBuffer& b, const std::vector<float>& v,
              bool bf16) {
    if (bf16) {
        std::vector<std::uint16_t> expected(v.size());
        std::transform(v.begin(), v.end(), expected.begin(), f32_to_bf16);
        return verify_exact(label.c_str(), from_device<std::uint16_t>(b.data(), v.size()),
                            expected);
    }
    return verify_exact(label.c_str(), from_device<float>(b.data(), v.size()), v);
}
} // namespace

void Fixture::upload(cudaStream_t stream) {
    cuda_synchronize(stream);
    upload_bf16(dx, x);
    upload_bf16(dy, y);
    upload_bf16(df, f);
    upload_bf16(db, base);
    upload_bf16(ds, scale);
    upload_bf16(dg, gamma);
    dq.copy_from_host(q.data(), q.size() * 4);
    dc.copy_from_host(c.data(), c.size() * 4);
    cuda_synchronize(nullptr);
    for (auto* b : {&dr, &du, &dnq, &dnc})
        CUDA_CHECK(cudaMemsetAsync(b->data(), 0xff, b->bytes(), stream));
}

void Fixture::invoke(CudaExecutionView e, MemoryArena* workspace) {
    auto& work = workspace ? *workspace : arena;
    switch (form) {
    case Form::Pre:
        ops::mhc_pre(tx, weights, parameters, outputs, work, e);
        break;
    case Form::Post:
        ops::mhc_post(tx, ty, coefficients, tr, e);
        break;
    case Form::ExpandPre:
        ops::mhc_expand_pre(ty, weights, parameters, tr, outputs, work, e);
        break;
    case Form::PostPre:
        ops::mhc_post_pre(tx, ty, coefficients, weights, parameters, tr, outputs, work, e);
        break;
    case Form::PostMeanNorm:
        ops::mhc_post_mean_norm(tx, ty, coefficients, weights.gamma, final_epsilon,
                                outputs.block_input, work, e);
        break;
    }
}

int Fixture::preserved(const std::string& label) const {
    int failed        = 0;
    const auto prefix = from_device<std::uint8_t>(scratch.data(), 256);
    for (auto byte : prefix)
        if (byte != 0xa5) {
            ++failed;
            break;
        }
    for (auto* b : {&dx, &dy, &df, &db, &ds, &dg, &dq, &dc, &dr, &du, &dnq, &dnc, &scratch})
        failed += b->verify_guards(label);
    if (!alias) failed += unchanged(label + " input", dx, x, true);
    failed +=
        unchanged(label + " block", dy, y, true) + unchanged(label + " projection", df, f, true) +
        unchanged(label + " base", db, base, true) + unchanged(label + " scale", ds, scale, true) +
        unchanged(label + " gamma", dg, gamma, true) + unchanged(label + " q", dq, q, false) +
        unchanged(label + " C", dc, c, false);
    return failed;
}
} // namespace ninfer::test::mhc
