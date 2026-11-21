#include "suite.h"
#include "models/qwen4_exp/execution/mtp.h"

#include <bit>
#include <cmath>
#include <numeric>

namespace ninfer::test::flash_next {
namespace {
template <class T>
std::vector<T> read(const Tensor& values) {
    check(values.is_contiguous(), "expected packed test storage");
    std::vector<T> data(values.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(data.data(), values.data, values.bytes(), cudaMemcpyDeviceToHost));
    return data;
}

template <class T>
DeviceBuffer upload(Device device, std::span<const T> values) {
    DeviceBuffer buffer(device, values.size_bytes());
    buffer.copy_from_host(values.data(), values.size_bytes());
    return buffer;
}

float bf16(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t(bits) << 16); }

double history_relative_tolerance(const q::execution::MtpParameters& mtp) {
    // Four BF16-rounded stages (stem projections, HC combine/norm, HC mix, QSA history) run at
    // different M extents between batched step-0/repair execution and single-row prefix replay.
    double tolerance = 0.02;
    for (const auto policy :
         {mtp.fc_embedding.policy, mtp.fc_hidden.policy, mtp.history.attention.input.policy}) {
        if (policy == ops::LinearPolicy::AllowA4) return 0.16;
        if (policy == ops::LinearPolicy::AllowA8) tolerance = 0.04;
    }
    return tolerance;
}

void close_words(const Tensor& a, const Tensor& b, double relative_tolerance, const char* field) {
    const auto x = read<std::uint16_t>(a), y = read<std::uint16_t>(b);
    check(x.size() == y.size(), "reference field lengths differ");
    double delta = 0, norm = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto lhs = bf16(x[i]), rhs = bf16(y[i]);
        delta += double(lhs - rhs) * (lhs - rhs);
        norm += double(rhs) * rhs;
    }
    // Full-prefix and single-step calls can select different T-dependent activation routes
    // within the parameter's declared LinearPolicy.
    if (!(delta <= relative_tolerance * relative_tolerance * norm + 1.e-12))
        throw std::runtime_error(std::string("full-prefix MTP ") + field + " relative L2 " +
                                 std::to_string(std::sqrt(delta / std::max(norm, 1.e-12))));
}
} // namespace

void HistoryReference::input(q::detail::RankProgramImpl& rank, const q::Commit& commit) {
    rank.device.synchronize();
    const auto& f = *rank.compute;
    adopted_.assign(f.rows.size(), 0);
    if (!rank.parameters.mtp || !f.conditioning.data) return;
    const auto states      = read<std::uint16_t>(f.conditioning);
    const auto ids         = read<TokenId>(f.batch.ids);
    const auto coordinates = read<std::int32_t>(f.batch.rope_positions);
    const auto h           = f.conditioning.ne[0];
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        const auto& row = f.rows[b];
        if (commit.rows[b].cancelled || commit.rows[b].failed) continue;
        auto& prefix = prefixes_[row.member.sequence.value];
        if (!row.main_frontier) prefix = {};
        check(prefix.ids.size() == row.main_frontier, "reference lost the committed target prefix");
        const auto count = f.sequence ? row.ids.size() : commit.rows[b].accepted;
        adopted_[b]      = count;
        for (std::size_t j = 0; j < count; ++j) {
            const auto at = b * f.tokens + j;
            prefix.ids.push_back(ids[at]);
            auto& rope = prefix.coordinates.emplace_back();
            for (int axis = 0; axis < 3; ++axis)
                rope[axis] = coordinates[(b * 3 + axis) * f.tokens + j];
            prefix.hidden.insert(prefix.hidden.end(), states.begin() + at * h,
                                 states.begin() + (at + 1) * h);
        }
    }
}

void HistoryReference::compare_repair(q::detail::RankProgramImpl& rank) {
    const auto& f    = *rank.compute;
    const auto first = [&](std::size_t b) {
        const auto frontier = f.rows[b].main_frontier;
        const bool reused   = frontier && adopted_[b] && f.verify && f.verify->extents[b];
        return frontier - (frontier != 0) + reused;
    };
    const auto count = [&](std::size_t b) {
        const auto end = f.rows[b].main_frontier + adopted_[b];
        return end - (end != 0) - first(b);
    };
    bool repaired = false;
    for (std::size_t b = 0; b < f.rows.size(); ++b) repaired |= count(b) != 0;
    if (!repaired) return;

    const int batch        = f.physical.rows;
    const int width        = std::max(1, int(f.tokens) - 1);
    const auto& frame      = *rank.resources.buffers().mtp;
    const auto layout      = q::history_control_view(frame.history, width, batch);
    const auto ids         = read<TokenId>(rank.resources.tensor(layout.ids));
    const auto positions   = read<int>(rank.resources.tensor(layout.positions));
    const auto coordinates = read<int>(rank.resources.tensor(layout.rope_positions));
    const auto counts      = read<int>(rank.resources.tensor(layout.valid_columns));
    const int h            = q::dimension(rank.parameters.profile->config.text.residual_width());
    const auto hidden =
        read<std::uint16_t>(rank.resources.packed(frame.alignment_conditioning, {h, width, batch}));
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        check(counts[b] == int(count(b)), "repair wrote a different number of causal pairs");
        if (!count(b)) continue;
        const auto& prefix = prefixes_.at(f.rows[b].member.sequence.value);
        for (unsigned j = 0; j < count(b); ++j) {
            const auto p   = first(b) + j;
            const auto col = b * width + j;
            check(ids[col] == prefix.ids[p + 1] && positions[col] == int(p),
                  "repair used a different adopted token or history position");
            for (unsigned axis = 0; axis < 3; ++axis)
                check(coordinates[(b * 3 + axis) * width + j] == prefix.coordinates[p][axis],
                      "repair changed the preceding RoPE coordinates");
            check(std::equal(hidden.begin() + col * h, hidden.begin() + (col + 1) * h,
                             prefix.hidden.begin() + p * h),
                  "repair changed the preceding BF16 conditioning words");
        }
    }
}

// Recompute every causal pair (H[p], x[p+1], RoPE[p]) from position zero into a private page.
// These inputs use only the adopted Target trace, independently of proposal step 0 or repair.
void HistoryReference::compare(q::detail::RankProgramImpl& rank) {
    const auto& f       = *rank.compute;
    const auto* forward = std::get_if<q::ForwardWork>(&f.request.work.operation);
    if (f.sequence || f.rows.empty() ||
        (!f.verify && (!forward || forward->kind != q::ForwardKind::Ordinary)))
        return;
    rank.device.synchronize();
    compare_repair(rank);
    const auto& p     = rank.parameters;
    const auto device = rank.cuda.device();
    const auto actual = rank.resources.mtp->layer(0);
    const auto h      = q::dimension(p.profile->config.text.residual_width());
    for (const auto& row : f.rows) {
        const auto found = prefixes_.find(row.member.sequence.value);
        if (found == prefixes_.end() || found->second.ids.size() < 2) continue;
        const auto& prefix = found->second;
        const int n        = q::dimension(prefix.ids.size() - 1);
        check(n <= kPagedKVPageSize, "history fixture exceeds its reference page");
        LayoutBuilder layout;
        const auto kv = q::plan_qsa_cache(layout, p.profile->local.qsa, 1, kPagedKVPageSize, 1,
                                          KvCacheStorage::BFloat16);
        DeviceBuffer fixed(device, layout.finish()), page(device, kv.pages.page_bytes);
        q::QsaCache cache(page.span(), fixed.span(), kv);
        const std::int64_t zero64 = 0;
        CUDA_CHECK(cudaMemcpy(cache.tables().row(0).data, &zero64, sizeof(zero64),
                              cudaMemcpyHostToDevice));
        // A single State image at offset zero is sufficient for the two oracle fields.
        const auto& state_layout = rank.resources.plan.persistent.state;
        DeviceBuffer state(device, state_layout.images.image_bytes);
        CUDA_CHECK(cudaMemset(state.data(), 0, state.size()));
        auto selector = upload<std::int64_t>(device, {&zero64, 1});
        Tensor offset(selector.span(), DType::I64, {1});
        q::StateViews state_view(state.span(), state_layout);
        auto tail = state_view.mtp_tail(offset);
        const int meta[]{n, 0, 1};
        auto control = upload<int>(device, meta);
        const Tensor scalar(control.span(), DType::I32, {3});
        const auto valid = scalar.slice(0, 0, 1), zero = scalar.slice(0, 1, 1),
                   live = scalar.slice(0, 2, 1);
        std::vector<int> positions(n), coordinates(3 * n);
        std::iota(positions.begin(), positions.end(), 0);
        for (int a = 0; a < 3; ++a)
            for (int j = 0; j < n; ++j) coordinates[a * n + j] = prefix.coordinates[j][a];
        auto x   = upload<TokenId>(device, std::span(prefix.ids).subspan(1));
        auto pos = upload<int>(device, positions), rope = upload<int>(device, coordinates);
        auto hidden =
            upload<std::uint16_t>(device, std::span(prefix.hidden).first(std::size_t(h) * n));
        const q::execution::BatchView batch{n,
                                            1,
                                            Tensor(x.span(), DType::I32, {n, 1}),
                                            Tensor(pos.span(), DType::I32, {n, 1}),
                                            Tensor(rope.span(), DType::I32, {n, 3, 1}),
                                            valid,
                                            offset,
                                            offset,
                                            zero,
                                            zero,
                                            zero,
                                            static_cast<unsigned>(n),
                                            live,
                                            static_cast<unsigned>(n)};
        q::execution::mtp_history(
            p.profile->config.text, p.profile->local.qsa, *p.mtp,
            {Tensor(hidden.span(), DType::BF16, {h, n}), models::TokenPartition(n)}, batch, tail,
            tail, cache.layer(0), {}, rank.resources.workspace, rank.cuda);
        rank.device.synchronize();
        const auto target   = prefix.ids.size() == row.main_frontier ? row.states.at(0).source
                                                                     : row.states.at(0).destination;
        const auto original = rank.resources.states->mtp_tail({});
        const auto tolerance = history_relative_tolerance(*p.mtp);
        check(read<int>(original.first_rope.tensor(target)) == read<int>(tail.first_rope.tensor(0)),
              "MTP tail first RoPE differs from the full prefix");
        close_words(original.raw_keys.tensor(target), tail.raw_keys.tensor(0), tolerance,
                    "unfinished index keys");
        const auto offset0 =
            read<std::int64_t>(rank.resources.mtp->tables().row(row.mtp_table.index)).front();
        const auto compare_plane = [&](const PagedPlaneView& device_plane,
                                       const PagedPlaneView& reference, int count,
                                       const char* name) {
            if (!count) return;
            const auto a = device_plane.tensor(offset0), b = reference.tensor(0);
            for (int head = 0; head < a.ne[2]; ++head) {
                auto x = a.slice(2, head, 1).slice(1, 0, count);
                auto y = b.slice(2, head, 1).slice(1, 0, count);
                close_words(x, y, tolerance, name);
            }
        };
        const auto ref = cache.layer(0);
        compare_plane(actual.main.k_pages, ref.main.k_pages, n, "K");
        compare_plane(actual.main.v_pages, ref.main.v_pages, n, "V");
        compare_plane(actual.index.keys, ref.index.keys, n / 4, "compressed index keys");
    }
}
} // namespace ninfer::test::flash_next
