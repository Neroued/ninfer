#include "models/qwen3_5/native_fixture.h"

#include "artifact/reader.h"
#include "core/transfer.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/load.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
namespace q35                     = models::qwen3_5;
namespace ctx                     = models::context;
using Native                      = test::qwen3_5::NativeRankFixture;
constexpr std::uint32_t kCapacity = 128, kBatch = 2, kDrafts = 3;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

PoolReservation reserve(ContextPool& pool, std::initializer_list<RecordRequest> demand) {
    auto result = pool.reserve(demand);
    require(result.reservation && result.reservation->ready(),
            "oracle Context capacity is insufficient");
    return std::move(*result.reservation);
}

template <class T>
std::vector<T> read(const Tensor& tensor) {
    require(tensor.is_contiguous() && tensor.bytes() % sizeof(T) == 0, "invalid oracle readback");
    std::vector<T> result(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

Tensor packed(const Tensor& storage, std::initializer_list<std::int32_t> shape) {
    return Tensor({storage.data, storage.bytes(), storage.device}, storage.dtype, shape);
}

template <class T>
DeviceBuffer upload(Device device, const std::vector<T>& values) {
    DeviceBuffer result(device, values.size() * sizeof(T));
    result.copy_from_host(values.data(), result.size());
    return result;
}

struct HistoryInput {
    std::int32_t width = 0, batch = 0;
    std::vector<std::uint16_t> conditioning, embeddings;
    std::vector<std::int32_t> ids, positions, rope, counts, lanes;
};

// The reference owns a separate cache and constructs (H[p], x[p+1], RoPE[p]) from
// observed Main outputs. It never calls the production alignment/window helpers.
class HistoryOracle {
public:
    explicit HistoryOracle(Native& fixture)
        : fixture_(fixture),
          workspace_(fixture.device().device(), std::max(fixture.rank().workspace_plan.mtp_prefill,
                                                         fixture.rank().workspace_plan.mtp_round)),
          arena_(workspace_.span()) {
        const auto& attention = *fixture.rank().parameters.profile->config.text.attention;
        LayoutBuilder builder;
        const auto layout = q35::plan_paged_kv_cache(
            builder, 1, kCapacity, q35::execution::dimension(attention.num_key_value_heads),
            q35::execution::dimension(attention.head_dim), KvCacheStorage::BFloat16, kBatch);
        backing_ = DeviceBuffer(fixture.device().device(), builder.finish(256, "reference MTP"));
        records_ = DeviceBuffer(fixture.device().device(), 4 * layout.pages.page_bytes);
        records_.fill();
        const std::array specs{RecordSpec{layout.pages.page_bytes, layout.pages.alignment}};
        pool_                   = std::make_unique<ContextPool>(0, records_.size(), specs);
        const auto record_class = pool_->record_class(0);
        auto reservation        = reserve(*pool_, {{record_class, 4}});
        cache_   = std::make_unique<q35::PagedKVCache>(records_.span(), backing_.span(), layout);
        mapping_ = PinnedHostBuffer(4 * sizeof(std::int64_t));
        for (std::uint32_t lane = 0; lane < kBatch; ++lane) {
            for (auto& offset : offsets_[lane])
                offset = pool_->resolve(reservation.materialize(record_class)).offset;
            cache_->tables().publish(
                lane, 0, offsets_[lane],
                mapping_.span().subspan(lane * 2 * sizeof(std::int64_t), 2 * sizeof(std::int64_t)),
                fixture.device().execution_view());
        }
        fixture.device().synchronize();
    }

    void history(const HistoryInput& input, bool forward = false) {
        const auto device = fixture_.device().device();
        auto h = upload(device, input.conditioning), ids = upload(device, input.ids);
        auto p = upload(device, input.positions), r = upload(device, input.rope);
        auto c = upload(device, input.counts), lanes = upload(device, input.lanes);
        auto e = input.embeddings.empty() ? DeviceBuffer{} : upload(device, input.embeddings);
        const auto hidden = fixture_.rank().resources.frame.spec.hidden;
        Tensor embeddings;
        if (e.size())
            embeddings = Tensor(e.span(), DType::BF16, {hidden, input.width, input.batch});
        const q35::execution::MtpInputs values{
            .ids              = Tensor(ids.span(), DType::I32, {input.width, input.batch}),
            .conditioning     = Tensor(h.span(), DType::BF16, {hidden, input.width, input.batch}),
            .cache_positions  = Tensor(p.span(), DType::I32, {input.width, input.batch}),
            .rope_positions   = Tensor(r.span(), DType::I32, {input.width * input.batch, 3}),
            .valid_columns    = Tensor(c.span(), DType::I32, {input.batch}),
            .kv_table_rows    = Tensor(lanes.span(), DType::I32, {input.batch}),
            .input_embeddings = e.size() ? &embeddings : nullptr,
        };
        q35::execution::MtpExecutionView execution{fixture_.rank().parameters, arena_,
                                                   fixture_.device().execution_view(),
                                                   cache_->batch_layer_view(0), ProposalHead::Full};
        if (forward) {
            DeviceBuffer output(device, input.conditioning.size() * sizeof(std::uint16_t));
            Tensor hidden_output(output.span(), DType::BF16, {hidden, input.width, input.batch});
            const auto maximum =
                *std::max_element(input.positions.begin(), input.positions.end()) + 1;
            q35::execution::mtp_forward(execution, values, {1, static_cast<std::uint32_t>(maximum)},
                                        hidden_output);
            fixture_.device().synchronize();
        } else {
            q35::execution::mtp_history(execution, values,
                                        {0, static_cast<std::uint32_t>(input.width)});
            fixture_.device().synchronize();
        }
    }

    void compare_inputs(const HistoryInput& expected) {
        const auto& frame = *fixture_.rank().resources.frame.mtp;
        const auto hidden = fixture_.rank().resources.frame.spec.hidden;
        require(read<std::uint16_t>(
                    packed(frame.history_conditioning, {hidden, expected.width, expected.batch})) ==
                    expected.conditioning,
                "MTP conditioning differs from independently indexed Main H[p]");
        require(read<std::int32_t>(packed(frame.history_ids, {expected.width, expected.batch})) ==
                    expected.ids,
                "MTP history token is not x[p+1]");
        require(read<std::int32_t>(packed(frame.history_positions,
                                          {expected.width, expected.batch})) == expected.positions,
                "MTP cache positions are misaligned");
        require(read<std::int32_t>(packed(frame.history_rope_positions,
                                          {expected.width * expected.batch, 3})) == expected.rope,
                "MTP three-axis RoPE does not describe conditioning positions");
    }

    void compare(std::uint32_t lane, std::uint32_t frontier) {
        fixture_.device().synchronize();
        const auto actual   = fixture_.rank().resources.decoder.mtp_cache()->batch_layer_view(0);
        const auto expected = cache_->batch_layer_view(0);
        const auto actual_table =
            read<std::int64_t>(actual.k_pages.records.page_offsets.slice(1, lane, 1));
        const auto expected_table =
            read<std::int64_t>(expected.k_pages.records.page_offsets.slice(1, lane, 1));
        for (const auto pair : {std::pair{actual.k_pages, expected.k_pages},
                                std::pair{actual.v_pages, expected.v_pages}}) {
            for (std::uint32_t page = 0; page * 64 < frontier; ++page) {
                const auto a = read<std::uint16_t>(pair.first.tensor(actual_table[page]));
                const auto e = read<std::uint16_t>(pair.second.tensor(expected_table[page]));
                for (std::uint32_t p = page * 64; p < std::min(frontier, (page + 1) * 64); ++p)
                    for (int head = 0; head < actual.num_kv_heads; ++head)
                        for (int d = 0; d < actual.head_dim; ++d) {
                            const auto offset =
                                d + std::size_t(actual.head_dim) * (p % 64 + 64 * head);
                            if (a[offset] != e[offset])
                                throw std::runtime_error("MTP committed KV differs at lane " +
                                                         std::to_string(lane) + " position " +
                                                         std::to_string(p));
                        }
            }
        }
        ++checks_;
    }

    void clone(std::uint32_t source, std::uint32_t destination) {
        for (std::uint32_t page = 0; page < 2; ++page)
            copy_bytes_async(
                records_.span().subspan(offsets_[destination][page],
                                        cache_->page_layout().page_bytes),
                records_.span().subspan(offsets_[source][page], cache_->page_layout().page_bytes),
                fixture_.device().execution_view());
        fixture_.device().synchronize();
    }

    std::size_t checks() const { return checks_; }

private:
    Native& fixture_;
    DeviceBuffer workspace_, backing_, records_;
    MemoryArena arena_;
    PinnedHostBuffer mapping_;
    std::unique_ptr<ContextPool> pool_;
    std::unique_ptr<q35::PagedKVCache> cache_;
    std::array<std::array<std::int64_t, 2>, kBatch> offsets_;
    std::size_t checks_ = 0;
};

class Cases {
public:
    explicit Cases(Native& native)
        : native_(native), oracle_(native), pool_(0, native.rank().resources.context.size(),
                                                  native.rank().resources.context_records) {
        const auto state_class = pool_.record_class(0);
        auto reservation       = reserve(
            pool_, {{state_class, 4}, {pool_.record_class(1), 4}, {pool_.record_class(2), 4}});
        for (auto& state : states_) state = pool_.resolve(reservation.materialize(state_class));
        for (std::uint32_t kind = 0; kind < 2; ++kind)
            for (auto& page : pages_[kind])
                page = pool_.resolve(reservation.materialize(pool_.record_class(kind + 1)));
        q35::ContextWork bindings;
        q35::Work work;
        for (std::uint32_t lane = 0; lane < kBatch; ++lane) {
            rows_[lane] = {.member            = {{lane + 1, 1}, {{lane}, 1}},
                           .source_state      = {lane, 1},
                           .destination_state = {lane, 1},
                           .state_offsets     = {static_cast<std::int64_t>(states_[lane].offset),
                                                 static_cast<std::int64_t>(states_[lane].offset)},
                           .main_table        = {0, lane, 1},
                           .backend_table     = ctx::TableRow{1, lane, 1}};
            work.context.operations.emplace_back(ctx::RecordClear{range(states_[lane])});
            bindings.bindings.push_back({rows_[lane].member, {}, {}});
            for (std::uint32_t pool = 0; pool < 2; ++pool)
                work.context.operations.emplace_back(ctx::MappingUpdate{
                    pool,
                    static_cast<std::int32_t>(lane),
                    0,
                    {static_cast<std::int64_t>(pages_[pool][lane * 2].offset),
                     static_cast<std::int64_t>(pages_[pool][lane * 2 + 1].offset)}});
        }
        work.operation = std::move(bindings);
        auto ticket    = submit(std::move(work), runtime::FrameKind::Context);
        (void)result(ticket);
        retire(ticket);
    }

    void forward(std::uint32_t lane, int count, q35::ForwardPurpose purpose, bool visual = false) {
        const auto base = rows_[lane].main_frontier;
        q35::ForwardWork work;
        work.purpose = purpose;
        work.rows    = {rows_[lane]};
        q35::InputWindow input;
        input.tokens.resize(count);
        input.rope_positions.resize(3 * count);
        for (int i = 0; i < count; ++i) {
            input.tokens[i] = 198 + (base + i) % 29;
            for (int axis = 0; axis < 3; ++axis)
                input.rope_positions[axis * count + i] = static_cast<int>(base + i) + 2 * axis;
        }
        work.inputs.push_back(input);
        if (purpose == q35::ForwardPurpose::Control) {
            work.committed_tokens = {input.tokens};
            work.sampling         = {{{}, false}};
        }
        const auto hidden = native_.rank().resources.frame.spec.hidden;
        std::optional<runtime::ExecutionTicket> ticket;
        std::unique_ptr<q35::detail::ComputeOperation> manual;
        std::vector<std::uint16_t> embeddings;
        DeviceBuffer visual_storage;
        if (visual) {
            // Isolate embedding-coordinate alignment from image encoding. These exact embeddings
            // are consumed by Main and must also feed the shifted MTP token positions.
            manual = std::make_unique<q35::detail::ComputeOperation>(
                runtime::ExecutionRequest<q35::ExecutionTypes>{{}, {}, {.operation = work}});
            native_.rank().initialize_inputs(*manual);
            visual_storage =
                DeviceBuffer(native_.device().device(), std::size_t(hidden) * count * 2);
            Tensor visual_input(visual_storage.span(), DType::BF16, {hidden, count});
            auto ids = native_.rank().resources.frame.input_view(count, 1).ids.view({count});
            ops::embedding(ids, native_.rank().parameters.text.token_embedding, visual_input,
                           native_.device().execution_view());
            native_.device().synchronize();
            embeddings = read<std::uint16_t>(visual_input);
            for (int column = 0; column < count; ++column)
                for (int d = 0; d < hidden; ++d)
                    if (column % 2 == 0)
                        embeddings[column * hidden + d] = (d % 2 ? 0xbe00 : 0x3e00);
            visual_storage.copy_from_host(embeddings.data(), visual_storage.size());
            manual->input_embeddings = visual_input.view({hidden, count, 1});
            native_.rank().begin_target(*manual);
            while (!native_.rank().advance_target(*manual)) native_.device().synchronize();
            native_.rank().align_mtp_history(*manual);
            native_.device().synchronize();
        } else {
            ticket = submit({.operation = std::move(work)}, runtime::FrameKind::Compute);
            (void)result(*ticket);
        }
        const auto produced =
            read<std::uint16_t>(native_.rank().resources.frame.input_view(count, 1).hidden);
        const int skip = base == 0 ? 1 : 0;
        if (count > skip) {
            HistoryInput expected;
            expected.width  = count - skip;
            expected.batch  = 1;
            expected.counts = {expected.width};
            expected.lanes  = {static_cast<int>(lane)};
            expected.conditioning.resize(std::size_t(hidden) * expected.width);
            expected.ids.resize(expected.width);
            expected.positions.resize(expected.width);
            expected.rope.resize(3 * expected.width);
            for (int i = 0; i < expected.width; ++i) {
                const auto p  = base + i + skip - 1;
                const auto* h = p < base ? hidden_[lane].data() + std::size_t(p) * hidden
                                         : produced.data() + std::size_t(p - base) * hidden;
                std::copy_n(h, hidden, expected.conditioning.data() + std::size_t(i) * hidden);
                expected.ids[i]       = input.tokens[i + skip];
                expected.positions[i] = p;
                for (int axis = 0; axis < 3; ++axis)
                    expected.rope[axis * expected.width + i] =
                        p < base ? rows_[lane].tail_rope_position[axis]
                                 : input.rope_positions[axis * count + p - base];
            }
            if (visual)
                expected.embeddings.assign(embeddings.begin() + std::size_t(skip) * hidden,
                                           embeddings.end());
            oracle_.compare_inputs(expected);
            oracle_.history(expected);
        }
        hidden_[lane].insert(hidden_[lane].end(), produced.begin(), produced.end());
        rows_[lane].main_frontier += count;
        rows_[lane].backend_frontier = rows_[lane].main_frontier - 1;
        for (int axis = 0; axis < 3; ++axis)
            rows_[lane].tail_rope_position[axis] = input.rope_positions[axis * count + count - 1];
        if (manual) {
            auto destination = native_.rank().resources.state_images.continuation_hidden(
                rows_[lane].state_offsets.destination);
            auto source =
                native_.rank().resources.frame.input_view(count, 1).hidden.slice(1, count - 1, 1);
            copy_bytes_async({destination.data, destination.bytes(), destination.device},
                             {source.data, source.bytes(), source.device},
                             native_.device().execution_view());
        } else {
            (void)native_.executor().commit(*ticket, {});
            retire(*ticket);
        }
        oracle_.compare(lane, rows_[lane].backend_frontier);
    }

    void checkpoint_clone() {
        q35::Work work;
        work.context.operations.emplace_back(
            ctx::RecordCopy{range(states_[0]), range(states_[2]), {}, 1});
        for (std::uint32_t pool = 0; pool < 2; ++pool)
            for (std::uint32_t page = 0; page < 2; ++page)
                work.context.operations.emplace_back(
                    ctx::RecordCopy{range(pages_[pool][page]),
                                    range(pages_[pool][page + 2]),
                                    {runtime::KvPoolId{static_cast<std::uint16_t>(pool)}},
                                    0,
                                    1});
        work.operation = q35::ContextWork{};
        auto ticket    = submit(std::move(work), runtime::FrameKind::Context);
        (void)result(ticket);
        retire(ticket);
        rows_[1].source_state = rows_[1].destination_state = {2, 1};
        rows_[1].state_offsets      = {static_cast<std::int64_t>(states_[2].offset),
                                       static_cast<std::int64_t>(states_[2].offset)};
        rows_[1].main_frontier      = rows_[0].main_frontier;
        rows_[1].backend_frontier   = rows_[0].backend_frontier;
        rows_[1].tail_rope_position = rows_[0].tail_rope_position;
        hidden_[1]                  = hidden_[0];
        oracle_.clone(0, 1);
        oracle_.compare(1, rows_[1].backend_frontier);
    }

    void sample_checkpoint() {
        q35::ForwardWork work;
        work.rows         = {rows_[1]};
        work.inputs       = {q35::InputWindow{}};
        work.sample       = q35::SamplePurpose::FirstToken;
        work.sampling     = {{{}, false}};
        const auto ticket = submit({.operation = std::move(work)}, runtime::FrameKind::Compute);
        auto value        = std::get<q35::ForwardResult>(result(ticket).output);
        require(value.generated && value.generated->counts == std::vector<std::uint32_t>{1},
                "exact checkpoint did not produce one anchor without Main input");
        anchors_[1] = value.generated->tokens[0];
        (void)native_.executor().commit(ticket, {{{1, false, false}}, {}});
        retire(ticket);
        oracle_.compare(1, rows_[1].backend_frontier);
    }

    void reset_for_visual_root() {
        q35::Work work;
        work.context.operations.emplace_back(ctx::RecordClear{range(states_[3])});
        work.operation    = q35::ContextWork{};
        const auto ticket = submit(std::move(work), runtime::FrameKind::Context);
        (void)result(ticket);
        retire(ticket);
        rows_[1].source_state = rows_[1].destination_state = {3, 1};
        rows_[1].state_offsets = {static_cast<std::int64_t>(states_[3].offset),
                                  static_cast<std::int64_t>(states_[3].offset)};
        rows_[1].main_frontier = rows_[1].backend_frontier = 0;
        rows_[1].tail_rope_position                        = {};
        hidden_[1].clear();
    }

    void decode(std::array<std::uint32_t, 2> extents, bool counterfactual) {
        q35::DecodeWork work;
        for (std::uint32_t lane = 0; lane < kBatch; ++lane)
            work.rows.push_back({rows_[lane], anchors_[lane], extents[lane], 8, {}});
        auto ticket          = submit({.operation = work}, runtime::FrameKind::Compute);
        const auto output    = std::get<q35::GeneratedResult>(result(ticket).output);
        auto& native         = native_.rank();
        auto& operation      = *native.compute;
        auto& frame          = native.resources.frame;
        const int width      = operation.width;
        const int hidden     = frame.spec.hidden;
        const auto main      = frame.input_view(width, kBatch);
        const auto target    = read<std::uint16_t>(main.hidden);
        const auto ids       = read<std::int32_t>(main.ids);
        const auto positions = read<std::int32_t>(main.cache_positions);
        const auto rope      = read<std::int32_t>(main.rope_positions);
        HistoryInput step;
        step.width  = 1;
        step.batch  = kBatch;
        step.counts = {1, 1};
        step.lanes  = {0, 1};
        step.conditioning.resize(hidden * kBatch);
        step.ids.assign(anchors_.begin(), anchors_.end());
        step.positions.resize(kBatch);
        step.rope.resize(kBatch * 3);
        for (std::uint32_t lane = 0; lane < kBatch; ++lane) {
            std::copy_n(hidden_[lane].data() + std::size_t(rows_[lane].main_frontier - 1) * hidden,
                        hidden, step.conditioning.data() + lane * hidden);
            step.positions[lane] = rows_[lane].main_frontier - 1;
            for (int axis = 0; axis < 3; ++axis)
                step.rope[axis * kBatch + lane] = rows_[lane].tail_rope_position[axis];
        }
        oracle_.history(step, operation.speculative);
        if (!operation.speculative) oracle_.compare_inputs(step);
        const auto rebuild = [&](std::array<std::uint32_t, 2> accepted, bool run_production) {
            if (!operation.speculative) return;
            HistoryInput expected;
            expected.width = kDrafts;
            expected.batch = kBatch;
            expected.lanes = {0, 1};
            expected.conditioning.resize(std::size_t(hidden) * expected.width * kBatch);
            expected.ids.resize(expected.width * kBatch);
            expected.positions.resize(expected.width * kBatch);
            expected.rope.resize(expected.width * kBatch * 3);
            for (std::uint32_t lane = 0; lane < kBatch; ++lane) {
                expected.counts.push_back(static_cast<int>(accepted[lane] - 1));
                for (int i = 0; i < expected.width; ++i) {
                    const auto dst = lane * expected.width + i, src = lane * width + i;
                    std::copy_n(target.data() + std::size_t(src) * hidden, hidden,
                                expected.conditioning.data() + std::size_t(dst) * hidden);
                    expected.ids[dst]       = ids[src + 1];
                    expected.positions[dst] = positions[src];
                    for (int axis = 0; axis < 3; ++axis)
                        expected.rope[axis * expected.width * kBatch + dst] =
                            rope[axis * width * kBatch + src];
                }
            }
            if (run_production) {
                operation.commit = q35::Commit{};
                for (std::uint32_t lane = 0; lane < kBatch; ++lane) {
                    operation.commit->rows.push_back({accepted[lane], true, false});
                    frame.host_commit->history_columns[lane] = expected.counts[lane];
                }
                CUDA_CHECK(cudaMemcpyAsync(frame.history_columns.data,
                                           frame.host_commit->history_columns.data(),
                                           kBatch * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                           native_.device().stream()));
                if (native.graphs)
                    native.graphs->rebuild_mtp_history(kBatch, native_.device().execution_view());
                else
                    native.rebuild_mtp_history(kBatch);
                native_.device().synchronize();
                oracle_.compare_inputs(expected);
            }
            oracle_.history(expected);
            for (std::uint32_t lane = 0; lane < kBatch; ++lane)
                oracle_.compare(lane, rows_[lane].main_frontier + accepted[lane] - 1);
        };
        if (counterfactual) {
            // Exercise actual-A rebuilding independently of the model's stochastic acceptance.
            // Every tested prefix belongs to the genuine target verify input and its Main H.
            for (const auto accepted : {1U, 2U, kDrafts + 1}) rebuild({accepted, 1}, true);
        }
        std::array<std::uint32_t, 2> accepted{output.counts[0], output.counts[1]};
        if (operation.speculative) rebuild(accepted, true);
        q35::Commit commit;
        for (auto count : accepted) commit.rows.push_back({count, false, false});
        (void)native_.executor().commit(ticket, std::move(commit));
        retire(ticket);
        for (std::uint32_t lane = 0; lane < kBatch; ++lane) {
            const auto begin = target.begin() + std::size_t(lane) * width * hidden;
            hidden_[lane].insert(hidden_[lane].end(), begin, begin + accepted[lane] * hidden);
            rows_[lane].main_frontier += accepted[lane];
            rows_[lane].backend_frontier = rows_[lane].main_frontier - 1;
            for (int axis = 0; axis < 3; ++axis)
                rows_[lane].tail_rope_position[axis] =
                    rope[axis * width * kBatch + lane * width + accepted[lane] - 1];
            anchors_[lane] = output.tokens[lane * output.row_stride + accepted[lane] - 1];
            oracle_.compare(lane, rows_[lane].backend_frontier);
        }
    }

    std::size_t checks() const { return oracle_.checks(); }

private:
    static ctx::ContextRange range(const RecordBinding& binding) {
        return {ctx::ContextMemory::Device, static_cast<std::int64_t>(binding.offset),
                binding.bytes};
    }

    runtime::ExecutionTicket submit(q35::Work work, runtime::FrameKind kind) {
        auto reservation = native_.executor().try_reserve(kind);
        require(bool(reservation), "native MTP frame is unexpectedly occupied");
        return native_.executor().submit(std::move(*reservation), std::move(work));
    }

    q35::Result result(runtime::ExecutionTicket ticket) {
        for (;;) {
            native_.device().synchronize();
            CUDA_CHECK(cudaStreamSynchronize(native_.device().transfer_stream()));
            auto event = native_.executor().poll();
            if (!event) continue;
            if (auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            if (auto* ready = std::get_if<runtime::ResultReady<q35::ExecutionTypes>>(&*event)) {
                require(ready->ticket == ticket, "unexpected native MTP result ticket");
                return std::move(ready->result);
            }
            throw std::runtime_error("unexpected native MTP event before result");
        }
    }

    void retire(runtime::ExecutionTicket ticket) {
        for (;;) {
            native_.device().synchronize();
            auto event = native_.executor().poll();
            if (!event) continue;
            if (auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failure->message);
            if (auto* retired = std::get_if<runtime::DeviceRetired>(&*event)) {
                require(retired->ticket == ticket, "unexpected native MTP retirement");
                return;
            }
            throw std::runtime_error("unexpected native MTP event before retirement");
        }
    }

    Native& native_;
    HistoryOracle oracle_;
    ContextPool pool_;
    std::array<RecordBinding, 4> states_;
    std::array<std::array<RecordBinding, 4>, 2> pages_;
    std::array<q35::ExecutionRow, 2> rows_;
    std::array<std::vector<std::uint16_t>, 2> hidden_;
    std::array<TokenId, 2> anchors_{198, 199};
};
} // namespace

int main() {
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (!path || !*path) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit MTP .ninfer artifact\n";
        return 77;
    }
    const char* graph    = std::getenv("NINFER_TEST_GRAPH");
    const bool use_graph = graph && std::string_view(graph) == "1";
    try {
        DeviceContext loading;
        artifact::Reader reader(path);
        auto definition = q35::read_definition(reader, {.speculative = SpeculativeBackend::Mtp});
        auto profile    = q35::execution::describe_model(*definition);
        std::shared_ptr<const q35::ModelShard> model = q35::materialize_model(
            q35::resolve_load(reader, definition, loading.device()), loading);
        EngineOptions options;
        options.max_context                       = kCapacity;
        options.max_concurrency                   = kBatch;
        options.prefill_chunk                     = 128;
        options.kv_cache                          = KvCacheStorage::BFloat16;
        options.use_cuda_graph                    = use_graph;
        options.speculative.backend               = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens          = kDrafts;
        options.context_cache.enabled             = false;
        options.context_cache.host_capacity_bytes = 0;
        Native fixture(definition, model, profile, options);
        Cases cases(fixture);
        cases.forward(0, 1, q35::ForwardPurpose::Prefill);
        cases.forward(0, 6, q35::ForwardPurpose::Prefill);
        cases.forward(0, 57, q35::ForwardPurpose::Prefill);
        cases.checkpoint_clone();
        cases.sample_checkpoint();
        cases.forward(0, 2, q35::ForwardPurpose::Control);
        cases.forward(1, 3, q35::ForwardPurpose::Replay);
        cases.decode({kDrafts, 0}, true);
        cases.decode({0, 0}, false);
        cases.forward(0, 4, q35::ForwardPurpose::Prefill, true);
        cases.reset_for_visual_root();
        cases.forward(1, 3, q35::ForwardPurpose::Prefill, true);
        std::cout << "OK MTP independent Main-conditioning oracle ["
                  << (use_graph ? "graph" : "eager") << "]: " << cases.checks() << " KV checks\n";
    } catch (const std::exception& error) {
        std::cerr << "MTP history oracle [" << (use_graph ? "graph" : "eager")
                  << "]: " << error.what() << '\n';
        return 1;
    }
}
