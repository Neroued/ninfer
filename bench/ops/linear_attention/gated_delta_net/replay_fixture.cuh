#pragma once
// ReplaySSM recurrent-Record and all-layer Fold benchmark.
//
// Each timed GPU body is exactly one public Op call. Row-control construction, buffer
// initialization, and L2 eviction are outside the timed interval. Both graph and eager launches
// report host submission time separately from CUDA-event GPU latency.

#include "core/buffer.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "op_measurement.cuh"
#include "replay_options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::bench::gdn_replay {
constexpr int kStateDim = 128;

using measurement::make_i32;

inline DeviceBuffer make_offsets(const std::vector<std::int64_t>& values) {
    DeviceBuffer result(bench::device_context().device(), values.size() * sizeof(std::int64_t));
    result.copy_from_host(values.data(), result.size());
    return result;
}

class RecordInputs {
public:
    RecordInputs(const Profile& profile, int width, int batch)
        : profile_(profile), width_(width), batch_(batch),
          q_(bench::make_bf16(qk_elements(), 101U)), k_(bench::make_bf16(qk_elements(), 103U)),
          v_(bench::make_bf16(value_elements(), 105U)),
          g_(bench::make_f32(gate_elements(), 211U, -1.5F, -.1F)),
          beta_(bench::make_f32(gate_elements(), 213U, .1F, .9F)),
          out_(bench::device_context().device(), value_elements() * sizeof(std::uint16_t)) {}

    void record(ConstStateFieldView source, const Tensor& valid, GdnRecurrenceRecordView records,
                CudaExecutionView cuda) {
        const Tensor q(q_.span(), DType::BF16, {kStateDim, profile_.qk_heads, width_, batch_});
        const Tensor k(k_.span(), DType::BF16, {kStateDim, profile_.qk_heads, width_, batch_});
        const Tensor v(v_.span(), DType::BF16, {kStateDim, profile_.value_heads, width_, batch_});
        const Tensor g(g_.span(), DType::FP32, {profile_.value_heads, width_, batch_});
        const Tensor beta(beta_.span(), DType::FP32, {profile_.value_heads, width_, batch_});
        Tensor out(out_.span(), DType::BF16, {kStateDim, profile_.value_heads, width_, batch_});
        ops::gated_delta_net_replay_record(q, k, v, g, beta, 1.0F / std::sqrt(128.0F), source,
                                           valid, records, out, cuda);
    }

private:
    std::size_t qk_elements() const {
        return static_cast<std::size_t>(kStateDim) * profile_.qk_heads * width_ * batch_;
    }

    std::size_t value_elements() const { return kStateDim * gate_elements(); }

    std::size_t gate_elements() const {
        return static_cast<std::size_t>(profile_.value_heads) * width_ * batch_;
    }

    Profile profile_;
    int width_, batch_;
    DeviceBuffer q_, k_, v_, g_, beta_, out_;
};

class FoldFixture {
public:
    FoldFixture(const Profile& profile, std::int32_t width, std::int32_t batch,
                DType state_dtype = DType::FP32)
        : profile_(profile), width_(width), batch_(batch), state_records_(std::max(11, batch + 3)),
          state_dtype_(state_dtype),
          record_storage_(ninfer::bench::device_context().device(),
                          record_bytes(profile, width, batch)),
          records_(record_storage_.span(), record_layout_),
          state_storage_(ninfer::bench::device_context().device(),
                         image_layout_.image_bytes * state_records_),
          images_(state_storage_.span(), image_layout_), states_(images_, state_layout_) {
        record_storage_.fill(0); // alignment gaps only; represented planes are filled below.
        state_storage_.fill(0);
        CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(records_.conv.data),
                                               records_.conv.numel(), 301U, -.5F, .5F));
        const auto conv_bytes      = states_.conv_tensor(0, 0).bytes();
        const auto recurrent_bytes = states_.recurrent_tensor(0, 0).bytes();
        DeviceBuffer initial_conv(bench::device_context().device(), conv_bytes * state_records_);
        DeviceBuffer initial_recurrent(bench::device_context().device(),
                                       recurrent_bytes * state_records_);
        for (int layer = 0; layer < profile.layers; ++layer) {
            CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(initial_conv.data()),
                                                   initial_conv.size() / 2, 401U + layer, -.5F,
                                                   .5F));
            if (state_dtype_ == DType::BF16) {
                CUDA_CHECK(bench::fixture::fill_values(
                    static_cast<__nv_bfloat16*>(initial_recurrent.data()),
                    initial_recurrent.size() / 2, 501U + layer, -.02F, .02F));
            } else {
                CUDA_CHECK(bench::fixture::fill_values(
                    static_cast<float*>(initial_recurrent.data()), initial_recurrent.size() / 4,
                    501U + layer, -.02F, .02F));
            }
            for (int record = 0; record < state_records_; ++record) {
                auto conv      = states_.conv_tensor(layer, byte_offset(record));
                auto recurrent = states_.recurrent_tensor(layer, byte_offset(record));
                CUDA_CHECK(cudaMemcpyAsync(conv.data,
                                           static_cast<const std::byte*>(initial_conv.data()) +
                                               record * conv_bytes,
                                           conv_bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(recurrent.data,
                                           static_cast<const std::byte*>(initial_recurrent.data()) +
                                               record * recurrent_bytes,
                                           recurrent_bytes, cudaMemcpyDeviceToDevice));
            }
        }
        auto selectors = make_offsets(offsets());
        const Tensor initial(selectors.span(), DType::I64, {batch_});
        RecordInputs inputs(profile, width, batch);
        const auto cuda = bench::device_context().execution_view();
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int layer = 0; layer < profile.layers; ++layer) {
            inputs.record(states_.layer_view(layer, initial).recurrent, {},
                          records_.layer(layer, batch).recurrence, cuda);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        initial_ = std::make_unique<bench::SavedBuffer>(state_storage_);
    }

    void restore(CudaExecutionView cuda) const { initial_->restore(cuda); }

    [[nodiscard]] std::vector<std::int64_t> offsets() const {
        constexpr int small_records[] = {10, 2, 8, 0, 6, 4, 9, 1};
        std::vector<std::int64_t> result(batch_);
        for (int b = 0; b < batch_; ++b)
            result[b] = byte_offset(batch_ <= 8 ? small_records[b] : b + 3);
        return result;
    }

    [[nodiscard]] const ops::GdnReplayFoldPlan& fold_plan() const { return fold_plan_; }

private:
    [[nodiscard]] std::int64_t byte_offset(int record) const {
        return static_cast<std::int64_t>(record * image_layout_.image_bytes);
    }

    static GdnReplayRecordLayout make_record_layout(const Profile& profile, std::int32_t width,
                                                    std::int32_t batch, LayoutBuilder& builder) {
        return plan_gdn_replay_records(builder, {.layers          = profile.layers,
                                                 .record_capacity = std::max(8, batch),
                                                 .width           = width,
                                                 .conv_channels   = profile.conv_channels,
                                                 .qk_heads        = profile.qk_heads,
                                                 .value_heads     = profile.value_heads,
                                                 .key_dim         = kStateDim,
                                                 .value_dim       = kStateDim});
    }

    static LinearAttentionStateLayout make_state_layout(const Profile& profile, DType state_dtype,
                                                        StateImageBuilder& builder) {
        return plan_linear_attention_state(builder,
                                           {.layers = static_cast<std::uint32_t>(profile.layers),
                                            .conv_channels   = profile.conv_channels,
                                            .conv_width      = 3,
                                            .value_heads     = profile.value_heads,
                                            .value_head_dim  = kStateDim,
                                            .key_head_dim    = kStateDim,
                                            .conv_dtype      = DType::BF16,
                                            .recurrent_dtype = state_dtype});
    }

    static std::size_t record_bytes(const Profile& profile, std::int32_t width,
                                    std::int32_t batch) {
        LayoutBuilder builder;
        (void)make_record_layout(profile, width, batch, builder);
        return builder.finish(256);
    }

    Profile profile_;
    std::int32_t width_, batch_, state_records_;
    DType state_dtype_;
    LayoutBuilder record_builder_;
    GdnReplayRecordLayout record_layout_ =
        make_record_layout(profile_, width_, batch_, record_builder_);
    StateImageBuilder state_builder_;
    LinearAttentionStateLayout state_layout_ =
        make_state_layout(profile_, state_dtype_, state_builder_);
    StateImageLayout image_layout_ = state_builder_.finish();
    DeviceBuffer record_storage_;
    GdnReplayRecords records_;
    DeviceBuffer state_storage_;
    StateImageStorage images_;
    LinearAttentionStateView states_;
    std::unique_ptr<bench::SavedBuffer> initial_;
    ops::GdnReplayFoldPlan fold_plan_{records_, states_.all_layers_view()};
};

class RecordFixture {
public:
    RecordFixture(const Profile& profile, int width, int batch, const std::vector<int>& valid,
                  DType state_dtype = DType::FP32)
        : value_heads_(profile.value_heads), batch_(batch), state_dtype_(state_dtype),
          inputs_(profile, width, batch),
          state_(state_dtype == DType::BF16
                     ? bench::make_bf16(state_elements() * batch_, 217U, -.02F, .02F)
                     : bench::make_f32(state_elements() * batch_, 217U, -.02F, .02F)),
          initial_(make_offsets(initial_offsets())),
          valid_(valid.empty() ? DeviceBuffer{} : make_i32(valid)),
          layout_(plan_gdn_replay_records(builder_,
                                          {1, batch, width, profile.conv_channels, profile.qk_heads,
                                           profile.value_heads, kStateDim, kStateDim})),
          storage_(bench::device_context().device(), builder_.finish(256)),
          records_(storage_.span(), layout_) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void launch_record(CudaExecutionView cuda) {
        Tensor initial(initial_.span(), DType::I64, {batch_});
        Tensor valid = valid_.data() ? Tensor(valid_.span(), DType::I32, {batch_}) : Tensor{};
        TensorRegion region{{0, state_elements() * dtype_size(state_dtype_), 16},
                            state_dtype_,
                            {kStateDim, kStateDim, value_heads_, 1}};
        const ConstStateFieldView source{{state_.span(), initial}, StateFieldLayout{region, 1, 0}};
        inputs_.record(source, valid, records_.layer(0, batch_).recurrence, cuda);
    }

private:
    std::size_t state_elements() const {
        return static_cast<std::size_t>(kStateDim) * kStateDim * value_heads_;
    }

    std::vector<std::int64_t> initial_offsets() const {
        std::vector<std::int64_t> offsets(batch_);
        for (int row = 0; row < batch_; ++row)
            offsets[row] = row * state_elements() * dtype_size(state_dtype_);
        return offsets;
    }

    int value_heads_, batch_;
    DType state_dtype_;
    RecordInputs inputs_;
    DeviceBuffer state_, initial_, valid_;
    LayoutBuilder builder_;
    GdnReplayRecordLayout layout_;
    DeviceBuffer storage_;
    GdnReplayRecords records_;
};

} // namespace ninfer::bench::gdn_replay
