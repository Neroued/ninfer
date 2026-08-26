// ReplaySSM recurrent-Record and all-layer Fold benchmark.
//
// Each timed GPU body is exactly one public Op call. Row-control construction, buffer
// initialization, and L2 eviction are outside the timed interval. Fold deliberately uses an
// ordinary eager launch; host submission time is reported separately from CUDA-event GPU latency.

#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kQkHeads        = 16;
constexpr std::int32_t kStateDim       = 128;
constexpr std::int32_t kRecordCapacity = 8;
constexpr std::int32_t kStateSlots     = 11;
constexpr std::size_t kDefaultFlush    = 256ULL << 20;

enum class ProfileSelection : std::uint8_t {
    Qwen27B,
    Qwen35B,
    All,
};

enum class CommitSelection : std::uint8_t {
    One,
    Dense,
    Mixed,
    All,
};

enum class ComponentSelection : std::uint8_t {
    Fold,
    Recurrent,
    All,
};

enum class ValidSelection : std::uint8_t {
    Dense,
    Mixed,
    All,
};

struct Profile {
    const char* name;
    std::int32_t layers;
    std::int32_t value_heads;
    std::int32_t conv_channels;
    std::vector<std::int32_t> widths;
};

struct Options {
    ProfileSelection profiles    = ProfileSelection::All;
    CommitSelection commits      = CommitSelection::All;
    ComponentSelection component = ComponentSelection::All;
    ValidSelection valid         = ValidSelection::All;
    std::int32_t exact_width     = 0;
    std::int32_t exact_batch     = 0;
    int warmup                   = 10;
    int repeat                   = 50;
    std::size_t flush_bytes      = kDefaultFlush;
};

struct Measurement {
    bench::ColdTiming warm;
    bench::ColdTiming cold;
    double host_median_us = 0.0;
};

std::int32_t parse_i32(std::string_view text, const char* label, std::int32_t minimum,
                       std::int32_t maximum) {
    const std::string value(text);
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::int32_t>(parsed);
}

ProfileSelection parse_profile(std::string_view value) {
    if (value == "27b") return ProfileSelection::Qwen27B;
    if (value == "35b") return ProfileSelection::Qwen35B;
    if (value == "all") return ProfileSelection::All;
    throw std::invalid_argument("--profile must be 27b, 35b, or all");
}

CommitSelection parse_commits(std::string_view value) {
    if (value == "one") return CommitSelection::One;
    if (value == "dense") return CommitSelection::Dense;
    if (value == "mixed") return CommitSelection::Mixed;
    if (value == "all") return CommitSelection::All;
    throw std::invalid_argument("--commits must be one, dense, mixed, or all");
}

ComponentSelection parse_component(std::string_view value) {
    if (value == "fold") return ComponentSelection::Fold;
    if (value == "recurrent") return ComponentSelection::Recurrent;
    if (value == "all") return ComponentSelection::All;
    throw std::invalid_argument("--component must be fold, recurrent, or all");
}

ValidSelection parse_valid(std::string_view value) {
    if (value == "dense") return ValidSelection::Dense;
    if (value == "mixed") return ValidSelection::Mixed;
    if (value == "all") return ValidSelection::All;
    throw std::invalid_argument("--valid must be dense, mixed, or all");
}

void print_help(const char* program) {
    std::printf("Usage: %s [options]\n\n"
                "  --profile 27b|35b|all       Registered all-layer geometry (default all).\n"
                "  --component fold|recurrent|all\n"
                "                              Measured component (default all).\n"
                "  --width T                   One physical width in [2,16].\n"
                "  --batch B                   One active row count in [1,8].\n"
                "  --commits one|dense|mixed|all\n"
                "                              Commit policy (default all).\n"
                "  --valid dense|mixed|all     Recurrent valid-prefix policy (default all).\n"
                "  --warmup N                  Warmups per point (default 10).\n"
                "  --repeat N                  Samples per point (default 50).\n"
                "  --flush-mib N               Cold-L2 eviction storage (default 256 MiB).\n"
                "  -h, --help                  Show this text.\n",
                program);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--profile") {
            options.profiles = parse_profile(next("profile"));
        } else if (argument == "--component") {
            options.component = parse_component(next("component"));
        } else if (argument == "--width") {
            options.exact_width = parse_i32(next("width"), "width", 2, 16);
        } else if (argument == "--batch") {
            options.exact_batch = parse_i32(next("batch"), "batch", 1, 8);
        } else if (argument == "--commits") {
            options.commits = parse_commits(next("commit policy"));
        } else if (argument == "--valid") {
            options.valid = parse_valid(next("valid policy"));
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("warmup"), "warmup", 0, INT32_MAX);
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("repeat"), "repeat", 1, INT32_MAX);
        } else if (argument == "--flush-mib") {
            const std::int32_t mib = parse_i32(next("flush MiB"), "flush MiB", 1, INT32_MAX >> 20);
            options.flush_bytes    = static_cast<std::size_t>(mib) << 20;
        } else if (argument == "--help" || argument == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    return options;
}

std::vector<Profile> selected_profiles(ProfileSelection selection) {
    std::vector<Profile> profiles;
    if (selection == ProfileSelection::Qwen27B || selection == ProfileSelection::All) {
        profiles.push_back({"27b", 48, 48, 10240, {2, 3, 4, 5, 6}});
    }
    if (selection == ProfileSelection::Qwen35B || selection == ProfileSelection::All) {
        profiles.push_back({"35b-a3b", 30, 32, 8192, {2, 6, 16}});
    }
    return profiles;
}

std::vector<std::int32_t> selected_widths(const Profile& profile, std::int32_t exact_width) {
    if (exact_width == 0) return profile.widths;
    if (std::find(profile.widths.begin(), profile.widths.end(), exact_width) ==
        profile.widths.end()) {
        throw std::invalid_argument("exact width is not in the selected profile workload");
    }
    return {exact_width};
}

std::vector<std::int32_t> selected_batches(std::int32_t exact_batch) {
    if (exact_batch != 0) return {exact_batch};
    return {1, 2, 4, 8};
}

std::vector<CommitSelection> selected_commit_policies(CommitSelection selection) {
    if (selection == CommitSelection::All) {
        return {CommitSelection::One, CommitSelection::Dense, CommitSelection::Mixed};
    }
    return {selection};
}

std::vector<ValidSelection> selected_valid_policies(ValidSelection selection) {
    if (selection == ValidSelection::All) { return {ValidSelection::Dense, ValidSelection::Mixed}; }
    return {selection};
}

const char* valid_name(ValidSelection selection) {
    switch (selection) {
    case ValidSelection::Dense:
        return "dense";
    case ValidSelection::Mixed:
        return "mixed";
    case ValidSelection::All:
        break;
    }
    return "invalid";
}

const char* commit_name(CommitSelection selection) {
    switch (selection) {
    case CommitSelection::One:
        return "one";
    case CommitSelection::Dense:
        return "dense";
    case CommitSelection::Mixed:
        return "mixed";
    case CommitSelection::All:
        break;
    }
    return "invalid";
}

std::vector<ops::GdnReplayFoldRow> make_rows(std::int32_t batch, std::int32_t width,
                                             CommitSelection selection) {
    constexpr std::int32_t kSlots[kRecordCapacity] = {10, 2, 8, 0, 6, 4, 9, 1};
    std::vector<ops::GdnReplayFoldRow> rows(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        std::int32_t commit = 1;
        if (selection == CommitSelection::Dense) {
            commit = width;
        } else if (selection == CommitSelection::Mixed) {
            constexpr std::int32_t kMixed[kRecordCapacity] = {0, 1, 2, 3, 16, 7, 12, 5};
            commit                                         = std::min(width, kMixed[row]);
        }
        rows[static_cast<std::size_t>(row)] = {kSlots[row], commit};
    }
    return rows;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <class Launch>
double measure_host_submission(Launch&& launch, cudaStream_t stream, int warmup, int repeat) {
    for (int index = 0; index < warmup; ++index) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeat));
    for (int index = 0; index < repeat; ++index) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto begin = std::chrono::steady_clock::now();
        launch(stream);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return median(std::move(samples));
}

class FoldResources {
public:
    FoldResources(const Profile& profile, std::int32_t width)
        : profile_(profile), width_(width), record_storage_(record_bytes(profile, width)),
          records_({record_storage_.p, record_storage_.bytes}, record_layout_),
          state_storage_(state_bytes(profile)),
          states_({state_storage_.p, state_storage_.bytes}, state_layout_) {
        record_storage_.fill(0);
        state_storage_.fill(0);
    }

    [[nodiscard]] const ops::GdnReplayFoldPlan& fold_plan() const { return fold_plan_; }

private:
    static GdnReplayRecordLayout make_record_layout(const Profile& profile, std::int32_t width,
                                                    LayoutBuilder& builder) {
        return plan_gdn_replay_records(builder, {.layers          = profile.layers,
                                                 .record_capacity = kRecordCapacity,
                                                 .width           = width,
                                                 .conv_channels   = profile.conv_channels,
                                                 .qk_heads        = kQkHeads,
                                                 .value_heads     = profile.value_heads,
                                                 .key_dim         = kStateDim,
                                                 .value_dim       = kStateDim});
    }

    static LinearAttentionStatePoolLayout make_state_layout(const Profile& profile,
                                                            LayoutBuilder& builder) {
        return plan_linear_attention_state_pool(
            builder, {.layers         = static_cast<std::uint32_t>(profile.layers),
                      .conv_channels  = profile.conv_channels,
                      .conv_width     = 3,
                      .value_heads    = profile.value_heads,
                      .value_head_dim = kStateDim,
                      .key_head_dim   = kStateDim,
                      .slot_count     = kStateSlots,
                      .conv_dtype     = DType::BF16});
    }

    static std::size_t record_bytes(const Profile& profile, std::int32_t width) {
        LayoutBuilder builder;
        (void)make_record_layout(profile, width, builder);
        return builder.finish(256);
    }

    static std::size_t state_bytes(const Profile& profile) {
        LayoutBuilder builder;
        (void)make_state_layout(profile, builder);
        return builder.finish(256);
    }

    Profile profile_;
    std::int32_t width_;
    LayoutBuilder record_builder_;
    GdnReplayRecordLayout record_layout_ = make_record_layout(profile_, width_, record_builder_);
    LayoutBuilder state_builder_;
    LinearAttentionStatePoolLayout state_layout_ = make_state_layout(profile_, state_builder_);
    DeviceBuffer record_storage_;
    GdnReplayRecords records_;
    DeviceBuffer state_storage_;
    LinearAttentionStatePool states_;
    ops::GdnReplayFoldPlan fold_plan_{records_, states_.all_layers_view()};
};

DeviceBuffer make_i32(const std::vector<std::int32_t>& values) {
    DeviceBuffer result(values.size() * sizeof(std::int32_t));
    result.copy_from_host(values.data(), result.bytes);
    return result;
}

DeviceBuffer make_f32(std::size_t elements, float value) {
    std::vector<float> values(elements, value);
    DeviceBuffer result(values.size() * sizeof(float));
    result.copy_from_host(values.data(), result.bytes);
    return result;
}

class RecurrentResources {
public:
    RecurrentResources(const Profile& profile, std::int32_t width, std::int32_t batch,
                       ValidSelection valid)
        : profile_(profile), width_(width), batch_(batch), valid_policy_(valid),
          q_(bench::make_bf16(qk_elements())), k_(bench::make_bf16(qk_elements())),
          v_(bench::make_bf16(value_elements())), g_(make_f32(gate_elements(), -0.7F)),
          beta_(make_f32(gate_elements(), 0.5F)),
          record_states_(record_state_elements() * sizeof(float)),
          record_initial_(make_i32(record_initial_slots())), valid_(make_valid()),
          record_out_(value_elements() * sizeof(std::uint16_t)),
          key_record_(qk_elements() * sizeof(std::uint16_t)),
          value_record_(value_elements() * sizeof(std::uint16_t)),
          gate_record_(gate_elements() * 2 * sizeof(float)) {
        record_states_.fill(0);
        record_out_.fill(0);
        key_record_.fill(0);
        value_record_.fill(0);
        gate_record_.fill(0);
    }

    void launch_record(cudaStream_t stream) {
        Tensor q(q_.p, DType::BF16, {kStateDim, kQkHeads, width_, batch_});
        Tensor k(k_.p, DType::BF16, {kStateDim, kQkHeads, width_, batch_});
        Tensor v(v_.p, DType::BF16, {kStateDim, profile_.value_heads, width_, batch_});
        Tensor g(g_.p, DType::FP32, {profile_.value_heads, width_, batch_});
        Tensor beta(beta_.p, DType::FP32, {profile_.value_heads, width_, batch_});
        Tensor states(record_states_.p, DType::FP32,
                      {kStateDim, kStateDim, profile_.value_heads, batch_});
        Tensor valid = valid_tensor();
        Tensor initial(record_initial_.p, DType::I32, {batch_});
        Tensor key_record(key_record_.p, DType::BF16, {kStateDim, kQkHeads, width_, batch_});
        Tensor value_record(value_record_.p, DType::BF16,
                            {kStateDim, profile_.value_heads, width_, batch_});
        Tensor gate_record(gate_record_.p, DType::FP32, {2, profile_.value_heads, width_, batch_});
        Tensor out(record_out_.p, DType::BF16, {kStateDim, profile_.value_heads, width_, batch_});
        ops::gated_delta_net_replay_record(q, k, v, g, beta, scale(), states, valid, initial,
                                           key_record, value_record, gate_record, out, stream);
    }

private:
    [[nodiscard]] static float scale() { return 1.0F / std::sqrt(128.0F); }

    [[nodiscard]] std::size_t qk_elements() const {
        return static_cast<std::size_t>(kStateDim) * kQkHeads * width_ * batch_;
    }

    [[nodiscard]] std::size_t value_elements() const {
        return static_cast<std::size_t>(kStateDim) * profile_.value_heads * width_ * batch_;
    }

    [[nodiscard]] std::size_t gate_elements() const {
        return static_cast<std::size_t>(profile_.value_heads) * width_ * batch_;
    }

    [[nodiscard]] std::size_t state_slot_elements() const {
        return static_cast<std::size_t>(kStateDim) * kStateDim * profile_.value_heads;
    }

    [[nodiscard]] std::size_t record_state_elements() const {
        return state_slot_elements() * static_cast<std::size_t>(batch_);
    }

    [[nodiscard]] std::vector<std::int32_t> record_initial_slots() const {
        std::vector<std::int32_t> slots(static_cast<std::size_t>(batch_));
        for (std::int32_t row = 0; row < batch_; ++row) { slots[row] = row; }
        return slots;
    }

    [[nodiscard]] DeviceBuffer make_valid() const {
        if (valid_policy_ == ValidSelection::Dense) return {};
        std::vector<std::int32_t> extents(static_cast<std::size_t>(batch_));
        for (std::int32_t row = 0; row < batch_; ++row) {
            extents[row] = std::max(1, width_ - row % width_);
        }
        return make_i32(extents);
    }

    [[nodiscard]] Tensor valid_tensor() const {
        if (valid_.p == nullptr) return {};
        return Tensor(valid_.p, DType::I32, {batch_});
    }

    Profile profile_;
    std::int32_t width_;
    std::int32_t batch_;
    ValidSelection valid_policy_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer g_;
    DeviceBuffer beta_;
    DeviceBuffer record_states_;
    DeviceBuffer record_initial_;
    DeviceBuffer valid_;
    DeviceBuffer record_out_;
    DeviceBuffer key_record_;
    DeviceBuffer value_record_;
    DeviceBuffer gate_record_;
};

Measurement measure_fold(const FoldResources& resources,
                         const std::vector<ops::GdnReplayFoldRow>& rows, DeviceBuffer& flush,
                         int warmup, int repeat) {
    cudaStream_t stream = nullptr;
    const auto launch   = [&](cudaStream_t launch_stream) {
        resources.fold_plan().execute(rows, launch_stream);
    };
    Measurement result;
    result.warm           = bench::measure_launch(launch, stream, warmup, repeat);
    result.cold           = bench::measure_cold_launch(launch, flush, stream, warmup, repeat);
    result.host_median_us = measure_host_submission(launch, stream, warmup, repeat);
    return result;
}

template <class Launch>
Measurement measure_component(Launch&& launch, DeviceBuffer& flush, int warmup, int repeat) {
    cudaStream_t stream = nullptr;
    Measurement result;
    result.warm           = bench::measure_launch(launch, stream, warmup, repeat);
    result.cold           = bench::measure_cold_launch(launch, flush, stream, warmup, repeat);
    result.host_median_us = measure_host_submission(launch, stream, warmup, repeat);
    return result;
}

void print_recurrent_result(const Profile& profile, std::int32_t width, std::int32_t batch,
                            ValidSelection valid, const char* form,
                            const Measurement& measurement) {
    std::printf("component=recurrent form=%-8s profile=%-8s T=%2d B=%d valid=%-5s "
                "gpu_warm=%8.2f us gpu_cold=%8.2f us host_submit=%7.2f us\n",
                form, profile.name, width, batch, valid_name(valid), measurement.warm.median_us,
                measurement.cold.median_us, measurement.host_median_us);
}

void run_recurrent_point(const Profile& profile, std::int32_t width, std::int32_t batch,
                         ValidSelection valid, DeviceBuffer& flush, const Options& options) {
    RecurrentResources resources(profile, width, batch, valid);
    const Measurement record =
        measure_component([&](cudaStream_t stream) { resources.launch_record(stream); }, flush,
                          options.warmup, options.repeat);
    print_recurrent_result(profile, width, batch, valid, "record", record);
}

void print_result(const Profile& profile, std::int32_t width, std::int32_t batch,
                  CommitSelection selection, const std::vector<ops::GdnReplayFoldRow>& rows,
                  const Measurement& measurement) {
    std::int64_t committed = 0;
    std::int32_t updated   = 0;
    for (const ops::GdnReplayFoldRow& row : rows) {
        committed += row.commit_columns;
        updated += row.commit_columns > 0 ? 1 : 0;
    }
    const double state_bytes = static_cast<double>(profile.layers) * updated * profile.value_heads *
                               kStateDim * kStateDim * sizeof(float) * 2.0;
    const double state_gbs         = state_bytes / (measurement.cold.median_us * 1.0e3);
    const double ns_per_transition = committed == 0
                                         ? 0.0
                                         : measurement.cold.median_us * 1000.0 /
                                               (static_cast<double>(profile.layers) * committed);
    std::printf("component=fold profile=%-8s T=%2d B=%d commits=%-5s gpu_warm=%8.2f us "
                "gpu_cold=%8.2f us "
                "host_submit=%7.2f us state_io=%7.1f GB/s ns/(layer,row,transition)=%7.2f\n",
                profile.name, width, batch, commit_name(selection), measurement.warm.median_us,
                measurement.cold.median_us, measurement.host_median_us, state_gbs,
                ns_per_transition);
}

int run(const Options& options) {
    DeviceBuffer flush(options.flush_bytes);
    for (const Profile& profile : selected_profiles(options.profiles)) {
        for (const std::int32_t width : selected_widths(profile, options.exact_width)) {
            const bool run_fold = options.component == ComponentSelection::Fold ||
                                  options.component == ComponentSelection::All;
            const bool run_recurrent = options.component == ComponentSelection::Recurrent ||
                                       options.component == ComponentSelection::All;
            if (run_fold) {
                FoldResources resources(profile, width);
                for (const std::int32_t batch : selected_batches(options.exact_batch)) {
                    for (const CommitSelection commits :
                         selected_commit_policies(options.commits)) {
                        const std::vector<ops::GdnReplayFoldRow> rows =
                            make_rows(batch, width, commits);
                        const Measurement measurement =
                            measure_fold(resources, rows, flush, options.warmup, options.repeat);
                        print_result(profile, width, batch, commits, rows, measurement);
                    }
                }
            }
            if (run_recurrent) {
                for (const std::int32_t batch : selected_batches(options.exact_batch)) {
                    for (const ValidSelection valid : selected_valid_policies(options.valid)) {
                        run_recurrent_point(profile, width, batch, valid, flush, options);
                    }
                }
            }
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            std::fprintf(stderr, "no CUDA device available\n");
            return 77;
        }
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
