#include "fixture.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::vda;

namespace {
int run(TestCase c) {
    Fixture fixture(c);
    int errors = 0;
    for (bool gate : {false, true}) {
        fixture.call(gate, cuda_environment().default_execution);
        errors += fixture.verify(gate);
    }
    return errors;
}

int seed_case(int tokens, int heads, int batch, bool pitched) {
    TestCase c{1, tokens, heads, batch, 5, 1, 1, true, pitched};
    auto in = make_input(c, 73);
    VectorData k(in.k, heads, tokens, batch, pitched), v(in.v, heads, tokens, batch, pitched);
    auto beta = to_device_f32(in.beta);
    if (!tokens) beta = DeviceBuffer(cuda_environment().context.device(), 256);
    GuardedDeviceBuffer out(std::size_t(heads) * batch * 128 * 128 * 4, pitched ? 260 : 256),
        scratch(std::max<std::size_t>(
            256, ops::video_delta_state_init_workspace_capacity_bytes(heads, 0, tokens, batch)));
    MemoryArena ws(scratch.span());
    Tensor s(out.span(), DType::FP32, {128, 128, heads, batch}),
        b(beta.data(), beta.device(), DType::FP32, {heads, std::max(tokens, 1), batch});
    b.ne[1] = tokens;
    ops::video_delta_state_init(k.tensor, v.tensor, b, .5f, ws, s,
                                cuda_environment().default_execution);
    cuda_synchronize();
    auto expected = vda_ref::initial(in, .5);
    auto floats   = from_device<float>(out.data(), expected.size());
    std::vector<double> got(floats.begin(), floats.end());
    int errors = 0;
    for (int group = 0; group < heads * batch; ++group) {
        const auto begin = std::size_t(group) * 128 * 128;
        errors += verify_reduction(
            "VDA text state per head/batch",
            std::vector<double>(got.begin() + begin, got.begin() + begin + 128 * 128),
            std::vector<double>(expected.begin() + begin, expected.begin() + begin + 128 * 128),
            StateCriterion);
    }
    if (!tokens) {
        errors += verify_exact("empty seed is positive zero",
                               from_device<std::uint32_t>(out.data(), expected.size()),
                               std::vector<std::uint32_t>(expected.size(), 0));
    }
    return errors + out.verify_guards("seed") + scratch.verify_guards("seed scratch") +
           k.intact("seed k") + v.intact("seed v");
}

int gates() {
    Fixture fixture({11, 17, 2});
    for (std::size_t i = 0; i < fixture.input.alpha.size(); ++i)
        fixture.input.alpha[i] = i % 3 == 0 ? 0.f : (i % 3 == 1 ? 1.f : 1e-15f);
    for (std::size_t i = 0; i < fixture.input.beta.size(); ++i)
        fixture.input.beta[i] = i % 2 ? 1.f : 0.f;
    fixture.alpha.copy_from_host(fixture.input.alpha.data(), fixture.input.alpha.size() * 4);
    fixture.beta.copy_from_host(fixture.input.beta.data(), fixture.input.beta.size() * 4);
    fixture.call(false, cuda_environment().default_execution);
    return fixture.verify(false);
}

int graph() {
    Fixture fixture({11, 17, 2, 2});
    auto ex = cuda_environment().context.execution_view();
    fixture.call(true, ex);
    cuda_synchronize();
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    CUDA_CHECK(cudaStreamBeginCapture(ex.stream(), cudaStreamCaptureModeThreadLocal));
    fixture.call(true, ex);
    CUDA_CHECK(cudaStreamEndCapture(ex.stream(), &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    int fail = 0;
    for (int i = 0; i < 2; ++i) {
        for (auto& x : fixture.input.alpha) x *= .97f;
        for (auto& x : fixture.input.beta) x = 1.f - x;
        for (auto& x : fixture.input.initial) x *= .7f;
        for (auto* values : {&fixture.input.q, &fixture.input.k, &fixture.input.v,
                             &fixture.input.gate, &fixture.input.weight}) {
            for (auto& x : *values) x *= -.9f;
            round_to_bf16(*values);
        }
        fixture.q.update(fixture.input.q);
        fixture.k.update(fixture.input.k);
        fixture.v.update(fixture.input.v);
        fixture.gate.update(fixture.input.gate);
        std::vector<std::uint16_t> weight_bits;
        for (float x : fixture.input.weight) weight_bits.push_back(f32_to_bf16(x));
        fixture.weight.copy_from_host(weight_bits.data(), weight_bits.size() * 2);
        fixture.alpha.copy_from_host(fixture.input.alpha.data(), fixture.input.alpha.size() * 4);
        fixture.beta.copy_from_host(fixture.input.beta.data(), fixture.input.beta.size() * 4);
        fixture.initial.copy_from_host(fixture.input.initial.data(),
                                       fixture.input.initial.size() * 4);
        fixture.output.fill(0xff);
        fixture.scratch.fill(0xff);
        CUDA_CHECK(cudaGraphLaunch(executable, ex.stream()));
        fail += fixture.verify(true);
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return fail;
}

int rejection() {
    Fixture fixture({4, 17, 2});
    auto ex      = cuda_environment().default_execution;
    int fail     = 0;
    auto rejects = [&](const char* label, auto fn) {
        try {
            fn();
            std::cerr << "did not reject " << label << '\n';
            ++fail;
        } catch (const std::exception&) {};
    };
    auto call = [&](const Tensor& q, Tensor& out, MemoryArena& ws,
                    ops::VideoDeltaAttentionWindow w) {
        ops::video_delta_attention(q, fixture.k.tensor, fixture.v.tensor, fixture.retention_tensor,
                                   fixture.beta_tensor, fixture.initial_tensor, 17, w, ws, out, ex);
    };
    auto bad  = fixture.q.tensor;
    bad.nb[1] = 2;
    rejects("overlapping strides",
            [&] { call(bad, fixture.output_tensor, fixture.workspace, {}); });
    bad       = fixture.q.tensor;
    bad.dtype = DType::FP16;
    rejects("wrong dtype", [&] { call(bad, fixture.output_tensor, fixture.workspace, {}); });
    auto aliased = fixture.q.tensor;
    rejects("output alias", [&] { call(fixture.q.tensor, aliased, fixture.workspace, {}); });
    rejects("zero chunk",
            [&] { call(fixture.q.tensor, fixture.output_tensor, fixture.workspace, {0, 1, 0}); });
    rejects("negative radius",
            [&] { call(fixture.q.tensor, fixture.output_tensor, fixture.workspace, {5, -1, 0}); });
    rejects("invalid offset",
            [&] { call(fixture.q.tensor, fixture.output_tensor, fixture.workspace, {5, 1, 5}); });
    DeviceBuffer tiny(ex.device(), 1);
    MemoryArena ws(tiny.span());
    rejects("short workspace", [&] { call(fixture.q.tensor, fixture.output_tensor, ws, {}); });
    rejects("unsupported heads", [&] {
        (void)ops::video_delta_attention_workspace_capacity_bytes(57, 17, 0, 4, 1, {});
    });
    rejects("token overflow", [&] {
        (void)ops::video_delta_attention_workspace_capacity_bytes(2, INT32_MAX, 0, 2, 1, {});
    });
    bad        = fixture.q.tensor;
    bad.device = Device::cpu();
    rejects("wrong device", [&] { call(bad, fixture.output_tensor, fixture.workspace, {}); });
    bad = fixture.q.tensor;
    bad.nb[2] += 1;
    rejects("odd outer stride", [&] { call(bad, fixture.output_tensor, fixture.workspace, {}); });
    bad       = fixture.q.tensor;
    bad.ne[0] = 64;
    rejects("wrong head dimension",
            [&] { call(bad, fixture.output_tensor, fixture.workspace, {}); });
    auto out = fixture.output_tensor;
    out.nb[2] += 2;
    rejects("strided output", [&] { call(fixture.q.tensor, out, fixture.workspace, {}); });
    MemoryArena aliased_workspace(
        {fixture.q.tensor.data, fixture.workspace.capacity(), ex.device()});
    rejects("workspace alias",
            [&] { call(fixture.q.tensor, fixture.output_tensor, aliased_workspace, {}); });
    const auto weight          = fixture.weight_tensor;
    fixture.weight_tensor.data = fixture.output_tensor.data;
    rejects("norm/output alias", [&] { fixture.call(true, ex); });
    fixture.weight_tensor = weight;
    rejects("invalid epsilon", [&] {
        ops::video_delta_attention_sigmoid_gated(
            fixture.q.tensor, fixture.k.tensor, fixture.v.tensor, fixture.retention_tensor,
            fixture.beta_tensor, fixture.initial_tensor, fixture.weight_tensor, fixture.gate.tensor,
            0, 17, {}, fixture.workspace, fixture.output_tensor, ex);
    });
    rejects("nonfinite seed scale", [&] {
        ops::video_delta_state_init(fixture.k.tensor, fixture.v.tensor, fixture.beta_tensor,
                                    std::numeric_limits<float>::infinity(), fixture.workspace,
                                    fixture.initial_tensor, ex);
    });
    cuda_synchronize();
    fail += verify_exact("rejected output untouched",
                         from_device<std::uint8_t>(fixture.output.data(), fixture.output.bytes()),
                         std::vector<std::uint8_t>(fixture.output.bytes(), 0xff));
    return fail + (fixture.workspace.used() != 0) +
           fixture.output.verify_guards("rejected output") + fixture.q.intact("rejected q") +
           fixture.k.intact("rejected k");
}

int empty_graph() {
    Fixture fixture({0, 17, 2});
    const auto execution = cuda_environment().context.execution_view();
    cudaGraph_t graph{};
    CUDA_CHECK(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeThreadLocal));
    fixture.call(false, execution);
    fixture.call(true, execution);
    CUDA_CHECK(cudaStreamEndCapture(execution.stream(), &graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return (nodes != 0) + fixture.verify(false);
}

int natural_alignment() {
    Fixture fixture({11, 17, 2, 2, 5, 1, 1, false, true});
    GuardedDeviceBuffer alpha(fixture.input.alpha.size() * 4, 260);
    GuardedDeviceBuffer beta(fixture.input.beta.size() * 4, 260);
    GuardedDeviceBuffer initial(fixture.input.initial.size() * 4, 260);
    GuardedDeviceBuffer output(fixture.input.q.size() * 2, 258);
    GuardedDeviceBuffer weight(256, 258);
    alpha.copy_from_host(fixture.input.alpha.data(), alpha.bytes());
    beta.copy_from_host(fixture.input.beta.data(), beta.bytes());
    initial.copy_from_host(fixture.input.initial.data(), initial.bytes());
    const auto bits = from_device<std::uint16_t>(fixture.weight.data(), 128);
    weight.copy_from_host(bits.data(), weight.bytes());
    fixture.retention_tensor.data = alpha.data();
    fixture.beta_tensor.data      = beta.data();
    fixture.initial_tensor.data   = initial.data();
    fixture.output_tensor.data    = output.data();
    fixture.weight_tensor.data    = weight.data();
    fixture.call(true, cuda_environment().default_execution);
    return fixture.verify(true) + alpha.verify_guards("unaligned alpha") +
           beta.verify_guards("unaligned beta") + initial.verify_guards("unaligned initial") +
           output.verify_guards("unaligned output") + weight.verify_guards("unaligned norm");
}

int read_only_aliases() {
    Fixture fixture({11, 17, 2});
    fixture.input.q     = fixture.input.k;
    fixture.q.tensor    = fixture.k.tensor;
    fixture.input.gate  = fixture.input.v;
    fixture.gate.tensor = fixture.v.tensor;
    fixture.call(true, cuda_environment().default_execution);
    return fixture.verify(true);
}

int workspace_interval() {
    constexpr int heads = 29, last_frame = 32;
    const ops::VideoDeltaAttentionWindow window{5, 1, 1};
    const auto capacity =
        ops::video_delta_attention_workspace_capacity_bytes(heads, 3, 0, last_frame, 1, window);
    GuardedDeviceBuffer backing(capacity + 1);
    int errors = 0;
    for (const int frames : {1, 4, 5, 6, 9, 10, 11, 31, 32}) {
        Fixture fixture({frames, 3, heads});
        fixture.workspace = MemoryArena(backing.span().subspan(1, capacity));
        fixture.call(false, cuda_environment().default_execution);
        errors += fixture.verify(false);
        errors += fixture.workspace.peak_used() > capacity;
    }
    return errors + backing.verify_guards("interval capacity");
}

int zero_controls() {
    Fixture fixture({11, 17, 2});
    std::fill(fixture.input.beta.begin(), fixture.input.beta.end(), 0.f);
    std::fill(fixture.input.alpha.begin(), fixture.input.alpha.end(), 1.f);
    fixture.beta.copy_from_host(fixture.input.beta.data(), fixture.input.beta.size() * 4);
    fixture.alpha.copy_from_host(fixture.input.alpha.data(), fixture.input.alpha.size() * 4);
    fixture.call(false, cuda_environment().default_execution);
    int errors = fixture.verify(false);
    std::fill(fixture.input.initial.begin(), fixture.input.initial.end(), 0.f);
    fixture.initial.copy_from_host(fixture.input.initial.data(), fixture.input.initial.size() * 4);
    fixture.call(true, cuda_environment().default_execution);
    return errors + fixture.verify(true);
}

int sigmoid_range() {
    Fixture fixture({11, 17, 2});
    int errors = 0;
    for (const auto [logit, weight] : std::array<std::pair<float, float>, 4>{
             {{-20.f, 1.f}, {20.f, 1.f}, {-100.f, 1e20f}, {-150.f, 1e35f}}}) {
        std::fill(fixture.input.gate.begin(), fixture.input.gate.end(), logit);
        std::fill(fixture.input.weight.begin(), fixture.input.weight.end(), weight);
        round_to_bf16(fixture.input.weight);
        fixture.gate.update(fixture.input.gate);
        std::vector<std::uint16_t> bits;
        for (float value : fixture.input.weight) bits.push_back(f32_to_bf16(value));
        fixture.weight.copy_from_host(bits.data(), bits.size() * 2);
        fixture.call(true, cuda_environment().default_execution);
        errors += fixture.verify(true);
    }
    return errors;
}

int head_partitions() {
    constexpr int frames = 11, per_frame = 3, total_heads = 56, tokens = frames * per_frame;
    Fixture full({frames, per_frame, total_heads});
    const auto expected  = vda_ref::evaluate(full.input, false);
    const auto execution = cuda_environment().default_execution;
    const ops::VideoDeltaAttentionWindow window{5, 1, 1};
    int errors = 0;
    for (const int partitions : {2, 4, 8}) {
        const int heads = total_heads / partitions;
        GuardedDeviceBuffer output(std::size_t(tokens) * heads * 128 * 2);
        GuardedDeviceBuffer scratch(ops::video_delta_attention_workspace_capacity_bytes(
            heads, per_frame, frames, frames, 1, window));
        MemoryArena workspace(scratch.span());
        for (int rank = 0; rank < partitions; ++rank) {
            const int first_head = rank * heads;
            auto view            = [&](const Tensor& tensor) {
                auto result  = tensor;
                result.ne[1] = heads;
                result.data  = static_cast<std::byte*>(tensor.data) + first_head * tensor.nb[1];
                return result;
            };
            std::vector<float> alpha(std::size_t(frames) * heads * 128),
                beta(std::size_t(tokens) * heads);
            for (int f = 0; f < frames; ++f)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < 128; ++d)
                        alpha[(f * heads + h) * 128 + d] =
                            full.input.alpha[(f * total_heads + first_head + h) * 128 + d];
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < heads; ++h)
                    beta[t * heads + h] = full.input.beta[t * total_heads + first_head + h];
            auto device_alpha = to_device_f32(alpha), device_beta = to_device_f32(beta);
            Tensor initial(static_cast<float*>(full.initial_tensor.data) + first_head * 128 * 128,
                           execution.device(), DType::FP32, {128, 128, heads});
            Tensor result(output.span(), DType::BF16, {128, heads, tokens});
            ops::video_delta_attention(
                view(full.q.tensor), view(full.k.tensor), view(full.v.tensor),
                Tensor(device_alpha.span(), DType::FP32, {128, heads, frames}),
                Tensor(device_beta.span(), DType::FP32, {heads, tokens}), initial, per_frame,
                window, workspace, result, execution);
            cuda_synchronize();
            const auto got = from_device_bf16(output.data(), std::size_t(tokens) * heads * 128);
            for (int f = 0; f < frames; ++f) {
                for (int h = 0; h < heads; ++h) {
                    std::vector<double> actual, reference;
                    for (int u = 0; u < per_frame; ++u)
                        for (int d = 0; d < 128; ++d) {
                            const int token = f * per_frame + u;
                            actual.push_back(got[(token * heads + h) * 128 + d]);
                            reference.push_back(
                                expected[(token * total_heads + first_head + h) * 128 + d]);
                        }
                    errors += verify_reduction("TP head partition vs FP64", actual, reference,
                                               OutputCriterion);
                }
            }
            errors += output.verify_guards("TP output") + scratch.verify_guards("TP workspace");
        }
    }
    return errors + full.q.intact("TP q") + full.k.intact("TP k") + full.v.intact("TP v");
}

int state_workspace_graph() {
    constexpr int heads = 7, maximum_tokens = 1024;
    auto input = make_input({1, maximum_tokens, heads}, 97);
    VectorData k(input.k, heads, maximum_tokens, 1, true),
        v(input.v, heads, maximum_tokens, 1, true);
    auto beta = to_device_f32(input.beta);
    const auto capacity =
        ops::video_delta_state_init_workspace_capacity_bytes(heads, 0, maximum_tokens, 1);
    GuardedDeviceBuffer backing(capacity + 1), output(heads * 128 * 128 * 4, 260);
    MemoryArena workspace(backing.span().subspan(1, capacity));
    Tensor state(output.span(), DType::FP32, {128, 128, heads});
    const auto execution = cuda_environment().context.execution_view();
    auto call            = [&](int length) {
        auto keys = k.tensor, values = v.tensor;
        keys.ne[2] = values.ne[2] = length;
        Tensor weights(beta.span(), DType::FP32, {heads, std::max(length, 1)});
        weights.ne[1] = length;
        ops::video_delta_state_init(keys, values, weights, .5f, workspace, state, execution);
    };
    auto verify = [&](int length) {
        cuda_synchronize();
        auto prefix      = input;
        prefix.per_frame = length;
        prefix.k.resize(std::size_t(length) * heads * 128);
        prefix.v.resize(prefix.k.size());
        prefix.beta.resize(std::size_t(length) * heads);
        const auto expected = vda_ref::initial(prefix, .5);
        const auto values   = from_device<float>(state.data, expected.size());
        int failures        = 0;
        for (int head = 0; head < heads; ++head) {
            const auto begin = std::size_t(head) * 128 * 128;
            failures += verify_reduction(
                "state interval/graph vs FP64",
                std::vector<double>(values.begin() + begin, values.begin() + begin + 128 * 128),
                std::vector<double>(expected.begin() + begin, expected.begin() + begin + 128 * 128),
                StateCriterion);
        }
        failures += workspace.used() != 0 || workspace.peak_used() > capacity;
        return failures + output.verify_guards("state graph output") +
               backing.verify_guards("state graph workspace");
    };
    int errors = 0;
    for (int length : {0, 31, 32, 33, 767, 768, 769, 1024}) {
        call(length);
        errors += verify(length);
    }
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    CUDA_CHECK(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeThreadLocal));
    call(maximum_tokens);
    CUDA_CHECK(cudaStreamEndCapture(execution.stream(), &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
        for (auto& value : input.k) value = -value;
        for (auto& value : input.v) value *= .5f;
        for (auto& value : input.beta) value = 1.f - value;
        round_to_bf16(input.v);
        k.update(input.k);
        v.update(input.v);
        beta.copy_from_host(input.beta.data(), input.beta.size() * 4);
        output.fill(0xff);
        backing.fill(0xff);
        CUDA_CHECK(cudaGraphLaunch(executable, execution.stream()));
        errors += verify(maximum_tokens);
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return errors + k.intact("state graph keys") + v.intact("state graph values") +
           verify_exact("state beta immutable", from_device<float>(beta.data(), input.beta.size()),
                        input.beta);
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (TestCase c : std::vector<TestCase>{{1, 1, 1},
                                            {4, 17, 7},
                                            {5, 127, 1},
                                            {6, 129, 2, 2},
                                            {11, 17, 14},
                                            {9, 3, 28},
                                            {7, 2, 56},
                                            {11, 3, 56, 2},
                                            {11, 17, 13, 3, 5, 1, 1, true, true},
                                            {11, 17, 2, 2, 5, 1, 1, true, true},
                                            {11, 32, 1, 1, 5, 0, 4, true},
                                            {17, 32, 1, 1, 3, 2, 2},
                                            {100, 1008, 1, 1, 5, 1, 1, true}}) {
        const auto id = "F" + std::to_string(c.frames) + "-U" + std::to_string(c.tokens_per_frame) +
                        "-H" + std::to_string(c.heads) + "-B" + std::to_string(c.batch) + "-o" +
                        std::to_string(c.frame_offset);
        suite.add(id, [c] { return run(c); }, {{"suite", c.frames == 100 ? "stress" : "math"}});
    }
    suite.add("empty-frames", empty_graph, {{"suite", "contract"}});
    suite.add("seed-empty", [] { return seed_case(0, 2, 1, false); }, {{"suite", "math"}});
    suite.add("seed-pitched", [] { return seed_case(17, 7, 2, true); }, {{"suite", "math"}});
    suite.add("seed-correlated-1008", [] { return seed_case(1008, 1, 1, false); },
              {{"suite", "stress"}});
    suite.add("seed-H3-prompt", [] { return seed_case(1024, 7, 1, false); }, {{"suite", "stress"}});
    suite.add("seed-H3-ref6", [] { return seed_case(6948, 7, 1, false); }, {{"suite", "stress"}});
    suite.add("seed-batch-wave-tail", [] { return seed_case(1024, 13, 5, true); },
              {{"suite", "stress"}});
    for (const int tokens : {31, 32, 33, 63, 64, 65, 129}) {
        suite.add("seed-partition-tail-" + std::to_string(tokens),
                  [tokens] { return seed_case(tokens, 14, 1, false); }, {{"suite", "math"}});
    }
    suite.add("gates", gates, {{"suite", "math"}});
    suite.add("graph-replay", graph, {{"suite", "contract"}});
    suite.add("rejections", rejection, {{"suite", "contract"}});
    suite.add("natural-alignment", natural_alignment, {{"suite", "contract"}});
    suite.add("read-only-aliases", read_only_aliases, {{"suite", "contract"}});
    suite.add("workspace-interval", workspace_interval, {{"suite", "contract"}});
    suite.add("zero-controls", zero_controls, {{"suite", "math"}});
    suite.add("sigmoid-range", sigmoid_range, {{"suite", "math"}});
    suite.add("TP-head-partitions", head_partitions, {{"suite", "contract"}});
    suite.add("state-workspace-graph", state_workspace_graph, {{"suite", "contract"}});
    return suite.main(argc, argv);
}
