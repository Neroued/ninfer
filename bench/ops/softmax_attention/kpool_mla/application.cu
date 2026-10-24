#include "application_fixture.cuh"
#include "ops/op_measurement.cuh"
#include "common/gpu_activity.cuh"
#include "common/roofline.h"
#include "core/decode_graph.h"
#include "../../../../third_party/nlohmann/json.hpp"

#include <set>
#include <unordered_set>
#include <fstream>

using namespace ninfer;
using namespace ninfer::bench;
namespace app = ninfer::bench::kpool_application;
namespace kp  = ninfer::tool::kpool;
using Json    = nlohmann::json;
using Clock   = std::chrono::steady_clock;

static double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static int bucket(int e) {
    int u = 128;
    while (u < e) u *= 2;
    return u;
}

static Json stats(const std::vector<double>& samples) {
    const auto s = summarize_timings(samples);
    return {{"p50_us", s.median_us},
            {"p95_us", s.p95_us},
            {"min_us", s.min_us},
            {"max_us", *std::max_element(samples.begin(), samples.end())},
            {"samples_us", samples}};
}

static kp::IndexerCase shape_of(const Json& c) {
    kp::IndexerCase s;
    s.valid      = c.at("valid_columns").get<std::vector<int>>();
    const auto f = c.at("frontiers").get<std::vector<int>>();
    if (s.valid.size() != f.size() || f.size() != c.at("B").get<std::size_t>())
        throw std::invalid_argument("frontiers and valid_columns must have B entries");
    for (std::size_t b = 0; b < f.size(); ++b) s.ends.push_back(f[b] + s.valid[b]);
    s.envelope         = c.at("envelope_max");
    s.shared_prefix    = c.value("shared_prefix_tokens", 0);
    s.contiguous_pages = c.value("contiguous_pages", false);
    if (c.contains("reserve_ends")) s.reserve_ends = c.at("reserve_ends").get<std::vector<int>>();
    return s;
}

static Json admission(CudaExecutionView cuda, const Json& c) {
    const auto shape = shape_of(c);
    const int T = c.at("T"), B = c.at("B"), L = c.value("layers", 11);
    const auto layout =
        ops::plan_kpool_mla_cache_layout({ops::KpoolMlaStorage::BFloat16, 512, 128, L});
    const auto page     = plan_kv_page_layout(layout.page.geometry);
    const auto bytes    = page.alignment + shape.pages(B, 0) * (page.page_bytes + page.alignment);
    const auto envelope = ops::KpoolExecutionEnvelope{std::uint32_t(shape.envelope)};
    const auto scratch =
        std::max({ops::kpool_indexer_workspace_capacity_bytes({32, 128}, envelope, B, 1, T,
                                                              cuda.multiprocessor_count()),
                  ops::kpool_sparse_mla_workspace_capacity_bytes(
                      {64, 256, 256, 512}, ops::KpoolMlaStorage::BFloat16, envelope, B, 1, T,
                      cuda.multiprocessor_count()),
                  ops::kpool_mla_prepare_record_workspace_capacity_bytes(
                      {kp::Projection::Full, 4096, 512, 128, 1536, 32, 4},
                      ops::KpoolMlaStorage::BFloat16, B, 1, T, cuda.multiprocessor_count()),
                  ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, 4096, 16384,
                                                       ops::LinearPolicy::A16Only, B, T * B)});
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    // Conservative fixture budget: device + Host mirrors of frame/weights, flush and
    // graph setup reserve. This is isolated-Op admission, not a complete-model budget.
    const std::uint64_t required = bytes + scratch + std::uint64_t(T) * B * 210000 + (2ULL << 30);
    return {{"native_kv_bytes", bytes},    {"estimated_total_bytes", required},
            {"cuda_free_bytes", free},     {"cuda_total_bytes", total},
            {"admitted", required < free}, {"scope", "isolated_attention_fixture"}};
}

static Json work(app::Fixture& f) {
    auto& i        = f.index;
    const auto ids = i.ids.download(), counts = i.counts.download(), ends = i.ends.download();
    std::unordered_set<std::size_t> latent, keys;
    std::uint64_t visits = 0, candidates = 0, active = 0, publications = 0, metadata = i.B * 4;
    bool factor = false;
    double mla  = 0;
    for (int b = 0; b < i.B; ++b) {
        int n = i.valid.at(b);
        active += n;
        if (!n) continue;
        int F = i.positions.at(0, b);
        metadata += 20 + std::uint64_t(n) * 8 + ((F + n + 63) / 64) * 8;
        if ((F + n) / 4 > 512)
            for (int g = 0; g < F / 4; ++g) keys.insert(f.physical_token(b, g * 4) / 4);
        publications += std::uint64_t(n) * 1024 + ((F + n) / 4 - F / 4) * 256;
        publications += f.record ? std::uint64_t(n) * 512 + (F % 4) * 512 : (F % 4 + 3) * 512;
        for (int t = 0; t < n; ++t) {
            int row = b * i.T + t, end = ends[row], size = counts[row] * 4 + end % 4;
            if (end / 4 > 512) candidates += end / 4;
            visits += size;
            factor |= size > 1;
            if (size) mla += 2. * 64 * 512 * 256;
            if (size > 1) mla += 2. * 64 * 512 * 256 + size * (4. * 64 * 512);
            for (int g = 0; g < counts[row]; ++g)
                for (int r = 0; r < 4; ++r) {
                    int token = ids[row * 512 + g] * 4 + r;
                    if (token < F) latent.insert(f.physical_token(b, token));
                }
            for (int token = end / 4 * 4; token < end; ++token)
                if (token < F) latent.insert(f.physical_token(b, token));
        }
    }
    double prepare = active * (2. * 4096 * 2336), wb = active * (2. * 1536 * 20480),
           idx = candidates * (2. * 32 * 128), o = active * (2. * 4096 * 16384);
    std::uint64_t weights = f.weights.weight.view.bytes() + f.w_b.view.bytes() +
                            f.value_up.view.bytes() + (factor ? f.key_up.view.bytes() : 0) +
                            f.o_weight.model_weight_bytes() + (512 + 1536 + 128 + 128 + 512) * 2;
    std::uint64_t bytes = metadata + publications + f.result.view.bytes() + latent.size() * 1024 +
                          keys.size() * 256 + (active ? weights + active * 4096 * 2 : 0);
    double flops            = prepare + wb + idx + mla + o;
    const double memory_us  = bytes / (kRoofline.sustained_read_gbs * 1000.);
    const double compute_us = flops / (kRoofline.bf16_f32_tflops * 1e6);
    return {{"minimum_external_bytes", bytes},
            {"useful_flops", flops},
            {"active_columns", active},
            {"selected_visits", visits},
            {"unique_old_latent_tokens", latent.size()},
            {"unique_old_index_pools", keys.size()},
            {"candidate_visits", candidates},
            {"weight_bytes", weights},
            {"flops_by_stage", {prepare, wb, idx, mla, o}},
            {"memory_reference_gbs", kRoofline.sustained_read_gbs},
            {"bf16_f32_reference_tflops", kRoofline.bf16_f32_tflops},
            {"memory_floor_us", memory_us},
            {"compute_floor_us", compute_us},
            {"roofline_us", std::max(memory_us, compute_us)}};
}

static Json static_case(CudaExecutionView cuda, const Json& c, int repeats, int warmup,
                        bool diagnostic) {
    Json out = {{"case", c}, {"admission", admission(cuda, c)}};
    if (!out["admission"]["admitted"].get<bool>()) {
        out["status"] = "not_admitted";
        return out;
    }
    const auto begin = Clock::now();
    app::Fixture f(cuda, c.at("T"), c.at("B"), shape_of(c),
                   c.value("prepare", "update") == "record", c.value("layers", 11));
    CUDA_CHECK(cudaDeviceSynchronize());
    L2FlushBuffer flush(256ULL << 20);
    measurement::BodyTimedGraph graph;
    f.launch(cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    graph.capture(cuda, [&](auto v) { f.launch(v); });
    out["setup_us"] = us(begin, Clock::now());
    std::vector<double> times, host;
    for (int r = -warmup; r < repeats; ++r) {
        condition_gpu(cuda, 200);
        flush_l2(flush, cuda);
        auto a = Clock::now();
        graph.launch(cuda);
        auto b = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        if (r >= 0) {
            times.push_back(graph.elapsed_us());
            host.push_back(us(a, b));
        }
    }
    out["body"]            = stats(times);
    out["host_launch"]     = stats(host);
    out["graph_kernels"]   = graph.kernels();
    out["workspace_bytes"] = f.index.workspace_bytes;
    out["workspace_peak"]  = f.index.workspace->peak_used();
    out["work"]            = work(f);
    if (diagnostic) {
        std::array<cudaEvent_t, 6> events;
        for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));
        {
            measurement::BodyTimedGraph staged;
            staged.capture(cuda, [&](auto v) {
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[0], v.stream(), cudaEventRecordExternal));
                f.prepare_phase(v);
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[1], v.stream(), cudaEventRecordExternal));
                f.queries(v);
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[2], v.stream(), cudaEventRecordExternal));
                f.index.launch(v);
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[3], v.stream(), cudaEventRecordExternal));
                f.attend(v);
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[4], v.stream(), cudaEventRecordExternal));
                f.project_o(v);
                CUDA_CHECK(
                    cudaEventRecordWithFlags(events[5], v.stream(), cudaEventRecordExternal));
            });
            std::array<std::vector<double>, 5> values;
            for (int r = -3; r < 20; ++r) {
                condition_gpu(cuda, 200);
                flush_l2(flush, cuda);
                staged.launch(cuda);
                CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
                if (r < 0) continue;
                for (int s = 0; s < 5; ++s) {
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, events[s], events[s + 1]));
                    values[s].push_back(ms * 1000.);
                }
            }
            for (int s = 0; s < 5; ++s)
                out["stages"][std::array{"prepare", "wb", "indexer", "mla", "o"}[s]] =
                    stats(values[s]);
        }
        for (auto e : events) cudaEventDestroy(e);
    }
    if (!f.index.scratch_guards() || !f.result.guards())
        throw std::runtime_error("application buffer guard");
    out["status"] = "measured";
    return out;
}

#include "trajectory.cuh"
#include "mtp.cuh"

int main(int argc, char** argv) {
    try {
        std::string path, outpath, jobs_path;
        int repeat = 100, warmup = 10;
        bool stages = true;
        for (int a = 1; a < argc; ++a) {
            std::string flag = argv[a];
            if (flag == "--help") {
                std::cout << "Usage: " << argv[0]
                          << " --case JSON --out JSON [--repeat N --warmup N --stages 0|1]\n"
                          << "       " << argv[0] << " --jobs JSON\n"
                          << "Case modes: static (default), trajectory, mtp.\n"
                          << "Jobs: array of {case, out, repeat, warmup, stages}.\n"
                          << "Defaults: repeat=100, warmup=10, stages=1.\n";
                return 0;
            }
            auto v = measurement::take(argc, argv, a);
            if (flag == "--case")
                path = v;
            else if (flag == "--out")
                outpath = v;
            else if (flag == "--jobs")
                jobs_path = v;
            else if (flag == "--repeat")
                repeat = measurement::integer(v);
            else if (flag == "--warmup")
                warmup = measurement::integer(v, 0);
            else if (flag == "--stages")
                stages = measurement::integer(v, 0, 1);
            else
                throw std::invalid_argument("unknown option " + flag);
        }
        Json jobs = Json::array();
        if (!jobs_path.empty())
            std::ifstream(jobs_path) >> jobs;
        else {
            Json c;
            std::ifstream(path) >> c;
            jobs.push_back({{"case", c},
                            {"out", outpath},
                            {"repeat", repeat},
                            {"warmup", warmup},
                            {"stages", stages}});
        }
        const auto& device = device_context();
        auto cuda          = device.execution_view();
        int runtime = 0, driver = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        const auto& properties = device.properties();
        const Json hardware    = {{"name", properties.name},
                                  {"sm_count", cuda.multiprocessor_count()},
                                  {"compute_capability", device.compute_capability()},
                                  {"cuda_runtime", runtime},
                                  {"cuda_driver", driver}};
        int completed          = 0;
        for (const auto& job : jobs) {
            const auto& c          = job.at("case");
            const auto begin       = Clock::now();
            const std::string mode = c.value("mode", "static");
            const int samples = job.value("repeat", 100), warmups = job.value("warmup", 10);
            if (samples < 1 || warmups < 0)
                throw std::invalid_argument("repeat must be positive and warmup nonnegative");
            if (mode != "static" && mode != "trajectory" && mode != "mtp")
                throw std::invalid_argument("unknown case mode: " + mode);
            Json result = mode == "trajectory" ? trajectory(cuda, c, c.value("repetitions", 5))
                          : mode == "mtp"
                              ? mtp_round(cuda, c, samples, warmups)
                              : static_case(cuda, c, samples, warmups, job.value("stages", true));
            result["hardware"]     = hardware;
            const std::string dest = job.value("out", std::string{});
            if (dest.empty())
                std::cout << result.dump(2) << '\n';
            else {
                std::ofstream output(dest);
                if (!(output << result.dump(2) << '\n'))
                    throw std::runtime_error("cannot write benchmark result: " + dest);
            }
            std::cerr << ++completed << "/" << jobs.size() << " " << c.at("id") << " "
                      << result.at("status") << " " << us(begin, Clock::now()) / 1e6 << "s"
                      << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
