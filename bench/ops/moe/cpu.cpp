#include "core/buffer.h"
#include "ninfer/ops/moe.h"

#include <time.h>
#include <emmintrin.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
using Clock = std::chrono::steady_clock;

std::string_view format_name(QType down) {
    return down == QType::NVFP4 ? "nvfp4" : down == QType::Q6_G64_FP16 ? "q4-q6" : "q4-q5";
}

std::uint32_t number(std::string_view text) {
    std::uint32_t value;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid integer");
    }
    return value;
}

std::vector<std::uint32_t> list(std::string_view text) {
    std::vector<std::uint32_t> result;
    for (;;) {
        const auto end = text.find(',');
        result.push_back(number(text.substr(0, end)));
        if (end == text.npos) { return result; }
        text.remove_prefix(end + 1);
    }
}

std::uint64_t random_word(std::uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545f4914f6cdd1dULL;
}

WeightGeometry geometry(QType format, std::uint64_t rows, std::uint64_t columns) {
    const std::array shape{rows, columns};
    return weight_geometry(
        format, format == QType::NVFP4 ? QuantLayout::BlockScaleK16M128x4 : QuantLayout::RowSplit,
        shape);
}

ops::RoutedSwiGLUProfile make_profile(QType down) {
    if (down == QType::NVFP4) {
        return {{2560, 640, down, down, QuantLayout::BlockScaleK16M128x4,
                 QuantLayout::BlockScaleK16M128x4, ops::LinearPolicy::AllowA4,
                 ops::LinearPolicy::AllowA4},
                512,
                10};
    }
    return {{2048, 512, QType::Q4_G64_FP16, down, QuantLayout::RowSplit, QuantLayout::RowSplit},
            256,
            8};
}

struct Bank {
    ops::RoutedSwiGLUProfile profile;
    HostBuffer gu_storage, down_storage;
    std::vector<float> gu_divisors, down_divisors;
    WeightParent gate_up_parent, down_parent;
    ops::CpuSwiGLUExperts weights;

    Bank(QType down_format, std::uint64_t seed) : profile(make_profile(down_format)) {
        const int H = profile.ffn.hidden, I = profile.ffn.intermediate, E = profile.experts;
        const auto fill = [&](HostBuffer& storage, WeightParent& parent,
                              std::vector<float>& divisors, QType format, int rows, int columns) {
            const bool nvfp4 = format == QType::NVFP4;
            if (nvfp4) {
                const std::array<std::uint64_t, 3> shape{unsigned(E), unsigned(rows),
                                                         unsigned(columns)};
                parent.geometry =
                    weight_geometry(format, QuantLayout::BatchedBlockScaleK16M128x4, shape);
            } else {
                parent.geometry = geometry(format, std::uint64_t(E) * rows, columns);
            }
            const auto& g = parent.geometry;
            storage       = HostBuffer(g.bytes);
            auto* data    = static_cast<std::byte*>(storage.data());
            for (std::size_t offset = 0; offset < g.scale_offset; offset += 8) {
                const auto word = random_word(seed);
                std::memcpy(data + offset, &word, sizeof(word));
            }
            if (nvfp4) {
                auto* scales = reinterpret_cast<std::uint8_t*>(data + g.scale_offset);
                for (std::size_t s = 0; s < g.scale_bytes; ++s) {
                    scales[s] = 32 + random_word(seed) % 48;
                }
                divisors.resize(E);
                for (auto& d : divisors) { d = 511.75F + random_word(seed) % 1024; }
                std::memcpy(data + g.divisor_offset, divisors.data(),
                            divisors.size() * sizeof(float));
            } else {
                auto* scales = reinterpret_cast<std::uint16_t*>(data + g.scale_offset);
                for (std::size_t s = 0; s < g.scale_bytes / 2; ++s) {
                    scales[s] = 0x1400 + (random_word(seed) & 2047);
                }
            }
            parent.device          = Device::cpu();
            parent.data            = data;
            parent.weight_divisors = divisors;
            return bind_weight_bank(
                parent, describe_weight_bank(g, {0, unsigned(E), unsigned(rows), unsigned(rows)}));
        };
        weights = {
            fill(gu_storage, gate_up_parent, gu_divisors, profile.ffn.gate_up_type, 2 * I, H),
            fill(down_storage, down_parent, down_divisors, profile.ffn.down_type, H, I)};
    }
};

struct Sample {
    std::vector<int> ids;
    std::vector<float> coefficients;
    std::uint32_t bank;
    int assignments = 0, unique = 0;
    ops::ExpertRoutes routes;

    Sample(const ops::RoutedSwiGLUProfile& profile, int tokens, int active, std::uint32_t bank,
           unsigned start, std::string_view distribution)
        : ids(static_cast<std::size_t>(tokens) * profile.top_k), coefficients(ids.size()),
          bank(bank) {
        const int E = profile.experts, K = profile.top_k;
        const int domain = active == K ? E : E / 2;
        std::vector<bool> seen(E);
        for (int t = 0; t < tokens; ++t) {
            int position = 0;
            if (distribution == "independent") {
                position = t;
            } else if (distribution == "skew") {
                position = t < tokens * 3 / 4 ? 0 : t - tokens * 3 / 4 + 1;
            }
            for (int j = 0; j < K; ++j) {
                const auto index    = static_cast<std::size_t>(t) * K + j;
                const bool cpu      = j < active;
                const int e         = cpu ? (start + position * active + j) % domain
                                          : E / 2 + (start + position * K + j) % (E / 2);
                ids[index]          = e;
                coefficients[index] = (1.0F + ((t * 19 + j * 7) % 31)) / 97.0F;
                if (cpu) {
                    ++assignments;
                    if (!seen[e]) {
                        ++unique;
                        seen[e] = true;
                    }
                }
            }
        }
        routes = {Tensor(ids.data(), Device::cpu(), DType::I32, {K, tokens}),
                  Tensor(coefficients.data(), Device::cpu(), DType::FP32, {K, tokens})};
    }
};

// Read-only reference: same native banks, eight-token grouping, column reuse and row owners.
// It removes decode/arithmetic, input/output traffic and public operand validation. This is an
// access-pattern reference, not a DRAM counter or a strict hardware lower bound.
struct ReadMatrix {
    const std::byte* planes[3];
    std::size_t row_bytes[3];
};

struct ReadTask {
    ReadMatrix gate_up, down;
};

using ReadChunks = std::vector<std::vector<ReadTask>>;

ReadChunks read_chunks(const Sample& sample, const Bank& bank, int tokens, int active,
                       int columns) {
    const int E = bank.profile.experts, K = bank.profile.top_k;
    ReadChunks chunks;
    const auto matrix = [](const WeightBankView& view, int expert) {
        ReadMatrix out{};
        int p = 0;
        for (const auto& plane : {view.layout.codes, view.layout.high, view.layout.scales}) {
            out.planes[p]      = static_cast<const std::byte*>(view.plane(plane, expert).data);
            out.row_bytes[p++] = plane.expert_bytes / view.layout.region.rows;
        }
        return out;
    };
    for (int begin = 0; begin < tokens; begin += std::min(8, tokens - begin)) {
        std::vector<int> counts(E);
        std::vector<int> order;
        for (int t = begin; t < begin + std::min(8, tokens - begin); ++t) {
            for (int j = 0; j < active; ++j) {
                const int e = sample.ids[std::size_t(t) * K + j];
                if (counts[e]++ == 0) { order.push_back(e); }
            }
        }
        auto& tasks = chunks.emplace_back();
        for (const int e : order) {
            for (int count = counts[e]; count > 0;) {
                tasks.push_back({matrix(bank.weights.gate_up, e), matrix(bank.weights.down, e)});
                count -= std::min(columns, count >= 4 ? 4 : count >= 2 ? 2 : 1);
            }
        }
    }
    return chunks;
}

struct alignas(64) ReadSink {
    std::uint64_t words[2];
};

struct ReadWork {
    const ReadChunks* chunks;
    ReadSink* sinks;
    int hidden, intermediate;
};

__m128i read_row(const ReadMatrix& matrix, int row, __m128i sum) noexcept {
    for (int plane = 0; plane < 3; ++plane) {
        const auto bytes = matrix.row_bytes[plane];
        if (!bytes) { continue; }
        const auto* base = matrix.planes[plane] + row * bytes;
        for (std::size_t offset = 0; offset < bytes; offset += 16) {
            sum = _mm_xor_si128(sum,
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + offset)));
        }
    }
    return sum;
}

void read_reference(void* argument, const CpuWorkerContext& worker) noexcept {
    const auto& work = *static_cast<ReadWork*>(argument);
    const int H = work.hidden, I = work.intermediate;
    auto sum = _mm_setzero_si128();
    for (std::size_t chunk = 0; chunk < work.chunks->size(); ++chunk) {
        const auto& tasks    = (*work.chunks)[chunk];
        const auto gate_rows = worker.range(tasks.size() * (I / 16));
        for (auto index = gate_rows.begin; index < gate_rows.end; ++index) {
            const auto& matrix = tasks[index / (I / 16)].gate_up;
            const int first    = (index % (I / 16)) * 16;
            for (int row = first; row < first + 16; ++row) {
                sum = read_row(matrix, row, sum);
                sum = read_row(matrix, I + row, sum);
            }
        }
        worker.barrier();
        const auto down_rows = worker.range(H / 16);
        for (auto block = down_rows.begin; block < down_rows.end; ++block) {
            for (const auto& task : tasks) {
                for (int row = block * 16; row < (block + 1) * 16; ++row) {
                    sum = read_row(task.down, row, sum);
                }
            }
        }
        if (chunk + 1 < work.chunks->size()) { worker.barrier(); }
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(work.sinks[worker.rank()].words), sum);
}

double sequential_read_gbps(const std::vector<std::unique_ptr<Bank>>& banks,
                            CpuExecutionView execution) {
    struct Work {
        std::vector<ConstMemorySpan> planes;
        std::vector<ReadSink> sinks;
    } work;

    work.sinks.resize(execution.worker_count());
    std::size_t bytes = 0;
    for (const auto& bank : banks) {
        for (const auto* buffer : {&bank->gu_storage, &bank->down_storage}) {
            work.planes.push_back(buffer->span());
            bytes += buffer->size();
        }
    }
    if (!bytes) { return 0; }
    const auto read = [](void* pointer, const CpuWorkerContext& worker) noexcept {
        auto& work = *static_cast<Work*>(pointer);
        __m128i a = _mm_setzero_si128(), b = a, c = a, d = a;
        for (const auto plane : work.planes) {
            const auto range  = worker.range(plane.bytes / 64);
            const auto* words = static_cast<const __m128i*>(plane.data);
            for (auto i = range.begin; i < range.end; ++i) {
                a = _mm_xor_si128(a, _mm_loadu_si128(words + i * 4));
                b = _mm_xor_si128(b, _mm_loadu_si128(words + i * 4 + 1));
                c = _mm_xor_si128(c, _mm_loadu_si128(words + i * 4 + 2));
                d = _mm_xor_si128(d, _mm_loadu_si128(words + i * 4 + 3));
            }
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(work.sinks[worker.rank()].words),
                         _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d)));
    };
    for (int i = 0; i < 3; ++i) { execution.run(read, &work); }
    std::array<double, 11> samples;
    for (auto& time : samples) {
        const auto start = Clock::now();
        execution.run(read, &work);
        time = std::chrono::duration<double>(Clock::now() - start).count();
    }
    std::sort(samples.begin(), samples.end());
    return bytes / samples[samples.size() / 2] / 1e9;
}

double cpu_us() {
    timespec usage{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &usage)) {
        throw std::runtime_error("process CPU clock failed");
    }
    return usage.tv_sec * 1e6 + usage.tv_nsec / 1000.0;
}

struct Options {
    QType down_format = QType::Q5_G64_FP16;
    std::vector<std::uint32_t> workers{8}, tokens{1}, active{1, 2, 4, 8};
    CpuAffinity cpus;
    std::optional<std::uint32_t> caller_cpu;
    bool smt = false, workers_given = false, read_reference = false, keep_active = false;
    std::string cache = "both", distribution = "same";
    std::uint32_t samples = 200, banks = 0, gap_us = 0;
};

void measure(const Options& options, std::vector<std::unique_ptr<Bank>>& banks,
             CpuExecutor& executor, int tokens, int active, std::string_view cache) {
    const auto profile = make_profile(options.down_format);
    const int H = profile.ffn.hidden, I = profile.ffn.intermediate, E = profile.experts,
              K             = profile.top_k;
    const auto expert_bytes = geometry(profile.ffn.gate_up_type, 2 * I, H).bytes +
                              geometry(profile.ffn.down_type, H, I).bytes;
    const auto kind = active == 0   ? ops::ExpertSubsetKind::None
                      : active == K ? ops::ExpertSubsetKind::All
                                    : ops::ExpertSubsetKind::MaskOnes;
    std::vector<int> mask(E);
    std::fill(mask.begin(), mask.begin() + E / 2, 1);
    const auto capacity = ops::routed_swiglu_workspace_capacity_bytes(profile, kind, 1, tokens);
    HostBuffer scratch(capacity), input(static_cast<std::size_t>(H) * tokens * 2),
        output(static_cast<std::size_t>(H) * tokens * 4);
    auto* words        = reinterpret_cast<std::uint16_t*>(input.data());
    std::uint64_t seed = 72913;
    for (std::size_t i = 0; i < input.size() / 2; ++i) {
        words[i] = 0x3d00 + (random_word(seed) & 127) + ((random_word(seed) & 1) << 15);
    }
    Tensor x(input.span(), DType::BF16, {H, tokens}), out(output.span(), DType::FP32, {H, tokens});
    MemoryArena arena(scratch.span());
    const Sample first(profile, tokens, active, 0, 0, options.distribution);
    const int step        = std::max(first.unique, 1);
    const auto bank_count = cache == "stream" && active ? banks.size() : 1;
    const int domain      = active == K ? E : E / 2;
    const auto cycle =
        cache == "stream" && active ? bank_count * domain / std::gcd(domain, step) : 1;
    std::vector<Sample> corpus;
    corpus.reserve(cycle);
    for (std::size_t s = 0; s < cycle; ++s) {
        corpus.emplace_back(profile, tokens, active, s % bank_count,
                            ((s / bank_count) * step) % domain, options.distribution);
    }
    const ops::CpuSwiGLUExperts no_weights;
    const ops::ExpertSubset subset{kind, Tensor(mask.data(), Device::cpu(), DType::I32, {E})};
    const auto execution = executor.execution_view();
    const auto activity =
        options.keep_active ? std::optional(execution.keep_active()) : std::nullopt;
    const auto call = [&](std::size_t index) {
        const auto& sample  = corpus[index % corpus.size()];
        const auto& weights = active ? banks[sample.bank]->weights : no_weights;
        ops::routed_swiglu(x, profile, weights, sample.routes, subset, out, arena, execution);
    };
    std::size_t warmup = 0;
    const auto warming = Clock::now();
    do {
        call(warmup++);
    } while (warmup < std::max<std::size_t>(20, corpus.size()) ||
             Clock::now() - warming < std::chrono::milliseconds(100));
    std::vector<double> times;
    times.reserve(options.samples);
    if (options.gap_us) { std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us)); }
    for (std::size_t s = 0; s < options.samples; ++s) {
        const auto start = Clock::now();
        call(warmup + s);
        const auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        if (options.gap_us) {
            std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
        }
    }
    // Account for spin/park and the chosen call cadence over an OS-visible interval.
    const double cpu_start    = cpu_us();
    const auto cpu_wall_start = Clock::now();
    std::size_t cpu_calls     = 0;
    do {
        call(warmup + cpu_calls++);
        if (options.gap_us) {
            std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
        }
    } while (Clock::now() - cpu_wall_start < std::chrono::milliseconds(250));
    const auto cpu_per_call = (cpu_us() - cpu_start) / cpu_calls;
    std::sort(times.begin(), times.end());
    const auto median       = times[times.size() / 2];
    const auto p95          = times[(times.size() * 95 + 99) / 100 - 1];
    const auto unique_bytes = static_cast<double>(first.unique) * expert_bytes;
    const auto footprint    = !active             ? 0
                              : cache == "stream" ? bank_count * domain * expert_bytes
                                                  : first.unique * expert_bytes;
    for (std::size_t i = 0; i < output.size() / sizeof(float); ++i) {
        if (!std::isfinite(static_cast<const float*>(out.data)[i])) {
            throw std::runtime_error("nonfinite benchmark output");
        }
    }
    double read_us            = 0;
    std::size_t encoded_reads = 0;
    if (active && options.down_format != QType::NVFP4) {
        const auto features = executor.execution_view().features();
        const bool wide = features.has(CpuFeature::Avx512F) && features.has(CpuFeature::Avx512Bw) &&
                          features.has(CpuFeature::Avx512Dq) && features.has(CpuFeature::Avx512Vl);
        std::vector<ReadChunks> read_corpus;
        for (const auto& sample : corpus) {
            read_corpus.push_back(
                read_chunks(sample, *banks[sample.bank], tokens, active, wide ? 4 : 2));
        }
        for (const auto& chunk : read_corpus.front()) {
            encoded_reads += chunk.size() * expert_bytes;
        }
        if (options.read_reference) {
            std::vector<ReadSink> sinks(executor.execution_view().worker_count());
            const auto read = [&](std::size_t index) {
                ReadWork work{&read_corpus[index % read_corpus.size()], sinks.data(), H, I};
                executor.execution_view().run(read_reference, &work);
            };
            std::size_t warm = 0;
            const auto start = Clock::now();
            do {
                read(warm++);
            } while (warm < read_corpus.size() ||
                     Clock::now() - start < std::chrono::milliseconds(100));
            std::vector<double> measured(options.samples);
            for (std::size_t i = 0; i < measured.size(); ++i) {
                const auto begin = Clock::now();
                read(warm + i);
                measured[i] =
                    std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
                if (options.gap_us) {
                    std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
                }
            }
            std::sort(measured.begin(), measured.end());
            read_us = measured[measured.size() / 2];
        }
    }
    std::cout << format_name(options.down_format) << ',' << tokens << ',' << active << ','
              << first.assignments << ',' << first.unique << ','
              << executor.plan().worker_cpus.size() << ',' << cache << ',' << options.distribution
              << ',' << options.samples << ',' << std::fixed << std::setprecision(3)
              << times.front() << ',' << median << ',' << p95 << ',' << times.back() << ','
              << unique_bytes / (median * 1e-6) / (1ULL << 30) << ','
              << (6.0 * H * I * first.assignments) / (median * 1e3) << ',' << cpu_per_call << ','
              << capacity << ',' << footprint << ',' << corpus.size() << ',' << options.gap_us
              << ',' << cpu_calls << ',';
    if (options.down_format == QType::NVFP4) {
        std::cout << "NA,NA\n";
    } else {
        std::cout << encoded_reads << ',' << read_us << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--help") {
                std::cout
                    << "CPU Q4+Q5/Q6 or NVFP4 routed_swiglu public Op; native Host parent planes.\n"
                       "--format q4-q5|q4-q6|nvfp4 native weight format pair (default q4-q5)\n"
                       "--tokens N[,N]          positive token counts (default 1)\n"
                       "--active N[,N]          CPU experts per token, 0..top_k (default 1,2,4,8)\n"
                       "--workers N[,N]         automatic worker limits; 0=all (default 8)\n"
                       "--cpus ID,ID,...        exact ordered CPU list, exclusive with "
                       "--workers/--smt\n"
                       "--caller-cpu ID         bind the submitting thread after topology "
                       "discovery\n"
                       "--smt                   append SMT siblings after physical cores\n"
                       "--cache repeat|stream|both  fixed subset or rotating native banks (default "
                       "both)\n"
                       "--distribution same|independent|skew  overlap within a call (default "
                       "same)\n"
                       "--samples N             measured calls per point (default 200)\n"
                       "--banks N               streaming banks (default auto: selected footprint "
                       ">=4x reported LLC)\n"
                       "--gap-us N              outside-call gap, included in process CPU "
                       "accounting (default 0)\n"
                       "--keep-active           keep workers polling between calls\n"
                       "--read-reference        sequential bandwidth; also integer-only "
                       "access-pattern timing\n";
                return 0;
            }
            if (arg == "--read-reference") {
                options.read_reference = true;
                continue;
            }
            if (arg == "--smt") {
                options.smt = true;
                continue;
            }
            if (arg == "--keep-active") {
                options.keep_active = true;
                continue;
            }
            if (++i == argc) { throw std::invalid_argument("missing option value"); }
            const std::string_view value(argv[i]);
            if (arg == "--format") {
                if (value != "q4-q5" && value != "q4-q6" && value != "nvfp4") {
                    throw std::invalid_argument("unsupported weight format pair");
                }
                options.down_format = value == "nvfp4"   ? QType::NVFP4
                                      : value == "q4-q6" ? QType::Q6_G64_FP16
                                                         : QType::Q5_G64_FP16;
            } else if (arg == "--tokens") {
                options.tokens = list(value);
            } else if (arg == "--active") {
                options.active = list(value);
            } else if (arg == "--workers") {
                options.workers       = list(value);
                options.workers_given = true;
            } else if (arg == "--cpus") {
                options.cpus = list(value);
            } else if (arg == "--caller-cpu") {
                options.caller_cpu = number(value);
            } else if (arg == "--cache") {
                options.cache = value;
            } else if (arg == "--distribution") {
                options.distribution = value;
            } else if (arg == "--samples") {
                options.samples = number(value);
            } else if (arg == "--banks") {
                options.banks = number(value);
            } else if (arg == "--gap-us") {
                options.gap_us = number(value);
            } else {
                throw std::invalid_argument("unknown option");
            }
        }
        if (!options.samples ||
            (options.cache != "repeat" && options.cache != "stream" && options.cache != "both") ||
            (options.distribution != "same" && options.distribution != "independent" &&
             options.distribution != "skew")) {
            throw std::invalid_argument("invalid sample/cache/distribution option");
        }
        for (auto t : options.tokens) {
            if (!t || t > 0x7fffffffU) { throw std::invalid_argument("invalid token extent"); }
        }
        for (auto active : options.active) {
            if (active > unsigned(make_profile(options.down_format).top_k)) {
                throw std::invalid_argument("active exceeds top_k");
            }
        }
        if (!options.cpus.empty() && (options.workers_given || options.smt)) {
            throw std::invalid_argument(
                "explicit CPUs cannot be combined with automatic team options");
        }
        CpuTopology topology;
        if (options.caller_cpu) {
            bind_current_thread(topology, std::span(&*options.caller_cpu, 1));
        }
        const auto identity     = current_cpu_identity();
        std::size_t llc_bytes   = 0;
        std::uint32_t llc_level = 0;
        for (const auto& c : topology.info().caches) { llc_level = std::max(llc_level, c.level); }
        for (const auto& c : topology.info().caches) {
            if (c.level == llc_level) { llc_bytes += c.bytes; }
        }
        const auto profile = make_profile(options.down_format);
        const int H = profile.ffn.hidden, I = profile.ffn.intermediate, E = profile.experts,
                  K             = profile.top_k;
        const auto expert_bytes = geometry(profile.ffn.gate_up_type, 2 * I, H).bytes +
                                  geometry(profile.ffn.down_type, H, I).bytes;
        const auto per_bank_working_set = (E / 2) * expert_bytes;
        const auto count =
            options.cache == "repeat" ? 1
            : options.banks           ? options.banks
                            : std::max<std::size_t>(2, (4 * llc_bytes + per_bank_working_set - 1) /
                                                           per_bank_working_set);
        std::vector<std::unique_ptr<Bank>> banks;
        if (std::any_of(options.active.begin(), options.active.end(),
                        [](auto n) { return n != 0; })) {
            banks.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                banks.push_back(std::make_unique<Bank>(options.down_format, 74967 + 81217 * i));
            }
        }
        std::cout << "# vendor=" << identity.vendor.data() << " family=" << identity.family
                  << " model=" << identity.model << " virtualized=" << identity.virtualized
                  << " reported_llc_bytes=" << llc_bytes << " banks=" << banks.size()
                  << " format=" << format_name(options.down_format)
                  << " expert_bytes=" << expert_bytes << " bank_bytes=" << E * expert_bytes
                  << " keep_active=" << options.keep_active
                  << " plane_layout=native_parent_banks cpu_clock=process_cputime compiler="
                  << __VERSION__ << '\n';
        std::cout << "# caller_cpu_ids=";
        for (auto cpu : current_thread_affinity(topology)) { std::cout << cpu << ' '; }
        std::cout << "\nformat,T,cpu_experts_per_token,assignments,unique_experts,workers,cache,"
                     "distribution,samples,min_us,median_us,p95_us,max_us,unique_weight_gib_s,"
                     "effective_gflops,process_cpu_us_per_call,workspace_bytes,reuse_footprint_"
                     "bytes,corpus_calls,gap_us,cpu_calls,encoded_read_bytes,read_reference_us\n";
        for (auto workers : options.workers) {
            const auto count =
                workers ? workers : cpu_worker_order(topology.info(), options.smt).size();
            const auto plan = options.cpus.empty()
                                  ? plan_cpu_team(topology.info(), count, options.smt)
                                  : CpuTeamPlan{options.cpus};
            CpuExecutor executor(topology, plan);
            std::cout << "# workers=" << executor.plan().worker_cpus.size() << " cpu_ids=";
            for (const auto cpu : executor.plan().worker_cpus) { std::cout << cpu << ' '; }
            std::cout << " common_features=0x" << std::hex
                      << executor.execution_view().features().bits << std::dec
                      << " automatic_smt=" << options.smt << '\n';
            if (options.read_reference) {
                std::cout << "# sequential_read_gb_s="
                          << sequential_read_gbps(banks, executor.execution_view())
                          << " read_corpus_bytes=" << banks.size() * E * expert_bytes << '\n';
            }
            for (auto tokens : options.tokens) {
                for (auto active : options.active) {
                    if (options.cache != "stream") {
                        measure(options, banks, executor, tokens, active, "repeat");
                    }
                    if (options.cache != "repeat") {
                        measure(options, banks, executor, tokens, active, "stream");
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
