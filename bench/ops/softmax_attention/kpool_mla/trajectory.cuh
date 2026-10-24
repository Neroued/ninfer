#pragma once

// The benchmark owns frontier policy, graph definitions, and metadata lifetimes.
// Ops receive only public views and a valid capacity envelope.
struct TraceStep {
    std::vector<int> frontiers, valid;
    bool reset = false;
};

static Json trajectory(CudaExecutionView cuda, const Json& c, int repetitions) {
    const int T = c.at("T"), B = c.at("B"), steps = c.value("steps", 512);
    if (steps < 1 || repetitions < 1 || (c.value("kind", "") == "churn" && steps < 4))
        throw std::invalid_argument(
            "trajectory requires positive repetitions/steps; churn needs >=4 steps");
    auto shape         = shape_of(c);
    const auto initial = c.at("frontiers").get<std::vector<int>>();
    std::vector<TraceStep> sequence;
    std::set<int> bounds;
    shape.reserve_ends.assign(B, 0);
    auto frontier = initial;
    for (int step = 0; step < steps; ++step) {
        std::vector<int> n(B, T);
        bool reset = false;
        if (c.value("kind", "") == "churn") {
            const int quarter = step / (steps / 4);
            if (quarter == 1)
                for (int b = 3; b < B; ++b) n[b] = 0;
            if (quarter == 2) std::fill(n.begin(), n.end(), 0);
            if (step == 3 * (steps / 4)) {
                reset = true;
                for (int b = 0; b < B; ++b) frontier[b] = b % 2 ? 262144 : 8192;
            }
        }
        sequence.push_back({frontier, n, reset});
        int max_end = 0;
        for (int b = 0; b < B; ++b) {
            frontier[b] += n[b];
            if (n[b]) max_end = std::max(max_end, frontier[b]);
            shape.reserve_ends[b] = std::max(shape.reserve_ends[b], frontier[b]);
        }
        bounds.insert(bucket(max_end));
    }
    shape.envelope         = *bounds.rbegin();
    Json budget            = c;
    budget["reserve_ends"] = shape.reserve_ends;
    budget["envelope_max"] = shape.envelope;
    Json out               = {{"case", c}, {"admission", admission(cuda, budget)}};
    if (!out["admission"]["admitted"].get<bool>()) {
        out["status"] = "not_admitted";
        return out;
    }
    auto setup = Clock::now();
    app::Fixture f(cuda, T, B, shape, c.value("prepare", "update") == "record");
    f.after = f.before;
    f.reserve(cuda, {bounds.begin(), bounds.end()});
    // One pinned, immutable upload frame per step. Reuse only after the final event.
    const std::size_t x_words     = std::size_t(T) * B * 4096;
    const std::size_t meta_words  = T * B + 4 * B;
    const std::size_t frame_bytes = x_words * 2 + meta_words * 4;
    void* pinned                  = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned, frame_bytes * steps));

    struct PinnedGuard {
        void* p;

        ~PinnedGuard() { cudaFreeHost(p); }
    } guard{pinned};

    for (int step = 0; step < steps; ++step) {
        auto* x =
            reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(pinned) + step * frame_bytes);
        auto* m       = reinterpret_cast<int*>(x + x_words);
        const auto& s = sequence[step];
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                m[b * T + t] = t < s.valid[b] ? s.frontiers[b] + t : -1;
                for (int d = 0; d < 4096; ++d)
                    x[(b * T + t) * 4096 + d] = kp::sample(
                        d + 4096U * (s.frontiers[b] + t) + kp::hash(b + 20261024), 1.F / 256);
            }
            m[T * B + b]         = s.valid[b];
            m[T * B + B + b]     = s.valid[b] ? B - 1 - b : -1;
            m[T * B + 2 * B + b] = s.frontiers[b];
            m[T * B + 3 * B + b] = s.valid[b];
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::map<int, DecodeGraphDefinition> definitions;
    for (int u : bounds) {
        f.index.envelope = {std::uint32_t(u)};
        definitions[u].capture(cuda.stream(), [&] {
            f.launch(cuda);
            if (f.record) f.extract(cuda);
        });
    }
    DecodeGraphDefinition parent;
    DecodeGraphChild child;
    parent.capture(cuda.stream(),
                   [&] { child = definitions.begin()->second.append_to(cuda.stream()); });
    DecodeGraphExecutable graph;
    graph.instantiate(parent);
    // Validate every child update and upload before entering the timed trajectory.
    for (auto& [u, definition] : definitions) graph.update_child(child, definition);
    graph.upload(cuda.stream());
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    out["setup_us"]              = us(setup, Clock::now());
    out["bounds"]                = bounds;
    out["workspace_bytes"]       = f.index.workspace_bytes;
    out["upload_bytes_per_step"] = frame_bytes;
    std::vector<cudaEvent_t> starts(steps), stops(steps);
    for (int k = 0; k < steps; ++k) {
        CUDA_CHECK(cudaEventCreate(&starts[k]));
        CUDA_CHECK(cudaEventCreate(&stops[k]));
    }
    for (int rep = -1; rep < repetitions; ++rep) {
        f.restore(cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        int current = -1;

        struct Sample {
            int end, u, active;
            bool changed;
            double select, update, launch;
        };

        std::vector<Sample> samples(steps);
        const auto begin = Clock::now();
        for (int k = 0; k < steps; ++k) {
            const auto& s = sequence[k];
            auto* x       = static_cast<std::byte*>(pinned) + k * frame_bytes;
            auto* m       = x + x_words * 2;
            auto a        = Clock::now();
            int end = 0, active = 0;
            for (int b = 0; b < B; ++b) {
                if (s.valid[b]) end = std::max(end, s.frontiers[b] + s.valid[b]);
                active += s.valid[b];
            }
            int u         = bucket(end);
            bool changed  = u != current;
            auto selected = Clock::now();
            if (changed) {
                graph.update_child(child, definitions.at(u));
                current = u;
            }
            auto updated = Clock::now();
            if (s.reset) f.restore(cuda);
            CUDA_CHECK(cudaMemcpyAsync(f.x.view.data, x, x_words * 2, cudaMemcpyHostToDevice,
                                       cuda.stream()));
            CUDA_CHECK(cudaMemcpyAsync(f.index.positions.view.data, m, T * B * 4,
                                       cudaMemcpyHostToDevice, cuda.stream()));
            CUDA_CHECK(cudaMemcpyAsync(f.index.valid.view.data, m + T * B * 4, B * 4,
                                       cudaMemcpyHostToDevice, cuda.stream()));
            CUDA_CHECK(cudaMemcpyAsync(f.index.selectors.view.data, m + (T * B + B) * 4, B * 4,
                                       cudaMemcpyHostToDevice, cuda.stream()));
            if (f.record) {
                CUDA_CHECK(cudaMemcpyAsync(f.bases.view.data, m + (T * B + 2 * B) * 4, B * 4,
                                           cudaMemcpyHostToDevice, cuda.stream()));
                CUDA_CHECK(cudaMemcpyAsync(f.accepted.view.data, m + (T * B + 3 * B) * 4, B * 4,
                                           cudaMemcpyHostToDevice, cuda.stream()));
            }
            CUDA_CHECK(cudaEventRecord(starts[k], cuda.stream()));
            auto launch = Clock::now();
            graph.launch(cuda.stream());
            auto launched = Clock::now();
            CUDA_CHECK(cudaEventRecord(stops[k], cuda.stream()));
            samples[k] = {end,
                          u,
                          active,
                          changed,
                          us(a, selected),
                          us(selected, updated),
                          us(launch, launched)};
        }
        CUDA_CHECK(cudaEventSynchronize(stops.back()));
        const auto end = Clock::now();
        if (rep >= 0) {
            Json trace = Json::array();
            for (int k = 0; k < steps; ++k) {
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, starts[k], stops[k]));
                const auto& h = samples[k];
                const auto& s = sequence[k];
                trace.push_back({{"step", k},
                                 {"end_max", h.end},
                                 {"envelope", h.u},
                                 {"active_columns", h.active},
                                 {"frontiers", s.frontiers},
                                 {"valid", s.valid},
                                 {"updated", h.changed},
                                 {"selection_us", h.select},
                                 {"update_us", h.update},
                                 {"launch_us", h.launch},
                                 {"gpu_interval_us", ms * 1000.}});
            }
            out["runs"].push_back({{"wall_us", us(begin, end)}, {"timeline", trace}});
        }
    }
    for (int k = 0; k < steps; ++k) {
        cudaEventDestroy(starts[k]);
        cudaEventDestroy(stops[k]);
    }
    if (!f.index.scratch_guards() || !f.result.guards())
        throw std::runtime_error("trajectory guard");
    out["status"] = "measured";
    return out;
}
