#pragma once

#include "ops/op_measurement.cuh"

#include <optional>

namespace ninfer::bench::kpool {

class JobResources {
public:
    DeviceContext& device() {
        if (!initialized_) {
            measurement::hardware();
            initialized_ = true;
        }
        return device_context();
    }

    L2FlushBuffer& flush(std::size_t bytes) {
        if (!flush_ || flush_->size() != bytes) flush_.emplace(bytes);
        return *flush_;
    }

private:
    bool initialized_ = false;
    std::optional<L2FlushBuffer> flush_;
};

template <class Run>
int run_jobs(int argc, char** argv, Run&& run) {
    try {
        JobResources resources;
        if (argc < 2 || std::string_view(argv[1]) != "--jobs-file") {
            run(argc, argv, resources);
            return 0;
        }
        if (argc != 3) throw std::invalid_argument("usage: --jobs-file PATH");
        std::ifstream file(argv[2]);
        if (!file) throw std::runtime_error("cannot open jobs file: " + std::string(argv[2]));
        std::string line;
        std::size_t number = 0, completed = 0;
        while (std::getline(file, line)) {
            ++number;
            if (line.ends_with('\r')) line.pop_back();
            if (line.empty()) continue;
            try {
                std::vector<char*> args{argv[0], line.data()};
                for (char& c : line) {
                    if (c == '\t') {
                        c = '\0';
                        args.push_back(&c + 1);
                    }
                }
                run(static_cast<int>(args.size()), args.data(), resources);
            } catch (const std::exception& error) {
                throw std::runtime_error(std::string(argv[2]) + " line " + std::to_string(number) +
                                         ": " + error.what());
            }
            if (++completed == 1 || completed % 10 == 0)
                std::cerr << "[RUN] " << argv[2] << " " << completed << " jobs complete\n";
        }
        if (!file.eof() || !completed)
            throw std::runtime_error("cannot read jobs from " + std::string(argv[2]));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

inline constexpr std::string_view packed_columns = ",graph_nodes,graph_kernels,ragged_segments,"
                                                   "ragged_histories";

inline std::string graph_csv(const measurement::Result& result) {
    return ',' + std::to_string(result.nodes) + ',' + std::to_string(result.kernels);
}

} // namespace ninfer::bench::kpool
