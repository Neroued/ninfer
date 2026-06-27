#include "qus/core/device.h"
#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_q5090_engine_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --profile model-bind --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    const std::filesystem::path fixture = make_fixture();
    qus::EngineOptions options;
    options.max_ctx    = 4;
    options.work_bytes = 64ULL * 1024ULL * 1024ULL;
    qus::Engine engine(options);
    engine.load(fixture.string());
    if (!engine.loaded()) {
        std::cerr << "engine did not report loaded\n";
        return 1;
    }
    if (engine.position() != 0) {
        std::cerr << "engine initial position mismatch: " << engine.position() << '\n';
        return 1;
    }
    if (engine.max_context() != 4) {
        std::cerr << "engine max context mismatch\n";
        return 1;
    }
    return 0;
}
