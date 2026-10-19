#include "runtime/engine/worker.h"

#include <iostream>

int main() {
    try {
        ninfer::runtime::run_worker(ninfer::distributed::read_worker_bootstrap());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer-worker: " << error.what() << '\n';
        return 1;
    }
}
