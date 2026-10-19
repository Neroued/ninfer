#pragma once

#include "distributed/launch.h"

namespace ninfer::runtime {
void run_worker(distributed::WorkerBootstrap);
} // namespace ninfer::runtime
