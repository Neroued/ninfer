#pragma once

#include "core/device.h"
#include "runtime/executor/contract.h"

#include <memory>
#include <utility>

namespace ninfer::runtime {

template <class Types>
struct RankModelInstance {
    std::unique_ptr<typename Types::ModelShard> model;
    const typename Types::Parameters parameters;
    std::unique_ptr<typename Types::RankProgram> program;

    RankModelInstance(std::unique_ptr<typename Types::ModelShard> model,
                      std::shared_ptr<const typename Types::Profile> profile)
        : model(std::move(model)), parameters(*this->model, std::move(profile)) {}
};

// The device object remains at the same address from startup through shutdown. Programs may
// borrow its streams, and every allocation owner is destroyed before the device.
template <class Types>
class RankWorker {
public:
    using Protocol = typename Types::ExecutionTypes;

    RankWorker(std::unique_ptr<DeviceContext> device,
               std::unique_ptr<RankModelInstance<Types>> instance)
        : device_(std::move(device)), instance_(std::move(instance)) {}

    ~RankWorker() {
        device_->bind_to_current_thread_noexcept();
        instance_.reset();
    }

    void bind_to_current_thread() { device_->bind_to_current_thread(); }

    [[nodiscard]] typename Types::RankProgram& program() noexcept { return *instance_->program; }

    [[nodiscard]] const typename Types::RankProgram& program() const noexcept {
        return *instance_->program;
    }

private:
    std::unique_ptr<DeviceContext> device_;
    std::unique_ptr<RankModelInstance<Types>> instance_;
};

} // namespace ninfer::runtime
