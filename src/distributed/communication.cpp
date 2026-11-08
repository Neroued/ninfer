#include "distributed/communication/implementation.h"
#include "core/platform.h"

#if NINFER_TARGET_SPARK
#    include "distributed/communication/native/rdma.h"
#endif

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ninfer::distributed {
namespace {

void normalize(GroupRequirements& requirements, const GroupDescription& group, std::uint32_t rank) {
    for (const auto& c : requirements.collectives) {
        if (!c.min_count || !c.step || c.max_count < c.min_count ||
            (c.max_count - c.min_count) % c.step)
            throw std::invalid_argument("communication count range is invalid");
        const auto width = dtype_size(c.dtype);
        if (!width || c.max_count > SIZE_MAX / width)
            throw std::overflow_error("communication payload size overflow");
        switch (c.operation) {
        case Collective::AllReduce:
            break;
        case Collective::Broadcast:
            if (c.root >= group.members.size())
                throw std::invalid_argument("communication root is outside the group");
            break;
        case Collective::AllGather:
        case Collective::ReduceScatter:
            if (c.max_count * width > SIZE_MAX / group.members.size())
                throw std::overflow_error("communication aggregate size overflow");
            break;
        default:
            throw std::invalid_argument("invalid collective operation");
        }
        if (c.reduction != Reduction::Sum && c.reduction != Reduction::Max)
            throw std::invalid_argument("invalid collective reduction");
    }
    const auto key = [](const CollectiveRequirement& c) {
        return std::tie(c.operation, c.dtype, c.reduction, c.root, c.min_count, c.max_count,
                        c.step);
    };
    std::sort(requirements.collectives.begin(), requirements.collectives.end(),
              [&](const auto& a, const auto& b) { return key(a) < key(b); });
    std::erase_if(requirements.peers,
                  [](const auto& p) { return !p.send_bytes && !p.receive_bytes; });
    std::sort(requirements.peers.begin(), requirements.peers.end(),
              [](const auto& a, const auto& b) { return a.peer < b.peer; });
    std::uint32_t previous = UINT32_MAX;
    for (const auto& p : requirements.peers) {
        if (p.peer >= group.members.size() || p.peer == rank || p.peer == previous)
            throw std::invalid_argument("invalid or duplicate communication peer");
        previous = p.peer;
    }
}

const GroupPreparation& prepared_group(const CommunicationPreparation& rank, GroupId id) {
    const auto found = std::find_if(rank.groups.begin(), rank.groups.end(),
                                    [&](const auto& g) { return g.requirements.group == id; });
    if (found == rank.groups.end())
        throw std::invalid_argument("communication preparation omitted a group");
    return *found;
}

} // namespace

bool CollectiveRequirement::contains(const CollectiveCall& call) const noexcept {
    return operation == call.operation && dtype == call.dtype && reduction == call.reduction &&
           root == call.root && call.count >= min_count && call.count <= max_count &&
           (call.count - min_count) % step == 0;
}

void validate_communication_preparation(const ExecutionTopology& topology,
                                        std::span<const CommunicationPreparation> preparations) {
    if (preparations.size() != topology.ranks.size())
        throw std::invalid_argument("communication preparation rank count differs");
    std::vector<const CommunicationPreparation*> ranks(topology.ranks.size());
    for (const auto& p : preparations) {
        if (p.rank >= ranks.size() || ranks[p.rank])
            throw std::invalid_argument("invalid or duplicate communication preparation rank");
        ranks[p.rank] = &p;
        if (p.policy > CommunicationPolicy::Auto || p.policy != preparations.front().policy)
            throw std::invalid_argument("communication policies differ across ranks");
        std::set<GroupId> seen;
        for (const auto& g : p.groups) {
            const auto& description = topology.group(g.requirements.group);
            (void)description.index(p.rank);
            if (!seen.insert(description.id).second)
                throw std::invalid_argument("duplicate prepared communication group");
        }
    }
    for (const auto& group : topology.groups) {
        const auto& first = prepared_group(*ranks.at(group.members.front()), group.id);
        for (const auto rank : group.members) {
            const auto& current = prepared_group(*ranks.at(rank), group.id);
            if (current.requirements.collectives != first.requirements.collectives)
                throw std::invalid_argument("group " + std::to_string(group.id) +
                                            ": collective requirements differ across ranks");
            if (group.members.size() > 1 &&
                preparations.front().policy == CommunicationPolicy::Native &&
                (!current.requirements.collectives.empty() ||
                 !current.requirements.peers.empty()) &&
                std::none_of(
                    current.connections.begin(), current.connections.end(),
                    [](const auto& c) { return c.backend == CommunicationBackend::Native; }))
                throw std::invalid_argument("native communication is unavailable on rank " +
                                            std::to_string(rank));
            for (const auto& peer : current.requirements.peers) {
                const auto& remote =
                    prepared_group(*ranks.at(group.members.at(peer.peer)), group.id).requirements;
                const auto reciprocal =
                    std::find_if(remote.peers.begin(), remote.peers.end(),
                                 [&](const auto& p) { return p.peer == group.index(rank); });
                if (reciprocal == remote.peers.end() ||
                    bool(peer.send_bytes) != bool(reciprocal->receive_bytes) ||
                    bool(peer.receive_bytes) != bool(reciprocal->send_bytes))
                    throw std::invalid_argument("peer communication requirements are unmatched");
            }
        }
    }
}

struct Group::Impl {
    enum class State { Created, Prepared, Selected, Connected, Ready, Closed };

    struct Route {
        std::unique_ptr<detail::Backend> implementation;
        CommunicationBackendStatistics statistics;
    };

    void ready(CudaExecutionView execution) const {
        if (execution.device() != device)
            throw std::invalid_argument("communication execution device differs");
        if (stop->load(std::memory_order_acquire)) owner->check();
        if (state != State::Ready) throw std::logic_error("communication group is not ready");
    }

    void observe(Route& route, std::size_t kind, std::size_t bytes) {
        auto* work = &route.statistics.eager;
        if (capture)
            work = &std::find_if(capture->begin(), capture->end(), [&](const auto& value) {
                        return value.backend == route.statistics.backend;
                    })->work;
        ++work->calls[kind];
        work->payload_bytes[kind] += bytes;
    }

    Route& select(const CollectiveCall& call) {
        for (auto& route : routes)
            if (route.implementation->supports(call)) return route;
        throw std::invalid_argument("no prepared communication backend supports this collective");
    }

    GroupDescription description;
    CommunicationContext* owner;
    std::uint32_t rank;
    Device device;
    std::shared_ptr<std::atomic<bool>> stop;
    State state = State::Created;
    GroupRequirements requirements;
    std::vector<Route> routes;
    CommunicationGraphWork* capture = nullptr;
};

Group::Group(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Group::~Group() = default;

GroupId Group::id() const noexcept { return impl_->description.id; }

std::uint32_t Group::rank() const noexcept { return impl_->rank; }

std::uint32_t Group::size() const noexcept { return impl_->description.members.size(); }

Device Group::device() const noexcept { return impl_->device; }

CommunicationStatistics Group::statistics() const {
    CommunicationStatistics out{id(), size(), {}};
    for (const auto& route : impl_->routes) out.backends.push_back(route.statistics);
    return out;
}

void Group::enqueue(const CollectiveCall& call, const void* source, void* destination,
                    CudaExecutionView execution) const {
    auto& p = *impl_;
    p.ready(execution);
    if (!source || !destination ||
        std::none_of(p.requirements.collectives.begin(), p.requirements.collectives.end(),
                     [&](const auto& r) { return r.contains(call); }))
        throw std::invalid_argument(
            "collective is outside the prepared communication requirements");
    if (size() == 1) {
        if (source != destination)
            CUDA_CHECK(cudaMemcpyAsync(destination, source, call.count * dtype_size(call.dtype),
                                       cudaMemcpyDeviceToDevice, execution.stream()));
        return;
    }
    auto& route = p.select(call);
    route.implementation->enqueue(call, source, destination, execution);
    p.observe(route, static_cast<std::size_t>(call.operation), call.count * dtype_size(call.dtype));
}

void Group::exchange(std::span<const PeerTransfer> transfers, CudaExecutionView execution) const {
    auto& p = *impl_;
    p.ready(execution);
    if (transfers.empty()) return;
    for (const auto& item : transfers) {
        const auto allowed = std::find_if(p.requirements.peers.begin(), p.requirements.peers.end(),
                                          [&](const auto& r) { return r.peer == item.peer; });
        if (std::count_if(transfers.begin(), transfers.end(),
                          [&](const auto& other) { return other.peer == item.peer; }) != 1)
            throw std::invalid_argument("peer exchange requires one entry per peer");
        if (allowed == p.requirements.peers.end() || item.send.bytes > allowed->send_bytes ||
            item.receive.bytes > allowed->receive_bytes)
            throw std::invalid_argument("peer transfer exceeds prepared communication capacity");
        for (auto memory : {item.send, static_cast<ConstMemorySpan>(item.receive)})
            if (memory.bytes && (!memory.data || memory.device != p.device))
                throw std::invalid_argument("peer communication buffer is not on the group device");
    }
    const auto overlaps = [](ConstMemorySpan a, ConstMemorySpan b) {
        if (!a.bytes || !b.bytes) return false;
        const auto x = reinterpret_cast<std::uintptr_t>(a.data);
        const auto y = reinterpret_cast<std::uintptr_t>(b.data);
        return x <= y ? y - x < a.bytes : x - y < b.bytes;
    };
    for (const auto& receiver : transfers)
        for (const auto& other : transfers)
            if (overlaps(receiver.receive, other.send) ||
                (&receiver != &other && overlaps(receiver.receive, other.receive)))
                throw std::invalid_argument(
                    "peer receive buffers must be disjoint from sends and other receives");
    auto found = std::find_if(p.routes.begin(), p.routes.end(), [&](const auto& r) {
        return r.implementation->supports_peers(p.requirements.peers);
    });
    if (found == p.routes.end()) throw std::logic_error("peer transfer has no prepared backend");
    auto& route = *found;
    route.implementation->exchange(transfers, execution);
    for (const auto& item : transfers) {
        if (item.send.bytes) p.observe(route, 4, item.send.bytes);
        if (item.receive.bytes) p.observe(route, 5, item.receive.bytes);
    }
}

void Group::record_graph(CommunicationGraphWork& into, const std::function<void()>& body) const {
    if (impl_->capture) throw std::logic_error("communication capture is already active");
    into.clear();
    for (const auto& route : impl_->routes) into.push_back({route.statistics.backend, {}});

    struct Recording {
        Impl& owner;

        ~Recording() { owner.capture = nullptr; }
    } recording{*impl_};

    impl_->capture = &into;
    body();
}

void Group::replayed(const CommunicationGraphWork& work) const noexcept {
    for (auto& route : impl_->routes) {
        route.implementation->wake();
        for (const auto& value : work)
            if (value.backend == route.statistics.backend) route.statistics.graph.add(value.work);
    }
}

CommunicationContext::CommunicationContext(const DeviceContext& device,
                                           const ExecutionTopology& topology, RankId rank,
                                           std::shared_ptr<std::atomic<bool>> stop)
    : device_(device), topology_(topology), rank_(rank),
      stop_(stop ? std::move(stop) : std::make_shared<std::atomic<bool>>(false)) {
    for (const auto& description : topology.groups) {
        if (std::find(description.members.begin(), description.members.end(), rank) ==
            description.members.end())
            continue;
        auto impl         = std::make_unique<Group::Impl>();
        impl->description = description;
        impl->owner       = this;
        impl->rank        = description.index(rank);
        impl->device      = device.device();
        impl->stop        = stop_;
        groups_.push_back(std::unique_ptr<Group>(new Group(std::move(impl))));
    }
}

CommunicationContext::~CommunicationContext() { abort(); }

CommunicationPreparation CommunicationContext::prepare(std::span<const GroupRequirements> needs,
                                                       CommunicationPolicy policy) {
    check();
    policy_ = policy;
    std::set<GroupId> seen;
    for (const auto& requirement : needs) {
        (void)group(requirement.group);
        if (!seen.insert(requirement.group).second)
            throw std::invalid_argument("duplicate communication group requirements");
    }
    const auto active_groups = std::count_if(needs.begin(), needs.end(), [](const auto& r) {
        return !r.collectives.empty() || !r.peers.empty();
    });
    const auto lane_budget =
        device_.multiprocessor_count() / std::max<std::size_t>(1, active_groups);
    CommunicationPreparation result{rank_, {}, policy};
    for (auto& group : groups_) {
        auto& p = *group->impl_;
        if (p.state != Group::Impl::State::Created)
            throw std::logic_error("communication resources were already prepared");
        const auto found = std::find_if(needs.begin(), needs.end(),
                                        [&](const auto& r) { return r.group == group->id(); });
        p.requirements   = found == needs.end() ? GroupRequirements{group->id(), {}, {}} : *found;
        normalize(p.requirements, p.description, p.rank);
        auto& prepared = result.groups.emplace_back(GroupPreparation{p.requirements, {}});
        if (group->size() > 1) {
#if NINFER_TARGET_SPARK
            if (policy != CommunicationPolicy::Nccl &&
                (!p.requirements.collectives.empty() || !p.requirements.peers.empty())) {
                auto backend = detail::make_native_backend(device_, p.description, p.rank,
                                                           lane_budget, stop_, native_progress_);
                if (backend) {
                    auto connection = backend->prepare(p.requirements);
                    if (!connection.data.empty()) {
                        prepared.connections.push_back(std::move(connection));
                        p.routes.push_back(
                            {std::move(backend), {.backend = CommunicationBackend::Native}});
                    }
                }
            }
#endif
            if (policy != CommunicationPolicy::Native) {
                auto backend = detail::make_nccl_backend(p.description, p.rank, stop_);
                prepared.connections.push_back(backend->prepare(p.requirements));
                p.routes.push_back({std::move(backend), {.backend = CommunicationBackend::Nccl}});
            }
        }
        p.state = Group::Impl::State::Prepared;
    }
    return result;
}

CommunicationPreparation
CommunicationContext::select(std::span<const CommunicationPreparation> peers,
                             std::chrono::steady_clock::time_point deadline) {
    check();
    validate_communication_preparation(topology_, peers);
    CommunicationPreparation selected{rank_, {}, policy_};
    for (auto& group : groups_) {
        auto& p = *group->impl_;
        if (p.state != Group::Impl::State::Prepared)
            throw std::logic_error("communication connection precedes local preparation");
        std::erase_if(p.routes, [&](const auto& route) {
            return std::any_of(
                p.description.members.begin(), p.description.members.end(), [&](RankId rank) {
                    const auto peer         = std::find_if(peers.begin(), peers.end(),
                                                           [&](const auto& r) { return r.rank == rank; });
                    const auto& connections = prepared_group(*peer, group->id()).connections;
                    return std::none_of(connections.begin(), connections.end(), [&](const auto& c) {
                        return c.backend == route.statistics.backend;
                    });
                });
        });
        for (auto& route : p.routes) route.implementation->resolve(peers, deadline);
        std::erase_if(p.routes, [&](const auto& route) {
            if (route.statistics.backend != CommunicationBackend::Native) return false;
            if (!p.requirements.peers.empty() &&
                route.implementation->supports_peers(p.requirements.peers))
                return false;
            return std::none_of(p.requirements.collectives.begin(),
                                p.requirements.collectives.end(), [&](const auto& r) {
                                    return route.implementation->supports(
                                        {r.operation, r.dtype, r.min_count, r.reduction, r.root});
                                });
        });
        std::vector<GroupRequirements> work(p.routes.size(), {group->id(), {}, {}});
        for (const auto& r : p.requirements.collectives) {
            if (group->size() == 1) break;
            std::size_t first = r.min_count;
            for (std::size_t i = 0; i < p.routes.size() && first <= r.max_count; ++i) {
                if (policy_ == CommunicationPolicy::Native &&
                    p.routes[i].statistics.backend != CommunicationBackend::Native)
                    continue;
                auto& backend = *p.routes[i].implementation;
                if (!backend.supports({r.operation, r.dtype, first, r.reduction, r.root})) continue;
                auto lo = (first - r.min_count) / r.step, hi = (r.max_count - r.min_count) / r.step;
                while (lo < hi) {
                    const auto mid = lo + (hi - lo + 1) / 2;
                    if (backend.supports({r.operation, r.dtype, r.min_count + mid * r.step,
                                          r.reduction, r.root}))
                        lo = mid;
                    else
                        hi = mid - 1;
                }
                auto range      = r;
                range.min_count = first;
                range.max_count = r.min_count + lo * r.step;
                work[i].collectives.push_back(range);
                if (range.max_count == r.max_count) {
                    first = 0;
                    break;
                }
                first = range.max_count + r.step;
            }
            if (first)
                throw std::invalid_argument(
                    "prepared backends do not cover the collective count range");
        }
        const bool peer_work = std::any_of(
            p.description.members.begin(), p.description.members.end(), [&](RankId rank) {
                const auto peer = std::find_if(peers.begin(), peers.end(),
                                               [&](const auto& v) { return v.rank == rank; });
                return !prepared_group(*peer, group->id()).requirements.peers.empty();
            });
        std::size_t peer_route = p.routes.size();
        if (peer_work) {
            auto route = std::find_if(p.routes.begin(), p.routes.end(), [&](const auto& r) {
                return r.implementation->supports_peers(p.requirements.peers);
            });
            if (route == p.routes.end())
                throw std::invalid_argument("prepared backends do not cover peer transfers");
            peer_route             = route - p.routes.begin();
            work[peer_route].peers = p.requirements.peers;
        }
        auto& description = selected.groups.emplace_back(GroupPreparation{p.requirements, {}});
        std::vector<Group::Impl::Route> used;
        for (std::size_t i = 0; i < p.routes.size(); ++i) {
            if (work[i].collectives.empty() && i != peer_route) continue;
            description.connections.push_back(p.routes[i].implementation->configure(work[i]));
            used.push_back(std::move(p.routes[i]));
        }
        p.routes = std::move(used);
        p.state  = Group::Impl::State::Selected;
    }
    return selected;
}

void CommunicationContext::connect(std::span<const CommunicationPreparation> peers,
                                   std::chrono::steady_clock::time_point deadline) {
    check();
    validate_communication_preparation(topology_, peers);
    std::size_t warmup_bytes = 0;
    for (auto& group : groups_) {
        auto& p = *group->impl_;
        if (p.state != Group::Impl::State::Selected)
            throw std::logic_error("communication connection precedes selection");
        for (const auto rank : p.description.members) {
            const auto peer         = std::find_if(peers.begin(), peers.end(),
                                                   [&](const auto& v) { return v.rank == rank; });
            const auto& connections = prepared_group(*peer, group->id()).connections;
            if (connections.size() != p.routes.size() ||
                std::any_of(p.routes.begin(), p.routes.end(), [&](const auto& route) {
                    return std::none_of(connections.begin(), connections.end(), [&](const auto& c) {
                        return c.backend == route.statistics.backend;
                    });
                }))
                throw std::invalid_argument("selected communication backends differ across ranks");
        }
        for (auto& route : p.routes) {
            route.implementation->connect(peers, deadline);
            warmup_bytes = std::max(warmup_bytes, route.implementation->warmup_bytes());
        }
        p.state = Group::Impl::State::Connected;
    }
    warmup_ = DeviceBuffer(device_.device(), warmup_bytes);
}

void CommunicationContext::warmup() {
    check();
#if NINFER_TARGET_SPARK
    if (native_progress_) native_progress_->start();
#endif
    set_active(true);
    for (auto& group : groups_) {
        auto& p = *group->impl_;
        if (p.state != Group::Impl::State::Connected)
            throw std::logic_error("communication warmup precedes connection");
        for (auto& route : p.routes)
            route.implementation->warmup(warmup_.span(), device_.execution_view());
        p.state = Group::Impl::State::Ready;
    }
    warmup_ = {};
    sample_startup_memory();
    set_active(false);
}

void CommunicationContext::set_active(bool active) noexcept {
#if NINFER_TARGET_SPARK
    if (native_progress_) native_progress_->set_active(active);
#endif
}

const Group& CommunicationContext::group(GroupId id) const {
    for (const auto& group : groups_)
        if (group->id() == id) return *group;
    throw std::invalid_argument("group is not present on this rank");
}

void CommunicationContext::request_stop() noexcept {
    stop_->store(true, std::memory_order_release);
}

void CommunicationContext::check() {
#if NINFER_TARGET_SPARK
    if (native_progress_) native_progress_->check();
#endif
    if (stop_->load(std::memory_order_acquire)) throw std::runtime_error("communication stopped");
    for (const auto& group : groups_)
        for (const auto& route : group->impl_->routes) route.implementation->check();
}

void CommunicationContext::abort() noexcept {
    request_stop();
#if NINFER_TARGET_SPARK
    if (native_progress_) native_progress_->stop();
#endif
    for (auto& group : groups_) {
        for (auto& route : group->impl_->routes) route.implementation->abort();
        group->impl_->state = Group::Impl::State::Closed;
    }
}

void CommunicationContext::close(std::chrono::steady_clock::time_point deadline) {
    for (auto& group : groups_) {
        for (auto& route : group->impl_->routes) route.implementation->close(deadline);
        group->impl_->state = Group::Impl::State::Closed;
    }
#if NINFER_TARGET_SPARK
    if (native_progress_) native_progress_->stop();
#endif
    request_stop();
}

void CommunicationContext::sample_startup_memory() {
    for (auto& group : groups_)
        for (auto& route : group->impl_->routes)
            route.statistics.memory = route.implementation->memory();
}

std::vector<CommunicationStatistics> CommunicationContext::statistics() const {
    std::vector<CommunicationStatistics> result;
    for (const auto& group : groups_) result.push_back(group->statistics());
    return result;
}

void CommunicationContext::reset_statistics() {
    for (auto& group : groups_)
        for (auto& route : group->impl_->routes) {
            route.statistics.eager = {};
            route.statistics.graph = {};
        }
}

void CommunicationContext::observe_totals(CommunicationWork& eager,
                                          CommunicationWork& graph) const noexcept {
    eager = {};
    graph = {};
    for (const auto& group : groups_)
        for (const auto& route : group->impl_->routes) {
            eager.add(route.statistics.eager);
            graph.add(route.statistics.graph);
        }
}

} // namespace ninfer::distributed
