#include "distributed/communication/native/rdma.h"

#include "core/cpu/system.h"
#include "distributed/wire.h"

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace ninfer::distributed::detail {
namespace native {
namespace {

std::uint64_t acquire(const Signal& value) {
    return std::atomic_ref(const_cast<std::uint64_t&>(value.value)).load(std::memory_order_acquire);
}

void release(Signal& value, std::uint64_t next) {
    std::atomic_ref(value.value).store(next, std::memory_order_release);
}

void check(int error, const char* operation) {
    if (error) throw std::runtime_error(std::string(operation) + ": " + std::strerror(error));
}

void require(const void* resource, const char* operation) {
    if (!resource) check(errno ? errno : ENOMEM, operation);
}

struct DeviceList {
    ibv_device** value = nullptr;
    int count          = 0;

    DeviceList() {
        value = ibv_get_device_list(&count);
        require(value, "enumerate RDMA devices");
    }

    ~DeviceList() { ibv_free_device_list(value); }
};

bool requested_device(std::string_view name) {
    const auto* environment = std::getenv("NINFER_COMM_HCA");
    if (!environment || !*environment) return true;
    std::string_view list(environment);
    if (list.starts_with('=')) list.remove_prefix(1);
    while (!list.empty()) {
        const auto comma = list.find(',');
        if (list.substr(0, comma) == name) return true;
        if (comma == list.npos) break;
        list.remove_prefix(comma + 1);
    }
    return false;
}

struct RemoteQueuePair {
    std::uint32_t qpn = 0, psn = 0;
};

struct RemoteRail {
    std::uint32_t network, mask, rkey;
    std::array<RemoteQueuePair, 2> queues;
    std::uint8_t mtu;
    std::uint64_t address;
    std::array<std::byte, 16> gid;
};

struct QueuePair {
    ibv_qp* qp                = nullptr;
    std::uint32_t psn         = 0;
    unsigned inline_bytes     = 0;
    std::uint64_t next_ticket = 0;
    std::atomic<std::uint64_t> completed{0}, queued{0};

    void close() noexcept {
        if (qp) {
            ibv_destroy_qp(qp);
            qp = nullptr;
        }
    }

    void receive(unsigned count) {
        std::array<ibv_recv_wr, 64> requests{};
        while (count) {
            const auto n = std::min<unsigned>(count, requests.size());
            for (unsigned i = 0; i < n; ++i)
                requests[i].next = i + 1 < n ? &requests[i + 1] : nullptr;
            ibv_recv_wr* bad = nullptr;
            check(ibv_post_recv(qp, requests.data(), &bad), "post RDMA receive credits");
            count -= n;
        }
    }
};

struct Rail {
    Port port;
    ibv_context* context = nullptr;
    ibv_pd* pd           = nullptr;
    ibv_cq *send_cq = nullptr, *receive_cq = nullptr;
    std::array<QueuePair, 2> queues;
    ibv_mr *send_mr = nullptr, *receive_mr = nullptr;
    std::uint8_t mtu = 0;
    RemoteRail remote{};

    ~Rail() { close(); }

    void close() noexcept {
        for (auto& queue : queues) queue.close();
        if (send_mr) {
            ibv_dereg_mr(send_mr);
            send_mr = nullptr;
        }
        if (receive_mr) {
            ibv_dereg_mr(receive_mr);
            receive_mr = nullptr;
        }
        if (send_cq) {
            ibv_destroy_cq(send_cq);
            send_cq = nullptr;
        }
        if (receive_cq) {
            ibv_destroy_cq(receive_cq);
            receive_cq = nullptr;
        }
        if (pd) {
            ibv_dealloc_pd(pd);
            pd = nullptr;
        }
        if (context) {
            ibv_close_device(context);
            context = nullptr;
        }
    }
};

constexpr unsigned kAckTag       = 1U << 31;
constexpr unsigned kRequestCount = 2 * kSlots * kMaxLanes;

struct Request {
    ibv_send_wr wr{};
    ibv_sge sge{};
    std::uint64_t sequence = 0;
};

} // namespace

std::vector<Port> discover_ports() {
    DeviceList devices;
    ifaddrs* addresses = nullptr;
    if (getifaddrs(&addresses)) check(errno, "inspect interface addresses");
    const std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> owner(addresses, freeifaddrs);
    std::vector<Port> result;
    for (int d = 0; d < devices.count; ++d) {
        const std::string name = ibv_get_device_name(devices.value[d]);
        if (!requested_device(name)) continue;
        auto* context = ibv_open_device(devices.value[d]);
        require(context, "open RDMA device");
        const std::unique_ptr<ibv_context, decltype(&ibv_close_device)> opened(context,
                                                                               ibv_close_device);
        ibv_device_attr device{};
        check(ibv_query_device(context, &device), "query RDMA device");
        for (unsigned p = 1; p <= device.phys_port_cnt; ++p) {
            ibv_port_attr port{};
            check(ibv_query_port(context, p, &port), "query RDMA port");
            if (port.state != IBV_PORT_ACTIVE || port.link_layer != IBV_LINK_LAYER_ETHERNET)
                continue;
            for (int g = 0; g < port.gid_tbl_len; ++g) {
                ibv_gid_entry entry{};
                if (ibv_query_gid_ex(context, p, g, &entry, 0) ||
                    entry.gid_type != IBV_GID_TYPE_ROCE_V2)
                    continue;
                for (auto* a = addresses; a; a = a->ifa_next) {
                    if (!a->ifa_addr || !a->ifa_netmask || a->ifa_addr->sa_family != AF_INET ||
                        if_nametoindex(a->ifa_name) != entry.ndev_ifindex)
                        continue;
                    const auto ip = reinterpret_cast<sockaddr_in*>(a->ifa_addr)->sin_addr.s_addr;
                    if (std::memcmp(entry.gid.raw + 12, &ip, sizeof(ip))) continue;
                    const auto mask =
                        reinterpret_cast<sockaddr_in*>(a->ifa_netmask)->sin_addr.s_addr;
                    Port value{name,
                               static_cast<std::uint8_t>(p),
                               static_cast<std::uint32_t>(g),
                               ntohl(ip) & ntohl(mask),
                               ntohl(mask),
                               {}};
                    std::memcpy(value.gid.data(), entry.gid.raw, value.gid.size());
                    result.push_back(std::move(value));
                    break;
                }
                if (!result.empty() && result.back().device == name && result.back().port == p)
                    break;
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return std::tie(a.network, a.device, a.port) < std::tie(b.network, b.device, b.port);
    });
    return result;
}

struct Link::Impl {
    Impl(const DeviceContext& device, GroupId id, unsigned local_rank, unsigned bytes,
         unsigned lane_count, std::span<const Port> ports)
        : group(id), rank(local_rank), capacity(bytes), allocated_lanes(lane_count),
          lanes(lane_count), packet_bytes(packet_stride(bytes)), stride(packet_bytes * lane_count),
          send(kSlots * stride + kSlots * lane_count * kHeaderBytes), receive(send.size()),
          control_storage(sizeof(Control)), state(device.device(), sizeof(DeviceState)) {
        std::memset(send.data(), 0, send.size());
        std::memset(receive.data(), 0, receive.size());
        control = new (control_storage.data()) Control{};
        state.fill();
        view = {static_cast<std::byte*>(
                    const_cast<void*>(send.cuda_view(device.device()).device_address)),
                static_cast<const std::byte*>(receive.cuda_view(device.device()).device_address),
                static_cast<Control*>(
                    const_cast<void*>(control_storage.cuda_view(device.device()).device_address)),
                static_cast<DeviceState*>(state.data()),
                capacity,
                lanes,
                0,
                packet_bytes,
                stride};
        DeviceList devices;
        for (const auto& port : ports) {
            auto rail  = std::make_unique<Rail>();
            rail->port = port;
            for (int i = 0; i < devices.count; ++i)
                if (port.device == ibv_get_device_name(devices.value[i]))
                    rail->context = ibv_open_device(devices.value[i]);
            require(rail->context, "open native RDMA device");
            rail->pd = ibv_alloc_pd(rail->context);
            require(rail->pd, "allocate native protection domain");
            rail->send_cq    = ibv_create_cq(rail->context, 512, nullptr, nullptr, 0);
            rail->receive_cq = ibv_create_cq(rail->context, 512, nullptr, nullptr, 0);
            require(rail->send_cq, "create native send CQ");
            require(rail->receive_cq, "create native receive CQ");
            rail->send_mr = ibv_reg_mr(rail->pd, send.data(), send.size(), IBV_ACCESS_LOCAL_WRITE);
            rail->receive_mr = ibv_reg_mr(rail->pd, receive.data(), receive.size(),
                                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            require(rail->send_mr, "register native TX window");
            require(rail->receive_mr, "register native RX window");
            ibv_port_attr attributes{};
            check(ibv_query_port(rail->context, port.port, &attributes), "query native port");
            rail->mtu = attributes.active_mtu;
            for (unsigned q = 0; q < (capacity > kSmallBytes ? 2U : 1U); ++q) {
                auto& queue = rail->queues[q];
                ibv_qp_init_attr init{};
                init.send_cq = rail->send_cq;
                init.recv_cq = rail->receive_cq;
                init.qp_type = IBV_QPT_RC;
                init.cap     = {256, 256, 1, 1, 128};
                queue.qp     = ibv_create_qp(rail->pd, &init);
                require(queue.qp, "create native RC queue pair");
                queue.inline_bytes = init.cap.max_inline_data;
                if (queue.inline_bytes < kHeaderBytes)
                    throw std::runtime_error("native RDMA requires inline credit packets");
                queue.psn = (queue.qp->qp_num * 0x9e3779b1U + rank) & 0xffffffU;
                ibv_qp_attr attr{};
                attr.qp_state        = IBV_QPS_INIT;
                attr.port_num        = port.port;
                attr.pkey_index      = 0;
                attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
                check(ibv_modify_qp(queue.qp, &attr,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                        IBV_QP_ACCESS_FLAGS),
                      "initialize native QP");
            }
            rails.push_back(std::move(rail));
        }
        for (unsigned i = 0; i < requests.size(); ++i) {
            requests[i].wr.sg_list = &requests[i].sge;
            requests[i].wr.num_sge = 1;
            requests[i].wr.opcode  = IBV_WR_RDMA_WRITE_WITH_IMM;
        }
    }

    ~Impl() { rails.clear(); }

    std::size_t offset(unsigned slot, unsigned lane, bool ack, bool remote = false) const {
        const auto n = remote ? remote_lanes : allocated_lanes;
        const auto s = packet_bytes * n;
        return ack ? kSlots * s + (slot * n + lane) * kHeaderBytes : slot * s + lane * packet_bytes;
    }

    bool queue(unsigned slot, unsigned lane, std::uint64_t sequence, bool ack,
               std::array<ibv_send_wr*, 2 * kMaxRails>& heads,
               std::array<ibv_send_wr*, 2 * kMaxRails>& tails) {
        const unsigned rail_index  = lane % rails.size();
        auto& rail                 = *rails[rail_index];
        const unsigned queue_index = ack && rail.queues[1].qp ? 1 : 0;
        auto& queue                = rail.queues[queue_index];
        const auto batch           = 2 * rail_index + queue_index;
        auto& request = requests[(ack ? kSlots * kMaxLanes : 0) + slot * kMaxLanes + lane];
        if (queue.next_ticket - queue.completed.load(std::memory_order_relaxed) >= 256)
            return false;
        auto* packet = reinterpret_cast<Packet*>(static_cast<std::byte*>(send.data()) +
                                                 offset(slot, lane, ack));
        if (ack) *packet = Packet{.sequence = sequence};
        if (packet->sequence != sequence || (!ack && packet->bytes > capacity))
            throw std::logic_error("native published packet has an invalid extent or generation");
        request.sequence      = sequence;
        request.wr.wr_id      = (++queue.next_ticket << 1) | queue_index;
        request.sge           = {reinterpret_cast<std::uintptr_t>(packet),
                                 kHeaderBytes + (ack ? 0U : packet->bytes), rail.send_mr->lkey};
        request.wr.imm_data   = htonl((ack ? kAckTag : 0U) | slot << 8 | lane);
        request.wr.send_flags = request.sge.length <= queue.inline_bytes ? IBV_SEND_INLINE : 0;
        request.wr.wr.rdma    = {rail.remote.address + offset(slot, lane, ack, true),
                                 rail.remote.rkey};
        request.wr.next       = nullptr;
        if (tails[batch])
            tails[batch]->next = &request.wr;
        else
            heads[batch] = &request.wr;
        tails[batch] = &request.wr;
        queue.queued.store(queue.next_ticket, std::memory_order_release);
        return true;
    }

    void check_completion(const ibv_wc& wc, const Rail& rail) const {
        if (wc.status != IBV_WC_SUCCESS)
            throw std::runtime_error("native group " + std::to_string(group) + " rank " +
                                     std::to_string(rank) + " rail " + rail.port.device + ": " +
                                     ibv_wc_status_str(wc.status) + " (vendor " +
                                     std::to_string(wc.vendor_err) + ")");
    }

    void completions() {
        std::array<ibv_wc, 32> completions;
        for (auto& owner : rails) {
            auto& rail = *owner;
            int n      = ibv_poll_cq(rail.receive_cq, completions.size(), completions.data());
            if (n < 0) throw std::runtime_error("poll native receive CQ failed");
            unsigned receives[2]{};
            for (int i = 0; i < n; ++i) {
                const auto& wc = completions[i];
                check_completion(wc, rail);
                if (!(wc.wc_flags & IBV_WC_WITH_IMM))
                    throw std::runtime_error("native receive completion omitted its packet tag");
                const unsigned tag = ntohl(wc.imm_data), slot = (tag & ~kAckTag) >> 8,
                               lane = tag & 255U;
                const bool ack      = tag & kAckTag;
                ++receives[ack && rail.queues[1].qp ? 1 : 0];
                if (slot >= kSlots || lane >= lanes)
                    throw std::runtime_error("invalid native packet tag");
                const auto* packet = reinterpret_cast<const Packet*>(
                    static_cast<const std::byte*>(receive.data()) + offset(slot, lane, ack));
                if (!packet->sequence || packet->sequence % kSlots != slot ||
                    (!ack && packet->bytes > capacity) ||
                    wc.byte_len != kHeaderBytes + (ack ? 0U : packet->bytes))
                    throw std::runtime_error(
                        "native received packet has an invalid extent or generation");
                if (ack) {
                    // Peer consumption proves that the NIC has finished reading this TX slot.
                    release(control->reusable[slot][lane], packet->sequence);
                } else
                    release(control->received[slot][lane], packet->sequence);
            }
            for (unsigned q = 0; q < 2; ++q)
                if (receives[q]) rail.queues[q].receive(receives[q]);
            n = ibv_poll_cq(rail.send_cq, completions.size(), completions.data());
            if (n < 0) throw std::runtime_error("poll native send CQ failed");
            for (int i = 0; i < n; ++i) {
                const auto& wc = completions[i];
                check_completion(wc, rail);
                auto& queue       = rail.queues[wc.wr_id & 1U];
                const auto ticket = wc.wr_id >> 1;
                if (!queue.qp || ticket <= queue.completed.load(std::memory_order_relaxed) ||
                    ticket > queue.next_ticket)
                    throw std::runtime_error("invalid native send completion");
                queue.completed.store(ticket, std::memory_order_release);
            }
        }
    }

    bool idle() const noexcept {
        for (unsigned lane = 0; lane < lanes; ++lane) {
            std::uint64_t ack = 0;
            for (unsigned slot = 0; slot < kSlots; ++slot) {
                const auto credited = credits_queued[slot][lane].load(std::memory_order_acquire);
                ack                 = std::max(ack, credited);
                if (acquire(control->ready[slot][lane]) > acquire(control->reusable[slot][lane]) ||
                    acquire(control->consumed[slot][lane]) > credited)
                    return false;
            }
            if (acquire(control->expected[lane]) > ack) return false;
        }
        for (const auto& rail : rails)
            for (const auto& queue : rail->queues)
                if (queue.queued.load(std::memory_order_acquire) !=
                    queue.completed.load(std::memory_order_acquire))
                    return false;
        return true;
    }

    GroupId group;
    unsigned rank, capacity, allocated_lanes, remote_lanes = 0, lanes;
    std::size_t packet_bytes, stride;
    PinnedHostBuffer send, receive, control_storage;
    DeviceBuffer state;
    Control* control;
    DeviceView view;
    std::vector<std::unique_ptr<Rail>> rails;
    std::array<Request, kRequestCount> requests;
    std::uint64_t transmit_queued[kMaxLanes]{};
    std::atomic<std::uint64_t> credits_queued[kSlots][kMaxLanes]{};
    std::atomic<std::uint64_t> drained{0};
    bool connected              = false;
    unsigned polls              = 0;
    std::uint64_t last_progress = 0;
    std::chrono::steady_clock::time_point waiting_since{};
};

Link::Link(const DeviceContext& device, GroupId group, unsigned rank, unsigned capacity,
           unsigned lanes, std::span<const Port> ports)
    : impl_(std::make_unique<Impl>(device, group, rank, capacity, lanes, ports)) {}

Link::~Link() = default;

std::vector<std::byte> Link::connection() const {
    wire::Encoder out;
    out.u32(impl_->capacity);
    out.u32(impl_->allocated_lanes);
    out.u32(impl_->rails.size());
    for (const auto& r : impl_->rails) {
        out.u32(r->port.network);
        out.u32(r->port.mask);
        out.raw(r->port.gid);
        for (const auto& queue : r->queues) {
            out.u32(queue.qp ? queue.qp->qp_num : 0);
            out.u32(queue.psn);
        }
        out.u8(r->mtu);
        out.u64(reinterpret_cast<std::uintptr_t>(impl_->receive.data()));
        out.u32(r->receive_mr->rkey);
    }
    return std::move(out).take();
}

void Link::connect(std::span<const std::byte> bytes) {
    auto& p = *impl_;
    wire::Decoder in(bytes);
    if (in.u32() != p.capacity)
        throw std::invalid_argument("native packet capacities differ across ranks");
    p.remote_lanes = in.u32();
    if (!p.remote_lanes || p.remote_lanes > kMaxLanes)
        throw std::invalid_argument("invalid native lane capacity");
    p.lanes          = std::min(p.allocated_lanes, p.remote_lanes);
    const auto count = in.u32();
    std::vector<RemoteRail> remote(count);
    for (auto& r : remote) {
        r.network      = in.u32();
        r.mask         = in.u32();
        const auto gid = in.raw(r.gid.size());
        std::copy(gid.begin(), gid.end(), r.gid.begin());
        for (auto& queue : r.queues) {
            queue.qpn = in.u32();
            queue.psn = in.u32();
        }
        r.mtu     = in.u8();
        r.address = in.u64();
        r.rkey    = in.u32();
    }
    in.finish();
    std::vector<bool> matched(remote.size());
    std::erase_if(p.rails, [&](const auto& local) {
        const auto peer = std::find_if(remote.begin(), remote.end(), [&](const auto& r) {
            return !matched[&r - remote.data()] && r.network == local->port.network &&
                   r.mask == local->port.mask;
        });
        if (peer == remote.end()) return true;
        matched[peer - remote.begin()] = true;
        local->remote                  = *peer;
        return false;
    });
    if (p.rails.size() > std::min(kMaxRails, p.lanes)) p.rails.resize(std::min(kMaxRails, p.lanes));
    for (auto& r : p.rails) {
        for (unsigned q = 0; q < 2; ++q) {
            auto& queue = r->queues[q];
            if (!queue.qp) continue;
            ibv_qp_attr attr{};
            attr.qp_state           = IBV_QPS_RTR;
            attr.path_mtu           = static_cast<ibv_mtu>(std::min(r->mtu, r->remote.mtu));
            attr.dest_qp_num        = r->remote.queues[q].qpn;
            attr.rq_psn             = r->remote.queues[q].psn;
            attr.max_dest_rd_atomic = 1;
            attr.min_rnr_timer      = 12;
            attr.ah_attr.is_global  = 1;
            attr.ah_attr.port_num   = r->port.port;
            std::memcpy(attr.ah_attr.grh.dgid.raw, r->remote.gid.data(), r->remote.gid.size());
            attr.ah_attr.grh.sgid_index = r->port.gid_index;
            attr.ah_attr.grh.hop_limit  = 1;
            check(ibv_modify_qp(queue.qp, &attr,
                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                                    IBV_QP_MIN_RNR_TIMER),
                  "connect native QP receiver");
            attr               = {};
            attr.qp_state      = IBV_QPS_RTS;
            attr.timeout       = 14;
            attr.retry_cnt     = 7;
            attr.rnr_retry     = 7;
            attr.sq_psn        = queue.psn;
            attr.max_rd_atomic = 1;
            check(ibv_modify_qp(queue.qp, &attr,
                                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
                  "connect native QP sender");
            queue.receive(256);
        }
    }
    p.view.lanes = p.lanes;
    p.view.rails = p.rails.size();
    p.connected  = true;
}

DeviceView Link::device_view() const { return impl_->view; }

unsigned Link::rails() const noexcept { return impl_->connected ? impl_->rails.size() : 0; }

CommunicationMemory Link::memory() const noexcept {
    const auto& p = *impl_;
    return {.owned_host_bytes      = p.send.size() + p.receive.size() + p.control_storage.size(),
            .owned_device_bytes    = p.state.size(),
            .registered_host_bytes = p.send.size() + p.receive.size()};
}

void Link::progress() {
    auto& p = *impl_;
    if (p.rails.empty()) return;
    const auto issued = acquire(p.control->issued);
    if (p.capacity > kSmallBytes && issued == p.drained.load(std::memory_order_relaxed)) return;
    if (acquire(p.control->abort) > 1)
        throw std::runtime_error("native collective received a mismatched operation or generation");
    p.completions();
    std::array<ibv_send_wr*, 2 * kMaxRails> heads{}, tails{};
    for (unsigned lane = 0; lane < p.lanes; ++lane) {
        for (unsigned slot = 0; slot < kSlots; ++slot) {
            const auto consumed = acquire(p.control->consumed[slot][lane]);
            auto& ack           = p.requests[(kSlots + slot) * kMaxLanes + lane];
            if (consumed > ack.sequence && p.queue(slot, lane, consumed, true, heads, tails))
                p.credits_queued[slot][lane].store(consumed, std::memory_order_release);
        }
        for (unsigned i = 0; i < kSlots; ++i) {
            const auto next = p.transmit_queued[lane] + 1;
            const auto slot = next % kSlots;
            if (acquire(p.control->ready[slot][lane]) < next ||
                !p.queue(slot, lane, next, false, heads, tails))
                break;
            p.transmit_queued[lane] = next;
        }
    }
    for (unsigned rail = 0; rail < p.rails.size(); ++rail) {
        // Credits have their own SQ, so queued payload cannot hold up slot reuse.
        for (const unsigned q : {1U, 0U}) {
            const auto batch = 2 * rail + q;
            if (!heads[batch]) continue;
            tails[batch]->send_flags |= IBV_SEND_SIGNALED;
            ibv_send_wr* bad = nullptr;
            check(ibv_post_send(p.rails[rail]->queues[q].qp, heads[batch], &bad),
                  "submit native RDMA batch");
        }
    }
    if (p.capacity > kSmallBytes && acquire(p.control->completed) == issued && p.idle())
        p.drained.store(issued, std::memory_order_release);
    if ((++p.polls & 1023U) == 0) {
        if (p.idle())
            p.waiting_since = {};
        else {
            std::uint64_t completed = 0;
            for (const auto& rail : p.rails)
                for (const auto& queue : rail->queues)
                    completed += queue.completed.load(std::memory_order_relaxed);
            const auto now = std::chrono::steady_clock::now();
            if (p.waiting_since == std::chrono::steady_clock::time_point{} ||
                completed != p.last_progress)
                p.waiting_since = now;
            else if (now - p.waiting_since > std::chrono::seconds(30))
                throw std::runtime_error("native group " + std::to_string(p.group) + " rank " +
                                         std::to_string(p.rank) + " made no progress");
            p.last_progress = completed;
        }
    }
}

bool Link::pending() const noexcept {
    const auto& p = *impl_;
    return (p.capacity > kSmallBytes &&
            acquire(p.control->issued) != p.drained.load(std::memory_order_acquire)) ||
           !p.idle();
}

void Link::signal_abort() noexcept {
    auto& flag             = impl_->control->abort.value;
    std::uint64_t expected = 0;
    std::atomic_ref(flag).compare_exchange_strong(expected, 1, std::memory_order_release);
}

void Link::close() noexcept { impl_->rails.clear(); }

} // namespace native

struct NativeProgress::Impl {
    explicit Impl(std::shared_ptr<std::atomic<bool>> signal) : shared_stop(std::move(signal)) {}

    void fail(std::string message) noexcept {
        {
            std::lock_guard lock(mutex);
            error = std::move(message);
        }
        for (auto* link : links) link->signal_abort();
        failed.store(true, std::memory_order_release);
        shared_stop->store(true, std::memory_order_release);
    }

    void run() noexcept {
        try {
            while (!stopping.load(std::memory_order_acquire) &&
                   !shared_stop->load(std::memory_order_acquire)) {
                bool pending = false;
                for (auto* link : links) {
                    link->progress();
                    if (!active.load(std::memory_order_relaxed)) pending |= link->pending();
                }
                if (!active.load(std::memory_order_acquire) && !pending) {
                    std::unique_lock lock(mutex);
                    changed.wait_for(lock, std::chrono::milliseconds(1), [&] {
                        return active.load(std::memory_order_acquire) ||
                               stopping.load(std::memory_order_acquire);
                    });
                }
            }
            for (auto* link : links) link->signal_abort();
        } catch (const std::exception& e) { fail(e.what()); }
    }

    std::shared_ptr<std::atomic<bool>> shared_stop;
    std::atomic<bool> active{true}, stopping{false}, failed{false};
    std::vector<native::Link*> links;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable changed;
    std::string error;
};

NativeProgress::NativeProgress(std::shared_ptr<std::atomic<bool>> stop)
    : impl_(std::make_unique<Impl>(std::move(stop))) {}

NativeProgress::~NativeProgress() { stop(); }

void NativeProgress::attach(native::Link& link) {
    if (!impl_->threads.empty()) throw std::logic_error("native link added after progress started");
    impl_->links.push_back(&link);
}

void NativeProgress::detach(native::Link& link) noexcept { std::erase(impl_->links, &link); }

void NativeProgress::start() {
    auto& p = *impl_;
    if (!p.threads.empty() || p.links.empty()) return;
    const auto topology = std::make_shared<CpuTopology>();
    auto cpus           = current_thread_affinity(*topology);
    if (cpus.empty()) throw std::runtime_error("native progress has no permitted CPU");
    const auto frequency = [](unsigned cpu) {
        std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/cpuinfo_max_freq");
        unsigned value = 0;
        in >> value;
        return value;
    };
    std::stable_sort(cpus.begin(), cpus.end(),
                     [&](auto a, auto b) { return frequency(a) < frequency(b); });
    auto selected = cpus.back();
    if (const auto* requested = std::getenv("NINFER_COMM_CPU")) {
        const std::string_view text(requested);
        const auto result = std::from_chars(text.data(), text.data() + text.size(), selected);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
            throw std::invalid_argument("NINFER_COMM_CPU must be an OS CPU index");
    }
    p.threads = cpu_detail::start_workers(
        1, [topology, selected](auto) { bind_current_thread(*topology, {&selected, 1}); },
        [](void* owner, auto) noexcept { static_cast<Impl*>(owner)->run(); }, &p);
}

void NativeProgress::set_active(bool active) noexcept {
    impl_->active.store(active, std::memory_order_release);
    if (active) impl_->changed.notify_one();
}

void NativeProgress::wake() noexcept {
    if (!impl_->active.exchange(true, std::memory_order_acq_rel)) impl_->changed.notify_one();
}

void NativeProgress::check() {
    if (!impl_->failed.load(std::memory_order_acquire)) return;
    std::lock_guard lock(impl_->mutex);
    if (!impl_->error.empty()) throw std::runtime_error(impl_->error);
}

void NativeProgress::stop() noexcept {
    impl_->stopping.store(true, std::memory_order_release);
    impl_->changed.notify_one();
    cpu_detail::join_workers(impl_->threads);
    impl_->threads.clear();
}

} // namespace ninfer::distributed::detail
