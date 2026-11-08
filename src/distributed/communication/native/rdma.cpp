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

struct RemoteRail {
    std::uint32_t network, mask, qpn, psn, rkey;
    std::uint8_t mtu;
    std::uint64_t address;
    std::array<std::byte, 16> gid;
};

struct Rail {
    Port port;
    ibv_context* context = nullptr;
    ibv_pd* pd           = nullptr;
    ibv_cq *send_cq = nullptr, *receive_cq = nullptr;
    ibv_qp* qp      = nullptr;
    ibv_mr *send_mr = nullptr, *receive_mr = nullptr;
    std::uint32_t psn     = 0;
    unsigned inline_bytes = 0;
    std::uint8_t mtu      = 0;
    RemoteRail remote{};

    ~Rail() { close(); }

    void close() noexcept {
        if (qp) {
            ibv_destroy_qp(qp);
            qp = nullptr;
        }
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

constexpr unsigned kAck          = kMaxParts;
constexpr unsigned kRequestCount = kSlots * (kMaxParts + 1);
constexpr unsigned kAckTag       = 1U << 31;

struct Request {
    ibv_send_wr wr{};
    ibv_sge sge{};
    std::uint64_t sequence = 0;
    bool pending           = false;
};

struct Flight {
    enum class Flow { Exchange, BroadcastSend, BroadcastReceive };
    std::uint64_t sequence = 0;
    unsigned parts = 0, rails = 1, posted = 0, completed = 0;
    Flow flow = Flow::Exchange;
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

std::size_t slot_stride(std::size_t capacity) {
    return align(capacity + kMaxParts * (kHeaderBytes + 256), 256);
}

struct Link::Impl {
    Impl(const DeviceContext& device, GroupId id, unsigned local_rank, std::size_t bytes,
         std::span<const Port> ports)
        : group(id), rank(local_rank), capacity(bytes), stride(slot_stride(bytes)),
          send(kSlots * stride + kSlots * kHeaderBytes),
          receive(kSlots * stride + kSlots * kHeaderBytes), control_storage(sizeof(Control)),
          state(device.device(), sizeof(DeviceState)) {
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
                stride,
                rank};
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
            ibv_qp_init_attr init{};
            init.send_cq = rail->send_cq;
            init.recv_cq = rail->receive_cq;
            init.qp_type = IBV_QPT_RC;
            init.cap     = {256, 256, 1, 1, 128};
            rail->qp     = ibv_create_qp(rail->pd, &init);
            require(rail->qp, "create native RC queue pair");
            rail->inline_bytes = init.cap.max_inline_data;
            rail->psn          = (rail->qp->qp_num * 0x9e3779b1U + rank) & 0xffffffU;
            ibv_qp_attr attr{};
            attr.qp_state        = IBV_QPS_INIT;
            attr.port_num        = port.port;
            attr.pkey_index      = 0;
            attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
            check(
                ibv_modify_qp(rail->qp, &attr,
                              IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
                "initialize native QP");
            rails.push_back(std::move(rail));
        }
        for (unsigned i = 0; i < requests.size(); ++i) {
            requests[i].wr.wr_id      = i;
            requests[i].wr.sg_list    = &requests[i].sge;
            requests[i].wr.num_sge    = 1;
            requests[i].wr.opcode     = IBV_WR_RDMA_WRITE_WITH_IMM;
            requests[i].wr.send_flags = IBV_SEND_SIGNALED;
        }
    }

    ~Impl() { rails.clear(); }

    Packet* tx(unsigned slot, unsigned part, unsigned parts) {
        return reinterpret_cast<Packet*>(static_cast<std::byte*>(send.data()) + slot * stride +
                                         packet_offset(capacity, part, parts));
    }

    const Packet* rx(unsigned slot, unsigned part, unsigned parts) const {
        return reinterpret_cast<const Packet*>(static_cast<const std::byte*>(receive.data()) +
                                               slot * stride +
                                               packet_offset(capacity, part, parts));
    }

    void publish_reusable(unsigned slot) {
        // Keep both data and credit descriptors alive until completion. The GPU may change
        // collective kind when it reuses this slot, including either broadcast role.
        release(control->reusable[slot],
                std::min({sent[slot], remote_consumed[slot],
                          acknowledged[slot].load(std::memory_order_relaxed)}));
    }

    void queue(unsigned slot, unsigned part, std::uint64_t sequence, unsigned parts,
               std::array<ibv_send_wr*, kMaxRails>& heads,
               std::array<ibv_send_wr*, kMaxRails>& tails) {
        const bool ack            = part == kAck;
        const unsigned rail_index = ack ? 0 : part % flights[slot].rails;
        auto& rail                = *rails[rail_index];
        auto& request             = requests[slot * (kMaxParts + 1) + part];
        if (request.pending) throw std::logic_error("native send slot reused before completion");
        const auto offset = ack ? kSlots * stride + slot * kHeaderBytes
                                : slot * stride + packet_offset(capacity, part, parts);
        auto* header = reinterpret_cast<Packet*>(static_cast<std::byte*>(send.data()) + offset);
        if (ack) *header = Packet{.sequence = sequence};
        if (header->sequence != sequence ||
            (!ack && (header->parts != parts || header->bytes > capacity ||
                      header->total_bytes > capacity)))
            throw std::logic_error("native published packet has an invalid extent or generation");
        request.sequence    = sequence;
        request.pending     = true;
        request.sge         = {reinterpret_cast<std::uintptr_t>(header),
                               static_cast<std::uint32_t>(kHeaderBytes + (ack ? 0 : header->bytes)),
                               rail.send_mr->lkey};
        request.wr.imm_data = htonl(ack ? kAckTag | slot : (slot << 10) | (parts << 5) | part);
        request.wr.send_flags =
            IBV_SEND_SIGNALED | (request.sge.length <= rail.inline_bytes ? IBV_SEND_INLINE : 0);
        request.wr.wr.rdma = {rail.remote.address + offset, rail.remote.rkey};
        request.wr.next    = nullptr;
        if (tails[rail_index])
            tails[rail_index]->next = &request.wr;
        else
            heads[rail_index] = &request.wr;
        tails[rail_index] = &request.wr;
    }

    void completions() {
        unsigned active_rails = 0;
        for (unsigned slot = 0; slot < kSlots; ++slot)
            if (flights[slot].sequence > acquire(control->reusable[slot]))
                active_rails = std::max(active_rails, flights[slot].rails);
        std::array<ibv_wc, 32> completions;
        for (unsigned r = 0; r < active_rails; ++r) {
            auto& rail = *rails[r];
            int n      = ibv_poll_cq(rail.receive_cq, completions.size(), completions.data());
            if (n < 0) throw std::runtime_error("poll native receive CQ failed");
            for (int i = 0; i < n; ++i) {
                const auto& wc = completions[i];
                check_completion(wc, rail);
                if (!(wc.wc_flags & IBV_WC_WITH_IMM))
                    throw std::runtime_error("native receive completion omitted its packet tag");
                const unsigned tag = ntohl(wc.imm_data);
                if (tag & kAckTag) {
                    const unsigned slot = tag & ~kAckTag;
                    if (slot >= kSlots || wc.byte_len != kHeaderBytes)
                        throw std::runtime_error("invalid native credit packet");
                    const auto* packet = reinterpret_cast<const Packet*>(
                        static_cast<const std::byte*>(receive.data()) + kSlots * stride +
                        slot * kHeaderBytes);
                    if (!packet->sequence || packet->sequence % kSlots != slot)
                        throw std::runtime_error("native credit generation differs from its slot");
                    remote_consumed[slot] = std::max(remote_consumed[slot], packet->sequence);
                    publish_reusable(slot);
                } else {
                    const unsigned slot = tag >> 10, parts = (tag >> 5) & 31U, part = tag & 31U;
                    if (slot >= kSlots || !parts || parts > kMaxParts || part >= parts)
                        throw std::runtime_error("invalid native fragment tag");
                    const auto* packet = rx(slot, part, parts);
                    if (!packet->sequence || packet->sequence % kSlots != slot ||
                        packet->parts != parts || packet->bytes > capacity ||
                        packet->total_bytes > capacity ||
                        wc.byte_len != packet->bytes + kHeaderBytes)
                        throw std::runtime_error(
                            "native received packet has an invalid extent or generation");
                    release(control->received[slot][part], packet->sequence);
                }
            }
            rail.receive(n);
            n = ibv_poll_cq(rail.send_cq, completions.size(), completions.data());
            if (n < 0) throw std::runtime_error("poll native send CQ failed");
            for (int i = 0; i < n; ++i) {
                const auto& wc = completions[i];
                check_completion(wc, rail);
                if (wc.wr_id >= requests.size())
                    throw std::runtime_error("invalid native send completion");
                auto& request       = requests[wc.wr_id];
                const unsigned slot = wc.wr_id / (kMaxParts + 1), part = wc.wr_id % (kMaxParts + 1);
                if (!request.pending) throw std::runtime_error("duplicate native send completion");
                request.pending = false;
                if (part == kAck) {
                    acknowledged[slot].store(request.sequence, std::memory_order_release);
                    if (flights[slot].flow == Flight::Flow::BroadcastReceive)
                        remote_consumed[slot] = request.sequence;
                    publish_reusable(slot);
                } else {
                    auto& flight = flights[slot];
                    if (request.sequence != flight.sequence)
                        throw std::runtime_error("native send completion changed generation");
                    flight.completed |= 1U << part;
                    if (flight.completed == (1U << flight.parts) - 1) {
                        sent[slot] = flight.sequence;
                        publish_reusable(slot);
                    }
                }
            }
        }
    }

    void check_completion(const ibv_wc& wc, const Rail& rail) const {
        if (wc.status != IBV_WC_SUCCESS)
            throw std::runtime_error("native group " + std::to_string(group) + " rank " +
                                     std::to_string(rank) + " rail " + rail.port.device + ": " +
                                     ibv_wc_status_str(wc.status) + " (vendor " +
                                     std::to_string(wc.vendor_err) + ")");
    }

    GroupId group;
    unsigned rank;
    std::size_t capacity, stride;
    PinnedHostBuffer send, receive, control_storage;
    DeviceBuffer state;
    Control* control;
    DeviceView view;
    std::vector<std::unique_ptr<Rail>> rails;
    std::array<Request, kRequestCount> requests;
    std::array<Flight, kSlots> flights;
    std::array<std::uint64_t, kSlots> sent{}, remote_consumed{};
    std::array<std::atomic<std::uint64_t>, kSlots> acknowledged{};
    bool connected                   = false;
    unsigned polls                   = 0;
    std::uint64_t completed_sequence = 0;
    std::chrono::steady_clock::time_point waiting_since{};
};

Link::Link(const DeviceContext& device, GroupId group, unsigned rank, std::size_t capacity,
           std::span<const Port> ports)
    : impl_(std::make_unique<Impl>(device, group, rank, capacity, ports)) {}

Link::~Link() = default;

std::vector<std::byte> Link::connection() const {
    wire::Encoder out;
    out.u64(impl_->capacity);
    out.u32(impl_->rails.size());
    for (const auto& r : impl_->rails) {
        out.u32(r->port.network);
        out.u32(r->port.mask);
        out.raw(r->port.gid);
        out.u32(r->qp->qp_num);
        out.u32(r->psn);
        out.u8(r->mtu);
        out.u64(reinterpret_cast<std::uintptr_t>(impl_->receive.data()));
        out.u32(r->receive_mr->rkey);
    }
    return std::move(out).take();
}

void Link::connect(std::span<const std::byte> bytes) {
    auto& p = *impl_;
    wire::Decoder in(bytes);
    if (in.u64() != p.capacity)
        throw std::invalid_argument("native window capacities differ across ranks");
    const auto count = in.u32();
    std::vector<RemoteRail> remote(count);
    for (auto& r : remote) {
        r.network      = in.u32();
        r.mask         = in.u32();
        const auto gid = in.raw(r.gid.size());
        std::copy(gid.begin(), gid.end(), r.gid.begin());
        r.qpn     = in.u32();
        r.psn     = in.u32();
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
    if (p.rails.size() > kMaxRails) p.rails.resize(kMaxRails);
    for (auto& r : p.rails) {
        ibv_qp_attr attr{};
        attr.qp_state           = IBV_QPS_RTR;
        attr.path_mtu           = static_cast<ibv_mtu>(std::min(r->mtu, r->remote.mtu));
        attr.dest_qp_num        = r->remote.qpn;
        attr.rq_psn             = r->remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer      = 12;
        attr.ah_attr.is_global  = 1;
        attr.ah_attr.port_num   = r->port.port;
        std::memcpy(attr.ah_attr.grh.dgid.raw, r->remote.gid.data(), r->remote.gid.size());
        attr.ah_attr.grh.sgid_index = r->port.gid_index;
        attr.ah_attr.grh.hop_limit  = 1;
        check(ibv_modify_qp(r->qp, &attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
              "connect native QP receiver");
        attr               = {};
        attr.qp_state      = IBV_QPS_RTS;
        attr.timeout       = 14;
        attr.retry_cnt     = 7;
        attr.rnr_retry     = 7;
        attr.sq_psn        = r->psn;
        attr.max_rd_atomic = 1;
        check(ibv_modify_qp(r->qp, &attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
              "connect native QP sender");
        r->receive(256);
    }
    p.connected = true;
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
    if (acquire(p.control->abort) > 1)
        throw std::runtime_error("native collective received a mismatched operation or generation");
    p.completions();
    std::array<ibv_send_wr*, kMaxRails> heads{}, tails{};
    for (unsigned slot = 0; slot < kSlots; ++slot) {
        const auto sequence = acquire(p.control->ready[slot][0]);
        auto& flight        = p.flights[slot];
        if (sequence > flight.sequence) {
            const auto* header = p.tx(slot, 0, 1);
            if (!header->parts || header->parts > kMaxParts || !header->rails ||
                header->rails > p.rails.size())
                throw std::logic_error("native producer declared an invalid fragment count");
            flight = {sequence, header->parts, header->rails, 0, 0};
            if (header->operation == static_cast<unsigned>(Collective::Broadcast)) {
                flight.flow = header->root == p.rank ? Flight::Flow::BroadcastSend
                                                     : Flight::Flow::BroadcastReceive;
                if (flight.flow == Flight::Flow::BroadcastReceive) {
                    flight.posted = flight.completed = (1U << flight.parts) - 1;
                    p.sent[slot]                     = sequence;
                }
            }
        }
        if (flight.sequence) {
            for (unsigned part = 0; part < flight.parts; ++part) {
                if (!(flight.posted & (1U << part)) &&
                    acquire(p.control->ready[slot][part]) == flight.sequence) {
                    p.queue(slot, part, flight.sequence, flight.parts, heads, tails);
                    flight.posted |= 1U << part;
                }
            }
        }
        const auto consumed = acquire(p.control->consumed[slot]);
        auto& ack           = p.requests[slot * (kMaxParts + 1) + kAck];
        if (consumed != flight.sequence) continue;
        if (flight.flow == Flight::Flow::BroadcastSend) {
            if (consumed > p.acknowledged[slot].load(std::memory_order_relaxed)) {
                p.acknowledged[slot].store(consumed, std::memory_order_release);
                p.publish_reusable(slot);
            }
        } else if (consumed > ack.sequence && !ack.pending)
            p.queue(slot, kAck, consumed, 1, heads, tails);
    }
    for (unsigned rail = 0; rail < p.rails.size(); ++rail) {
        if (!heads[rail]) continue;
        ibv_send_wr* bad = nullptr;
        check(ibv_post_send(p.rails[rail]->qp, heads[rail], &bad), "submit native RDMA batch");
    }
    if ((++p.polls & 1023U) == 0) {
        const auto completed =
            std::max(acquire(p.control->consumed[0]), acquire(p.control->consumed[1]));
        if (!pending())
            p.waiting_since = {};
        else {
            const auto now = std::chrono::steady_clock::now();
            if (p.waiting_since == std::chrono::steady_clock::time_point{} ||
                completed != p.completed_sequence)
                p.waiting_since = now;
            else if (now - p.waiting_since > std::chrono::seconds(30))
                throw std::runtime_error(
                    "native group " + std::to_string(p.group) + " rank " + std::to_string(p.rank) +
                    " made no progress after generation " + std::to_string(completed));
        }
        p.completed_sequence = completed;
    }
}

bool Link::pending() const noexcept {
    const auto& p = *impl_;
    for (unsigned slot = 0; slot < kSlots; ++slot) {
        const auto issued = acquire(p.control->ready[slot][0]);
        if (issued > acquire(p.control->reusable[slot])) return true;
    }
    return false;
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
