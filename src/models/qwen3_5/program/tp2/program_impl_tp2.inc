// Translation-unit fragment of src/models/qwen3_5/program/program_impl.cpp, included once at its
// end (after its closing namespace): the two-device (tp 2) parts of ProgramImpl's construction,
// rank 1's runtime, the KV/StateImage mirrors, the mailbox probe and step-down, and the
// vocabulary-split scoring head. The constructor calls construct_tensor_parallel() and
// bind_tensor_parallel_execution() at the points where this code used to stand inline. Not a
// header. Our file; see docs/maintainer/upstream-merge.md.

#include "models/qwen3_5/execution/tp2/workspace_split.h"
#include "models/qwen3_5/execution/tp2/prefill_slices.h"
#include "models/qwen3_5/program/tp2/eager_transport.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::detail {

namespace {

// The mailbox slots. Every captured exchange of a round is a [hidden, columns] BF16 activation
// whose columns are at most (K+1) per request: the verification forward's 128, and the split
// DFlash2 drafter's layer sums over its K+1 block columns. The ordinary slot carries one request's
// exchange, hidden x (K+1); with `batch` the wide slots (PeerMailbox, WIDE SLOTS) carry a full
// batch of `max_concurrency` requests, so the batched rounds stay on the mailbox instead of the
// staged copies inside the same graph. Measured on two RTX 5070 Ti without P2P
// (tools/tp2/mailbox_probe.cu, graph chain of 128): pipelined kernel 17.2 / 23.7 / 31.5 us against
// copies 31.4 / 40.2 / 45.4 us per exchange at 160 / 240 / 320 KiB. The original kernel is slower
// than the copies at those widths (84 / 131 / 196 us), so it gets no wide slots. Returns the wide
// slot's bytes, 0 for none.
std::size_t tp2_mailbox_wide_slot_bytes(std::size_t request_bytes, std::uint32_t max_concurrency,
                                        bool batch) {
    return batch && max_concurrency > 1 ? request_bytes * max_concurrency : 0;
}

// Bytes of the next GEMM's weights one decode exchange prefetches into L2 (PeerMailbox,
// set_l2_prefetch): about what the DRAM moves during the exchange's PCIe round trip.
// NINFER_TP_EXCHANGE_PREFETCH_KIB overrides it, 0 turns the prefetch off.
std::size_t tp2_exchange_prefetch_bytes() {
    constexpr std::size_t kDefaultKib = 12288;
    const char* env                   = std::getenv("NINFER_TP_EXCHANGE_PREFETCH_KIB");
    if (env == nullptr || *env == '\0') { return kDefaultKib << 10; }
    char* end               = nullptr;
    const unsigned long kib = std::strtoul(env, &end, 10);
    if (*end != '\0' || kib > (1UL << 20)) {
        throw std::invalid_argument(
            "NINFER_TP_EXCHANGE_PREFETCH_KIB: expected KiB in [0, 1048576]");
    }
    return static_cast<std::size_t>(kib) << 10;
}

} // namespace

const PersistentLayout& ProgramImpl::PeerRuntime::layout(const SequencePlanImpl& plan) {
    // Rank 1 holds drafter state only for the split DFlash2 drafter, and then all of it.
    if (!plan.peer_persistent || plan.peer_persistent->dflash.has_value() !=
                                     plan.peer_persistent->state_images.dflash_local.has_value()) {
        throw std::logic_error("tensor-parallel plan has no rank 1 persistent layout");
    }
    return *plan.peer_persistent;
}

ProgramImpl::PeerRuntime::PeerRuntime(DeviceContext& peer_device, const SequencePlanImpl& plan)
    : device(peer_device),
      elastic_arena(make_elastic_arena(plan, layout(plan), peer_device.device)),
      persistent(layout(plan).bytes, ZeroFill::Yes),
      workspace_storage(plan.workspace.rank_capacity(1, plan.features.vision_rank), ZeroFill::Yes),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}) {
    const PersistentLayout& own = layout(plan);
    const DeviceSpan backing    = persistent.alloc_bytes(own.bytes, 256);
    decoder                     = std::make_unique<qwen3_5::DecoderState>(
        peer_device.device, backing, own.decoder,
        elastic_arena ? elastic_arena->span() : DeviceSpan{});
    state_images = std::make_unique<qwen3_5::StateImageDevicePool>(backing, own.state_images);
    if (own.replay_records) {
        replay_records.emplace(backing, *own.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    io             = qwen3_5::RoundState(backing, own.round);
    prefill_hidden = own.prefill_hidden.bind(backing);
    if (own.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) { throw std::logic_error("rank 1 DFlash StateImage has no rings"); }
        dflash.emplace(backing, *own.dflash, *local);
    }
    if (own.score_hidden) { score_hidden = own.score_hidden->bind(backing); }
    if (own.qwen4) {
        if (own.qwen4->expert_table) {
            qwen4_expert_table = own.qwen4->expert_table->bind(backing);
        }
        if (own.qwen4->expert_slots > 0) {
            qwen4_cache_pool =
                own.qwen4->bind_pool(elastic_arena ? elastic_arena->span() : backing);
            const auto& q1 = *own.qwen4;
            if (q1.cache_mirror) {
                qwen4_cache_mirror =
                    static_cast<std::int32_t*>(q1.cache_mirror->bind(backing).data);
            }
            qwen4_cache_reads = static_cast<std::int32_t*>(q1.cache_reads->bind(backing).data);
            if (q1.cache_slot_offsets) {
                qwen4_slot_offsets =
                    static_cast<std::int64_t*>(q1.cache_slot_offsets->bind(backing).data);
            }
            if (q1.stage_table) {
                qwen4_stage_table = static_cast<std::int32_t*>(q1.stage_table->bind(backing).data);
                qwen4_stage_picks = static_cast<std::int32_t*>(q1.stage_picks->bind(backing).data);
                for (std::size_t plane = 0; plane < qwen4_staging.size(); ++plane) {
                    qwen4_staging[plane] =
                        static_cast<std::byte*>(q1.stage_pool[plane].bind(backing).data);
                }
            }
            if (own.qwen4->cache_stream_tables) {
                qwen4_stream_tables =
                    static_cast<std::int32_t*>(own.qwen4->cache_stream_tables->bind(backing).data);
            }
            qwen4_cache_table =
                static_cast<std::int32_t*>(own.qwen4->cache_table->bind(backing).data);
            qwen4_cache_divisors =
                static_cast<float*>(own.qwen4->cache_divisors->bind(backing).data);
        }
        if (own.qwen4->decode_ple_rows) {
            qwen4_decode_ple_rows = own.qwen4->decode_ple_rows->bind(backing);
        }
        if (own.qwen4->ple_record) { qwen4_ple_record = own.qwen4->ple_record->bind(backing); }
        if (own.qwen4->draft_extra > 0) {
            for (std::size_t p = 0; p < qwen4_draft_planes.size(); ++p) {
                qwen4_draft_planes[p] =
                    static_cast<std::byte*>(own.qwen4->draft_planes[p].bind(backing).data);
            }
        }
    }
    // A scoring Program decodes nothing: its round state has no decode frame.
    if (static_cast<int>(io.ordinary.has_value()) + static_cast<int>(io.mtp_decode.has_value()) +
            static_cast<int>(io.dflash_decode.has_value()) !=
        (plan.causal_scoring ? 0 : 1)) {
        throw std::logic_error("tensor-parallel rank 1 needs exactly one of the ordinary, MTP and "
                               "DFlash decode frames");
    }
    if (io.ordinary) { ordinary = execution::ordinary_peer_frame(*io.ordinary); }
}

void ProgramImpl::construct_tensor_parallel(const SequencePlanImpl& plan,
                                            ExecutionContext* execution_in,
                                            const execution::Parameters* peer_parameters_in) {
    if (plan.tp != 1 && plan.tp != execution::kTensorParallelWidth) {
        throw std::invalid_argument("Qwen3.5 Program tensor-parallel width must be 1 or 2");
    }
    if ((plan.tp == execution::kTensorParallelWidth) !=
            (execution_in != nullptr && peer_parameters_in != nullptr) ||
        (execution_in != nullptr && execution_in->tp != plan.tp)) {
        throw std::invalid_argument(
            "Qwen3.5 Program execution context does not match the sequence plan width");
    }
    if (execution_in != nullptr) {
        if (!execution_in->dev[0] || !execution_in->dev[1] || &*execution_in->dev[0] != &device ||
            &peer_parameters_in->model != &parameters.model || peer_parameters_in->device != 1 ||
            parameters.device != 0 || parameters.model.device_count() != plan.tp) {
            throw std::invalid_argument(
                "tensor-parallel Program requires rank 0 and rank 1 Parameters of one two-device "
                "Model on its ExecutionContext");
        }
        if (vision_enabled && (!workspace_plan.vision_receiver ||
                               !(vision_rank == 0 ? parameters : *peer_parameters_in).vision ||
                               (vision_rank == 0 ? *peer_parameters_in : parameters).vision)) {
            throw std::invalid_argument(
                "tensor-parallel Vision requires the tower on exactly its vision rank");
        }
        execution_context = execution_in;
        peer_parameters   = peer_parameters_in;
        if ((speculative_backend == SpeculativeBackend::Mtp ||
             speculative_backend == SpeculativeBackend::DFlash2) &&
            proposal_head == ProposalHead::Optimized) {
            tp_transport_status.proposal_head =
                peer_parameters->proposal ? "split by vocabulary" : "rank 0";
        }
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            tp_transport_status.drafter =
                !execution::dflash2_drafter_split(*peer_parameters) ? "rank 0"
                : execution::dflash2_selector_rank(parameters) == 1
                    ? "split (16/4 attention heads, MLP halves per rank; selector on rank 1)"
                    : "split (16/4 attention heads, MLP halves per rank; selector on rank 0)";
        }
        {
            // cudaMalloc is neither stream-ordered nor capturable: rank 1's storage is allocated
            // once, here, with rank 1 current.
            const ScopedCurrentDevice rank1(execution_context->dev[1]->device);
            peer = std::make_unique<PeerRuntime>(*execution_context->dev[1], plan);
            peer_events.emplace(*execution_context);
            gdn_gating_overlap.emplace(*execution_context);
            // T56: eager exchanges (prefill chunks) through pinned host buffers, asynchronously,
            // instead of the driver-staged peer copy that blocks the host per call. That copy is
            // only staged without P2P: with peer access it is a direct device-to-device copy and
            // faster than two host hops (issue #3, P2P board: prefill 5.3k tok/s direct vs 2.0k
            // through the host). So with P2P the direct copy stays, except with elastic KV, whose
            // lent workspace is VMM memory the peer pull cannot read (T64/T68; startup.cpp keeps
            // the lending on that assumption). NINFER_TP_HOST_EXCHANGE=0 / =1 force either way.
            // enable_peer_access() only reports here: the Engine enabled it already, idempotently.
            // With peer access and no HostExchange, the 27B prefill overlap's split-phase sums take
            // ops::PeerLanes (the direct copy on a side stream per rank) instead of falling back
            // to one call per chunk; NINFER_TP_PEER_OVERLAP=0 keeps the one-call schedule, =force
            // attaches the lanes without P2P too (correctness test).
            // The decision table is eager_transport.h (unit-tested on the host).
            const char* host_exchange_env = std::getenv("NINFER_TP_HOST_EXCHANGE");
            const char* peer_overlap_env  = std::getenv("NINFER_TP_PEER_OVERLAP");
            const bool peer_access        = ops::enable_peer_access(*execution_context);
            const Tp2EagerTransport eager = tp2_eager_transport(
                peer_access, plan.elastic_kv.enabled, host_exchange_env ? host_exchange_env : "",
                peer_overlap_env ? peer_overlap_env : "");
            if (eager.host_exchange) {
                host_exchange.emplace(*execution_context);
                peer_events->attach_host_exchange(&*host_exchange);
            }
            if (eager.peer_lanes) {
                peer_lanes.emplace(*execution_context, eager.peer_lanes_direct);
                peer_events->attach_peer_lanes(&*peer_lanes);
            }
            tp_transport_status.eager_transport = eager.host_exchange ? "pinned host exchange"
                                                  : peer_access       ? "direct peer copies"
                                                                      : "driver-staged copies";
            // The plain residual (dense Qwen3.5) overlaps each prefill chunk's exchanges
            // (Qwen4Forward::run_plain); the hyper-connection residual does not.
            if (!parameters.model.config().text.hyper_connection) {
                const bool overlap_off = !execution::prefill_overlap_enabled();
                tp_transport_status.prefill_overlap =
                    overlap_off ? "off (NINFER_TP_PREFILL_OVERLAP=0)"
                    : eager.peer_lanes
                        ? (eager.peer_lanes_direct ? "direct peer lanes"
                                                   : "driver-staged peer lanes (test only)")
                    : eager.host_exchange ? "pinned host lanes"
                                          : "off (one call per chunk)";
                // Whether chunks the halves do not take run sliced.
                if (!overlap_off && (eager.peer_lanes || eager.host_exchange)) {
                    const auto mode = execution::prefill_slices_mode_env();
                    tp_transport_status.prefill_overlap +=
                        mode == execution::PrefillSlicesMode::Force  ? " + slices (force)"
                        : mode == execution::PrefillSlicesMode::Auto ? " + slices"
                                                                     : " (slices off)";
                }
            }
            if (use_cuda_graph) {
                graph_bridge.emplace(execution_context->dev[0]->device,
                                     execution_context->dev[1]->device);
                if (plan.tp_mailbox) {
                    // NINFER_TP_MAILBOX_LEGACY=1 keeps the original exchange kernel, for A/B
                    // runs on one binary; both kernels produce the same bits.
                    const char* legacy_env = std::getenv("NINFER_TP_MAILBOX_LEGACY");
                    const ops::PeerExchangeKernel kernel =
                        legacy_env != nullptr && std::string_view(legacy_env) == "1"
                            ? ops::PeerExchangeKernel::Legacy
                            : ops::PeerExchangeKernel::Pipelined;
                    // One slot carries one request's widest exchange, the [hidden, K+1]
                    // verification activation; the wide slots a full batch's.
                    // NINFER_TP_MAILBOX_SLOT=request leaves the wide slots out (the batched
                    // exchanges then take the staged copies), for A/B runs on one binary.
                    // Qwen4Exp exchanges its MoE partials in FP32 (allreduce_sum_f32, one
                    // rounding after the sum): its slot carries [hidden, K+1] FP32.
                    const std::size_t element = parameters.model.config().text.qwen4()
                                                    ? sizeof(float)
                                                    : sizeof(std::uint16_t);
                    const std::size_t slot_bytes =
                        static_cast<std::size_t>(parameters.model.config().text.hidden_size) *
                        (static_cast<std::size_t>(draft_window) + 1U) * element;
                    const char* slot_env              = std::getenv("NINFER_TP_MAILBOX_SLOT");
                    const std::size_t wide_slot_bytes = tp2_mailbox_wide_slot_bytes(
                        slot_bytes, max_concurrency,
                        kernel == ops::PeerExchangeKernel::Pipelined &&
                            (slot_env == nullptr || std::string_view(slot_env) != "request"));
                    peer_mailbox.emplace(*execution_context, slot_bytes, 2, kernel,
                                         wide_slot_bytes);
                    // Qwen4Exp: the exchanges release the next hyper-connection mixer early
                    // (its Down launch streams its weights under the exchange, T45).
                    peer_mailbox->set_release_dependents(parameters.model.config().text.qwen4());
                    // q27/dexch: decode exchanges prefetch the leading rows of the next GEMM's
                    // weights into L2 while they wait on PCIe (PeerMailbox::set_l2_prefetch;
                    // the dense text forward names them). NINFER_TP_EXCHANGE_PREFETCH_KIB sets
                    // the bytes per exchange, 0 turns it off; timings only, never a result bit.
                    peer_mailbox->set_l2_prefetch_budget(tp2_exchange_prefetch_bytes());
                    peer_events->attach_mailbox(&*peer_mailbox);
                    tp_transport_status.exchange_kernel = ops::peer_exchange_kernel_name(kernel);
                    tp_transport_status.slot_bytes      = peer_mailbox->max_bytes();
                    probe_peer_mailbox();
                } else {
                    tp_transport_status.transport = "copies";
                }
            } else {
                tp_transport_status.transport = "copies";
            }
        }
    }
}

void ProgramImpl::bind_tensor_parallel_execution() {
    set_peer_i32(peer->io.text_kv_table_row, 0);
    if (!causal_scoring) { set_peer_i32(peer->io.backend_kv_table_row, 0); }
    const bool mtp         = speculative_backend == SpeculativeBackend::Mtp;
    const bool speculative = speculative_backend != SpeculativeBackend::None;
    if (mtp && (!peer->io.mtp || !peer->io.mtp_decode || !peer->replay_records ||
                peer->decoder->mtp_cache() == nullptr)) {
        throw std::logic_error("tensor-parallel rank 1 has no MTP storage");
    }
    // Rank 1 verifies DFlash2 proposals in its own copy of the decode frame and records its
    // GDN heads for the commit fold. With the split drafter it also holds its half of the drafter
    // and all of the drafter's state (rings of its four KV heads, prefill and pending features).
    const bool split_drafter = speculative_backend == SpeculativeBackend::DFlash2 &&
                               execution::dflash2_drafter_split(*peer_parameters);
    if (speculative_backend == SpeculativeBackend::DFlash2 &&
        (!peer->io.dflash_decode || !peer->replay_records ||
         peer->dflash.has_value() != split_drafter ||
         (peer->state_images->dflash_local() != nullptr) != split_drafter ||
         execution::dflash2_drafter_split(parameters) != split_drafter)) {
        throw std::logic_error("tensor-parallel rank 1 has no DFlash2 verification storage");
    }
    if (split_drafter) {
        if (!peer->io.dflash_prefill) {
            throw std::logic_error("tensor-parallel rank 1 has no DFlash2 prefill controls");
        }
        peer->drafter.emplace(execution::DFlashPeerDrafter{
            .execution = {peer->device, *peer_parameters, peer->work, peer->state_images->linear(),
                          &*peer->replay_records, peer->io, peer->prefill_hidden, prefill_chunk,
                          proposal_head},
            .dflash    = *peer->dflash,
        });
    }
    tp_execution.emplace(execution::TpExecution{
        .execution            = execution_context,
        .events               = &*peer_events,
        .parameters           = peer_parameters,
        .work                 = &peer->work,
        .linear_attention     = &peer->state_images->linear(),
        .text_cache           = &peer->decoder->text_kv,
        .text_kv_table_row    = peer->io.text_kv_table_row,
        .rope_delta           = peer->io.rope_delta,
        .ordinary             = peer->io.ordinary ? &peer->ordinary : nullptr,
        .replay_records       = speculative ? &*peer->replay_records : nullptr,
        .mtp_cache            = mtp ? peer->decoder->mtp_cache() : nullptr,
        .mtp_kv               = {},
        .backend_kv_table_row = mtp ? peer->io.backend_kv_table_row : Tensor{},
        .mtp                  = mtp ? &*peer->io.mtp : nullptr,
        .prefill_hidden       = mtp ? peer->prefill_hidden : Tensor{},
    });
    tp_execution->gdn_gating = gdn_gating_overlap ? &*gdn_gating_overlap : nullptr;
    if (peer->drafter) { tp_execution->dflash = &*peer->drafter; }
    const char* draft_env = std::getenv("NINFER_TP_MAILBOX_DRAFT");
    if (mtp && peer_mailbox && draft_env != nullptr && std::string_view(draft_env) == "copies") {
        tp_execution->staged_draft_collectives = true;
        tp_transport_status.transport          = "mailbox, MTP draft on copies";
    }
}

void ProgramImpl::attach_tensor_parallel_mirrors() {
    // Rank 0's pools own all page, row and slot bookkeeping; every physical mutation they issue
    // (page zero/copy, execution-row acquire/release/publication, StateImage zero/copy) is
    // replayed at the same indices on rank 1's pool, on rank 1's stream. Attached before any
    // mutation, so both ranks start from the same, empty addressing. Host transfers move rank 1's
    // half on rank 1's transfer stream; the context transfer events order and await it.
    const DeviceContext& rank1 = peer->device;
    {
        const ScopedCurrentDevice scope(rank1.device);
        peer_transfer_order_.emplace(rank1);
        peer_context_completion_.emplace(rank1);
    }
    const DeviceKVMirror where{.device          = rank1.device,
                               .stream          = rank1.stream,
                               .transfer_stream = rank1.transfer_stream,
                               .transfer_order  = &*peer_transfer_order_};
    decoder->text_kv.page_pool().attach_mirror(peer->decoder->text_kv.page_pool(), where);
    decoder->text_kv.execution_tables().attach_mirror(peer->decoder->text_kv.execution_tables(),
                                                      where);
    if (qwen3_5::PagedKVCache* mtp = decoder->mtp_cache()) {
        // The MTP rows are leased with the sequence's backend address space; the mirror takes and
        // releases rank 1's lease on the same row with rank 0's.
        qwen3_5::PagedKVCache* peer_mtp = peer->decoder->mtp_cache();
        if (peer_mtp == nullptr) { throw std::logic_error("tensor-parallel rank 1 has no MTP KV"); }
        mtp->page_pool().attach_mirror(peer_mtp->page_pool(), where);
        mtp->execution_tables().attach_mirror(peer_mtp->execution_tables(), where);
    }
    state_images->attach_mirror(*peer->state_images, rank1.device, rank1.stream,
                                rank1.transfer_stream, &*peer_transfer_order_);
}

void ProgramImpl::record_context_completion() {
    context_completion_.record(device.transfer_stream);
    if (peer_context_completion_) {
        peer_context_completion_->record(peer->device.transfer_stream);
    }
}

bool ProgramImpl::context_completion_ready() const {
    return context_completion_.ready() &&
           (!peer_context_completion_ || peer_context_completion_->ready());
}

void ProgramImpl::synchronize_context_completion() const {
    context_completion_.synchronize();
    if (peer_context_completion_) { peer_context_completion_->synchronize(); }
}

void ProgramImpl::synchronize_context_transfer_streams() const {
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (peer && peer->device.transfer_stream != nullptr) {
        (void)cudaStreamSynchronize(peer->device.transfer_stream);
    }
}

std::optional<execution::TpExecution>
ProgramImpl::prefill_tp_binding(const SequenceState& sequence) const {
    if (!tp_execution) { return std::nullopt; }
    execution::TpExecution binding = *tp_execution;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
            !backend_kv_addresses->active(*sequence.kv->backend)) {
            throw std::logic_error("sequence has no active MTP KV execution mapping");
        }
        const KVExecutionRowLease& row = decoder->mtp_cache()->execution_tables().mirror_row(
            backend_kv_addresses->execution_row(*sequence.kv->backend).handle());
        binding.mtp_kv = peer->decoder->mtp_cache()->execution_view(row);
    }
    return binding;
}

void ProgramImpl::probe_peer_mailbox() {
    const char* setting_env        = std::getenv("NINFER_TP_MAILBOX_PROBE");
    const std::string_view setting = setting_env != nullptr ? setting_env : "";
    if (setting == "off") {
        tp_transport_status.transport = "mailbox";
        return;
    }
    // A slow-but-alive mailbox is worse than the copies: every decode step crosses it about
    // twice per layer, so a round trip past this bound would cost more than the staged path.
    constexpr double kMaxRoundTripMs = 50.0;
    constexpr std::size_t kBytes     = 4096; // whole 16-byte vectors, inside every slot
    constexpr std::int32_t kElements = static_cast<std::int32_t>(kBytes / sizeof(std::uint16_t));
    DeviceContext& rank0             = *execution_context->dev[0];
    DeviceContext& rank1             = *execution_context->dev[1];

    struct Scratch {
        int device    = 0;
        void* buffer  = nullptr;
        void* staging = nullptr;

        explicit Scratch(int device_in) : device(device_in) {
            const ScopedCurrentDevice current(device);
            CUDA_CHECK(cudaMalloc(&buffer, kBytes));
            CUDA_CHECK(cudaMalloc(&staging, kBytes));
            CUDA_CHECK(cudaMemset(buffer, 0, kBytes));
            CUDA_CHECK(cudaMemset(staging, 0, kBytes));
        }

        ~Scratch() {
            const ScopedCurrentDevice current(device);
            cudaFree(buffer);
            cudaFree(staging);
        }

        Scratch(const Scratch&)            = delete;
        Scratch& operator=(const Scratch&) = delete;
    };

    bool hang         = false;
    bool selected     = true;
    double elapsed_ms = 0.0;
    {
        const Scratch scratch0(rank0.device);
        const Scratch scratch1(rank1.device);
        const std::array<Tensor, 2> buffer{Tensor(scratch0.buffer, DType::BF16, {1, kElements}),
                                           Tensor(scratch1.buffer, DType::BF16, {1, kElements})};
        const std::array<Tensor, 2> staging{Tensor(scratch0.staging, DType::BF16, {1, kElements}),
                                            Tensor(scratch1.staging, DType::BF16, {1, kElements})};
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        {
            const ScopedCurrentDevice current(rank0.device);
            CUDA_CHECK(cudaDeviceSynchronize());
            {
                const ScopedCurrentDevice peer_sync(rank1.device);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            definition.capture(
                rank0.stream,
                [&] {
                    if (setting == "fail") {
                        // Only rank 0 enqueues its half: the poller must give up and report.
                        peer_mailbox->enqueue_exchange_sum(0, peer_mailbox->take_capture_slot(),
                                                           scratch0.buffer, kBytes, rank0.stream);
                    } else {
                        ops::allreduce_sum(buffer, staging, *execution_context, *peer_events);
                    }
                    CUDA_CHECK(cudaSetDevice(rank0.device));
                },
                DecodeGraphPeerCapture{.bridge = &*graph_bridge, .stream = rank1.stream});
            // The mailbox exchange is one kernel node per device; the staged path is more.
            selected = definition.node_count() == 2;
            executable.instantiate(definition);
            const auto start = std::chrono::steady_clock::now();
            executable.launch(rank0.stream);
            CUDA_CHECK(cudaStreamSynchronize(rank0.stream));
            {
                const ScopedCurrentDevice peer_sync(rank1.device);
                CUDA_CHECK(cudaStreamSynchronize(rank1.stream));
            }
            elapsed_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
        }
        hang = peer_mailbox->hang_reported();
    }
    tp_transport_status.probe_ms = elapsed_ms;
    if (!hang && selected && elapsed_ms <= kMaxRoundTripMs) {
        tp_transport_status.transport = "mailbox";
        return;
    }
    if (hang) {
        tp_transport_status.fallback = "the startup probe exchange timed out";
    } else if (!selected) {
        tp_transport_status.fallback = "the startup probe did not capture a mailbox exchange";
    } else {
        tp_transport_status.fallback = "the startup probe exchange took " +
                                       std::to_string(static_cast<long>(elapsed_ms)) + " ms";
    }
    tp_transport_status.transport       = "copies";
    tp_transport_status.exchange_kernel = {};
    tp_transport_status.slot_bytes      = 0;
    peer_events->attach_mailbox(nullptr);
    peer_mailbox.reset();
}

void ProgramImpl::synchronize_devices() const {
    if (peer) { peer->device.synchronize(); }
    device.synchronize();
    if (peer_mailbox && peer_mailbox->hang_reported()) {
        std::string transport = tp_transport_status.transport;
        if (tp_transport_status.probe_ms > 0.0) {
            transport += ", startup probe " +
                         std::to_string(static_cast<long>(tp_transport_status.probe_ms * 1000.0)) +
                         " us";
        }
        throw std::runtime_error("tensor-parallel mailbox exchange timed out waiting for the peer "
                                 "device (captured all-reduces: " +
                                 transport + "); the two ranks' results diverged");
    }
}

bool ProgramImpl::degrade_peer_mailbox(std::string_view when) {
    if (!peer_mailbox || !tp_execution) { return false; }
    // The executables bake in the hung mailbox's slab and words: they go first.
    ordinary_graphs                      = {};
    mtp_graphs                           = {};
    dflash_graphs                        = {};
    const std::size_t slot_bytes         = peer_mailbox->slot_bytes();
    const std::size_t wide_slot_bytes    = peer_mailbox->wide_slot_bytes();
    const ops::PeerExchangeKernel kernel = peer_mailbox->kernel();
    const bool release_dependents        = peer_mailbox->release_dependents();
    const std::size_t prefetch_budget    = peer_mailbox->l2_prefetch_budget();
    peer_events->attach_mailbox(nullptr);
    peer_mailbox.reset();
    const std::string previous = tp_transport_status.transport;
    const std::string reason = "an exchange timed out " + std::string(when) + " (" + previous + ")";
    tp_transport_status.fallback +=
        (tp_transport_status.fallback.empty() ? "" : "; then ") + reason;
    if ((speculative_backend == SpeculativeBackend::Mtp ||
         speculative_backend == SpeculativeBackend::DFlash2) &&
        !tp_execution->staged_draft_collectives) {
        tp_execution->staged_draft_collectives = true;
        peer_mailbox.emplace(*execution_context, slot_bytes, 2, kernel, wide_slot_bytes);
        peer_mailbox->set_release_dependents(release_dependents);
        peer_mailbox->set_l2_prefetch_budget(prefetch_budget);
        peer_events->attach_mailbox(&*peer_mailbox);
        tp_transport_status.transport = speculative_backend == SpeculativeBackend::Mtp
                                            ? "mailbox, MTP draft on copies"
                                            : "mailbox, DFlash2 draft on copies";
        return true;
    }
    tp_execution->staged_draft_collectives = false;
    tp_transport_status.transport          = "copies";
    tp_transport_status.exchange_kernel    = {};
    tp_transport_status.slot_bytes         = 0;
    return true;
}

std::string ProgramImpl::recover_peer_transport() {
    if (!peer_mailbox || !peer_mailbox->hang_reported()) {
        throw std::logic_error("no tensor-parallel exchange timed out");
    }
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("the transport cannot step down during a context transaction");
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] < continuation_capacity ||
            requests[lane].lifecycle != Lifecycle::Empty) {
            throw std::logic_error("the transport cannot step down while a request holds a lane");
        }
    }
    // The timed-out poller gave up and the rest of the work retired: both devices are idle.
    if (peer) { peer->device.synchronize(); }
    device.synchronize();
    if (!degrade_peer_mailbox("while serving")) {
        throw std::logic_error("the tensor-parallel transport has nothing left to step down to");
    }
    // The Program holds no request and no retained context: the capture runs as at startup (its
    // StateImages, KV rows and execution-table rows are free, and zeroing every StateImage loses
    // nothing). A hang in a first launch steps down again inside it.
    prepare_graphs();
    return tp_transport_status.transport;
}

void ProgramImpl::set_peer_i32(Tensor& tensor, std::int32_t value) {
    if (!peer) { throw std::logic_error("tensor-parallel rank 1 is unavailable"); }
    const ScopedCurrentDevice rank1(peer->device.device);
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice,
                               peer->device.stream));
}

void ProgramImpl::project_score_tile_split(const Tensor& hidden, const Tensor& logits) {
    // The vocabulary-split output head over the staged columns, as project_split_output_head does
    // for one column: rank 1 copies rank 0's hidden once rank 0's stream has staged it, each rank
    // projects its half, and rank 0 gathers both halves into `logits`.
    const DeviceContext& rank1 = peer->device;
    // The staged columns are contiguous [rows, columns]; rank 1's persistent staging may have
    // more rows per column (Qwen4Exp stages the wide residual, then projects its mixed hidden).
    if (static_cast<std::int64_t>(peer->score_hidden.bytes()) <
            static_cast<std::int64_t>(hidden.bytes()) ||
        !hidden.is_contiguous()) {
        throw std::logic_error("tensor-parallel score staging is smaller than the tile");
    }
    Tensor peer_hidden(peer->score_hidden.data, hidden.dtype, {hidden.ne[0], hidden.ne[1]});
    CUDA_CHECK(cudaEventRecord(peer_events->inputs_ready(0), device.stream));
    {
        const ScopedCurrentDevice scope(rank1.device);
        CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, peer_events->inputs_ready(0), 0));
        CUDA_CHECK(::ninfer::memcpy_device_async(peer_hidden.data, hidden.data, hidden.bytes(),
                                                 rank1.stream));
    }
    const execution::LinearParameters& head0 = parameters.text.output_head;
    const execution::LinearParameters& head1 = peer_parameters->text.output_head;
    const std::int32_t columns               = hidden.ne[1];
    const auto part0 =
        execution::workspace::tp_logits(work, head0.weight.n, head1.weight.n, columns, true);
    const auto part1 =
        execution::workspace::tp_logits(peer->work, head1.weight.n, head0.weight.n, columns, false);
    execution::output_logits_split_rank0({hidden, peer_hidden}, {&head0, &head1},
                                         {part0.partial, part1.partial}, logits, part0.staging,
                                         {&work, &peer->work}, *execution_context, *peer_events);
}

} // namespace ninfer::models::qwen3_5::detail
