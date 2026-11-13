// Class-body fragment of ProgramImpl (program/program_impl.h), included in its public section
// right after `workspace_plan`: the two-device (tp 2) runtime. The position matters: rank 1's
// storage and the mailbox are declared before rank 0's storage and the graph families so that
// they are destroyed after them. Not a header; our file, see docs/maintainer/upstream-merge.md.

public:
// Tensor-parallel rank 1 (width 2 only). It holds rank 0's workspace layout (with Vision, only
// the tower's rank holds the encode region; WorkspacePlan::rank_capacity) and rank 0's
// persistent layout without the masked drafter's state (SequencePlanImpl::peer_persistent) on
// ExecutionContext::dev[1]; its KV page pools, KV execution tables and StateImages are mirrors
// that rank 0's pools drive, so rank 0's bookkeeping names both ranks' storage. Rank 1 reads
// the Text and MTP KV, the GDN/hidden StateImages, its ReplaySSM records, the prefill hidden
// and KV rows, and the ordinary, MTP or DFlash2 decode frame. Declared before rank 0's
// storage so it is destroyed after it: rank 0's execution tables hold rank 1's row leases
// until they are destroyed, and rank 0's graphs hold rank 1 nodes and events.
struct PeerRuntime {
    PeerRuntime(DeviceContext& device, const SequencePlanImpl& plan);

    [[nodiscard]] static const PersistentLayout& layout(const SequencePlanImpl& plan);

    DeviceContext& device;
    std::unique_ptr<ElasticArena> elastic_arena; // T64, as ProgramImpl::elastic_arena
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
    std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    // Causal scoring only: rank 1's copy of the staged hidden columns, the input of its half
    // of the vocabulary-split output head.
    Tensor score_hidden;
    // Views rank 1's ordinary decode frame; empty under a speculative backend.
    execution::OrdinaryPeerFrame ordinary;
    // The split DFlash2 drafter (load/sharding.h): rank 1's drafter state (its DFlash rings
    // live in state_images, mirrored with rank 0's) and its drafter operands, bound by
    // bind_tensor_parallel_execution(); empty otherwise.
    std::optional<DFlashPersistentState> dflash;
    std::optional<execution::DFlashPeerDrafter> drafter;
    // Qwen4Exp: rank 1's in-place expert table (the odd expert IDs) and its decode PLE rows
    // (empty otherwise); construct_qwen4 fills and binds them.
    Tensor qwen4_expert_table;
    Tensor qwen4_decode_ple_rows;
    Tensor qwen4_ple_record; // MTP rounds: rank 1's PLE convolution inputs
    // Offloaded experts (T33): rank 1's slot pool, slot table and divisor copy.
    std::array<std::byte*, 4> qwen4_cache_pool{};
    std::int32_t* qwen4_cache_table   = nullptr;
    float* qwen4_cache_divisors       = nullptr;
    std::int32_t* qwen4_stream_tables = nullptr; // prefill streaming tables (T42)
    // Miss-split share on rank 1 (T42): residency mirror, reads, staged scratch.
    std::int32_t* qwen4_cache_mirror = nullptr;
    std::int32_t* qwen4_cache_reads  = nullptr;
    std::int64_t* qwen4_slot_offsets = nullptr; // ggml block banks (Qwen4PersistentLayout)
    std::int32_t* qwen4_stage_table  = nullptr;
    std::int32_t* qwen4_stage_picks  = nullptr;
    std::array<std::byte*, 4> qwen4_staging{};
    // T43 split draft shortlist: rank 1's gathered head rows (code, high, scale planes).
    std::array<std::byte*, 3> qwen4_draft_planes{};
};

ExecutionContext* execution_context          = nullptr;
const execution::Parameters* peer_parameters = nullptr;
std::unique_ptr<PeerRuntime> peer;
// Host tier at tp 2: rank 1's half of every context transfer runs on its transfer stream.
// peer_transfer_order_ orders each one after rank 1's compute stream (the mirrored pools
// record and wait it); peer_context_completion_ is context_completion_'s rank 1 twin.
std::optional<CudaCompletionEvent> peer_transfer_order_;
std::optional<CudaCompletionEvent> peer_context_completion_;
// The captured all-reduces' pinned-host transport (tp 2 with CUDA Graphs, unless disabled),
// attached to peer_events. Declared before peer_events, which points to it, and before the
// graph families, whose kernels address its slab: both are destroyed first.
std::optional<ops::PeerMailbox> peer_mailbox;
// The eager all-reduces' pinned-host transport (T56: prefill chunks), attached to peer_events
// without P2P or with elastic KV (NINFER_TP_HOST_EXCHANGE=0 / =1 force it off / on). Declared
// before peer_events, which points to it.
std::optional<ops::HostExchange> host_exchange;
// v07/p2p-overlap: the split-phase sums' direct-copy lanes, attached to peer_events with peer
// access and no HostExchange (NINFER_TP_PEER_OVERLAP=0 / =force: off / also without P2P).
// Declared before peer_events, which points to it.
std::optional<ops::PeerLanes> peer_lanes;
std::optional<ops::PeerEvents> peer_events;
// PR #64: the side streams of the tp 2 GDN gating overlap. Declared before the graph families,
// which capture work on its streams and events: they are destroyed first.
std::optional<execution::GdnGatingOverlap> gdn_gating_overlap;
std::optional<DecodeGraphPeerBridge> graph_bridge;
std::optional<execution::TpExecution> tp_execution;
// Filled at construction: which transport the captured all-reduces ended up with.
TpTransportStatus tp_transport_status;

// Exchanges one probe payload through peer_mailbox inside a two-device graph before any
// decode graph is captured. A hang (WSL2's GPU virtualization is the known case) or a slow
// round trip drops the mailbox: peer_events falls back to the staged copies for good.
// NINFER_TP_MAILBOX_PROBE=off skips the probe, =fail runs it with a missing peer (test aid).
void probe_peer_mailbox();

// Called by prepare_graphs() after a mailbox exchange timed out in a graph's first launch (and
// by recover_peer_transport() after one timed out while serving), with both devices idle:
// discards every captured graph and steps the transport down once, from the mailbox everywhere
// to the mailbox with the MTP draft phase on the staged copies (MTP only), then to the staged
// copies everywhere. `when` completes the recorded reason ("an exchange timed out <when>").
// Returns false when nothing is left to step down to. The replacement mailbox is fresh: the
// hung one's words have diverged.
bool degrade_peer_mailbox(std::string_view when);

// Program::recover_peer_transport: steps down after an exchange timed out while serving and
// captures the graphs again, as at startup. Requires an empty Program.
std::string recover_peer_transport();

[[nodiscard]] bool tensor_parallel() const noexcept { return peer != nullptr; }

// The Parameters that hold the Vision tower: rank 1's when it holds the tower, else rank 0's.
[[nodiscard]] const execution::Parameters& vision_parameters() const noexcept {
    return vision_rank == 1 && peer_parameters != nullptr ? *peer_parameters : parameters;
}

[[nodiscard]] const execution::TpExecution* tp_binding() const noexcept {
    return tp_execution ? &*tp_execution : nullptr;
}

[[nodiscard]] const DecodeGraphPeerBridge* graph_peer_bridge() const noexcept {
    return graph_bridge ? &*graph_bridge : nullptr;
}

// A prefill call's copy of tp_execution naming `sequence`'s rank-1 MTP KV row; empty at
// width 1.
[[nodiscard]] std::optional<execution::TpExecution>
prefill_tp_binding(const SequenceState& sequence) const;

// Waits for rank 1 and then rank 0; at width 1 only rank 0. Throws std::runtime_error when a
// captured mailbox exchange of the retired work timed out: the two ranks' results diverged.
void synchronize_devices() const;

// MemorySummary::cuda_graph_observed_bytes, measured once by prepare_graphs().
std::array<std::size_t, 2> graph_observed_bytes{};
