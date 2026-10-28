#include "models/glm5next/parallel/preparation.h"
#include "models/glm5next/program/resources.h"
#include "core/platform.h"

#include <algorithm>
#include <iostream>

namespace {
using namespace ninfer;
namespace glm = models::glm5next;
namespace ctx = models::context;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Storage geometry without model weights or arithmetic admission. Workspace is a separate
// contract qualified by Op tests and the real startup check.
glm::ExecutionPlan storage_plan(unsigned rank, bool mtp, KvCacheStorage storage,
                                KvCacheVectorFormat index_format) {
    auto profile           = std::make_shared<glm::execution::ModelProfile>();
    auto& c                = profile->config.text;
    c.hidden               = 4096;
    c.vocabulary           = 154880;
    c.max_context          = 1048576;
    c.kda                  = {64, 128, 4, 256, -5.0F};
    c.mla                  = {64, 256, 256, 1536, 512, 32, 128, 4, 512, true};
    c.mhc                  = {4, 20, 1e-6F};
    c.kda_layers           = 34;
    c.mla_layers           = 11;
    profile->public_tokens = c.vocabulary;
    profile->local         = {
        rank, 2, {rank * 32, 32}, {rank * 77440, 77440}, {rank * 6144, 6144}, {rank * 1024, 1024}};
    profile->config.mtp = mtp;
    if (mtp) profile->mtp.emplace();
    glm::ExecutionPlan p;
    p.profile = std::move(profile);
    p.shape   = {129,  3,    128, mtp ? 3U : 0U, storage, index_format, EnginePurpose::Generation,
                 true, false};
    p.state   = glm::plan_state(*p.profile);
    LayoutBuilder b;
    p.main = glm::plan_cache(b, c.mla, 11, 129, 3, glm::latent_storage(storage), index_format);
    if (mtp)
        p.mtp = glm::plan_cache(b, c.mla, 1, 129, 3, glm::latent_storage(storage), index_format);
    p.frame            = glm::plan_frame(b, *p.profile, p.shape);
    p.fixed_bytes      = b.finish(256);
    p.mapping_elements = p.main.tables.page_offsets.region.bytes / 8;
    if (p.mtp) p.mapping_elements += p.mtp->tables.page_offsets.region.bytes / 8;
    p.pinned_output_bytes = p.frame.egress.bytes;
    p.workspace.bytes = p.workspace.general_bytes = 256;
    return p;
}

struct Fixture {
    std::array<glm::ExecutionPlan, 2> plans;
    glm::ParallelPlan parallel;
    ctx::ContextDirectorySpec context;
    std::vector<glm::RankResourceDescription> descriptions;

    explicit Fixture(bool mtp, KvCacheStorage storage = KvCacheStorage::BFloat16,
                     KvCacheVectorFormat index_format = KvCacheVectorFormat::BFloat16)
        : plans{storage_plan(0, mtp, storage, index_format),
                storage_plan(1, mtp, storage, index_format)} {
        parallel.ranks.resize(2);
        parallel.ranks[1].rank = 1;
        const std::array offers{
            runtime::RankCapacityOffer{glm::describe_execution_capacity(plans[0]), 1ULL << 30,
                                       2ULL << 30, NINFER_SHARED_PHYSICAL_MEMORY != 0},
            runtime::RankCapacityOffer{glm::describe_execution_capacity(plans[1]), 2ULL << 30,
                                       2ULL << 30, NINFER_SHARED_PHYSICAL_MEMORY != 0}};
        const auto capacities = runtime::resolve_group_context_capacity(
            ContextCapacityPolicy::automatic(1ULL << 30), offers, 0);
        for (unsigned rank = 0; rank < 2; ++rank) {
            auto& p    = plans[rank];
            p.capacity = capacities[rank];
            descriptions.push_back({rank,
                                    p.shape,
                                    glm::describe_state(p.state),
                                    glm::describe_context(p),
                                    {},
                                    {},
                                    0,
                                    p.mtp
                                        ? std::vector<KVPageGeometry>{p.main.cache.page.geometry,
                                                                      p.mtp->cache.page.geometry}
                                        : std::vector<KVPageGeometry>{p.main.cache.page.geometry}});
        }
        context = glm::plan_group_context(parallel, descriptions);
    }
};

void host() {
    for (const bool mtp : {false, true})
        for (const auto storage :
             {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0})
            for (const auto index_format :
                 {KvCacheVectorFormat::BFloat16, KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0}) {
                Fixture f(mtp, storage, index_format);
                const auto& p              = f.plans[0];
                const auto recurrent_bytes = 34ULL * 32 * 128 * 128 * 4;
                const auto conv_bytes      = 34ULL * 3 * 32 * 128 * 3 * 2;
                const auto tails           = (11ULL + mtp) * 128 * 3 * 2 * 2;
                require(p.state.common.image_bytes == recurrent_bytes + conv_bytes + 4096 * 2 &&
                            p.state.owner.image_bytes == p.state.common.image_bytes + tails,
                        "GLM State does not contain the required KDA/hidden/owner-tail payload");
                for (const auto& plan : f.plans) {
                    const auto& reference = plan.capacity.reference;
                    require(reference.tokens_per_request == 129 && reference.request_count == 2 &&
                                reference.device_bytes == (5ULL + mtp) * p.state.owner.image_bytes,
                            "ADP capacity reference omitted remote-owner State or local KV");
                }
                const auto latent_bytes = storage == KvCacheStorage::BFloat16 ? 1024ULL : 516ULL;
                const auto index_bytes =
                    index_format == KvCacheVectorFormat::BFloat16 ? 256ULL : 129ULL;
                const auto index_page_bytes = (index_bytes * 16 + 255) / 256 * 256;
                require(p.main.cache.page.page_bytes ==
                                11ULL * (latent_bytes * 64 + index_page_bytes) &&
                            p.main.tables.spec.logical_page_capacity == 3,
                        "latent/index pages or partial-page table capacity differ");
                require(f.context.ranks[1].device_bytes > f.context.ranks[0].device_bytes &&
                            f.context.pools.size() == 1U + mtp,
                        "group preparation discarded rank capacity or optional history");
                for (const unsigned tail : {3U, 4U, 63U}) {
                    const auto transfer = glm::history_conversion(p.main.cache.page, 0, 1, tail);
                    std::size_t copied  = 0;
                    for (const auto& field : transfer.partial_page) copied += field.bytes;
                    require(
                        copied == 11ULL * (latent_bytes * tail + index_bytes * (tail / 4)),
                        "partial MLA migration includes invalid latent or pooled-index columns");
                }
                auto prepared   = glm::prepare_context(f.parallel, f.descriptions);
                auto& directory = *prepared.directory;
                ctx::ContextDemand demand;
                const auto state0 = demand.add_state({.count = 1, .variant = 0});
                const auto state1 = demand.add_state({.count = 1, .variant = 1});
                const auto pages0 = demand.add_history(0, 2, ctx::ContextMemory::Device, {0});
                const auto pages1 = demand.add_history(0, 1, ctx::ContextMemory::Device, {1});
                auto reserved     = directory.reserve(demand);
                require(reserved && reserved.reservation->ready(),
                        "mixed owner demand did not fit");
                {
                    auto a  = reserved.reservation->take(state0).materialize();
                    auto b  = reserved.reservation->take(state1).materialize();
                    auto h0 = reserved.reservation->take(pages0).materialize();
                    auto h1 = reserved.reservation->take(pages1).materialize();
                    require(a.size() == 2 && b.size() == 2 && h0.size() == 1 && h1.size() == 1 &&
                                a[0].binding().bytes == p.state.owner.image_bytes &&
                                b[0].binding().bytes == p.state.common.image_bytes &&
                                h0[0].domain().rank == 0 && h1[0].domain().rank == 1,
                            "ADP State/history placement lost owner-only parts");
                }
                reserved.reservation.reset();
                for (const auto& pool : directory.usage().pools)
                    require(pool.live_bytes == 0 && pool.reserved_bytes == 0,
                            "released GLM records retained physical ownership");
                const auto copies = glm::state_conversion(prepared.states, 0, 1);
                require(copies.size() == 3 && copies[0].source_rank == copies[0].destination_rank &&
                            copies[1].source_rank == copies[1].destination_rank &&
                            copies[2].source_rank == 0 && copies[2].destination_rank == 1 &&
                            copies[2].bytes == tails,
                        "State conversion moved KDA shards or omitted owner tails");
            }
}

void gpu() {
    DeviceContext device;
    Fixture f(true);
    glm::Resources r(f.plans[0], device);
    const auto memory = r.observe();
    require(memory.allocations.device_bytes() == r.plan.device_bytes() &&
                memory.allocations.io_pinned_bytes == r.plan.pinned_bytes(),
            "allocated GLM backing differs from its plan");
    ctx::ContextDirectory directory(f.context);
    ctx::ContextDemand demand;
    const auto page    = demand.add_history(0, 1, ctx::ContextMemory::Device, {0});
    auto reserve       = directory.reserve(demand);
    auto records       = reserve.reservation->take(page).materialize();
    const auto binding = records.front().binding();
    ctx::ContextActions actions;
    actions.operations.emplace_back(
        ctx::RecordClear{{ctx::ContextMemory::Device, binding.offset, binding.bytes}});
    actions.operations.emplace_back(ctx::MappingUpdate{0, 2, 2, {binding.offset}});
    r.context_actions->submit(actions);
    r.context_actions->drain();
    const auto cache = r.main.layer(10);
    const auto data  = cache.latent.data.tensor(binding.offset);
    const auto index = cache.index.keys.tensor(binding.offset);
    require(data.ne[0] == 512 && data.ne[1] == 64 && index.ne[0] == 128 && index.ne[1] == 16,
            "MLA layer views changed latent/index geometry");
    std::array<std::int64_t, 9> offsets{};
    CUDA_CHECK(cudaMemcpy(offsets.data(), r.main.tables().matrix().data, sizeof(offsets),
                          cudaMemcpyDeviceToHost));
    require(offsets[8] == binding.offset &&
                std::all_of(offsets.begin(), offsets.begin() + 8, [](auto n) { return n == -1; }),
            "Context actions did not publish arena-relative offsets to the selected row");
    const auto tail = r.mtp_tail(2, 2);
    std::array<std::int64_t, 2> tails{};
    CUDA_CHECK(cudaMemcpy(tails.data(), tail.raw_keys.records.byte_offsets.data, sizeof(tails),
                          cudaMemcpyDeviceToHost));
    const auto bytes = r.plan.frame.mtp->tail_keys.shape[0] * 3ULL * sizeof(std::uint16_t);
    require(tails[0] == 6 * bytes && tails[1] == 7 * bytes && r.kda_fold.has_value(),
            "MTP tail slots overlap or KDA fold was not prepared");
    r.synchronize();
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: glm-resource-test --host|--gpu");
        if (std::string_view(argv[1]) == "--host")
            host();
        else {
            require(std::string_view(argv[1]) == "--gpu", "invalid resource test mode");
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
            gpu();
        }
        std::cout << "PASS GLM Context placement and execution resources\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
