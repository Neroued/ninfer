#include "ops/mhc/kernel/projection_mma.cuh"
#include "ops/mhc/kernel/projection_partition.cuh"
#include "ops/mhc/kernel/projection_vector.cuh"
#include "ops/mhc/kernel/pre_finish.cuh"
#include "ops/mhc/kernel/expand_finish.cuh"
#include "ops/mhc/kernel/post.cuh"
#include "ops/mhc/kernel/post_mean_norm.cuh"
#include "ops/mhc/kernel/post_mean_norm_cluster.cuh"
#include "ops/mhc/kernel/post_mean_norm_split.cuh"
#include "ops/mhc/kernel/pre_cluster.cuh"
#include "ops/mhc/kernel/expand_cluster.cuh"
#include "ops/common/host/launch.h"
#include <algorithm>

namespace ninfer::ops::detail::mhc {
namespace {
constexpr std::int64_t GridLimit = 2147483647;

template <int Threads, auto Kernel, class... Args>
void launch_dependent_finish(int columns, CudaExecutionView e, Args... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(columns);
    config.blockDim = dim3(Threads);
    config.stream   = e.stream();
    config.attrs    = &attribute;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, Kernel, args...));
}

template <Form F, int Hidden, int Tokens>
void launch_projection(Operands o, float* scratch, CudaExecutionView e) {
    const auto tiles = ((std::int64_t(o.columns) + Tokens - 1) / Tokens) * (kHidden / Hidden);
    for (std::int64_t offset = 0; offset < tiles; offset += GridLimit) {
        const auto count = static_cast<unsigned>(std::min(GridLimit, tiles - offset));
        if constexpr (Hidden == 64)
            projection_partition_kernel<F, Hidden, 256>
                <<<count, 256, 0, e.stream()>>>(o, scratch, offset);
        else
            projection_mma_kernel<F, Hidden, Tokens>
                <<<count, ProjectionSchedule<Tokens>::kThreads, 0, e.stream()>>>(o, scratch,
                                                                                 offset);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class Tile, auto Kernel, bool Persistent = false>
void launch_cluster_tiles(Operands o, CudaExecutionView e) {
    constexpr int Tokens = 8, Cluster = 8;
    constexpr int Shared = Tile::kSharedBytes;
    (void)common::prepare_shared<Shared, Kernel, true>();
    const auto tiles = (std::int64_t(o.columns) + Tokens - 1) / Tokens;
    if constexpr (Persistent) {
        cudaLaunchConfig_t config{};
        config.gridDim          = dim3(e.multiprocessor_count() * Cluster);
        config.blockDim         = dim3(Tile::kThreads);
        config.dynamicSmemBytes = Shared;
        int resident            = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveClusters(&resident, Kernel, &config));
        if (resident <= 0) throw std::runtime_error("mhc: clustered projection cannot reside");
        const auto count = static_cast<unsigned>(std::min<std::int64_t>(resident, tiles));
        Kernel<<<count * Cluster, Tile::kThreads, Shared, e.stream()>>>(o, 0);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr auto MaxTiles = GridLimit / Cluster;
    for (std::int64_t offset = 0; offset < tiles; offset += MaxTiles) {
        const auto count = static_cast<unsigned>(std::min(MaxTiles, tiles - offset));
        Kernel<<<count * Cluster, Tile::kThreads, Shared, e.stream()>>>(o, offset);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <Form F>
void launch_cluster(Operands o, CudaExecutionView e) {
    if constexpr (F == Form::ExpandPre) {
        launch_cluster_tiles<ExpandClusterTile<512, 128>, expand_cluster_kernel<512, 128>>(o, e);
    } else {
        constexpr int BK = F == Form::Pre ? 128 : 64, Stages = F == Form::Pre ? 2 : 4;
        launch_cluster_tiles<ClusterPreTile<8, 8, BK, Stages>,
                             pre_cluster_kernel<F, 8, 8, BK, Stages>, F == Form::Pre>(o, e);
    }
}

template <Form F>
void launch_pre(Operands o, Plan p, float* scratch, CudaExecutionView e) {
    if (p.method == Method::VectorProjection) {
        if constexpr (F == Form::Pre || F == Form::ExpandPre) {
            constexpr int Parts = 8, Threads = 256;
            projection_vector_kernel<F, Parts, Threads>
                <<<dim3(Parts, kRows), Threads, 0, e.stream()>>>(o, scratch);
            CUDA_CHECK(cudaGetLastError());
            if constexpr (F == Form::Pre)
                launch_dependent_finish<544, pre_finish_kernel<544, 8, Parts, true, true>>(
                    o.columns, e, o, scratch, Parts);
            else
                launch_dependent_finish<544, expand_finish_kernel<544, 8, Parts, true>>(
                    o.columns, e, o, scratch, Parts);
        } else {
            throw std::logic_error("mhc: invalid vector projection plan");
        }
        return;
    }
    if (p.hidden_tile == 64)
        launch_projection<F, 64, 8>(o, scratch, e);
    else if (p.hidden_tile == 128)
        launch_projection<F, 128, 16>(o, scratch, e);
    else if constexpr (F == Form::Pre || F == Form::ExpandPre)
        launch_projection<F, 256, 32>(o, scratch, e);
    else
        throw std::logic_error("mhc: invalid post-pre projection plan");
    if constexpr (F == Form::ExpandPre) {
        if (p.output_threads == 288)
            launch_dependent_finish<288, expand_finish_kernel<288, 16, 64, true>>(o.columns, e, o,
                                                                                  scratch, 64);
        else if (p.output_threads == 128)
            expand_finish_kernel<128><<<o.columns, 128, 0, e.stream()>>>(o, scratch, p.partials);
        else
            expand_finish_kernel<256><<<o.columns, 256, 0, e.stream()>>>(o, scratch, p.partials);
    } else {
        if constexpr (F == Form::PostPre) o.residual = o.updated;
        if (p.output_threads == 544)
            launch_dependent_finish<544, pre_finish_kernel<544, 8, 64, true, F == Form::Pre>>(
                o.columns, e, o, scratch, 64);
        else if (p.output_threads == 288)
            launch_dependent_finish<288, pre_finish_kernel<288, 16, 64, true, F == Form::Pre>>(
                o.columns, e, o, scratch, 64);
        else if (p.output_threads == 128)
            pre_finish_kernel<128, 8, 0, false, false, true>
                <<<o.columns, 128, 0, e.stream()>>>(o, scratch, p.partials);
        else if (p.output_threads == 512)
            pre_finish_kernel<512, 4><<<o.columns, 512, 0, e.stream()>>>(o, scratch, p.partials);
        else
            throw std::logic_error("mhc: invalid pre finish plan");
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch(Form form, Operands o, Plan p, void* scratch, CudaExecutionView e) {
    auto* partial = static_cast<float*>(scratch);
    switch (form) {
    case Form::Pre:
        if (p.method == Method::ClusterProjection)
            launch_cluster<Form::Pre>(o, e);
        else
            launch_pre<Form::Pre>(o, p, partial, e);
        return;
    case Form::ExpandPre:
        if (p.method == Method::ClusterProjection)
            launch_cluster<Form::ExpandPre>(o, e);
        else
            launch_pre<Form::ExpandPre>(o, p, partial, e);
        return;
    case Form::PostPre:
        if (p.method == Method::ClusterProjection) {
            launch_cluster<Form::PostPre>(o, e);
        } else {
            launch_pre<Form::PostPre>(o, p, partial, e);
        }
        return;
    case Form::PostMeanNorm:
        if (p.method == Method::SplitMean) {
            if (p.partials == 32) {
                post_mean_norm_partial_kernel<128, 1>
                    <<<o.columns * 32, 128, 0, e.stream()>>>(o, partial);
                CUDA_CHECK(cudaGetLastError());
                post_mean_norm_finish_kernel<32, 128, 4>
                    <<<o.columns * 8, 128, 0, e.stream()>>>(o, partial);
            } else {
                post_mean_norm_partial_kernel<128, 2>
                    <<<o.columns * 16, 128, 0, e.stream()>>>(o, partial);
                CUDA_CHECK(cudaGetLastError());
                post_mean_norm_finish_kernel<16, 128, 4>
                    <<<o.columns * 8, 128, 0, e.stream()>>>(o, partial);
            }
        } else if (p.method == Method::ClusterMean) {
            post_mean_norm_cluster_kernel<4, 128, 4><<<o.columns * 4, 128, 0, e.stream()>>>(o);
        } else if (p.output_threads == 256) {
            post_mean_norm_vector_kernel<256, 8><<<o.columns, 256, 0, e.stream()>>>(o);
        } else {
            post_mean_norm_vector_kernel<1024, 4><<<o.columns, 1024, 0, e.stream()>>>(o);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    case Form::Post:
        const auto tiles = std::int64_t(o.columns) * (kHidden / p.hidden_tile);
        for (std::int64_t offset = 0; offset < tiles; offset += GridLimit) {
            const auto count = static_cast<unsigned>(std::min(GridLimit, tiles - offset));
            if (p.output_vector == 1) {
                if (p.output_threads == 128)
                    post_kernel<128><<<count, 128, 0, e.stream()>>>(o, offset);
                else
                    post_kernel<512><<<count, 512, 0, e.stream()>>>(o, offset);
            } else if (p.output_vector == 2) {
                post_vector_kernel<128, 2><<<count, 128, 0, e.stream()>>>(o, offset);
            } else if (p.output_threads == 128) {
                post_vector_kernel<128, 4><<<count, 128, 0, e.stream()>>>(o, offset);
            } else {
                post_vector_kernel<256, 4><<<count, 256, 0, e.stream()>>>(o, offset);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
}
} // namespace ninfer::ops::detail::mhc
