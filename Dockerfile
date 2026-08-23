# syntax=docker/dockerfile:1

# Default to the CUDA minor that produced the figures in docs/deployment.md.
# Override for an older driver, e.g. --build-arg CUDA_VERSION=13.2.1.
ARG CUDA_VERSION=13.3.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Only the trees the application build reads. Copying the whole repository would
# make an edit to the entrypoint, compose file, or any document invalidate this
# layer and recompile 129 .cu and 133 .cpp translation units behind a link job
# pool of one. .dockerignore cannot express this: the entrypoint has to stay in
# the context for the runtime stage to COPY it.
COPY CMakeLists.txt ./
COPY include/     ./include/
COPY src/         ./src/
COPY apps/        ./apps/
COPY third_party/ ./third_party/

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/* \
    && (id ubuntu > /dev/null 2>&1 || useradd --uid 1000 --user-group --create-home ubuntu) \
    && install -d -o ubuntu -g ubuntu /var/log/ninfer

COPY --from=build /build/apps/ninfer       /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY --chmod=0755 docker-entrypoint.sh     /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace
USER ubuntu
EXPOSE 11434
STOPSIGNAL SIGTERM

# /health answers only after the artifact is loaded and a real warmup generation
# has run, so the start period covers a cold ~20 GiB O_DIRECT read rather than a
# process launch. Probes during it are free: they do not count toward retries.
# A TCP probe would be wrong -- the socket is bound before the model loads.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10m --start-interval=5s \
    CMD curl -fsS "http://127.0.0.1:${NINFER_PORT:-11434}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
