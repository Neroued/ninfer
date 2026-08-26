# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

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
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

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
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

# /health is the only unauthenticated endpoint, so this needs no API key. It answers
# 503 once the inference executor has failed (issue #10) and 200 otherwise, which is
# what makes an alive-but-unusable server visible from outside the process.
#
# NINFER_PORT must match --port; HEALTHCHECK is static but the port is not. The
# server binds before loading the model and only listens afterwards, so expect
# connection-refused for the whole load — that is what start-period covers.
#
# NOTE: a restart policy does NOT act on health. Docker restarts on EXIT, not on
# unhealthy, so this makes the failed state observable and alertable; it does not
# by itself recover the container. Recovery needs a watcher (Swarm, an external
# monitor, or the native supervisor's health-restart path).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${NINFER_PORT:-8080}/health" || exit 1

CMD ["ninfer-serve", "--help"]
