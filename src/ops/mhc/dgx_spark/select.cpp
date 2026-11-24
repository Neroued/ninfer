#include "ops/mhc/implementation.h"
#include <limits>

namespace ninfer::ops::detail::mhc {
namespace {
constexpr int Unlimited = std::numeric_limits<int>::max();

Plan projection(int hidden, int tokens, int threads, int vector, int last) {
    return {Method::Projection, hidden, tokens, kHidden / hidden, threads, vector, last};
}

Plan post(int threads, int vector, int last) {
    return {Method::Post, threads * vector, 1, 0, threads, vector, last};
}
} // namespace

Plan select(Form form, int columns, int) {
    // Each public form owns its cache-independent schedule and reduction crossovers.
    switch (form) {
    case Form::Pre:
        if (columns == 1) return {Method::VectorProjection, 0, 1, 8, 544, 8, 1};
        if (columns <= 8) return projection(64, 8, 544, 8, 8);
        if (columns <= 16) return projection(64, 8, 288, 16, 16);
        if (columns <= 128) return projection(128, 16, 512, 4, 128);
        if (columns <= 256) return projection(128, 16, 128, 8, 256);
        if (columns <= 608) return projection(256, 32, 128, 8, 608);
        return {Method::ClusterProjection, 512, 8, 0, 512, 4, Unlimited};
    case Form::ExpandPre:
        if (columns == 1) return {Method::VectorProjection, 0, 1, 8, 544, 8, 1};
        if (columns <= 16) return projection(64, 8, 288, 16, 16);
        if (columns <= 64) return projection(128, 16, 256, 8, 64);
        if (columns <= 512) return projection(256, 32, 128, 8, 512);
        return {Method::ClusterProjection, 512, 8, 0, 512, 8, Unlimited};
    case Form::PostPre:
        if (columns <= 16) return projection(64, 8, 288, 16, 16);
        if (columns <= 128) return projection(128, 16, 512, 4, 128);
        if (columns <= 512) return projection(128, 16, 128, 8, 512);
        return {Method::ClusterProjection, 512, 8, 0, 512, 4, Unlimited};
    case Form::Post:
        if (columns <= 4) return post(128, 1, 4);
        if (columns <= 8) return post(128, 2, 8);
        if (columns <= 128) return post(128, 4, 128);
        if (columns <= 1024) return post(256, 4, 1024);
        return post(512, 1, Unlimited);
    case Form::PostMeanNorm:
        if (columns <= 4) return {Method::SplitMean, 128, 1, 32, 128, 4, 4};
        if (columns <= 8) return {Method::SplitMean, 256, 1, 16, 128, 4, 8};
        if (columns <= 16) return {Method::ClusterMean, 1024, 1, 0, 128, 4, 16};
        if (columns <= 512) return {Method::Mean, kHidden, 1, 0, 256, 8, 512};
        return {Method::Mean, kHidden, 1, 0, 1024, 4, Unlimited};
    }
    throw std::logic_error("mhc: invalid form");
}
} // namespace ninfer::ops::detail::mhc
