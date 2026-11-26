#include "ninfer/ops/relative_attention.h"

#include <cuda_fp16.h>
#include "core/decode_graph.h"
#include "ops/op_tester.h"
#include "ops/ragged_fixture.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <array>
#include <limits>
#include <memory>
#include <numeric>

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

constexpr AttentionHeadGeometry kGeometry{128, 24, 8};
constexpr int kDim = 128, kHeads = 24, kKvHeads = 8, kWindow = 1024, kRel = 16;
constexpr int kCacheRecords = 5;
constexpr float kScale      = 0.08838834764831845f;
constexpr ReductionCriterion kAccuracy{0.0045, 0.0001, 0.010};

std::uint32_t hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

float sample(std::size_t i, unsigned seed) {
    return float(int(hash(unsigned(i) + seed) % 2001) - 1000) / 1000.0f;
}

// Generate represented inputs independently of the production codec.
template <class Upload>
void populate_kv(bool fp8, bool minimum_key_scale, int features, unsigned seed,
                 std::vector<double>& key, std::vector<double>& value, Upload upload) {
    for (int plane = 0; plane < 2; ++plane) {
        auto& represented  = plane ? value : key;
        const auto vectors = represented.size() / features;
        if (fp8) {
            std::vector<std::uint8_t> codes(vectors * kDim), scales(vectors);
            for (std::size_t row = 0; row < vectors; ++row) {
                const auto exponent = std::uint8_t(
                    minimum_key_scale && plane == 0 ? 0 : 117 + hash(row + seed + 19 + plane) % 5);
                scales[row] = exponent;
                for (int d = 0; d < features; ++d) {
                    const auto u    = hash(row * features + d + seed + plane * 301);
                    const auto code = std::uint8_t((u % 110) | ((u >> 9) & 128));
                    codes[row * kDim + d * kDim / features] = code;
                    const int e = (code >> 3) & 15, m = code & 7;
                    const double significand = e ? 1.0 + m / 8.0 : m / 8.0;
                    represented[row * features + d] =
                        std::ldexp(code & 128 ? -significand : significand,
                                   std::max(1, e) - 7 + exponent - 127);
                }
            }
            upload(plane, codes);
            upload(plane + 2, scales);
        } else {
            std::vector<std::uint16_t> words(vectors * kDim);
            for (std::size_t row = 0; row < vectors; ++row) {
                for (int d = 0; d < features; ++d) {
                    const auto f = sample(row * features + d, seed + 17 + plane * 711);
                    const auto w = plane ? __half_as_ushort(__float2half_rn(f)) : f32_to_bf16(f);
                    words[row * kDim + d * kDim / features] = w;
                    represented[row * features + d] =
                        plane ? __half2float(__ushort_as_half(w)) : bf16_to_f32(w);
                }
            }
            upload(plane, words);
        }
    }
}

class ChunkFixture {
public:
    ChunkFixture(KvCacheStorage storage, int width, int batch, int features, bool minimum_key_scale)
        : features_(features), key_(std::size_t(width) * batch * kKvHeads * features),
          value_(key_.size()) {
        const auto layout = paged_kv_storage_layout(storage, kDim);
        view.storage      = storage;
        std::array<Tensor*, 4> planes{&view.k, &view.v, &view.k_scales, &view.v_scales};
        std::array dtypes{layout.key.data_dtype, layout.value.data_dtype, DType::U8, DType::U8};
        populate_kv(layout.key.has_scale(), minimum_key_scale, features, 17409, key_, value_,
                    [&](int plane, auto& data) {
                        buffers_[plane] = to_device(data);
                        *planes[plane]  = Tensor(buffers_[plane].span(), dtypes[plane],
                                                 {plane < 2 ? kDim : 1, kKvHeads, width, batch});
                    });
    }

    const double* data(bool value, int token, int head) const {
        return (value ? value_ : key_).data() + (token * kKvHeads + head) * features_;
    }

    void poison_token(int token) {
        const auto write = [&]<class Word>(Word key_nan, Word value_nan) {
            std::array<Word, kDim * kKvHeads> words;
            for (int plane : {0, 1}) {
                words.fill(plane ? value_nan : key_nan);
                buffers_[plane].copy_from_host(words.data(), sizeof(words), token * sizeof(words));
                auto& represented = plane ? value_ : key_;
                std::fill_n(represented.begin() + token * kKvHeads * features_,
                            kKvHeads * features_, std::numeric_limits<double>::quiet_NaN());
            }
        };
        if (view.storage == KvCacheStorage::BFloat16)
            write(std::uint16_t{0x7fc0}, std::uint16_t{0x7e00});
        else
            write(std::uint8_t{0x7f}, std::uint8_t{0x7f});
    }

    RelativeAttentionChunkKV view;

private:
    int features_;
    std::vector<double> key_, value_;
    std::array<DeviceBuffer, 4> buffers_;
};

class CacheFixture {
public:
    CacheFixture(bool swa, KvCacheStorage storage, int batch, int history, int features,
                 bool minimum_key_scale, int padded_capacity)
        : swa_(swa), storage_(storage), rows_(batch),
          slots_(swa ? padded_capacity : kPagedKVPageSize), pages_((history + 63) / 64),
          key_(std::size_t(kCacheRecords) * kKvHeads * slots_ * features), value_(key_.size()),
          features_(features), mapping_(batch * (swa ? 1 : pages_)), row_ids_(batch) {
        const auto dev    = cuda_environment().context.device();
        const auto schema = paged_kv_storage_layout(storage, kDim);
        KVPageGeometry geometry{
            static_cast<unsigned>(slots_),
            {{schema.key.data_dtype, kDim, kKvHeads}, {schema.value.data_dtype, kDim, kKvHeads}}};
        if (schema.key.has_scale()) {
            geometry.planes.push_back({DType::U8, 1, kKvHeads});
            geometry.planes.push_back({DType::U8, 1, kKvHeads});
        }
        packed_ = std::make_unique<tool::PackedKVFixture>(dev, geometry, kCacheRecords);
        populate_kv(schema.key.has_scale(), minimum_key_scale, features, 0, key_, value_,
                    [&](int plane, auto& data) { upload(plane, data); });
        offsets_ = DeviceBuffer(dev, mapping_.size() * sizeof(std::int64_t));
        tables_  = DeviceBuffer(dev, batch * sizeof(int));
    }

    void remap(int epoch, std::span<const int> live, std::span<const int> positions, int width) {
        const int row_pages = swa_ ? 1 : pages_;
        for (int i = 0; i < rows_; ++i) {
            row_ids_[i] = swa_ ? i : (rows_ - 1 - i + epoch) % rows_;
            if (!live.empty() && live[i] == 0) {
                if (swa_)
                    mapping_[i] = -1;
                else
                    row_ids_[i] = -99;
                continue;
            }
            int allocated = row_pages;
            if (!swa_ && !live.empty()) {
                const auto first = positions.begin() + i * width;
                allocated        = *std::max_element(first, first + live[i]) / 64 + 1;
            }
            for (int p = 0; p < row_pages; ++p) {
                const int record                      = strong_pages_ ? (p < strong_pages_ ? 0 : 1)
                                                                      : (3 * p + i + epoch) % kCacheRecords;
                mapping_[row_ids_[i] * row_pages + p] = p < allocated ? record : -1;
            }
        }
        const auto offsets = packed_->table_offsets(mapping_);
        offsets_.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int64_t));
        tables_.copy_from_host(row_ids_.data(), row_ids_.size() * sizeof(int));
    }

    const double* data(bool value, int sequence, int p, int head) const {
        const int physical =
            swa_ ? mapping_[sequence] : mapping_[row_ids_[sequence] * pages_ + p / slots_];
        const auto row = (physical * kKvHeads + head) * slots_ + p % (swa_ ? kWindow : slots_);
        return (value ? value_ : key_).data() + row * features_;
    }

    void poison_history(std::span<const int> first, std::span<const int> positions) {
        const auto offsets = packed_->table_offsets(mapping_);
        auto poisoned      = offsets;
        for (int i = 0; i < rows_; ++i)
            if (positions[first[i]] <= 0) poisoned[i] = INT64_MAX;
        offsets_.copy_from_host(poisoned.data(), poisoned.size() * sizeof(std::int64_t));
    }

    void poison_slot(int slot) {
        const auto write = [&]<class Word>(Word key_nan, Word value_nan) {
            std::array<Word, kDim * kKvHeads> words;
            for (int plane : {0, 1}) {
                words.fill(plane ? value_nan : key_nan);
                const auto& layout = packed_->layout().planes[plane];
                const auto width   = kDim * sizeof(Word);
                auto& represented  = plane ? value_ : key_;
                for (int record = 0; record < kCacheRecords; ++record) {
                    auto* dest = static_cast<unsigned char*>(packed_->storage().data) +
                                 packed_->physical_offset(record) + layout.tensor.region.offset;
                    CUDA_CHECK(cudaMemcpy2D(dest + slot * width, layout.head_stride_bytes,
                                            words.data(), width, width, kKvHeads,
                                            cudaMemcpyHostToDevice));
                    for (int head = 0; head < kKvHeads; ++head) {
                        const int row = (record * kKvHeads + head) * slots_ + slot;
                        std::fill_n(represented.begin() + row * features_, features_,
                                    std::numeric_limits<double>::quiet_NaN());
                    }
                }
            }
        };
        if (storage_ == KvCacheStorage::BFloat16)
            write(std::uint16_t{0x7fc0}, std::uint16_t{0x7e00});
        else
            write(std::uint8_t{0x7f}, std::uint8_t{0x7f});
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void set_bf16_weak_tail(int strong) {
        strong_pages_ = (strong + slots_ - 1) / slots_;
        std::vector<std::uint16_t> k(key_.size() * kDim), v(k.size());
        for (std::size_t row = 0; row < key_.size(); ++row) {
            const bool first = row < kKvHeads * slots_ && row % slots_ < std::size_t(strong);
            key_[row]        = first ? 1.0 : 0.0;
            value_[row]      = first ? 0.0 : 1.0;
            k[row * kDim]    = f32_to_bf16(float(key_[row]));
            v[row * kDim]    = __half_as_ushort(__float2half_rn(float(value_[row])));
        }
        upload(0, k);
        upload(1, v);
    }

    PagedKVBatchLayerView paged() {
        Tensor table(offsets_.span(), DType::I64, {pages_, rows_});
        PagedKVBatchLayerView v{
            packed_->plane(0, table), packed_->plane(1, table), {}, {}, kDim, kKvHeads, storage_};
        if (packed_->layout().planes.size() == 4) {
            v.k_scale_pages = packed_->plane(2, table);
            v.v_scale_pages = packed_->plane(3, table);
        }
        return v;
    }

    CyclicKVCacheLayerView cyclic() {
        const Tensor offsets(offsets_.span(), DType::I64, {rows_});
        const auto field = [&](int index) -> StateFieldView {
            const auto region = packed_->layout().planes[index].tensor;
            return {{packed_->storage(), offsets}, {region, 1, region.region.bytes}};
        };
        CyclicKVCacheLayerView v{
            .k               = field(0),
            .v               = field(1),
            .k_scales        = packed_->layout().planes.size() == 4 ? field(2) : StateFieldView{},
            .v_scales        = packed_->layout().planes.size() == 4 ? field(3) : StateFieldView{},
            .capacity        = kWindow,
            .padded_capacity = static_cast<unsigned>(slots_),
            .num_kv_heads    = kKvHeads,
            .head_dim        = kDim,
            .storage         = storage_};
        return v;
    }

    Tensor table_rows() { return Tensor(tables_.span(), DType::I32, {rows_}); }

    bool guards_intact() const { return packed_->guards_intact(); }

private:
    template <class T>
    void upload(int i, std::vector<T>& data) {
        const auto& plane = packed_->layout().planes[i].tensor;
        const auto& shape = plane.shape;
        packed_->upload_plane(i,
                              Tensor({data.data(), data.size() * sizeof(T), Device::cpu()},
                                     plane.dtype, {shape[0], shape[1], shape[2], kCacheRecords}));
    }

    bool swa_;
    int strong_pages_ = 0;
    KvCacheStorage storage_;
    int rows_, slots_, pages_;
    std::vector<double> key_, value_;
    int features_;
    std::vector<int> mapping_, row_ids_;
    std::unique_ptr<tool::PackedKVFixture> packed_;
    DeviceBuffer offsets_, tables_;
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

class Fixture {
public:
    Fixture(bool swa, KvCacheStorage storage, int batch, int width, int history, int features,
            std::unique_ptr<RaggedFixture> ragged = {}, bool minimum_key_scale = false,
            bool chunk_decode = false, int padded_capacity = kWindow)
        : swa(swa), storage(storage), batch(batch), width(width), features(features),
          ragged(std::move(ragged)), bucket(select_context_bucket(history)),
          cache(swa, storage, this->ragged ? this->ragged->segments() : batch, history, features,
                minimum_key_scale, padded_capacity),
          positions(batch * width), live(batch, width), q(positions.size() * kHeads * kDim),
          r(positions.size() * kHeads * kRel), projection(kRel * kWindow), gate(q.size()),
          sink(positions.size() * kHeads), d_pos(to_device(positions)), d_live(to_device(live)),
          workspace_bytes(capacity()),
          d_workspace(cuda_environment().context.device(), workspace_bytes),
          workspace(d_workspace.span()), output(cuda_environment().context.device(), q.size() * 2) {
        if (swa && (width > 1 || this->ragged || chunk_decode))
            chunk =
                std::make_unique<ChunkFixture>(storage, width, batch, features, minimum_key_scale);
        parallel_elements(q.size(), [&](std::size_t i) {
            const auto query = std::ldexp(sample(i, 399), minimum_key_scale ? 120 : 0);
            q[i]             = i % (kDim / features) == 0 ? f32_to_bf16(query) : 0;
            gate[i]          = f32_to_bf16(i % 23 == 0   ? 90.0f
                                           : i % 19 == 0 ? -90.0f
                                                         : 3.0f * sample(i, 181));
        });
        parallel_elements(r.size(), [&](std::size_t i) { r[i] = f32_to_bf16(sample(i, 601)); });
        parallel_elements(projection.size(), [&](std::size_t i) {
            projection[i] = f32_to_bf16(sample(i, 1901) * 0.8f);
        });
        parallel_elements(sink.size(), [&](std::size_t i) {
            sink[i] = f32_to_bf16(i % 13 == 0   ? -100.0f
                                  : i % 11 == 0 ? 100.0f
                                                : 3.0f * sample(i, 870) + 3.0f);
        });
        d_q          = to_device(q);
        d_r          = to_device(r);
        d_projection = to_device(projection);
        d_gate       = to_device(gate);
        d_sink       = to_device(sink);
    }

    void upload(int epoch, bool masked = true) {
        cache.remap(epoch, ragged ? std::span<const int>{} : std::span<const int>(live), positions,
                    width);
        if (chunk) {
            std::vector<int> first(ragged ? ragged->segments() : batch);
            for (int seq = 0; seq < int(first.size()); ++seq) first[seq] = sequence_first(seq);
            cache.poison_history(first, positions);
        }
        d_pos.copy_from_host(positions.data(), positions.size() * sizeof(int));
        d_live.copy_from_host(live.data(), live.size() * sizeof(int));
        use_mask = masked;
        output.fill(0xa5);
    }

    void launch(const RelativeAttentionChunkKV* override_chunk = nullptr) {
        const auto execution = cuda_environment().context.execution_view();
        RelativeAttentionQueryInputs in{
            Tensor(d_q.span(), DType::BF16, {kDim, kHeads, width, batch}),
            Tensor(d_r.span(), DType::BF16, {kRel, kHeads, width, batch}),
            Tensor(d_projection.span(), DType::BF16, {kRel, kWindow}),
            Tensor(d_gate.span(), DType::BF16, {kDim, kHeads, width, batch}),
            Tensor(d_sink.span(), DType::BF16, {kHeads, width, batch}),
            Tensor(d_pos.span(), DType::I32, {width, batch})};
        Tensor out(output.span(), DType::BF16, {kDim, kHeads, width, batch});
        const Tensor valid = use_mask ? Tensor(d_live.span(), DType::I32, {batch}) : Tensor{};
        const RelativeAttentionChunkKV empty;
        const auto& chunk_view = override_chunk ? *override_chunk : chunk ? chunk->view : empty;
        if (ragged) {
            if (swa)
                relative_softmax_attention(in, ragged->view, kGeometry, kWindow, kScale,
                                           cache.cyclic(), chunk_view, bucket, workspace, out,
                                           execution);
            else
                relative_softmax_attention(in, ragged->view, cache.table_rows(), kGeometry, kScale,
                                           cache.paged(), bucket, workspace, out, execution);
        } else if (swa) {
            relative_softmax_attention(in, valid, kGeometry, kWindow, kScale, cache.cyclic(),
                                       chunk_view, bucket, workspace, out, execution);
        } else {
            relative_softmax_attention(in, valid, cache.table_rows(), kGeometry, kScale,
                                       cache.paged(), bucket, workspace, out, execution);
        }
    }

    int check(const std::string& label) {
        cuda_environment().context.synchronize();
        const auto words = from_device<std::uint16_t>(output, q.size());
        const auto q_val = decode_bf16_words(q), r_val = decode_bf16_words(r);
        const auto pr = decode_bf16_words(projection), g = decode_bf16_words(gate);
        const auto s = decode_bf16_words(sink);
        std::vector<double> ref(words.size());
        parallel_rows(positions.size() * kHeads, positions.size() * kHeads * 1024,
                      [&](int begin, int end) {
                          std::vector<double> scores;
                          for (int row = begin; row < end; ++row) {
                              const int token = row / kHeads, head = row % kHeads;
                              if (inactive(token)) continue;
                              const int seq      = ragged ? ragged->mapping[token] : token / width;
                              const int position = positions[token];
                              const int first    = swa ? std::max(0, position - kWindow + 1) : 0;
                              const int kv       = head / 3;
                              scores.resize(position - first + 1);
                              const auto* qr = q_val.data() + row * kDim;
                              const auto* rr = r_val.data() + row * kRel;
                              for (auto p = std::int64_t(first); p <= position; ++p) {
                                  const auto* key = kv_data(false, seq, p, kv);
                                  double dot = 0, bias = 0;
                                  for (int d = 0; d < features; ++d)
                                      dot += qr[d * kDim / features] * key[d];
                                  const int distance = position - p;
                                  if (distance < kWindow)
                                      for (int u = 0; u < kRel; ++u)
                                          bias += rr[u] * pr[distance * kRel + u];
                                  scores[p - first] = dot * double(kScale) + bias;
                              }
                              const auto max = *std::max_element(scores.begin(), scores.end());
                              std::array<double, kDim> value{};
                              double norm = 0;
                              for (auto p = std::int64_t(first); p <= position; ++p) {
                                  const auto* v       = kv_data(true, seq, p, kv);
                                  const double weight = std::exp(scores[p - first] - max);
                                  norm += weight;
                                  for (int d = 0; d < features; ++d) value[d] += weight * v[d];
                              }
                              const double factor = sigmoid(max + std::log(norm) - s[row]) / norm;
                              for (int d = 0; d < features; ++d) {
                                  const auto i = row * kDim + d * kDim / features;
                                  ref[i]       = value[d] * sigmoid(g[i]) * factor;
                              }
                          }
                      });
        for (std::size_t i = 0; i < words.size(); ++i) {
            if (inactive(i / (kHeads * kDim)) && words[i] != 0) {
                std::cerr << label << " inactive output was not exact BF16 zero\n";
                return 1;
            }
        }
        if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes ||
            !cache.guards_intact())
            throw std::runtime_error("workspace cursor, capacity, or cache guards violated");
        const auto got   = decode_bf16_words(words);
        const auto stats = compute_reduction_stats(got.data(), ref.data(), got.size());
        report_reduction_stats(label, got.size(), stats, kAccuracy);
        if (reduction_passes(stats, got.size(), kAccuracy)) return 0;
        std::cerr << label << " worst output " << stats.maximum_error_index << " expected "
                  << stats.reference_at_maximum << " got " << stats.actual_at_maximum << '\n';
        return 1;
    }

    bool swa;
    KvCacheStorage storage;
    int batch, width, features;
    std::unique_ptr<RaggedFixture> ragged;
    ContextBucket bucket;
    CacheFixture cache;
    std::unique_ptr<ChunkFixture> chunk;
    std::vector<int> positions, live;
    std::vector<std::uint16_t> q, r, projection, gate, sink;
    DeviceBuffer d_pos, d_live, d_q, d_r, d_projection, d_gate, d_sink;
    std::size_t workspace_bytes;
    DeviceBuffer d_workspace;
    MemoryArena workspace;
    DeviceBuffer output;
    bool use_mask = true;

private:
    int sequence_first(int seq) const { return ragged ? ragged->prefix[seq] : seq * width; }

    const double* kv_data(bool value, int seq, int position, int head) const {
        const int first = sequence_first(seq);
        if (chunk && position >= positions[first])
            return chunk->data(value, first + (position - positions[first]), head);
        return cache.data(value, seq, position, head);
    }

    bool inactive(int token) const { return !ragged && token % width >= live[token / width]; }

    std::size_t capacity() const {
        const int sms = cuda_environment().context.multiprocessor_count();
        if (ragged)
            return relative_softmax_attention_ragged_workspace_capacity_bytes(
                kGeometry, storage, swa, {ContextBucket::K2, ContextBucket::M1}, 1, width, 1,
                ragged->segments(), sms);
        if (swa)
            return relative_softmax_attention_swa_workspace_capacity_bytes(
                kGeometry, storage, kWindow, {ContextBucket::K2, ContextBucket::M1}, batch, 1, 16,
                sms);
        return relative_softmax_attention_full_workspace_capacity_bytes(
            kGeometry, storage, {ContextBucket::K2, ContextBucket::M1}, batch, 1, 16, sms);
    }
};

int graph_case(bool swa, KvCacheStorage storage, int width, bool chunk_decode = false) {
    constexpr int batch = 5, history = 2070;
    Fixture f(swa, storage, batch, width, history, 32, {}, false, chunk_decode);
    f.live.assign(batch, 0);
    std::fill(f.positions.begin(), f.positions.end(), INT32_MIN);
    f.upload(0);
    const auto stream = cuda_environment().context.stream();
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    f.launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    int errors = 0;
    for (int replay = 0; replay < 3; ++replay) {
        for (int b = 0; b < batch; ++b) {
            const int count = replay == 0 ? 0 : b == batch - 1 ? width : (b + replay) % (width + 1);
            f.live[b]       = count;
            for (int t = 0; t < width; ++t) {
                const int base[] = {0, 63, 1023, 1024, 2049};
                f.positions[b * width + t] =
                    t < count ? (swa ? base[b] + t : std::max(0, base[b] - t)) : INT32_MIN;
            }
        }
        f.upload(replay);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        errors += f.check("replay " + std::to_string(replay));
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return errors;
}

int graph_bucket_patch(bool swa, KvCacheStorage storage, int width) {
    constexpr int batch = 6;
    Fixture captured(swa, storage, batch, width, 1041, 8);
    Fixture updated(swa, storage, batch, width, 2077, 8);
    const auto prepare = [](Fixture& f, int end, int epoch) {
        for (int b = 0; b < f.batch; ++b) {
            f.live[b] = b == 1 ? 0 : b == 2 ? 1 : f.width;
            for (int t = 0; t < f.width; ++t)
                f.positions[b * f.width + t] =
                    t < f.live[b] ? end - 2 * b - f.width + t : INT32_MIN;
        }
        f.upload(epoch);
    };
    prepare(captured, 1041, 1);
    prepare(updated, 2077, 3);
    const auto& context = cuda_environment().context;
    const int sms       = context.multiprocessor_count();
    const auto profile  = [&](ContextBucket bucket) {
        return swa ? relative_softmax_attention_swa_graph_profile(kGeometry, storage, kWindow,
                                                                   bucket, batch, width, sms, true)
                    : relative_softmax_attention_full_graph_profile(kGeometry, storage, bucket,
                                                                    batch, width, sms, true);
    };
    if (captured.bucket == updated.bucket || profile(captured.bucket) != profile(updated.bucket))
        throw std::runtime_error("graph patch fixture requires two buckets with a shared profile");

    const auto stream = context.stream();
    DecodeGraphDefinition definition;
    DecodeGraphKernelSlice slice;
    definition.capture(stream, [&] { slice.capture(stream, [&] { captured.launch(); }); });
    if (slice.size() != 2) throw std::runtime_error("graph patch fixture must use split attention");
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.launch(stream);
    int errors = captured.check("captured split attention");

    DecodeGraphKernelPatch patch;
    patch.capture(stream, [&] { updated.launch(); });
    executable.update_slice(slice, patch);
    executable.launch(stream);
    errors += updated.check("patched split attention");
    prepare(updated, 2065, 4);
    executable.launch(stream);
    errors += updated.check("patched split attention with new positions and offsets");
    return errors;
}

int swa_direct_decode_replay() {
    Fixture f(true, KvCacheStorage::BFloat16, 1, 1, 1027, 8);
    f.workspace    = MemoryArena({nullptr, 0, cuda_environment().context.device()});
    f.live[0]      = 0;
    f.positions[0] = INT32_MIN;
    f.upload(0);
    const auto stream = cuda_environment().context.stream();
    DecodeGraphDefinition graph;
    DecodeGraphKernelSlice slice;
    graph.capture(stream, [&] { slice.capture(stream, [&] { f.launch(); }); });
    if (slice.size() != 1) return 1;
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    executable.launch(stream);
    int errors = f.check("direct SWA inactive row");
    f.live[0]  = 1;
    for (const int position : {63, 383, 511, 1026}) {
        f.positions[0] = position;
        f.upload(position);
        executable.launch(stream);
        errors += f.check("direct SWA position " + std::to_string(position));
    }
    return errors;
}

int full_fp8_split_replay() {
    constexpr int history = context_bucket_tokens(ContextBucket::K128);
    Fixture f(false, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 1, 1, history, 8);
    f.positions[0] = history / 2;
    f.upload(1);
    const auto stream = cuda_environment().context.stream();
    DecodeGraphDefinition graph;
    graph.capture(stream, [&] { f.launch(); });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    executable.launch(stream);
    int errors     = f.check("full FP8 shorter context");
    f.positions[0] = history - 1;
    f.upload(2);
    executable.launch(stream);
    errors += f.check("full FP8 longer context on replay");
    f.positions[0] = history / 2;
    f.upload(3);
    executable.launch(stream);
    return errors + f.check("full FP8 shorter context on replay");
}

int batched_case(bool swa, KvCacheStorage storage, int batch, int width, int history) {
    Fixture f(swa, storage, batch, width, history, history < 129 ? kDim : 8);
    for (int b = 0; b < batch; ++b) {
        // Include deliberately nonconsecutive positions to exercise the unsheared bias path.
        for (int t = 0; t < width; ++t) {
            const int gap              = b % 3 == 1 ? 19 : 1;
            f.positions[b * width + t] = swa ? std::max(0, history - width - b % 3) + t
                                             : std::max(0, history - 1 - t * gap - b % 3);
        }
    }
    f.upload(1, batch > 1);
    f.launch();
    return f.check("batched B=" + std::to_string(batch) + " W=" + std::to_string(width) +
                   " history=" + std::to_string(history));
}

int ragged_case(bool swa, KvCacheStorage storage, const std::vector<int>& lengths,
                const std::vector<int>& starts = {}) {
    auto ragged      = std::make_unique<RaggedFixture>(lengths);
    const int n      = ragged->tokens();
    int history      = 0;
    const auto start = [&](int s) { return starts.empty() ? (s % 2 ? 73 : 0) : starts[s]; };
    for (int s = 0; s < int(lengths.size()); ++s)
        history = std::max(history, lengths[s] + start(s));
    Fixture f(swa, storage, 1, n, history, n < 100 ? kDim : 8, std::move(ragged));
    for (int s = 0; s < f.ragged->segments(); ++s)
        for (int t = 0; t < lengths[s]; ++t) f.positions[f.ragged->prefix[s] + t] = t + start(s);
    f.upload(2);
    f.launch();
    return f.check("ragged N=" + std::to_string(n));
}

int bf16_ragged_position_replay() {
    constexpr int tokens = 449;
    Fixture f(false, KvCacheStorage::BFloat16, 1, tokens, 2 * tokens, 8,
              std::make_unique<RaggedFixture>(std::vector{tokens}));
    std::iota(f.positions.begin(), f.positions.end(), 0);
    f.upload(1);
    const auto stream = cuda_environment().context.stream();
    DecodeGraphDefinition graph;
    graph.capture(stream, [&] { f.launch(); });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    executable.launch(stream);
    int errors = f.check("ragged contiguous positions");

    for (auto& position : f.positions) position *= 2;
    f.upload(2);
    executable.launch(stream);
    return errors + f.check("ragged nonconsecutive positions on replay");
}

int chunk_batched(KvCacheStorage storage, int width, int start) {
    // The two sources use distinct samples. In particular slot 0 and column 1
    // differ in the [1023, 1024] case.
    Fixture f(true, storage, 3, width, kWindow, kDim, {}, false, true);
    for (int b = 0; b < f.batch; ++b)
        std::iota(f.positions.begin() + b * width, f.positions.begin() + (b + 1) * width, start);
    f.live = {width, width - 1, 0};
    std::fill(f.positions.begin() + 2 * width, f.positions.end(), INT32_MIN);
    f.upload(1);
    f.launch();
    return f.check("two-segment SWA");
}

int expired_slot(KvCacheStorage storage) {
    int errors = 0;
    for (int width : {1, 2, 3}) {
        // With three columns, only [1024, 1025] are live; the third may contain NaNs.
        Fixture f(true, storage, 1, width, 1026, 8, {}, false, true);
        f.positions[0] = 1024;
        f.live[0]      = std::min(width, 2);
        if (width > 1) f.positions[1] = 1025;
        if (width > 2) {
            f.positions[2] = INT32_MIN;
            f.chunk->poison_token(2);
        }
        f.cache.poison_slot(0);
        f.upload(2);
        f.launch();
        errors += f.check("SWA expired cyclic slot");
    }
    Fixture ragged(true, storage, 1, 2, 1026, 8, std::make_unique<RaggedFixture>(std::vector{2}));
    ragged.positions = {1024, 1025};
    ragged.cache.poison_slot(0);
    ragged.upload(1);
    ragged.launch();
    return errors + ragged.check("ragged SWA expired cyclic slot");
}

int fp8_odd_scale_stride() {
    int errors = 0;
    for (int width : {1, 5, 49}) {
        auto layout = width > 16 ? std::make_unique<RaggedFixture>(std::vector{width}) : nullptr;
        Fixture f(true, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 1, width, 1021 + width, 8,
                  std::move(layout), false, false, 1025);
        std::iota(f.positions.begin(), f.positions.end(), 1021);
        f.upload(2);
        f.launch();
        errors += f.check("SWA cyclic FP8 padded_capacity=1025 W=" + std::to_string(width));
    }
    return errors;
}

int bf16_weak_tail(int history, int strong, float query) {
    Fixture f(false, KvCacheStorage::BFloat16, 1, 2, history, 1,
              std::make_unique<RaggedFixture>(std::vector{2}));
    f.positions = {history - 2, history - 1};
    f.cache.set_bf16_weak_tail(strong);
    std::fill(f.q.begin(), f.q.end(), 0);
    for (std::size_t row = 0; row < f.sink.size(); ++row) f.q[row * kDim] = f32_to_bf16(query);
    std::fill(f.r.begin(), f.r.end(), 0);
    std::fill(f.gate.begin(), f.gate.end(), f32_to_bf16(100.0f));
    std::fill(f.sink.begin(), f.sink.end(), f32_to_bf16(-100.0f));
    f.d_q    = to_device(f.q);
    f.d_r    = to_device(f.r);
    f.d_gate = to_device(f.gate);
    f.d_sink = to_device(f.sink);
    f.upload(0);
    f.launch();
    return f.check("Full BF16 small weights across " + std::to_string(history) + " keys");
}

int reject_invalid_chunk(KvCacheStorage storage) {
    Fixture f(true, storage, 1, 2, 2, 8);
    f.positions = {0, 1};
    f.upload(1);
    const auto rejected = [&](const RelativeAttentionChunkKV& chunk) {
        try {
            f.launch(&chunk);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    const auto good  = f.chunk->view;
    auto mismatch    = good;
    mismatch.storage = storage == KvCacheStorage::BFloat16 ? KvCacheStorage::Fp8E4m3fnGroup128Ue8m0
                                                           : KvCacheStorage::BFloat16;
    auto missing     = good;
    missing.k        = {};
    const bool rejected_invalid = rejected({}) && rejected(missing) && rejected(mismatch);
    // Reject missing required chunk metadata before borrowing scratch.
    f.workspace = MemoryArena({nullptr, 0, cuda_environment().context.device()});
    const bool rejected_without_scratch = rejected({});
    Fixture decode(true, storage, 1, 1, 1, 8, {}, false, true);
    decode.upload(1);
    auto half_empty       = decode.chunk->view;
    half_empty.k          = {};
    bool rejected_partial = false;
    try {
        decode.launch(&half_empty);
    } catch (const std::invalid_argument&) { rejected_partial = true; }
    return !(rejected_invalid && rejected_without_scratch && rejected_partial);
}

int profiles(KvCacheStorage storage) {
    const int sms = cuda_environment().context.multiprocessor_count();
    for (const bool swa : {false, true}) {
        for (const int batch : {1, 3, 8, 1024}) {
            const auto bytes = [&](ContextBucketRange buckets, int lo, int hi) {
                if (swa)
                    return relative_softmax_attention_swa_workspace_capacity_bytes(
                        kGeometry, storage, kWindow, buckets, batch, lo, hi, sms);
                return relative_softmax_attention_full_workspace_capacity_bytes(
                    kGeometry, storage, buckets, batch, lo, hi, sms);
            };
            std::size_t peak = 0;
            for (auto bucket : kAllContextBuckets)
                for (int w = 1; w <= 16; ++w) peak = std::max(peak, bytes(bucket, w, w));
            if (peak != bytes({ContextBucket::K2, ContextBucket::M1}, 1, 16)) return 1;
        }
    }
    const auto profile = [&](ContextBucket b, int w, bool masked) {
        return relative_softmax_attention_swa_graph_profile(kGeometry, storage, kWindow, b, 1, w,
                                                            sms, masked);
    };
    if (profile(ContextBucket::K2, 7, true) != profile(ContextBucket::M1, 7, true) ||
        profile(ContextBucket::K2, 7, true) == profile(ContextBucket::K2, 7, false))
        return 1;
    const auto full = [&](ContextBucket b) {
        return relative_softmax_attention_full_graph_profile(kGeometry, storage, b, 1, 1, sms);
    };
    return full(ContextBucket::K32) != full(ContextBucket::M1);
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0}) {
        const std::string format = kv_cache_profile_info(storage).name;
        suite.add(format + "/profiles", [=] { return profiles(storage); }, {{"format", format}});
        for (bool swa : {true, false}) {
            const auto name = format + (swa ? "/swa" : "/full");
            for (auto dims : {std::array{1, 1, 1},
                              {3, 2, 2},
                              {5, 3, 63},
                              {8, 7, 64},
                              {13, 8, 65},
                              {1, 13, 1023},
                              {3, 16, 1024},
                              {5, 1, 1025},
                              {8, 2, 2049},
                              {1, 3, 4099},
                              {1, 1, 16387},
                              {1, 2, 32771}}) {
                const auto [b, w, l] = dims;
                suite.add(name + "/batched/" + std::to_string(l),
                          [=] { return batched_case(swa, storage, b, w, l); },
                          {{"format", format}});
            }
            for (const int w : {1, 7})
                suite.add(name + "/graph/" + std::to_string(w),
                          [=] { return graph_case(swa, storage, w); }, {{"format", format}});
            for (const int w : {1, 3})
                suite.add(name + "/graph/bucket_patch/" + std::to_string(w),
                          [=] { return graph_bucket_patch(swa, storage, w); },
                          {{"format", format}});
            suite.add(name + "/direct", [=] { return batched_case(swa, storage, 97, 3, 17); },
                      {{"format", format}});
            if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
                suite.add(name + "/minimum_ue8m0_scale",
                          [=] {
                              int errors = 0;
                              for (int width : {1, 3}) {
                                  Fixture f(swa, storage, 1, width, 37, 8, {}, true);
                                  std::iota(f.positions.begin(), f.positions.end(), 37 - width);
                                  f.upload(1);
                                  f.launch();
                                  errors += f.check("minimum UE8M0 scale");
                              }
                              return errors;
                          },
                          {{"format", format}});
            }
            if (!swa) {
                if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0)
                    suite.add(name + "/graph/split_replay", full_fp8_split_replay,
                              {{"format", format}});
                suite.add(name + "/largest_bucket",
                          [=] {
                              return batched_case(false, storage, 1, 1,
                                                  context_bucket_tokens(ContextBucket::M1));
                          },
                          {{"format", format}});
            }
            if (swa) {
                if (storage == KvCacheStorage::BFloat16)
                    suite.add(name + "/direct_decode_replay", swa_direct_decode_replay,
                              {{"format", format}});
                suite.add(name + "/expired_slot", [=] { return expired_slot(storage); },
                          {{"format", format}});
                if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0)
                    suite.add(name + "/odd_scale_stride", fp8_odd_scale_stride,
                              {{"format", format}});
                suite.add(name + "/chunk/graph_decode",
                          [=] { return graph_case(true, storage, 1, true); }, {{"format", format}});
                for (const auto [width, start] :
                     {std::pair{2, 1023}, {8, 1020}, {8, 0}, {1, 0}, {1, 1024}, {3, INT32_MAX - 3}})
                    suite.add(
                        name + "/chunk/" + std::to_string(width) + "/" + std::to_string(start),
                        [=] { return chunk_batched(storage, width, start); }, {{"format", format}});
                suite.add(name + "/chunk/rejected", [=] { return reject_invalid_chunk(storage); },
                          {{"format", format}});
                suite.add(name + "/chunk/fresh4096",
                          [=] { return ragged_case(true, storage, {4096}); }, {{"format", format}});
                suite.add(name + "/chunk/continuation",
                          [=] { return ragged_case(true, storage, {1024, 1500}, {1024, 1020}); },
                          {{"format", format}});
            }
            if (!swa && storage == KvCacheStorage::BFloat16) {
                suite.add(name + "/ragged/position_replay", bf16_ragged_position_replay,
                          {{"format", format}});
                suite.add(name + "/weak_tail", [] { return bf16_weak_tail(131072, 1, 204.0f); },
                          {{"format", format}});
                suite.add(name + "/weak_tail_after_strong_prefix",
                          [] { return bf16_weak_tail(65536, 32768, 430.0f); },
                          {{"format", format}});
            }
            suite.add(name + "/ragged/single", [=] { return ragged_case(swa, storage, {65}); },
                      {{"format", format}});
            suite.add(name + "/ragged/mixed",
                      [=] { return ragged_case(swa, storage, {1, 17, 65, 313, 977, 1025, 2053}); },
                      {{"format", format}});
        }
    }
    return suite.main(argc, argv);
}
