#include "support.h"
#include "core/decode_graph.h"
#include "core/platform.h"

namespace ninfer::test::hc {
void add_fp8_mix_cases(OpSuite& suite) {
#if NINFER_TARGET_RTX5090
    namespace f = mix::fp8;

    struct Case {
        int m;
        bool final, graph;
        ops::LinearPolicy policy;
        bool mixed = false;
    };

    for (auto pattern :
         {f::Pattern::Dense, f::Pattern::Mapping, f::Pattern::Saturation, f::Pattern::Zero}) {
        std::vector<Case> cases;
        for (bool final : {false, true}) {
            if (pattern == f::Pattern::Dense) {
                for (int m : {1, 32, 33, 34, 64, 65, 66, 128, 129, 130, 320, 321, 322, 1024, 1025,
                              1026, 2048})
                    cases.push_back({m, final, m == 1 || m == 128 || m == 129 || m == 2048,
                                     ops::LinearPolicy::AllowA8});
                cases.push_back({1, final, false, ops::LinearPolicy::A16Only});
                cases.push_back({512, final, true, ops::LinearPolicy::A16Only});
                cases.push_back({48, final, true, ops::LinearPolicy::A16Only, true});
                cases.push_back({48, final, true, ops::LinearPolicy::AllowA8, true});
            } else {
                cases.push_back({3, final, false, ops::LinearPolicy::AllowA8});
                cases.push_back({33, final, true, ops::LinearPolicy::AllowA8});
            }
        }
        std::vector<OpCase> catalogue;
        for (auto c : cases)
            catalogue.push_back({std::string(c.final ? "final-mix" : "block-mix") + "/fp8/h5120/p" +
                                     std::to_string(int(pattern)) + "/m" + std::to_string(c.m) +
                                     (c.policy == ops::LinearPolicy::A16Only ? "/a16" : "/a8") +
                                     (c.mixed ? "/mixed-up" : "") + (c.graph ? "/graph" : "/eager"),
                                 {{"format", "fp8"},
                                  {"width", "5120"},
                                  {"execution", c.graph ? "graph" : "eager"}}});
        suite.add(std::move(catalogue), [=](auto selected) {
            f::Weights weights(pattern);
            const auto values    = f::input(pattern);
            const auto ref       = f::reference(weights, values);
            const bool any_graph = std::any_of(selected.begin(), selected.end(),
                                               [&](auto i) { return cases[i].graph; });
            auto next            = any_graph ? f::input(pattern, 1) : std::vector<float>{};
            auto next_ref        = any_graph ? f::reference(weights, next) : f::Ref{};
            auto e               = cuda_environment().context.execution_view();
            int failed           = 0;
            for (auto i : selected) {
                auto c = cases[i];
                f::Fixture fixture(weights, c.m, c.final, c.policy,
                                   c.mixed ? (c.policy == ops::LinearPolicy::A16Only
                                                  ? ops::LinearPolicy::AllowA8
                                                  : ops::LinearPolicy::A16Only)
                                           : c.policy);
                fixture.upload(values, e.stream());
                DecodeGraphDefinition definition;
                DecodeGraphExecutable graph;
                if (c.graph) {
                    definition.capture(e.stream(), [&] { fixture.invoke(e); });
                    graph.instantiate(definition);
                    graph.launch(e.stream());
                } else
                    fixture.invoke(e);
                cuda_synchronize(e.stream());
                std::string label =
                    "FP8 HC pattern=" + std::to_string(int(pattern)) + " M=" + std::to_string(c.m);
                failed += fixture.check(values, ref, label);
                if (c.graph) {
                    fixture.upload(next, e.stream());
                    graph.launch(e.stream());
                    cuda_synchronize(e.stream());
                    failed += fixture.check(next, next_ref, label + " changed input");
                }
            }
            failed += verify_exact("HC FP8 down preserved",
                                   from_device<std::uint8_t>(weights.dd.data(), weights.dd.bytes()),
                                   weights.down.payload);
            failed += verify_exact("HC FP8 up preserved",
                                   from_device<std::uint8_t>(weights.du.data(), weights.du.bytes()),
                                   weights.up.payload);
            failed += verify_exact(
                "HC BF16 injection preserved",
                from_device<std::uint16_t>(weights.dj.data(), weights.inject.bits.size()),
                weights.inject.bits);
            return failed + weights.dd.verify_guards("down") + weights.du.verify_guards("up") +
                   weights.dj.verify_guards("inject");
        });
    }
    suite.add("block-mix/fp8/h5120/contract",
              [] {
                  f::Weights w(f::Pattern::Mapping);
                  auto e = cuda_environment().context.execution_view();
                  f::Fixture v(w, 3, false, ops::LinearPolicy::AllowA8);
                  auto z = f::input(w.pattern);
                  v.upload(z, e.stream());
                  int failed        = 0;
                  const auto reject = [&](auto call) {
                      try {
                          call();
                          ++failed;
                      } catch (const std::invalid_argument&) {
                      } catch (const std::out_of_range&) {}
                  };
                  auto saved = w.bound;
                  w.bound.down.weight.n -= 1;
                  reject([&] { v.invoke(e); });
                  w.bound             = saved;
                  w.bound.down.policy = ops::LinearPolicy::AllowA4;
                  reject([&] { ops::hc_mix(v.z, w.bound, v.arena, v.x, v.j, e); });
                  w.bound                 = saved;
                  w.bound.injection.qtype = QType::FP8_E4M3FN_ROW_BF16;
                  reject([&] { v.invoke(e); });
                  w.bound  = saved;
                  auto old = v.x;
                  v.x.data = v.z.data;
                  reject([&] { v.invoke(e); });
                  v.x      = old;
                  old      = v.j;
                  v.j.data = v.x.data;
                  reject([&] { v.invoke(e); });
                  v.j = old;
                  old = v.z;
                  v.z.nb[1] += 16;
                  reject([&] { v.invoke(e); });
                  v.z        = old;
                  old        = v.z;
                  v.z.device = Device::cpu();
                  reject([&] { v.invoke(e); });
                  v.z = old;
                  MemoryArena too_small(v.scratch.span().subspan(0, v.scratch.bytes() - 1));
                  try {
                      ops::hc_mix(v.z, w.bound, too_small, v.x, v.j, e);
                      ++failed;
                  } catch (const std::bad_alloc&) {}
                  MemoryArena overlap(w.dd.span());
                  reject([&] { ops::hc_mix(v.z, w.bound, overlap, v.x, v.j, e); });
                  reject([&] { (void)ops::hc_mix_workspace_capacity_bytes(w.bound, 0, 3); });
                  const ops::HcFp8MixProfile p{
                      ops::linear_profile(w.bound.down),
                      ops::linear_profile(w.bound.up),
                      {QType::BF16, 4, f::K, ops::LinearPolicy::A16Only, QuantLayout::Contiguous}};
                  for (int m : {1, 32, 33, 64, 65, 320, 321, 1024, 1025, 2048})
                      if (ops::hc_mix_workspace_capacity_bytes(p, 1, m) <
                          ops::hc_mix_workspace_capacity_bytes(w.bound, m, m))
                          ++failed;
                  return failed;
              },
              {{"format", "fp8"}, {"width", "5120"}});
#endif
}
} // namespace ninfer::test::hc
