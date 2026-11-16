#include "models/common/vision/window.h"
#include <iostream>
#include <stdexcept>

namespace v = ninfer::models::vision;
namespace r = ninfer::runtime;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        using ninfer::models::frontend::VisionItem;
        std::vector<VisionItem> media(2);
        media[0].grid        = {1, 4, 4};
        media[0].patch_count = 16;
        media[0].token_spans = {{2, 4}};
        media[1].grid        = {2, 2, 4};
        media[1].patch_count = 16;
        media[1].token_spans = {{10, 2}, {18, 2}};
        const auto view      = [](const auto& item) -> const auto& { return item; };
        const auto all       = v::recovery_work(media, view, 0, 24, 24, 4);
        require(all.tokens == 24 && all.attention_pairs == 300 && all.vision_items == 3 &&
                    all.vision_patches == 32 && all.chunks == 2,
                "recovery must price image/frame units and capacity-limited chunks");
        const auto partial = v::recovery_work(media, view, 11, 19, 4, 4);
        require(partial.tokens == 8 && partial.attention_pairs == 124 &&
                    partial.vision_items == 2 && partial.vision_patches == 16 &&
                    partial.chunks == 2,
                "partial media replay lost a whole encoder unit");
        const auto gap = v::recovery_work(media, view, 12, 18, 4, 4);
        require(gap.vision_items == 0 && gap.vision_patches == 0 && gap.chunks == 2,
                "video text gap was charged for unreferenced frames");
        const auto empty = v::recovery_work(media, view, 11, 11, 4, 4);
        require(!empty.tokens && !empty.chunks && !empty.vision_items,
                "empty recovery range retained media work");
        std::vector<VisionItem> held(2);
        for (auto& item : held) {
            item.grid        = {1, 4, 4};
            item.patch_count = 16;
        }
        held[0].token_spans  = {{6, 4}};
        held[1].token_spans  = {{13, 4}};
        const auto held_work = v::recovery_work(held, view, 0, 24, 8, 8);
        require(held_work.chunks == 4 && held_work.vision_items == 2 &&
                    held_work.vision_patches == 32,
                "a held window must stop the next chunk before an unencoded frame");
        const std::vector<v::Unit> held_units{{0, 0, 0, 16, 6, 4, 1}, {1, 0, 0, 16, 13, 4, 1}};
        v::Window held_window;
        const r::ExecutionMember held_owner{{3, 1}, {{0}, 1}};
        std::vector<std::uint32_t> boundaries;
        for (std::uint32_t begin = 0; begin < 24;) {
            const auto end =
                held_window.limit(held_owner, held_units, begin, std::min(24U, begin + 8), 8);
            const auto planned =
                held_window.current() ? nullptr : v::plan_window(held_units, begin, end, 8, 1);
            if (planned || held_window.current()) {
                const auto use = held_window.reserve(held_owner, planned);
                const r::ExecutionTicket ticket{.epoch = {1}, .id = boundaries.size() + 1};
                held_window.submitted(use.id, ticket);
                held_window.consumed(use.id, end >= use.plan->consume_end);
                held_window.retired(ticket);
            }
            boundaries.push_back(end);
            begin = end;
        }
        require(
            boundaries == std::vector<std::uint32_t>{8, 13, 21, 24} &&
                held_work.chunks == boundaries.size(),
            "reference chunk quotation disagrees with Native window publication and retirement");
        const std::vector<v::Unit> units{
            {0, 0, 0, 8, 3, 2, 8}, {1, 0, 0, 12, 7, 3, 20}, {2, 0, 0, 12, 14, 3, 20}};
        v::validate_units(units, 5, 20);
        const auto plan = v::plan_window(units, 0, 20, 5, 2);
        require(plan->features == 5 && plan->consume_end == 10 && plan->outputs.size() == 2,
                "window must stop at whole-unit capacity");
        const auto slices = v::embedding_slices(*plan, 4, 5);
        require(slices.size() == 2 && slices[0].source_begin == 0 &&
                    slices[0].destination_columns == std::vector<std::int32_t>({3, 4}) &&
                    slices[1].source_begin == 4 &&
                    slices[1].destination_columns == std::vector<std::int32_t>{0},
                "rank-major output must scatter into chronological token positions");
        const r::ExecutionMember a{{1, 1}, {{0}, 1}}, b{{2, 1}, {{1}, 1}};
        v::Window window;
        require(window.limit(a, units, 0, 20, 5) == 14, "capacity limits the next prefill range");
        const auto first = window.reserve(a, plan);
        require(first.produce && !window.available(b), "window has one producer");
        require(window.limit(b, units, 0, 20, 5) == 3, "other requests may consume preceding text");
        window.submitted(first.id, r::ExecutionTicket{.epoch = {1}, .id = 1});
        window.consumed(first.id, false);
        window.retired(r::ExecutionTicket{.epoch = {1}, .id = 1});
        const auto next = window.reserve(a, {});
        require(!next.produce && next.id == first.id, "later chunks reuse encoded features");
        window.submitted(next.id, r::ExecutionTicket{.epoch = {1}, .id = 2});
        window.consumed(next.id, true);
        window.retired(r::ExecutionTicket{.epoch = {1}, .id = 1});
        require(!window.available(b), "logical completion must wait for its own reader");
        window.retired(r::ExecutionTicket{.epoch = {1}, .id = 2});
        require(window.available(b) && !window.owner(), "last reader releases the window");
        const auto cancelled = window.reserve(b, plan);
        window.submitted(cancelled.id, r::ExecutionTicket{.epoch = {1}, .id = 3});
        window.revoke(b);
        require(!window.available(a), "cancellation must retain in-flight output");
        window.retired(r::ExecutionTicket{.epoch = {1}, .id = 3});
        require(window.available(a), "cancelled reader releases on retirement");
        (void)window.reserve(a, plan);
        window.revoke(a);
        require(window.available(b), "unused reservation can be released immediately");
        std::cout << "Vision window capacity, mapping and retirement passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
