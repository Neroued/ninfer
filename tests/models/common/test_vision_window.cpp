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
