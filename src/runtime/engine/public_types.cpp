#include "ninfer/types.h"

#include <utility>

namespace ninfer {

CancellationView::CancellationView(std::function<bool()> requested)
    : requested_(std::move(requested)) {}

bool CancellationView::requested() const { return requested_ && requested_(); }

} // namespace ninfer
