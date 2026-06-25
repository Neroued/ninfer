// qus::model — Qwen3.6-27B model card: the hand-written static forward schedule (the compute
// graph). embed -> 64-layer loop (3:1 GDN/full dispatch) -> final norm -> lm_head, plus the
// prefill/decode drivers and KV/state lifecycle. Implements include/qus/model/model.h.
// See docs/qwen3.6-27b-architecture.md §9 and docs/design.md §5/§7.
#include "qus/model/model.h"
#include "qus/model/config.h"

// TODO(impl): issue L1 kernel calls in order for prefill and decode steps.
