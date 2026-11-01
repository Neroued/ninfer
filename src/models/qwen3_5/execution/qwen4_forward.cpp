// The Text forward's two instantiations: execution (WorkspaceArena) and startup workspace sizing
// (WorkspaceLayoutBuilder). Its callers include only the declarations (qwen4_forward.h), so the
// member definitions compile here once.
#include "models/qwen3_5/execution/qwen4_forward.h"

#include "models/qwen3_5/execution/qwen4_forward/attention.h"
#include "models/qwen3_5/execution/qwen4_forward/core.h"
#include "models/qwen3_5/execution/qwen4_forward/ffn.h"
#include "models/qwen3_5/execution/qwen4_forward/gdn.h"
#include "models/qwen3_5/execution/qwen4_forward/moe.h"
#include "models/qwen3_5/execution/qwen4_forward/mtp.h"
#include "models/qwen3_5/execution/qwen4_forward/plain.h"
#include "models/qwen3_5/execution/qwen4_forward/projection.h"
#include "models/qwen3_5/execution/qwen4_forward/qsa.h"
#include "models/qwen3_5/execution/qwen4_forward/residual.h"

namespace ninfer::models::qwen3_5::execution {

template class Qwen4Forward<WorkspaceArena>;
template class Qwen4Forward<WorkspaceLayoutBuilder>;

} // namespace ninfer::models::qwen3_5::execution
