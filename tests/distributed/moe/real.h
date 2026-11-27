#pragma once

#include "models/common/moe/exchange.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "distributed/bootstrap.h"

namespace ninfer::test::exchange {
void real_layer(distributed::WorkerConnection&, const models::moe::ExpertExchangeStage&,
                const models::qwen4_exp::RankLoadRequest&, const artifact::Reader&,
                std::shared_ptr<const models::qwen4_exp::ModelDefinition>);
}
