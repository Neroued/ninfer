#pragma once
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5 {

// The common request controller is instantiated once for this concrete model implementation.
struct RuntimeTypes {
    using Frontend                   = qwen3_5::Frontend;
    using PreparedPrompt             = qwen3_5::PreparedPrompt;
    using OutputSession              = qwen3_5::OutputSession;
    using PublishedOutput            = qwen3_5::PublishedOutput;
    using SequencePlanner            = qwen3_5::SequencePlanner;
    using SequencePlan               = qwen3_5::SequencePlan;
    using RequestBasePlan            = qwen3_5::RequestBasePlan;
    using PersistentBackfillProof    = qwen3_5::PersistentBackfillProof;
    using SequenceHandle             = qwen3_5::SequenceHandle;
    using MaterializationResult      = qwen3_5::MaterializationResult;
    using HybridAdmissionQuote       = qwen3_5::HybridAdmissionQuote;
    using ContextTransactionProgress = qwen3_5::ContextTransactionProgress;
    using PendingBatch               = qwen3_5::PendingBatch;
    using StartResult                = qwen3_5::StartResult;
    using PrefillProgress            = qwen3_5::PrefillProgress;
    using CommitResult               = qwen3_5::CommitResult;
    using DiscardResult              = qwen3_5::DiscardResult;
    using FinishResult               = qwen3_5::FinishResult;
    using AbortResult                = qwen3_5::AbortResult;
    using Program                    = qwen3_5::Program;
};

} // namespace ninfer::models::qwen3_5
