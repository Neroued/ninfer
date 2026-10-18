#pragma once

#include "models/common/qwen/frontend/frontend.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/session/session.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5 {
struct RuntimeTypes {
    using ModelDefinition   = qwen3_5::ModelDefinition;
    using ModelShard        = qwen3_5::ModelShard;
    using Parameters        = execution::Parameters;
    using Profile           = execution::ModelProfile;
    using Frontend          = qwen::Frontend;
    using PreparedPrompt    = qwen::PreparedPrompt;
    using OutputSession     = qwen::OutputSession;
    using PublishedOutput   = qwen::PublishedOutput;
    using SequencePlanner   = qwen3_5::SequencePlanner;
    using SequencePlan      = qwen3_5::SequencePlan;
    using RequestBasePlan   = qwen3_5::RequestBasePlan;
    using SequenceHandle    = qwen3_5::SequenceHandle;
    using CheckpointHandle  = qwen3_5::CheckpointHandle;
    using CheckpointSummary = qwen3_5::CheckpointSummary;
    using SourceCandidate   = qwen3_5::SourceCandidate;
    using ResumeState       = qwen3_5::ResumeState;
    using ExecutionUnit     = qwen3_5::ExecutionUnit;
    using ExecutionUnitKind = qwen3_5::ExecutionUnitKind;
    using ContextProgress   = qwen3_5::ContextProgress;
    using PendingBatch      = qwen3_5::PendingBatch;
    using PrefillProgress   = qwen3_5::PrefillProgress;
    using ReplayProgress    = qwen3_5::ReplayProgress;
    using CommitResult      = qwen3_5::CommitResult;
    using DiscardResult     = qwen3_5::DiscardResult;
    using FinishResult      = qwen3_5::FinishResult;
    using AbortResult       = qwen3_5::AbortResult;
    using ExecutionTypes    = qwen3_5::ExecutionTypes;
    using SessionPlan       = qwen3_5::SessionPlan;
    using PreparedSession   = qwen3_5::PreparedSession;
    using RankProgram       = qwen3_5::RankProgram;
    using Session           = qwen3_5::Session;
    using SessionEvent      = qwen3_5::SessionEvent;
    using ControlComplete   = qwen3_5::ControlComplete;
    using ScoreComplete     = qwen3_5::ScoreComplete;
    using CacheSessionKey   = qwen::PreparedSessionKey;
};
} // namespace ninfer::models::qwen3_5
