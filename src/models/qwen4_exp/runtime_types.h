#pragma once

#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/program/program.h"
#include "models/qwen4_exp/session/session.h"

namespace ninfer::models::qwen4_exp {

struct RuntimeTypes {
    using ModelDefinition   = qwen4_exp::ModelDefinition;
    using ModelShard        = qwen4_exp::ModelShard;
    using Parameters        = execution::Parameters;
    using Profile           = execution::ModelProfile;
    using Frontend          = qwen::Frontend;
    using PreparedPrompt    = qwen::PreparedPrompt;
    using OutputSession     = qwen::OutputSession;
    using PublishedOutput   = qwen::PublishedOutput;
    using SequencePlan      = ExecutionPlan;
    using SessionPlan       = qwen4_exp::SessionPlan;
    using ExecutionTypes    = qwen4_exp::ExecutionTypes;
    using RankProgram       = qwen4_exp::RankProgram;
    using Session           = qwen4_exp::Session;
    using PreparedSession   = qwen4_exp::PreparedSession;
    using ContextResources  = qwen4_exp::ContextResources;
    using RequestBasePlan   = qwen4_exp::RequestBasePlan;
    using SequenceHandle    = qwen4_exp::SequenceHandle;
    using CheckpointHandle  = qwen4_exp::CheckpointHandle;
    using CheckpointSummary = qwen4_exp::CheckpointSummary;
    using SourceCandidate   = qwen4_exp::SourceCandidate;
    using ResumeState       = qwen4_exp::ResumeState;
    using ExecutionUnit     = qwen4_exp::ExecutionUnit;
    using ExecutionUnitKind = qwen4_exp::ExecutionUnitKind;
    using ContextProgress   = qwen4_exp::ContextProgress;
    using PendingBatch      = qwen4_exp::PendingBatch;
    using PrefillProgress   = qwen4_exp::PrefillProgress;
    using ReplayProgress    = qwen4_exp::ReplayProgress;
    using ControlComplete   = qwen4_exp::ControlComplete;
    using ScoreComplete     = qwen4_exp::ScoreComplete;
    using SessionEvent      = qwen4_exp::SessionEvent;
    using CommitResult      = qwen4_exp::CommitResult;
    using DiscardResult     = qwen4_exp::DiscardResult;
    using FinishResult      = qwen4_exp::FinishResult;
    using AbortResult       = qwen4_exp::AbortResult;
    using CacheSessionKey   = qwen::PreparedSessionKey;
};

} // namespace ninfer::models::qwen4_exp
