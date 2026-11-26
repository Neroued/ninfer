#pragma once

#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/program/program.h"
#include "models/inkling2/session/session.h"

namespace ninfer::models::inkling2 {

struct RuntimeTypes {
    using ModelDefinition   = inkling2::ModelDefinition;
    using ModelShard        = inkling2::ModelShard;
    using Parameters        = execution::Parameters;
    using Profile           = execution::ModelProfile;
    using Frontend          = inkling2::Frontend;
    using PreparedPrompt    = inkling2::PreparedPrompt;
    using OutputSession     = models::frontend::OutputSession;
    using PublishedOutput   = models::frontend::PublishedOutput;
    using SequencePlan      = ExecutionPlan;
    using SessionPlan       = inkling2::SessionPlan;
    using ExecutionTypes    = inkling2::ExecutionTypes;
    using RankProgram       = inkling2::RankProgram;
    using Session           = inkling2::Session;
    using PreparedSession   = inkling2::PreparedSession;
    using ContextDemand     = inkling2::ContextDemand;
    using ContextAssessment = inkling2::ContextAssessment;
    using ContextReleaseSet = inkling2::ContextReleaseSet;
    using RequestBasePlan   = inkling2::RequestBasePlan;
    using SequenceHandle    = inkling2::SequenceHandle;
    using CheckpointHandle  = inkling2::CheckpointHandle;
    using CheckpointSummary = inkling2::CheckpointSummary;
    using SourceCandidate   = inkling2::SourceCandidate;
    using ResumeState       = inkling2::ResumeState;
    using ExecutionUnit     = inkling2::ExecutionUnit;
    using ExecutionUnitKind = inkling2::ExecutionUnitKind;
    using ContextProgress   = inkling2::ContextProgress;
    using PendingBatch      = inkling2::PendingBatch;
    using PrefillProgress   = inkling2::PrefillProgress;
    using ReplayProgress    = inkling2::ReplayProgress;
    using ControlComplete   = inkling2::ControlComplete;
    using ScoreComplete     = inkling2::ScoreComplete;
    using SessionEvent      = inkling2::SessionEvent;
    using CommitResult      = inkling2::CommitResult;
    using DiscardResult     = inkling2::DiscardResult;
    using FinishResult      = inkling2::FinishResult;
    using AbortResult       = inkling2::AbortResult;
    using CacheSessionKey   = models::frontend::PreparedSessionKey;
};

} // namespace ninfer::models::inkling2
