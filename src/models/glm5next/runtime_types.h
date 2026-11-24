#pragma once

#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/program/program.h"
#include "models/glm5next/session/session.h"

namespace ninfer::models::glm5next {

struct RuntimeTypes {
    using ModelDefinition   = glm5next::ModelDefinition;
    using ModelShard        = glm5next::ModelShard;
    using Parameters        = execution::Parameters;
    using Profile           = execution::ModelProfile;
    using Frontend          = glm5next::Frontend;
    using PreparedPrompt    = glm5next::PreparedPrompt;
    using OutputSession     = models::frontend::OutputSession;
    using PublishedOutput   = models::frontend::PublishedOutput;
    using SequencePlan      = ExecutionPlan;
    using SessionPlan       = glm5next::SessionPlan;
    using ExecutionTypes    = glm5next::ExecutionTypes;
    using RankProgram       = glm5next::RankProgram;
    using Session           = glm5next::Session;
    using PreparedSession   = glm5next::PreparedSession;
    using ContextDemand     = glm5next::ContextDemand;
    using ContextAssessment = glm5next::ContextAssessment;
    using ContextReleaseSet = glm5next::ContextReleaseSet;
    using RequestBasePlan   = glm5next::RequestBasePlan;
    using SequenceHandle    = glm5next::SequenceHandle;
    using CheckpointHandle  = glm5next::CheckpointHandle;
    using CheckpointSummary = glm5next::CheckpointSummary;
    using SourceCandidate   = glm5next::SourceCandidate;
    using ResumeState       = glm5next::ResumeState;
    using ExecutionUnit     = glm5next::ExecutionUnit;
    using ExecutionUnitKind = glm5next::ExecutionUnitKind;
    using ContextProgress   = glm5next::ContextProgress;
    using PendingBatch      = glm5next::PendingBatch;
    using PrefillProgress   = glm5next::PrefillProgress;
    using ReplayProgress    = glm5next::ReplayProgress;
    using ControlComplete   = glm5next::ControlComplete;
    using ScoreComplete     = glm5next::ScoreComplete;
    using SessionEvent      = glm5next::SessionEvent;
    using CommitResult      = glm5next::CommitResult;
    using DiscardResult     = glm5next::DiscardResult;
    using FinishResult      = glm5next::FinishResult;
    using AbortResult       = glm5next::AbortResult;
    using CacheSessionKey   = models::frontend::PreparedSessionKey;
};

} // namespace ninfer::models::glm5next
