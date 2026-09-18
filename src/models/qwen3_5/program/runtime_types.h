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
    using AdmissionCandidate         = qwen3_5::AdmissionCandidate;
    using ResourcePlan               = qwen3_5::ResourcePlan;
    using PersistentBackfillProof    = qwen3_5::PersistentBackfillProof;
    using SequenceHandle             = qwen3_5::SequenceHandle;
    using ContinuationHandle         = qwen3_5::ContinuationHandle;
    using SharedPrefixHandle         = qwen3_5::SharedPrefixHandle;
    using CaptureOffer               = qwen3_5::CaptureOffer;
    using CacheSessionKey            = qwen3_5::PreparedSessionKey;
    using ContinuationSummary        = qwen3_5::ContinuationSummary;
    using SharedPrefixSummary        = qwen3_5::SharedPrefixSummary;
    using PressurePlanningSession    = qwen3_5::PressurePlanningSession;
    using PressureTargetHandle       = qwen3_5::PressureTargetHandle;
    using AssessedPressureTarget     = qwen3_5::AssessedPressureTarget;
    using CapturePressurePlan        = qwen3_5::CapturePressurePlan;
    using MaterializationResult      = qwen3_5::MaterializationResult;
    using ContextTransactionProgress = qwen3_5::ContextTransactionProgress;
    using CaptureAssessment          = qwen3_5::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_5::ActiveCaptureResult;
    using PendingBatch               = qwen3_5::PendingBatch;
    using StartResult                = qwen3_5::StartResult;
    using PrefillProgress            = qwen3_5::PrefillProgress;
    using CommitResult               = qwen3_5::CommitResult;
    using DiscardResult              = qwen3_5::DiscardResult;
    using FinishResult               = qwen3_5::FinishResult;
    using AbortResult                = qwen3_5::AbortResult;
    using ReleaseResult              = qwen3_5::ReleaseResult;
    using Program                    = qwen3_5::Program;
};

} // namespace ninfer::models::qwen3_5
