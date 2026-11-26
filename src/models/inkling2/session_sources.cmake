target_sources(ninfer_model_control PRIVATE
  inkling2/work.cpp
  inkling2/parallel/preparation.cpp
  inkling2/session/planning.cpp
  inkling2/session/session.cpp
  inkling2/session/request.cpp
  inkling2/session/sequence.cpp
  inkling2/session/binding.cpp
  inkling2/session/checkpoints.cpp
  inkling2/session/context.cpp
  inkling2/session/execution.cpp
  inkling2/session/commit.cpp
  inkling2/session/pause.cpp
  inkling2/session/reclaim.cpp
  inkling2/session/scoring.cpp)
