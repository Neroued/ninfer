/*!
 * \file xgrammar/support/work_budget.h
 * \brief Cooperative deadline for the grammar work of one thread.
 *
 * ninfer/capy-flash addition (structured output limits): a caller that must bound the work of a
 * compilation, a token-mask fill or an accept installs a deadline for the current thread; the
 * parser (and the schema normalization and conversion) charges units of work and the clocks are
 * read every kCheckInterval units. When a deadline has passed, WorkBudgetExceeded is thrown from
 * inside the work. A parser (and the matcher that owns it) is then in an unspecified state and
 * must not be used again, except for destruction. Without an installed deadline the charge is a
 * thread-local load and a branch.
 *
 * A deadline has a wall-clock part and, optionally, a CPU-time part measured on the calling
 * thread (CLOCK_THREAD_CPUTIME_ID where available): the matcher budget is CPU time, so a busy
 * machine does not make a legitimate grammar look expensive, with the wall clock as a wider
 * safety net.
 */

#ifndef XGRAMMAR_SUPPORT_WORK_BUDGET_H_
#define XGRAMMAR_SUPPORT_WORK_BUDGET_H_

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <time.h>
#endif

namespace xgrammar {

class WorkBudgetExceeded final : public std::runtime_error {
 public:
  WorkBudgetExceeded() : std::runtime_error("grammar work exceeded its time budget") {}
};

/*! \brief CPU time consumed by the calling thread (wall time where not available). */
inline std::chrono::nanoseconds ThreadCpuTime() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
  timespec now{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
  return std::chrono::seconds(now.tv_sec) + std::chrono::nanoseconds(now.tv_nsec);
#else
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
  );
#endif
}

class WorkDeadline {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::int32_t kCheckInterval = 1024;

  explicit WorkDeadline(
      Clock::time_point wall, std::optional<std::chrono::nanoseconds> cpu = std::nullopt
  )
      : wall_(wall), cpu_(cpu) {}

  void Charge(std::int32_t units) {
    countdown_ -= units;
    if (countdown_ > 0) return;
    countdown_ = kCheckInterval;
    Check();
  }

  /*! \brief Throws if a deadline has passed (also used at phase boundaries outside the parser). */
  void Check() const {
    if (Clock::now() >= wall_) throw WorkBudgetExceeded();
    if (cpu_ && ThreadCpuTime() >= *cpu_) throw WorkBudgetExceeded();
  }

  Clock::time_point WallDeadline() const { return wall_; }

 private:
  Clock::time_point wall_;
  std::optional<std::chrono::nanoseconds> cpu_;  // absolute ThreadCpuTime() limit
  std::int32_t countdown_ = kCheckInterval;
};

inline thread_local WorkDeadline* current_work_deadline = nullptr;

/*! \brief One unit is one processed parser state; heavier steps charge more. */
inline void ChargeWork(std::int32_t units = 1) {
  if (WorkDeadline* deadline = current_work_deadline) deadline->Charge(units);
}

/*! \brief Installs a deadline for the current thread for the lifetime of the scope. */
class ScopedWorkDeadline {
 public:
  explicit ScopedWorkDeadline(
      WorkDeadline::Clock::time_point wall,
      std::optional<std::chrono::nanoseconds> cpu = std::nullopt
  )
      : deadline_(wall, cpu), previous_(current_work_deadline) {
    current_work_deadline = &deadline_;
  }
  ~ScopedWorkDeadline() { current_work_deadline = previous_; }
  ScopedWorkDeadline(const ScopedWorkDeadline&) = delete;
  ScopedWorkDeadline& operator=(const ScopedWorkDeadline&) = delete;

  void Check() const { deadline_.Check(); }

 private:
  WorkDeadline deadline_;
  WorkDeadline* previous_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_WORK_BUDGET_H_
