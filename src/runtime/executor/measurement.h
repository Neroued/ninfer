#pragma once

#include <cstdint>

namespace ninfer::runtime {

enum class MeasurementAction : std::uint8_t { Reset, Begin, End, Cancel };

struct MeasurementCommand {
    MeasurementAction action;
    bool profile = false;
};

// Used on the rank's CUDA thread, after earlier execution and its continuations retire.
class RankMeasurement {
public:
    RankMeasurement() = default;
    ~RankMeasurement();

    RankMeasurement(const RankMeasurement&)            = delete;
    RankMeasurement& operator=(const RankMeasurement&) = delete;

    template <class Program>
    void apply(MeasurementCommand command, Program& program) {
        validate(command);
        switch (command.action) {
        case MeasurementAction::Reset:
            program.reset_measurement_peaks();
            break;
        case MeasurementAction::Begin:
            program.reset_measurement_peaks();
            begin(command.profile);
            break;
        case MeasurementAction::End:
            end();
            break;
        case MeasurementAction::Cancel:
            end();
            break;
        }
    }

    void cancel() noexcept;

private:
    void validate(MeasurementCommand command) const;
    void begin(bool profile);
    void end();

    bool active_    = false;
    bool profiling_ = false;
};

} // namespace ninfer::runtime
