#pragma once

namespace ninfer::ops {

/**
 * Multiprocessor count of the active CUDA device, queried once and cached.
 *
 * Persistent-grid launchers size one resident wave from this value. Sizing from
 * a hardcoded reference-part count leaves multiprocessors idle on a device with
 * a wider die (or oversubscribes a narrower one); both are sm_120a parts and
 * differ only in enabled SM count.
 *
 * Returns the reference RTX 5090 count if the device query fails, so a launcher
 * always receives a positive, usable value.
 */
int device_sm_count();

} // namespace ninfer::ops
