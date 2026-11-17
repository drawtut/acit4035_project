# Methodology and architecture

## Signal input and timing

Two identical shoe-dorsum mounted USB–RS485 IMUs, +Y toward toes and +Z upward, return measured accel XYZ (m/s²) and gyro XYZ (degrees/s). The system does not use on-device magnetometer/Euler orientations. Device polls are independently scheduled; host monotonic request/response midpoint timestamps share a clock but do **not** imply hardware-synchronized samples. Missing data are rejected when original sample distance or bracket gap exceeds the limits used by the synchronous analysis.

## Orientation and FPA

The orientation processor estimates gyroscope bias from an initial stationary segment (robust median), low-pass filters gyro at 10 Hz and accelerometer at 5 Hz, integrates angular velocity as a quaternion, and uses quality-gated gravity/tilt correction. A separate functional calibration provides a shoe-forward unit vector for EACH sensor. Quaternion-rotate it, project into the horizontal plane, and compare landing orientation to the initial taped 0° forward reference. FPA is signed to make toe-out positive on either foot. This is **not** a double-integration/ZUPT walking-displacement estimate.

The impact detector uses 4th-order Butterworth band-pass 11–24 Hz on acceleration Z and a 75 ms RMS envelope with 1.0 m/s² prominence, combined with gyroscope swing and stance evidence. Candidate A is confirmed impact+swing+stance; B is swing+stance without clear impact; C is weaker, review-only evidence. The production profile remains frozen.

## Synchronous differential heading

With the same initial taped 0° heading, interpolate both independently estimated shoe headings to a common 40 ms clock grid, reject under-supported moments and calculate their signed relative heading. This complements per-foot FPA but is not its replacement, nor an absolute compass reference.

## Experimental Continuous Relative Phase (CRP)

Compute pitch of the calibrated shoe-forward axis from each internally estimated six-axis quaternion (vertical component of the forward vector; yaw-invariant), then interpolate left and right onto identical host timestamps every 20 ms. Only supported original raw timestamps are accepted (nearest ≤26 ms, original sample gap ≤60 ms). For each side, apply a **separate third-order 0.35–1.25 Hz Butterworth band-pass** and `sosfiltfilt` zero-phase smoothing; obtain Hilbert analytic phases φ_L(t), φ_R(t). One combined CRP(t) is the wrapped right-minus-left phase difference. Report circular mean, phase dispersion and absolute angular deviation from ±180° (anti-phase). Short/low-amplitude segments and filter edges are excluded.

CRP is **retrospective**, because `sosfiltfilt`/Hilbert use the full recorded window. It estimates coordination of foot-*pitch oscillations*, not anatomical swing duration, force/weight support or a clinical diagnosis. The 0.35–1.25 Hz choice is a research parameter, not a validated universal gait band.

## Processing layout

- `fpa_unified_pipeline.py` orchestrates unchanged FPA/detector and the supplemental bilateral measurements.
- `online_unified/acquisition_phase2.py` receives/timestamps two serial streams.
- `online_unified/processing_phase2.py` calls the same canonical engine for preview/finalization.
- `online_unified/fpa_online.py` exposes the browser API, start/stop workflow and downloadable session results.
- Windows `fpa_filter_studio.py` is an offline research GUI, not an alternate production engine.

The two platform folders deliberately duplicate identical processing files so either remains independently deployable. Changes must be tested together.
