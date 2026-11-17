# Development history (short overview)

**Final maintained code:** Frozen signed reference-line FPA and the Phase 13 A/B/C detector, the Phase 16 simultaneous inter-foot relative heading, and exploratory bilateral foot-pitch CRP. Earlier research experiments are preserved in private source archives; they are not installed or run from this clean repository.

| Research stage | Contribution / finding |
|---|---|
| Acquisition + mounts | Two wired dorsum-mounted IMUs, common host timestamps, per-sensor calibration and rate diagnostics |
| Filter Studio | Compared high/low-pass choices, impact RMS/peak settings, gyro motion and stance evidence |
| Timebase/stance | Regridding, quality gates and rejection of USB dropouts or asymmetric recording tails |
| Early FPA methods | Tested start/landing orientations, trajectory reconstruction, ZUPT and local path headings; displacement methods were drift-sensitive in short walking trials |
| Reference-line FPA | Adopted a 0° taped starting position with landing heading from raw six-axis orientation; fixed filters |
| Multievidence detector | Three confidence classes: A confirmed, B provisional, C review-only; candidate count must not be forced to match external footprint marks |
| Same-time relative heading | Calculates bilateral relative yaw from the two interpolated shoe directions (not pairwise completed steps) |
| Rotational-intensity experiment | Tested a gyro-RMS difference/alternation metric; abandoned because it discarded direction and was difficult to interpret clinically |
| Current CRP | Uses both synchronized foot-pitch oscillations, Hilbert phases and a circular right-minus-left phase relationship; an experimental coordination extension |

## Why keep the experimental stages separate?

The project was not developed in a single pass. Earlier approaches informed filter choices, calibration, detector evidence, stance gates and eventual selection of the fixed reference. **Final participant outcomes must be described as results of reprocessing using the final pipeline**, not as if every module was already present during initial data collection.
