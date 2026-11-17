# Verification status and limitations

## Completed software checks

- Original paired raw recordings from **five anonymized trials** were replayed with the restored Phase 16 + CRP pipeline. A total of **53** FPA candidate events had identical ordering, timestamps, FPA values and A/B/C classifications compared with the untouched Phase 16 baseline.
- CRP produced exploratory output for all five recordings; it was **not** independently validated against motion capture or annotated pitch phase.
- The source package includes automated synthetic/regression tests for both platform folders. Run them locally or through GitHub Actions; passing tests demonstrates code correctness on test cases, not clinical accuracy.
- The old differential swing/RMS experiment was intentionally **excluded** from final code.

## FPA versus CRP ground truth

Paper/painted footprint orientations, measured relative to the marked walking line, serve as an external FPA reference but depend on mark quality, protractor measurement and the participant following the straight line. Footprints do **not** provide time-resolved CRP ground truth. A/B/C contact detections are internal IMU results and are not independent event annotations.

The five trials contained roughly 6–9 seconds of usable CRP walking sections each. The resulting foot-pitch phase differences varied substantially, but participants must **not** be clinically ranked by them. The chosen 0.35–1.25 Hz CRP band, Hilbert edges, short bouts and lack of ground truth limit confidence.

## Before clinical claims or a deployment claim

1. Check CRP filter-band and segmentation-window sensitivity on existing recordings.
2. Plot CRP circularly or show deviation from ±180°, avoiding fake wrap jumps.
3. Verify Pi 5 acquisition, dual-port mappings, browser behavior, and sample-gap rates with final hardware.
4. Independently validate temporal gait events/CRP against synchronized reference measurements if medical accuracy is to be claimed.

This software is a research prototype and is not intended for diagnosis or patient treatment decisions.
