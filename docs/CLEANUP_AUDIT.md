# Source tree cleanup audit

The uploaded project checkout contained approximately **1,383 ZIP entries** and **277 MiB of compressed content**, including an embedded `.git` history, historic development branches copied into folders, videos, dozens of experimental outputs and original trials. Its root `fpa_unified_pipeline.py` was an older displacement/landing implementation, not the final CRP pipeline.

The cleanup **does not delete or modify that upload**. A compact source-only repository was instead reconstructed using the most recent verified **Phase 16 + CRP** release for Windows and Raspberry Pi, plus a small number of useful sensor/filter diagnostics from the uploaded project.

## Included

- Windows project (offline Filter Studio, raw-six-axis step detection/FPA/CRP and web dashboard)
- Self-contained Pi 5 project (same core algorithms, web dashboard, setup/start scripts)
- Frozen, separately determined left/right calibration and project configuration
- Optional read-only hardware diagnostics and saved exploratory filter preset
- All regression test modules originally distributed with the current build
- README, implementation notes, privacy/data handling guidance and GitHub Actions tests

## Intentionally excluded from GitHub source

- Embedded `.git` files and all Git object history; use your existing clone to preserve remote identity/history
- Original participant sessions/CSV and trial ZIPs, recordings, private or identifiable videos, photos and audio
- Generated figures, intermediate output directories and test logs
- Obsolete GUI/backend copies (`app2`, `online2`, older `online detector`) and ZUPT/displacement prototypes
- Previous Phase 17 gyro-RMS swing differential (explicitly rejected)
- Old chronological README patches and obsolete checkbox instructions

## Source structure decisions

Two platform folders remain self-contained and duplicate the current processing code. This is **deliberate**: rather than risk breaking paths before the final submission, both existing, tested implementations are packaged unchanged. Platform parity is checked by `python scripts/verify_repo.py`.

## Important Git/history finding

The uploaded Git index contained **31 tracked paths**, including at least one video and several generated plots. The uploaded `.git/objects` directory also included a single very large object. `.gitignore` cannot erase such items from the *history*. Removing a tracked asset from the newest commit does not remove the blob from older commits. Check the existing public/private remote and follow approved research data retention/publishing rules before making `main` public. No history rewrite or force push is performed by this package.
