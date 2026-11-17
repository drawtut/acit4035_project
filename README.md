# ACIT4035 — Bilateral Foot Progression Angle and Continuous Relative Phase

Research prototype for **signed foot progression angle (FPA)** at each detected landing and **exploratory bilateral Continuous Relative Phase (CRP)** from two foot-mounted WT901C485-E USB–RS485 IMUs. Runs offline on Windows and records/processes sessions through a Raspberry Pi 5 web dashboard.

The research implementation uses **six measured IMU channels only** (accelerometer XYZ, gyroscope XYZ). The onboard sensor Euler angles and magnetometer are **not used**. The shoes must begin aligned to the same physical straight reference line. Both sensors are time-aligned using shared host-clock timestamps; **they are not hardware-synchronized**.

> **Research software, not a medical device.** Footprint measurements can validate FPA in the recorded straight-line trials. CRP is an experimental foot-pitch phase relationship, **not** an independently validated clinical coordination score, a direct heel-strike/swing-duration measurement, or a diagnosis.

## Quick start

### Windows — offline data and Filter Studio

From `windows/` (Windows PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python fpa_filter_studio.py
```

To process an existing paired recording with the taped-reference protocol:

```powershell
python run_reference_line_replay.py "PATH_TO_SESSION.zip" --aligned-start --output reference_line_replay
```

Alternatively, for already extracted CSV files:

```powershell
python run_offline_stage4.py --left "PATH_TO_LEFT.csv" --right "PATH_TO_RIGHT.csv" --out offline_stage4_results
```

The generated Stage 4 files include per-foot FPA and `bilateral_continuous_relative_phase.csv` plus its JSON summary. CRP requires original **shared-host** timestamp data from both sides; if missing, it should report unavailable, not invent simultaneous samples.

To start the Windows web dashboard (hardware required):

```powershell
python online_unified/fpa_online.py --help
python online_unified/fpa_online.py --left COM5 --right COM4 --baud 230400 --hz 150 --timing precise --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> locally. Verify which COM port is connected to each foot before recording.

### Raspberry Pi 5 — live capture and browser dashboard

Copy the `raspberry_pi/` directory to the Pi, enter it, and run:

```bash
bash setup_pi.sh
./start_pi.sh /dev/ttyUSB0 /dev/ttyUSB1
```

The two device paths must refer to **different ports**, in left/right order. Prefer `/dev/serial/by-id/` persistent paths if available. Open the dashboard URL printed by the script on another device connected to the same network. **Actual sampling rate is measured, not guaranteed by the 150 Hz poll target.**

For a demo without connected hardware:

```bash
./start_demo.sh
```

### Run tests

Windows:

```powershell
cd windows
python -m unittest discover -p "test_*.py" -v
```

Raspberry Pi source (can also be tested on a desktop):

```bash
cd raspberry_pi
python -m unittest discover -p 'test_*.py' -v
```

## Repository layout

```text
.
├── windows/          # Windows offline Filter Studio, processing and web backend
├── raspberry_pi/     # Self-contained Pi 5 processing, dashboard and start scripts
├── docs/             # Architecture, development, validation and data-handling notes
├── .github/workflows/# Source regression tests on pushes/PRs
├── .gitignore
└── README.md
```

The two platform folders deliberately each contain the same frozen core processing implementation. This avoids platform-only path hacks and preserves the tested Windows/Pi behavior; **changes to shared algorithms should be made consistently in both copies and regression-tested**. We have not performed a risky module/package refactor as part of this cleanup.

## Key source files

| Component | Windows entry point / file | What it does |
|---|---|---|
| Filter development | `fpa_filter_studio.py` | Visual inspection and offline tuning |
| Saved profile | `fpa_frozen_profile.py` | Fixed production filter and detector settings |
| Footstep logic | `fpa_phase13_detector.py` | Confirmed A, provisional B and review-only C candidates |
| Orientation | `fpa_raw6_orientation.py` | Bias-corrected, gravity-leveled six-axis orientation |
| FPA | `fpa_reference_line.py` | Signed landing FPA vs initial taped-zero line |
| Synchronous heading | `fpa_synchronous.py` | Phase 16 same-timestamp inter-foot relative heading |
| Bilateral CRP | `fpa_continuous_relative_phase.py` | Band-passed foot pitch and Hilbert phase difference |
| Unified pipeline | `fpa_unified_pipeline.py` | Runs the shared processing and exports |
| Dashboard | `online_unified/fpa_online.py` | Capture, API, preview and result display |
| Offline replay | `run_reference_line_replay.py` | Reprocesses saved session ZIPs |

Details: [Project guide](docs/PROJECT_GUIDE.md), [Implementation](docs/IMPLEMENTATION.md), [Development history](docs/DEVELOPMENT.md), [Validation](docs/VALIDATION.md), [Data policy](docs/DATA_AND_PRIVACY.md).

## Scope and limitations

- Positive FPA means **toe-out for either foot**, negative means toe-in. FPA is a **separate per-foot** output.
- CRP is **one combined inter-foot phase signal**, at 20 ms common host timestamps; its Hilbert phase calculation is retrospective and non-causal. A plot jump at −180°/+180° is a wrap boundary, not necessarily a gait event.
- Physical taped-start alignment and functional per-foot calibration are required.
- Recorded-data replay and synthetic unit tests establish software behavior; independent clinical CRP ground truth and final Pi hardware testing are still required.
- Keep all identifiable trials, images, videos and original ZIPs **outside Git**. See [Data policy](docs/DATA_AND_PRIVACY.md).
- A license has **not** been selected. Add one only after agreeing to the terms for distributing the code.
