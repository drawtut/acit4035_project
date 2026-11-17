# Which files to run and why

This guide covers **workflow files**, not every helper module. Run commands from inside the relevant platform folder.

## 1. Explore signals with Filter Studio (Windows)

`python fpa_filter_studio.py`

Load left/right raw CSV files; optionally open a locally stored reference video. Inspect the accelerometer impact signal, gyro movement, stable sections and the A/B/C candidate steps. The original reference movies/recordings are **not shipped** in this repository.

Use `filterstudio_settings/best_exploratory_filter_profile.json` as a historical exploration preset. Production parameters are held in `fpa_frozen_profile.py`; manually changing a Filter Studio slider does not change published validation results or the fixed production configuration.

## 2. Evaluate existing sessions offline

- Use `python run_reference_line_replay.py SESSION.zip --aligned-start --output reference_line_replay` for recorded ZIPs from the taped 0° protocol.
- Or use `python run_offline_stage4.py --left LEFT.csv --right RIGHT.csv --out offline_stage4_results` for extracted paired CSVs.
- Inspect `reference_line_fpa.csv`, Stage 4 FPA outputs, `bilateral_continuous_relative_phase.csv` and its summary JSON. A/B/C detector confidence must be preserved; C is review-only.
- The processing chain is `fpa_unified_pipeline.py`, using `fpa_phase13_detector.py`, `fpa_raw6_orientation.py`, `fpa_reference_line.py`, `fpa_synchronous.py` and `fpa_continuous_relative_phase.py`.

## 3. Run the Windows hardware dashboard

`python online_unified/fpa_online.py --left COM5 --right COM4 --baud 230400 --hz 150 --timing precise --host 127.0.0.1 --port 8000`

Open <http://127.0.0.1:8000>. Check physical left/right device mappings and supply the five-second taped-aligned stationary period before walking. Phase/CRP previews may change until finalization.

## 4. Run the Pi dashboard

Copy the entire `raspberry_pi/` folder to a Raspberry Pi 5, run `bash setup_pi.sh`, then `./start_pi.sh LEFT_PORT RIGHT_PORT`. Device access may require membership in `dialout` and a new login. Use the URL printed by the script from a phone/laptop on the same network. Do not assume port numbers identify the same foot each reboot.

## 5. Check regressions and hardware settings

`python -m unittest discover -p 'test_*.py' -v` runs synthetic and regression tests. The optional `windows/sensor_config.py` inspects/changes Modbus settings; **hardware writes require an explicit --yes**. Use `python sensor_config.py --help` before configuring devices. `imu_benchmark.py` measures actual serial rates, not theoretical sensor rates.

## Historical vs current components

Early trajectory integration/ZUPT, displacement-based FPA and the prior rotational-intensity swing differential were research experiments. **They are not the final FPA or CRP production algorithms.** Only the clean Phase 16 + CRP implementation is included in the current run path. Older phase READMEs belong to the private archive, not the main README.
