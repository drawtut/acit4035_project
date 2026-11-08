ACIT4035 | Dual-IMU scheduler fix | 2026-11-08
===============================================

WHY THIS PATCH
--------------
Your simultaneous 200-Hz 'precise' test achieved only 112 Hz left and
116 Hz right, even though 'max' reached 254 Hz left and 254 Hz right.
The original `wait_until` contained multi-millisecond Python busy-spin
loops (`pass`) that contend for the GIL between two acquisition threads.
This patch replaces that busy-spin with cooperative yielding and asks
Windows for 1-ms timer resolution, when available.

FILES
-----
imu_acquisition.py - UPDATED scheduling; same CLI, same output CSV columns.
fpa_reader.py     - Your current uploaded version, UNCHANGED, included to
                    make this a complete matched pair. It imports the new
                    `wait_until` automatically.
fpa_estimation.py - Your current uploaded version, UNCHANGED. The FPA
                    mathematics are still EXPERIMENTAL and NOT VALIDATED.

INSTALLATION
------------
1. Stop any running fpa_reader.py or imu_acquisition.py process.
2. Make backups of the current scripts.
3. Copy the three scripts into:
   C:\Users\abdis\Documents\ACIT4035_project
4. Keep imu_reader.py and sensor_config.py as they currently are.
5. Keep both IMUs stationary, same orientation, on separate USB-RS485 ports.

BENCHMARK COMMAND
-----------------
cd C:\Users\abdis\Documents\ACIT4035_project
py imu_acquisition.py --sensor both --left COM5 --right COM4 --baud 230400 --poll-hz 200 --duration 30 --data-only --timing precise --label dual_scheduler_fix_200hz

A GOOD RESULT: ~200 saved rows/s per sensor, 0 read errors, 0 queue drops,
~5 ms median intervals, and p95 gaps close to 5 ms (some jitter expected).
If below target, send the terminal output plus summary.json. Don't infer
200 unique IMU updates/s from 200 successful polls/s; also check repeated
motion samples on a movement trial.

NEXT: FPA READER TWO-IMU TEST
-----------------------------
Once the acquisition benchmark is satisfactory:
py fpa_reader.py --sensor both --left COM5 --right COM4 --baud 230400 --poll-hz 200 --label bilateral_stationary

IMPORTANT
---------
- The FPA reader shows reference-relative fused yaw, NOT measured FPA.
- No smoothing is applied to raw measurements.
- Host-side midpoint timestamps do not guarantee hardware synchronization.
- The original script and raw recording CSV formats are preserved.
- The synthetic test of two threads is NOT a replacement for testing with
  your actual two WT901C485-E sensors and Windows USB-RS485 adapters.
