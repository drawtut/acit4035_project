"""ACIT4035 Stage 2A: offline, reproducible IMU signal processing.

Input: raw FPA reader left_raw.csv or right_raw.csv (200 Hz recommended).
Output: separate processed CSV, stance/swing candidate CSV, plots, JSON report.
No sensor settings or source CSVs are modified.

Examples:
    py gait_processing.py --input "recordings\\fpa_trials\\TRIAL\\left_raw.csv"
    py gait_processing.py --input left_raw.csv --plot
    py gait_processing.py --input left_raw.csv --gyro-threshold 30 --accel-threshold 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

G = 9.80665
ACC = ('ax_mps2', 'ay_mps2', 'az_mps2')
GYRO = ('gx_dps', 'gy_dps', 'gz_dps')


def spans(mask: np.ndarray, t: np.ndarray, min_seconds: float = 0.0):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(i), int(j), float(t[i]), float(t[j-1])) for i,j in zip(starts, ends)
            if t[j-1] - t[i] >= min_seconds]


def analyze(path: Path, output: Path, args):
    df = pd.read_csv(path)
    for key in ('time_s', *ACC, *GYRO):
        if key not in df.columns:
            raise ValueError(f'Missing required column: {key}')
    t = df['time_s'].to_numpy(dtype=float)
    if len(t) < 50 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Require >= 50 rows with finite, strictly increasing time_s')
    accel = df.loc[:, ACC].to_numpy(dtype=float)
    gyro = df.loc[:, GYRO].to_numpy(dtype=float)
    if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)):
        raise ValueError('Sensor channels contain NaN or infinity')
    dt = np.diff(t)
    fs = float(1.0 / np.median(dt))
    if args.cutoff >= fs/2:
        raise ValueError(f'Low-pass cutoff must be below Nyquist ({fs/2:.1f} Hz)')

    # Bias estimation uses only samples showing low rotation in the first
    # specified interval. Fails explicitly if the foot is moving too much.
    cal_mask = (t <= t[0] + args.bias_window) & (np.linalg.norm(gyro, axis=1) < args.bias_max_gyro)
    if cal_mask.sum() < max(20, int(fs * 0.2)):
        raise ValueError('Insufficient stationary samples at recording start. '
                         'Start with >= 2 seconds motionless or adjust --bias-window.')
    bias = np.median(gyro[cal_mask], axis=0)
    gyro_corrected = gyro - bias

    # Filtering is an offline diagnostic step. Do NOT substitute a constant
    # gravity subtraction or integrate these signals into positions directly.
    sos = butter(4, args.cutoff, btype='lowpass', fs=fs, output='sos')
    a_f = sosfiltfilt(sos, accel, axis=0)
    w_f = sosfiltfilt(sos, gyro_corrected, axis=0)
    gyro_norm = np.linalg.norm(w_f, axis=1)
    accel_residual = np.abs(np.linalg.norm(a_f, axis=1) - G)
    # Windowed stationary scores smooth isolated peaks without using the
    # device Euler yaw or magnetic heading.
    window = max(3, round(args.score_window * fs))
    gyro_score = np.sqrt(uniform_filter1d(gyro_norm**2, size=window, mode='nearest'))
    accel_score = np.sqrt(uniform_filter1d(accel_residual**2, size=window, mode='nearest'))
    stance = (gyro_score < args.gyro_threshold) & (accel_score < args.accel_threshold)

    # Short non-stance gaps inside a stance get bridged; very short stance
    # islands are removed. These are heuristic candidate labels only.
    bridge = max(1, round(args.bridge_gap * fs))
    min_stance = max(1, round(args.min_stance * fs))
    stance = binary_closing(stance, structure=np.ones(bridge), border_value=0)
    stance = binary_opening(stance, structure=np.ones(min_stance), border_value=0)
    swing = ~stance
    stance_events = spans(stance, t, args.min_stance)
    swing_events = spans(swing, t, args.min_swing)

    result = pd.DataFrame({'time_s': t, 'gyro_norm_raw_dps': np.linalg.norm(gyro, axis=1),
                           'gyro_norm_filtered_dps': gyro_norm,
                           'accel_norm_raw_mps2': np.linalg.norm(accel, axis=1),
                           'accel_norm_filtered_mps2': np.linalg.norm(a_f, axis=1),
                           'gyro_stationary_score_dps': gyro_score,
                           'accel_stationary_score_mps2': accel_score,
                           'stance_candidate': stance.astype(int),
                           'swing_candidate': swing.astype(int)})
    for i, k in enumerate(GYRO):
        result[k.replace('_dps', '_bias_corrected_dps')] = gyro_corrected[:, i]
        result[k.replace('_dps', '_filtered_dps')] = w_f[:, i]
    for i, k in enumerate(ACC):
        result[k.replace('_mps2', '_filtered_mps2')] = a_f[:, i]

    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'processed_signals.csv', index=False)
    event_rows = ([{'event':'stance_candidate', 'start_s': s, 'end_s': e, 'duration_s':e-s} for _,_,s,e in stance_events]
                  + [{'event':'swing_candidate', 'start_s':s, 'end_s':e,'duration_s':e-s} for _,_,s,e in swing_events])
    pd.DataFrame(event_rows, columns=['event','start_s','end_s','duration_s']).sort_values('start_s').to_csv(
        output / 'candidate_events.csv', index=False)
    report = {
        'source':str(path), 'samples':len(df), 'duration_s':float(t[-1]-t[0]),
        'median_rate_hz':fs, 'mean_rate_hz':float((len(t)-1)/(t[-1]-t[0])),
        'p95_gap_ms':float(np.percentile(dt*1000, 95)), 'max_gap_ms':float(dt.max()*1000),
        'bias_samples':int(cal_mask.sum()),
        'gyro_bias_dps':dict(zip(GYRO, map(float, bias))),
        'filter':'4th order zero-phase Butterworth low-pass for exploratory event detection',
        'filter_cutoff_hz':args.cutoff,
        'stance_gyro_score_threshold_dps':args.gyro_threshold,
        'stance_accel_score_threshold_mps2':args.accel_threshold,
        'stance_candidate_intervals':len(stance_events),
        'swing_candidate_intervals':len(swing_events),
        'stance_candidate_percentage':float(stance.mean()*100),
        'limitations': ['These are heuristic stance/swing candidates, not validated foot contacts or physical step counts.',
                        'Bias estimate requires a stationary opening interval.',
                        'No gravity-compensated translation, ZUPT, functional foot-axis calibration, or actual FPA is computed.',
                        'Unmodified raw data remains in the input CSV.']
    }
    (output / 'processing_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    import matplotlib
    if not args.plot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for i,k in enumerate(GYRO):
        axs[0].plot(t, gyro[:,i], lw=.65, alpha=.55, label=f'{k} raw')
        axs[1].plot(t, w_f[:,i], lw=.9, label=f'{k} bias corrected + filtered')
    for i,k in enumerate(ACC):
        axs[2].plot(t, accel[:,i], lw=.65, alpha=.28, label=f'{k} raw')
        axs[2].plot(t, a_f[:,i], lw=1, label=f'{k} filtered')
    axs[3].plot(t, gyro_score, lw=1.1, label='Angular-rate RMS (filtered)')
    axs[3].axhline(args.gyro_threshold, color='black', lw=.8, ls='--', label='Gyro threshold')
    for ax in axs:
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', ncol=3, fontsize=8)
    for _,_,s,e in stance_events:
        axs[3].axvspan(s,e,color='tab:green',alpha=.22)
    for _,_,s,e in swing_events:
        axs[3].axvspan(s,e,color='tab:orange',alpha=.10)
    axs[0].set_ylabel('Gyro raw (°/s)')
    axs[1].set_ylabel('Gyro processed (°/s)')
    axs[2].set_ylabel('Acceleration (m/s²)')
    axs[3].set_ylabel('Angular rate (°/s)')
    axs[3].set_xlabel('Time since trial start (s)')
    fig.suptitle('ACIT4035 Stage 2A: raw / processed signals and candidate gait phases')
    fig.savefig(output / 'gait_processing_overview.png', dpi=160)
    if args.plot:
        plt.show()
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f'\nSaved analysis to: {output}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True, help='FPA reader left_raw.csv/right_raw.csv')
    p.add_argument('--out', type=Path, default=None, help='Output folder (default: next to raw CSV)')
    p.add_argument('--cutoff', type=float, default=15.0, help='Exploratory lowpass cutoff, Hz')
    p.add_argument('--bias-window', type=float, default=1.0, help='Initial stationary window, seconds')
    p.add_argument('--bias-max-gyro', type=float, default=8.0, help='Maximum gyro norm for bias samples, deg/s')
    p.add_argument('--gyro-threshold', type=float, default=35.0, help='Stance candidate gyro RMS limit, deg/s')
    p.add_argument('--accel-threshold', type=float, default=3.0, help='Stance candidate accel residual RMS limit, m/s²')
    p.add_argument('--score-window', type=float, default=0.045, help='RMS score window, seconds')
    p.add_argument('--bridge-gap', type=float, default=0.04, help='Bridge short gaps in candidate stance, seconds')
    p.add_argument('--min-stance', type=float, default=0.1, help='Minimum stance interval, seconds')
    p.add_argument('--min-swing', type=float, default=0.17, help='Minimum swing interval, seconds')
    p.add_argument('--plot', action='store_true', help='Show interactive plot after saving files')
    args = p.parse_args()
    if args.cutoff <= 0 or args.bias_window <= 0 or args.gyro_threshold <= 0 or args.accel_threshold <= 0:
        p.error('Cutoff, bias window and thresholds must be positive')
    if not args.input.is_file():
        p.error(f'File not found: {args.input}')
    out = args.out or args.input.parent / (args.input.stem + '_stage2a')
    analyze(args.input, out, args)

if __name__ == '__main__':
    main()
