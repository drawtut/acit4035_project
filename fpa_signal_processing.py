#!/usr/bin/env python3
"""ACIT4035 bilateral FPA signal-processing / event-detection lab.

Offline diagnostic only: does not calculate FPA or identify ground-truth heel strikes.
Unmodified source CSVs remain untouched.

Usage:
  py fpa_signal_processing.py --left "path/to/left_raw.csv" --right "path/to/right_raw.csv"
  py fpa_signal_processing.py --left "..." --right "..." --cutoff 15 --gyro-threshold 50

Dependencies: numpy pandas scipy matplotlib
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d, binary_closing, binary_opening
import matplotlib.pyplot as plt

G = 9.80665
ACC = ['ax_mps2','ay_mps2','az_mps2']
GYR = ['gx_dps','gy_dps','gz_dps']


def signal_intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_foot(filename, side, args):
    df = pd.read_csv(filename)
    for c in ACC + GYR:
        if c not in df:
            raise ValueError(f'{side}: required column {c} missing in {filename}')
    tc = 'time_s' if 'time_s' in df else 'timestamp_midpoint_monotonic_s'
    if tc not in df:
        raise ValueError(f'{side}: expected time_s or timestamp_midpoint_monotonic_s')
    t = df[tc].to_numpy(dtype=float, copy=True)
    t -= t[0]
    accel = df[ACC].to_numpy(dtype=float)
    gyro = df[GYR].to_numpy(dtype=float)
    if len(t) < 300 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f'{side}: invalid or too-short timestamps')
    if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
        raise ValueError(f'{side}: non-finite accelerometer or gyroscope data')
    fs = 1 / np.median(np.diff(t))
    if not 40 < fs < 1000:
        raise ValueError(f'{side}: unexpected sampling rate {fs:.1f} Hz')
    initial = t < args.bias_seconds
    if np.sum(initial) < 30:
        raise ValueError(f'{side}: recording requires a stationary opening interval')
    bias = np.median(gyro[initial], axis=0)
    initial_gyro_dev = np.linalg.norm(gyro[initial] - bias, axis=1)
    if np.percentile(initial_gyro_dev, 90) > 10:
        print(f'WARNING: {side}: the first {args.bias_seconds:g} s may not be stationary; gyro bias is unreliable')
    gyro_corrected = gyro - bias

    def lowpass(x, cutoff):
        sos = butter(4, cutoff, btype='low', fs=fs, output='sos')
        return sosfiltfilt(sos, x, axis=0)

    filtered = {freq: (lowpass(accel, freq), lowpass(gyro_corrected, freq))
                for freq in (10, 15, 30) if freq < 0.42 * fs}
    if args.cutoff not in filtered:
        filtered[args.cutoff] = (lowpass(accel, args.cutoff), lowpass(gyro_corrected, args.cutoff))
    af, wf = filtered[args.cutoff]
    anorm = np.linalg.norm(af, axis=1)
    gnorm = np.linalg.norm(wf, axis=1)
    nwin = max(3, int(round(args.variance_window_ms * 0.001 * fs)) | 1)
    amean = uniform_filter1d(anorm, size=nwin, mode='nearest')
    avar = np.maximum(0., uniform_filter1d(anorm**2, size=nwin, mode='nearest') - amean**2)
    raw_candidates = ((gnorm < args.gyro_threshold) &
                      (np.abs(anorm - G) < args.gravity_tolerance) &
                      (avar < args.accel_variance))
    close_n = max(1, int(round(args.bridge_ms * 0.001 * fs)))
    min_n = max(2, int(round(args.min_stationary_ms * 0.001 * fs)))
    stationary = binary_closing(raw_candidates, structure=np.ones(close_n), border_value=0)
    stationary = binary_opening(stationary, structure=np.ones(min_n), border_value=0)
    raw_ints = signal_intervals(stationary)
    ev = []
    for k, (i, j) in enumerate(raw_ints, 1):
        if (t[j - 1] - t[i]) * 1000 < args.min_stationary_ms * .8:
            continue
        ev.append(dict(side=side, interval=k, start_s=round(float(t[i]), 4),
                       end_s=round(float(t[j - 1]), 4),
                       duration_ms=round(float((t[j - 1] - t[i]) * 1000), 1),
                       mid_s=round(float((t[i] + t[j - 1]) / 2), 4),
                       kind='stationary-foot candidate (not verified)'))
    # For signal-inspection only, do not call any acceleration peak an actual heel strike.
    derived = pd.DataFrame({
        'time_s': t, 'ax_filtered_mps2': af[:,0], 'ay_filtered_mps2': af[:,1],
        'az_filtered_mps2': af[:,2], 'gx_filtered_dps': wf[:,0],
        'gy_filtered_dps': wf[:,1], 'gz_filtered_dps': wf[:,2],
        'accel_norm_filtered_mps2': anorm, 'gyro_norm_filtered_dps': gnorm,
        'accel_norm_variance': avar, 'stationary_candidate': stationary.astype(int)
    })
    return dict(side=side, t=t, fs=fs, accel=accel, gyro=gyro, bias=bias,
                filtered=filtered, af=af, wf=wf, derived=derived,
                events=ev, stance=stationary,
                gyro_rate=float(gnorm.max()),
                avg_hz=(len(t)-1)/(t[-1]-t[0]))


def make_figures(out, samples, args):
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    for i, s in enumerate(samples):
        t = s['t']
        ax = axes[i]
        gy_raw = np.linalg.norm(s['gyro'] - s['bias'], axis=1)
        gy_filtered = np.linalg.norm(s['wf'], axis=1)
        ax.plot(t, gy_raw, color='0.72', linewidth=.65, label=f"{s['side']} gyro magnitude raw (bias-corrected)")
        ax.plot(t, gy_filtered, linewidth=1.1, label=f"{s['side']} gyro magnitude {args.cutoff:g} Hz")
        for e in s['events']:
            ax.axvspan(e['start_s'], e['end_s'], color='#68a58b', alpha=.28)
        ax.axhline(args.gyro_threshold, color='tab:red', linestyle=':', linewidth=.8)
        ax.set_ylabel(f"{s['side'].title()} gyro (°/s)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=.15)
    for i, s in enumerate(samples):
        ax = axes[2+i]
        t=s['t']
        raw=np.linalg.norm(s['accel'],axis=1)
        af=np.linalg.norm(s['af'],axis=1)
        ax.plot(t,raw,color='0.72',linewidth=.65,label='Raw accel magnitude')
        ax.plot(t,af,linewidth=1.1,label=f'{args.cutoff:g} Hz filtered accel magnitude')
        for e in s['events']:
            ax.axvspan(e['start_s'], e['end_s'], color='#68a58b', alpha=.28)
        ax.axhline(G, color='tab:red', linestyle=':', linewidth=.8)
        ax.set_ylabel(f"{s['side'].title()} accel (m/s²)")
        ax.legend(loc='upper right',fontsize=8)
        ax.grid(alpha=.15)
    axes[-1].set_xlabel('Time from start of each sensor recording (s)')
    fig.suptitle('ACIT4035 | candidate stationary foot intervals (green) — unvalidated', fontsize=14)
    fig.savefig(out/'bilateral_stationary_detection.png',dpi=170)
    plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(15,9),sharex='col',constrained_layout=True)
    for idx,s in enumerate(samples):
        t=s['t']; sl=(t>=args.zoom_start)&(t<=args.zoom_end)
        for j,(key,raw) in enumerate([('gyro',np.linalg.norm(s['gyro']-s['bias'],axis=1)),
                                      ('accel',np.linalg.norm(s['accel'],axis=1))]):
            ax=axs[j,idx]; ax.plot(t[sl],raw[sl],color='0.65',lw=.9,label='Raw')
            for hz,(a,w) in sorted(s['filtered'].items()):
                if hz in (10,15,30):
                    arr=np.linalg.norm(w,axis=1) if key=='gyro' else np.linalg.norm(a,axis=1)
                    ax.plot(t[sl],arr[sl],lw=1.0,label=f'Butterworth {hz} Hz')
            ax.set_title(f"{s['side'].title()}: {key} magnitude")
            ax.set_ylabel('°/s' if key=='gyro' else 'm/s²')
            ax.grid(alpha=.15); ax.legend(fontsize=7,loc='upper right')
    axs[1,0].set_xlabel('Time (s)');axs[1,1].set_xlabel('Time (s)')
    fig.suptitle('Offline zero-phase filtering; compare event shape, not only smoothness')
    fig.savefig(out/'filter_comparison.png',dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--left',type=Path,required=True)
    ap.add_argument('--right',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,default=Path('fpa_signal_processing_output'))
    ap.add_argument('--cutoff',type=float,default=15,help='Butterworth filter for event detection (Hz)')
    ap.add_argument('--bias-seconds',type=float,default=1.25)
    ap.add_argument('--gyro-threshold',type=float,default=50,help='candidate stance gyro norm cutoff deg/s')
    ap.add_argument('--gravity-tolerance',type=float,default=1.8,help='m/s²')
    ap.add_argument('--accel-variance',type=float,default=.8,help='(m/s²)^2')
    ap.add_argument('--variance-window-ms',type=float,default=60)
    ap.add_argument('--bridge-ms',type=float,default=40)
    ap.add_argument('--min-stationary-ms',type=float,default=70)
    ap.add_argument('--zoom-start',type=float,default=2)
    ap.add_argument('--zoom-end',type=float,default=11)
    args = ap.parse_args()
    if args.cutoff <= 0 or args.zoom_end <= args.zoom_start:
        ap.error('Filter cutoff must be positive and zoom end must exceed zoom start')
    out=args.outdir;out.mkdir(parents=True, exist_ok=True)
    samples = [load_foot(args.left,'left',args),load_foot(args.right,'right',args)]
    for s in samples:
        s['derived'].to_csv(out/f"{s['side']}_processed.csv",index=False)
    evs = [e for s in samples for e in s['events']]
    pd.DataFrame(evs,columns=['side','interval','start_s','end_s','duration_ms','mid_s','kind']).to_csv(out/'stationary_candidates.csv',index=False)
    make_figures(out,samples,args)
    report = dict(status='EXPERIMENTAL - stationary intervals NOT ground truth; no FPA calculation',
                  sources={'left':str(args.left),'right':str(args.right)},
                  filters=dict(type='4th order zero-phase Butterworth (offline)',cutoff_hz=args.cutoff,
                               comparison_cutoffs_hz=[10,15,30]),
                  detector=dict(gyro_threshold_dps=args.gyro_threshold,
                                gravity_tolerance_mps2=args.gravity_tolerance,
                                accel_variance_limit=args.accel_variance,
                                min_stationary_ms=args.min_stationary_ms),
                  feet={s['side']:dict(rows=len(s['t']),median_sample_rate_hz=round(s['fs'],2),
                           average_rate_hz=round(s['avg_hz'],2),
                           gyro_bias_dps=[round(float(x),4) for x in s['bias']],
                           stationary_candidate_intervals=len(s['events'])) for s in samples})
    (out/'analysis_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for s in samples:
        print(f"{s['side'].upper():5s}: {len(s['t'])} samples, ~{s['avg_hz']:.1f} Hz, "
              f"{len(s['events'])} candidate stationary intervals")
    print('Generated:',out.resolve())
    print('Caution: markers are hypotheses, not confirmed heel strikes or measured FPA.')

if __name__=='__main__':
    main()
