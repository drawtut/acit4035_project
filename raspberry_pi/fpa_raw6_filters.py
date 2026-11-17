"""Independent raw-six-axis filtering laboratory for ACIT4035 FPA research.

Only IMU timestamps, accelerometer XYZ and gyroscope XYZ are inputs.
This module DOES NOT estimate FPA and DOES NOT change footstep detection.
Offline Butterworth ``sosfiltfilt`` is for experimental comparison, not real time.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

ACC_COLUMNS = ('ax_mps2', 'ay_mps2', 'az_mps2')
GYRO_COLUMNS = ('gx_dps', 'gy_dps', 'gz_dps')


def _select_stationary_bias_window(t, accel, gyro, requested_start, requested_end):
    """Use the configured window if stationary; else find a real initial still window.

    Only measured accelerometer/gyroscope channels and IMU timestamps are used.
    Search is limited to the first 6 seconds; never use post-walk stillness to
    initialize a pre-walk quaternion. No video or derived yaw enters selection.
    """
    def check(lo, hi):
        idx=(t >= lo) & (t < hi)
        if np.count_nonzero(idx) < 80 or hi-lo < 1.2:
            return None
        g=gyro[idx]; a=accel[idx]
        bias=np.median(g,axis=0)
        moving=np.linalg.norm(g-bias,axis=1)
        accel_norm=np.linalg.norm(a,axis=1)
        q90=float(np.percentile(moving,90))
        accel_std=float(np.std(accel_norm))
        accel_med=float(np.median(accel_norm))
        if q90 > 5.0 or accel_std > 0.15 or not 7.5 < accel_med < 12.0:
            return None
        return (q90,accel_std)

    if check(requested_start,requested_end) is not None:
        return requested_start,requested_end,False
    duration=float(t[-1])
    # Prefer early initial rest, and require >=1.5 seconds.  The trial's
    # physical stillness may be shorter than the planned 5-second wait.
    windows=[]
    for duration_s in (2.0,1.75,1.5):
        for begin in np.arange(0., max(0., min(4.5,duration-1.5))+0.001,.25):
            end=float(begin+duration_s)
            if end > min(6.,duration):
                continue
            quality=check(float(begin),end)
            if quality is not None:
                windows.append((float(begin),end,quality[0],quality[1]))
        if windows:
            # First viable window; priority goes to longer candidate duration.
            windows.sort(key=lambda item:(item[0],item[2],item[3]))
            break
    if not windows:
        raise ValueError('No stationary initial gyro-bias interval found in first 6 seconds. '
                         'Record at least 2 seconds of stillness before walking; '
                         'do not force a moving reference.')
    lo,hi,_,_=windows[0]
    return lo,hi,True


def filter_raw_six_axis(t, accel, gyro, *, gyro_hz=10.0, accel_hz=5.0,
                        order=4, bias_start_s=1.0, bias_end_s=4.0,
                        correct_gyro_bias=True, filter_gyro=True,
                        filter_accel=True):
    """Return six raw and filtered series, plus stationary bias diagnostics.

    Inputs and outputs are per-IMU. Neither reference video nor sensor Euler
    angles can enter this function. Bias window is defined in IMU time only.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(accel, dtype=float)
    g = np.asarray(gyro, dtype=float)
    if t.ndim != 1 or len(t) < 20 or a.shape != (len(t),3) or g.shape != (len(t),3):
        raise ValueError('Expect t[N], accel[N,3], gyro[N,3], with at least 20 samples')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(g)):
        raise ValueError('All IMU samples must be finite')
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('IMU timestamps must be strictly increasing')
    fs = 1.0 / np.median(dt)
    start, end = float(bias_start_s), float(bias_end_s)
    if not (0 <= start < end <= t[-1] + 1e-8):
        raise ValueError('Gyro bias window must fall inside the IMU recording')
    requested_start,requested_end=start,end
    start,end,auto_reference = _select_stationary_bias_window(t,a,g,start,end)
    stationary = (t >= start) & (t < end)
    if stationary.sum() < 10:
        raise ValueError('Gyro bias window contains fewer than 10 IMU samples')
    b = np.median(g[stationary], axis=0)
    raw_still = np.linalg.norm(g[stationary]-b,axis=1)
    moving_bias_warning = bool(np.percentile(raw_still, 90) > 5.0)
    corrected = g - b if correct_gyro_bias else g.copy()
    order = int(order)
    if order not in (2,3,4,5,6):
        raise ValueError('Filter order must be 2–6')
    def lp(x, cutoff, enabled):
        if not enabled:
            return x.copy()
        cutoff=float(cutoff)
        if not 0 < cutoff < fs/2:
            raise ValueError(f'Low-pass cutoff must be between 0 and Nyquist {fs/2:.1f} Hz')
        sos=butter(order,cutoff,fs=fs,btype='lowpass',output='sos')
        return sosfiltfilt(sos,x,axis=0)
    gf=lp(corrected,gyro_hz,filter_gyro)
    af=lp(a,accel_hz,filter_accel)
    rms_before=float(np.sqrt(np.mean(np.sum((g[stationary]-b)**2,axis=1))))
    rms_after=float(np.sqrt(np.mean(np.sum((gf[stationary])**2,axis=1))))
    return {'time_s':t,'gyro_raw':g, 'gyro_bias_corrected':corrected,
            'gyro_filtered':gf, 'accel_raw':a, 'accel_filtered':af,
            'gyro_bias_dps':b, 'fs_hz':float(fs),
            'bias_start_s':start, 'bias_end_s':end,
            'requested_bias_window_s':[requested_start,requested_end],
            'reference_window_auto_selected':auto_reference,
            'rms_gyro_before_dps':rms_before, 'rms_gyro_after_dps':rms_after,
            'bias_window_movement_warning':moving_bias_warning}
