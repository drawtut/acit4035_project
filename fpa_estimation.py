"""ACIT4035: experimental, OFFLINE, magnetometer-independent foot FPA estimator.

Dependencies: numpy pandas scipy matplotlib.
Raw CSV is NEVER overwritten. This is a prototype, not a validated clinical device.

Two commands:
  py fpa_estimation.py inspect --input left_raw.csv --plot
  py fpa_estimation.py toe-axis --input toe_lifts.csv --side left --output toe_axis_left.json
  py fpa_estimation.py calibrate --input calibration_left_raw.csv --side left \
      --start 5.05 --end 12.05 --toe-axis toe_axis_left.json --output foot_calibration.json
  py fpa_estimation.py estimate --input trial_left_raw.csv --side left \
      --calibration foot_calibration.json --plot

Functional calibration must be a *separate, intentionally performed* trial:
stand still at start; take >=4 straight walking steps with forward-aligned toes;
mark the time interval containing the steps with --start and --end.
The script uses PCA on angular velocity to estimate a foot mediolateral axis
and gravity to estimate a forward axis. Use the toe-axis command to derive a measured 3D shoe-forward vector from three
heel-planted toe-UP lifts; a measured vector replaces the old forced single-axis mounting assumption.

Estimator: initial gravity leveling, gyro integration, motion segmentation,
rotate acceleration to local level frame, linear zero-velocity drift correction
between same-foot stances, integrate corrected velocity for stride displacement.
FPA uses midstance shoe direction versus same-foot stride travel direction.
This intentionally avoids fused yaw and magnetometer for FPA calculation.
The reconstructed displacements and FPA are preliminary and require ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.spatial.transform import Rotation

G = 9.80665
ACC = ['ax_mps2', 'ay_mps2', 'az_mps2']
GYR = ['gx_dps', 'gy_dps', 'gz_dps']


def load(path):
    df = pd.read_csv(path)
    required = ACC + GYR
    for col in required:
        if col not in df: raise ValueError(f'Missing required column: {col}')
    time_col = 'timestamp_midpoint_monotonic_s' if 'timestamp_midpoint_monotonic_s' in df else 'time_s'
    if time_col not in df: raise ValueError('Missing timestamps')
    t = df[time_col].to_numpy(dtype=float, copy=True)
    t -= t[0]
    a, w = df[ACC].to_numpy(float), df[GYR].to_numpy(float)
    if len(t) < 200 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Need >=200 rows and finite strictly increasing timestamps')
    if not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError('Invalid accelerometer or gyroscope values')
    fs = 1 / np.median(np.diff(t))
    if not 20 <= fs <= 1000: raise ValueError(f'Unexpected sample rate: {fs:.1f} Hz')
    return df, t, a, w, fs


def intervals(mask, t, min_dur=0):
    start = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == 1)
    end = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == -1)
    return [(int(i), int(j)) for i, j in zip(start, end)
            if t[j-1] - t[i] >= min_dur]


def process(t, accel, gyro, fs, args):
    start = (t - t[0] < args.bias_seconds)
    norm = np.linalg.norm(gyro, axis=1)
    quiet = start & (norm < args.bias_max_gyro)
    if quiet.sum() < max(30, int(fs * 0.2)):
        raise ValueError('No sufficiently motionless starting interval for gyro bias. '
                         'Record >=2 stationary seconds before walking.')
    bias = np.median(gyro[quiet], axis=0)
    gyro_corr = gyro - bias
    cutoff = min(args.cutoff, fs * 0.40)
    sos = butter(4, cutoff, btype='low', fs=fs, output='sos')
    a_filter = sosfiltfilt(sos, accel, axis=0)
    w_filter = sosfiltfilt(sos, gyro_corr, axis=0)
    # Detection: conservative midstance candidates. Separate from integration.
    window = max(3, int(round(args.score_window * fs)) | 1)
    anorm = np.linalg.norm(a_filter, axis=1)
    gnorm = np.linalg.norm(w_filter, axis=1)
    avar = uniform_filter1d(anorm**2, window, mode='nearest') - uniform_filter1d(anorm, window, mode='nearest')**2
    avar = np.maximum(0, avar)
    stance = ((gnorm < args.stance_gyro) & (anorm > args.accel_min) &
              (anorm < args.accel_max) & (avar < args.accel_variance))
    # Conservative morphology: close only very short gaps, remove isolated hits.
    close = max(1, round(args.bridge_ms / 1000 * fs))
    keep = max(2, round(args.min_stance * fs))
    if close > 1: stance = binary_closing(stance, structure=np.ones(close), border_value=0)
    stance = binary_opening(stance, structure=np.ones(keep), border_value=0)
    events = intervals(stance, t, args.min_stance * 0.85)
    return dict(bias=bias, gyro_corr=gyro_corr, gyro_filter=w_filter, acc_filter=a_filter,
                stance=stance, events=events, gyro_norm=gnorm, accel_norm=anorm,
                accel_variance=avar, quiet=quiet)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9: raise ValueError('Cannot normalize near-zero vector')
    return v/n


def toe_lift_axis(t, accel, gyro, fs, side, source_sha256, minimum_lifts=3):
    """Estimate shoe-forward in sensor coordinates using signed toes-UP tilt.

    At rest the accelerometer measures specific force (approximately world up).
    During a toes-up tilt with heel held in place, the change in that vector,
    projected perpendicular to baseline up, points roughly heel->toe in sensor
    coordinates. This assumes toe-up lifts, not toe-down presses or translations.
    """
    baseline = (t <= min(0.55, 0.07*t[-1]))
    if baseline.sum() < 20:
        raise ValueError('Toe-axis test must begin with >=0.5 seconds stationary.')
    initial_w = np.linalg.norm(gyro[baseline], axis=1)
    if np.percentile(initial_w, 90) > 8:
        raise ValueError('Toe-axis baseline was moving; repeat with foot flat and still at start.')
    up = normalize(np.median(accel[baseline], axis=0))
    # Filtering only for tilt-event detection; original accelerometer is retained.
    smoothed = uniform_filter1d(accel, size=max(3,int(fs*.12)), axis=0, mode='nearest')
    un = smoothed / np.maximum(np.linalg.norm(smoothed, axis=1, keepdims=True),1e-8)
    tilt = np.degrees(np.arccos(np.clip(un @ up,-1,1)))
    peaks,props=find_peaks(tilt, prominence=12, distance=max(1,int(fs*1.5)),height=16)
    # Use the most prominent separated peaks. Three are expected in the user trial.
    selected=[]
    for idx in peaks[np.argsort(props['prominences'])[::-1]]:
        if len(selected)>=minimum_lifts: break
        if any(abs(t[idx]-t[other])<1.5 for other in selected):continue
        selected.append(int(idx))
    selected.sort()
    if len(selected)<minimum_lifts:
        raise ValueError(f'Found {len(selected)} toe-up tilts; need {minimum_lifts} distinct lifts. '
                         'Keep heel planted, lift toes, pause, lower, and repeat.')
    forwards=[]; detail=[]
    for k in selected:
        region=(t >= t[k]-.12) & (t <= t[k]+.12)
        # The plateau gives a stable gravity direction (if present).
        good=region & (np.linalg.norm(gyro,axis=1)<15) & (np.abs(np.linalg.norm(accel,axis=1)-G)<1.25)
        if good.sum()<max(3,int(fs*.025)):
            raise ValueError(f'Toe lift near {t[k]:.2f}s has no sufficiently stationary raised pose.')
        v=normalize(np.median(accel[good],axis=0))
        change=v-np.dot(v,up)*up
        forward=normalize(change)
        forwards.append(forward)
        detail.append(dict(peak_time_s=float(t[k]),tilt_deg=float(np.degrees(np.arccos(np.clip(up@v,-1,1)))),
                           stable_samples=int(good.sum()),forward_unit=[float(x) for x in forward]))
    mean=normalize(np.mean(forwards,axis=0))
    consistency=np.array([np.dot(f,mean) for f in forwards])
    if float(consistency.min())<np.cos(np.deg2rad(15)):
        raise ValueError('Toe-lift forward direction differs by more than 15 deg across lifts.')
    return dict(schema='ACIT4035_TOE_AXIS_V1',side=side,
                sensor_forward_unit=[float(x) for x in mean],
                sensor_up_reference=[float(x) for x in up],
                lift_details=detail,minimum_direction_cosine=float(consistency.min()),
                source_sha256=source_sha256,
                assumptions=['Heel remains planted during all toe-UP lifts',
                             'Sensor remains fixed to shoe between toe-axis and walking calibration',
                             'Accelerometer magnitude approximates gravity in the raised poses'],
                warning='Experimental toe-lift anatomical-axis estimate; verify with shoe axis markings or reference video.')


def foot_axis_calibration(t, accel, gyro, fs, proc, start, end, side, mount_forward=None, toe_axis=None):
    if not 0 <= start < end <= t[-1] + .01:
        raise ValueError('Invalid calibration interval')
    mask = (t >= start) & (t <= end) & (~proc['stance'])
    w = proc['gyro_corr'][mask]
    w = w[np.linalg.norm(w, axis=1) > 35]
    if len(w) < max(50, fs * .5):
        raise ValueError('Calibration needs >=4 distinct controlled walking steps. Insufficient rotation samples.')
    _, vals, vt = np.linalg.svd(w - np.mean(w, axis=0), full_matrices=False)
    ml_pca = normalize(vt[0])
    anisotropy = float((vals[0] / max(vals[1], 1e-9)) ** 2)
    sensor_up = normalize(np.median(accel[proc['quiet']], axis=0))
    ml_pca = normalize(ml_pca - np.dot(ml_pca, sensor_up) * sensor_up)
    f_pca = normalize(np.cross(sensor_up, ml_pca))
    if anisotropy < 1.3:
        raise ValueError(f'PCA mediolateral axis ambiguous (eigenvalue ratio {anisotropy:.2f}); '
                         'repeat calibration with >=4 clear forward walking steps.')
    if toe_axis is not None:
        if toe_axis.get('schema')!='ACIT4035_TOE_AXIS_V1' or toe_axis.get('side')!=side:
            raise ValueError('Toe-axis JSON schema/side mismatch')
        toe_forward = normalize(np.array(toe_axis['sensor_forward_unit'],float))
        toe_up = normalize(np.array(toe_axis['sensor_up_reference'],float))
        up_error=float(np.degrees(np.arccos(np.clip(np.dot(toe_up,sensor_up),-1,1))))
        if up_error>15:
            raise ValueError(f'Toe-axis and walking baseline gravity differ by {up_error:.1f}°. '
                             'IMU may have moved on shoe, or shoe pose changed. Recalibrate.')
        toe_forward=normalize(toe_forward-np.dot(toe_forward,sensor_up)*sensor_up)
        if np.dot(f_pca,toe_forward)<0:f_pca=-f_pca
        agreement=float(np.degrees(np.arccos(np.clip(np.dot(f_pca,toe_forward),-1,1))))
        if agreement>35:
            raise ValueError(f'Toe-up and walking-PCA forward axes disagree by {agreement:.1f}°. '
                             'Check mounting and whether four forward steps were performed.')
        # Toe lift directly informs the shoe-forward direction; PCA supplies
        # an independent alignment check and stabilizes the combined estimate.
        forward=normalize(toe_forward+f_pca)
        mount_type='toe_lift_3d_plus_gyro_pca'
        toe_sha=toe_axis['source_sha256']
    else:
        if mount_forward is None:
            raise ValueError('Calibration requires --toe-axis (recommended) or an explicitly verified --mount-forward.')
        mount_axes = {'x':[1.,0.,0.],'-x':[-1.,0.,0.],'y':[0.,1.,0.],'-y':[0.,-1.,0.],'z':[0.,0.,1.],'-z':[0.,0.,-1.]}
        mount_vector=np.array(mount_axes[mount_forward],float)
        if np.dot(f_pca,mount_vector)<0:f_pca=-f_pca
        alignment=float(np.dot(f_pca,mount_vector))
        if alignment<.5:
            raise ValueError(f'PCA axis poorly aligned with claimed {mount_forward} mount (cos={alignment:.2f})')
        forward=f_pca
        agreement=None;up_error=None;toe_sha=None
        mount_type='verified_single_mount_axis'
    foot_ml=normalize(np.cross(forward,sensor_up))
    forward=normalize(np.cross(sensor_up,foot_ml))
    return dict(schema='ACIT4035_FPA_CALIBRATION_V2',side=side,
                sensor_forward_unit=[float(x) for x in forward],
                sensor_up_reference=[float(x) for x in sensor_up],
                foot_mediolateral_unit=[float(x) for x in foot_ml],
                pca_mediolateral_raw=[float(x) for x in ml_pca],
                pca_eigenvalue_ratio=anisotropy,
                calibration_method=mount_type,forward_axis_mount=mount_forward,
                toe_pca_agreement_deg=agreement,toe_walking_gravity_difference_deg=up_error,
                toe_axis_source_sha256=toe_sha,calibration_seconds=[start,end],
                warning='Experimental functional alignment. Verify shoe axes and validate with reference video.')


def integrate_orientation(t, accel, w, quiet):
    # Initial world +Z aligned with the measured stationary specific-force vector.
    up_sensor = normalize(np.median(accel[quiet], axis=0))
    r0, _ = Rotation.align_vectors(np.array([[0.,0.,1.]]), up_sensor.reshape(1,3))
    q = np.empty((len(t), 4), float)
    q[0] = r0.as_quat()
    rotations = [r0]
    for i in range(1,len(t)):
        dt = t[i]-t[i-1]
        # Body frame angular rates; midpoint integration for modest sample intervals.
        omega = np.deg2rad((w[i-1] + w[i])*0.5)
        ri = rotations[-1] * Rotation.from_rotvec(omega*dt)
        rotations.append(ri)
        q[i] = ri.as_quat()
    rots = Rotation.from_quat(q)
    specific_world = rots.apply(accel)
    linear_world = specific_world - np.array([0., 0., G])
    return rots, linear_world


def stride_results(t, linear_world, rotations, events, foot_axis, side, args):
    rows=[]
    paths=[]
    for n, ((a,b),(c,d)) in enumerate(zip(events,events[1:]),1):
        # Consecutive same-foot midstances (endpoints in the interior).
        i=(a+b-1)//2; j=(c+d-1)//2
        duration=t[j]-t[i]
        if not args.min_stride <= duration <= args.max_stride: continue
        if j-i < 20: continue
        dt = np.diff(t[i:j+1]); acc = linear_world[i:j+1]
        # Trapezoidal initial integral of acceleration.
        velocity=np.zeros((len(acc),3))
        velocity[1:] = np.cumsum((acc[1:]+acc[:-1])*.5*dt[:,None], axis=0)
        # ZUPT both ends: linear velocity drift correction to force v(end)=0.
        frac=(t[i:j+1]-t[i])/duration
        velocity -= frac[:,None]*velocity[-1]
        pos=np.zeros_like(velocity)
        pos[1:] = np.cumsum((velocity[1:]+velocity[:-1])*.5*dt[:,None], axis=0)
        delta=pos[-1]
        horiz=float(np.linalg.norm(delta[:2]))
        # Foot axis at the FINAL midstance; no fused yaw used.
        f=rotations[j].apply(foot_axis)
        horizontal_axis_norm=float(np.linalg.norm(f[:2]))
        if horizontal_axis_norm > 1e-9: f[:2] = f[:2]/horizontal_axis_norm
        valid=(horiz >= args.min_displacement and
               horizontal_axis_norm > .75 and
               np.max(dt) <= args.max_gap and
               np.isfinite(pos).all())
        reason='accepted' if valid else 'low displacement, nonhorizontal foot axis, or timing gap'
        angle=np.nan
        if valid:
            direction=delta[:2]/horiz
            cross=direction[0]*f[1]-direction[1]*f[0]
            dot=float(np.clip(np.dot(direction,f[:2]),-1,1))
            ccw=np.degrees(np.arctan2(cross,dot))
            # Left toe-out = counterclockwise, right toe-out = clockwise.
            angle=float(ccw if side=='left' else -ccw)
            # Extremely large estimates are almost always a bad trajectory or calibration.
            if abs(angle) > args.max_abs_fpa:
                valid=False; reason='implausible large angle; check trajectory / calibration'; angle=np.nan
        rows.append(dict(stride=n,side=side,start_s=float(t[i]),end_s=float(t[j]),
                         duration_s=float(duration),delta_x_m=float(delta[0]),
                         delta_y_m=float(delta[1]),delta_z_m=float(delta[2]),
                         horizontal_displacement_m=horiz,
                         end_speed_before_zupt_mps=float(np.linalg.norm(np.sum((acc[1:]+acc[:-1])*.5*dt[:,None],axis=0))),
                         max_sample_gap_ms=float(max(dt)*1000),
                         fpa_deg=angle,accepted=bool(valid),reason=reason))
        paths.append((n,pos.copy(),valid))
    return rows, paths


def create_plots(out,t,proc,linear,events,paths,report,show):
    import matplotlib
    if not show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True, constrained_layout=True)
    axes[0].plot(t,proc['gyro_norm'],lw=.8,label='Filtered angular speed')
    axes[0].axhline(report['parameters']['stance_gyro_dps'],ls='--',color='gray',label='Stationary gyro threshold')
    axes[1].plot(t,proc['accel_norm'],lw=.75,label='Filtered acceleration magnitude')
    axes[1].axhline(G,ls='--',color='gray',label='Gravity')
    axes[2].plot(t,linear[:,0],lw=.75,label='World-frame ax (gravity removed)')
    axes[2].plot(t,linear[:,1],lw=.75,label='World-frame ay (gravity removed)')
    for ax in axes[:2]:
        for a,b in events: ax.axvspan(t[a],t[b-1],color='tab:green',alpha=.2)
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Angular speed (°/s)')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[2].set_ylabel('Linear accel (m/s²)');axes[2].set_xlabel('Time (s)')
    fig.suptitle('ACIT4035 | candidate stationary foot intervals (NOT verified ground contacts)')
    fig.savefig(out/'stationary_diagnostics.png',dpi=140)
    if show: plt.show(block=False)
    plt.close(fig)
    if paths:
        fig,ax=plt.subplots(figsize=(7,7))
        for n,pos,valid in paths:
            ax.plot(pos[:,0],pos[:,1],label=f'{n}'+('' if valid else ' rejected'),alpha=.7,
                    ls='-' if valid else ':')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
        ax.set_xlabel('Local X displacement (m)');ax.set_ylabel('Local Y displacement (m)')
        ax.set_title('Experimental same-foot ZUPT stride displacement (local heading arbitrary)')
        if len(paths)<=12:ax.legend(fontsize=8)
        fig.savefig(out/'stride_trajectories.png',dpi=140)
        if show:plt.show(block=False)
        plt.close(fig)


def run(args):
    df,t,a,w,fs=load(args.input)
    if args.mode=='toe-axis':
        calib=toe_lift_axis(t,a,w,fs,args.side,hashlib.sha256(args.input.read_bytes()).hexdigest())
        destination=args.output or (args.out or args.input.parent/(args.input.stem+'_fpa_estimation'))/'toe_axis.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(calib,indent=2),encoding='utf-8')
        print('Toe-lift direction:',np.round(calib['sensor_forward_unit'],4).tolist())
        print('Consistency minimum:',round(calib['minimum_direction_cosine'],4))
        print('Saved experimental toe-axis calibration:',destination)
        return
    proc=process(t,a,w,fs,args)
    out=args.out or args.input.parent/(args.input.stem+'_fpa_estimation')
    out.mkdir(parents=True,exist_ok=True)
    intervals_rows=[dict(start_s=float(t[i]),end_s=float(t[j-1]),
                         duration_s=float(t[j-1]-t[i]),start_index=i,end_index_exclusive=j)
                    for i,j in proc['events']]
    pd.DataFrame(intervals_rows,columns=['start_s','end_s','duration_s','start_index','end_index_exclusive']).to_csv(
        out/'stationary_intervals.csv',index=False)
    calib=None
    if args.mode=='calibrate':
        if args.session:
            meta=json.loads(args.session.read_text(encoding='utf-8'))
            if meta.get('schema') != 'ACIT4035_FUNCTIONAL_SESSION_V1' or 'walking_window_s' not in meta:
                raise ValueError('Invalid functional_session.json schema or missing walking window')
            if args.side not in meta.get('sides',[]):
                raise ValueError('Requested side is not in functional session')
            args.start,args.end=map(float,meta['walking_window_s'])
        if args.start is None or args.end is None:
            raise ValueError('calibrate requires --session functional_session.json or --start and --end')
        toe=json.loads(args.toe_axis.read_text(encoding='utf-8')) if args.toe_axis else None
        calib=foot_axis_calibration(t,a,w,fs,proc,args.start,args.end,args.side,args.mount_forward,toe)
        calib['calibration_source_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        destination=args.output or (out/'foot_calibration.json')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(calib,indent=2),encoding='utf-8')
        print('Saved experimental functional calibration:',destination)
    elif args.mode=='estimate' and args.calibration:
        calib=json.loads(args.calibration.read_text(encoding='utf-8'))
        if calib.get('schema') not in ('ACIT4035_FPA_CALIBRATION_V1','ACIT4035_FPA_CALIBRATION_V2') or calib.get('side') !=args.side:
            raise ValueError('Calibration schema/side mismatch')
        if calib.get('calibration_source_sha256') == hashlib.sha256(args.input.read_bytes()).hexdigest():
            raise ValueError('Estimation trial must differ from functional calibration recording. '
                             'Record a new straight-walking trial after calibrating.')
    elif args.mode=='estimate':
        print('NO FPA: --calibration required. Stationary detection/trajectories only.')

    rots,linear=integrate_orientation(t,a,proc['gyro_corr'],proc['quiet'])
    rows=[];paths=[]
    if args.mode=='estimate':
        if calib:
            rows,paths=stride_results(t,linear,rots,proc['events'],
                                      normalize(np.array(calib['sensor_forward_unit'],float)),args.side,args)
        else:
            # Without foot calibration cannot calculate an angle, but can still inspect paths.
            rows,paths=stride_results(t,linear,rots,proc['events'],np.array([0.,1.,0.]),args.side,args)
            for row in rows:
                row['fpa_deg']=np.nan;row['accepted']=False;row['reason']='functional calibration missing'
            paths=[(n,p,False) for n,p,_ in paths]
        pd.DataFrame(rows,columns=['stride','side','start_s','end_s','duration_s','delta_x_m',
                                  'delta_y_m','delta_z_m','horizontal_displacement_m',
                                  'end_speed_before_zupt_mps','max_sample_gap_ms','fpa_deg',
                                  'accepted','reason']).to_csv(out/'fpa_results.csv',index=False)
    pd.DataFrame({'time_s':t,'gyro_norm_processed_dps':proc['gyro_norm'],
                  'accel_norm_processed_mps2':proc['accel_norm'],
                  'accel_norm_local_variance':proc['accel_variance'],
                  'stationary_candidate':proc['stance'].astype(int),
                  'linear_ax_local_mps2':linear[:,0],
                  'linear_ay_local_mps2':linear[:,1],
                  'linear_az_local_mps2':linear[:,2]}).to_csv(out/'diagnostic_signals.csv',index=False)
    accepted=[r['fpa_deg'] for r in rows if r['accepted']]
    report=dict(status='EXPERIMENTAL; NOT VALIDATED FPA',mode=args.mode,source=str(args.input),
                side=args.side,rows=len(t),median_rate_hz=float(fs),
                achieved_rate_hz=float((len(t)-1)/(t[-1]-t[0])),
                max_gap_ms=float(np.diff(t).max()*1000),
                gyro_bias_dps=[float(x) for x in proc['bias']],
                stationary_candidate_count=len(proc['events']),
                calibration_loaded=bool(calib),calibration=calib,
                stride_candidates=len(rows),accepted_provisional_fpa_count=len(accepted),
                provisional_mean_fpa_deg=float(np.mean(accepted)) if accepted else None,
                parameters=dict(filter_cutoff_hz=args.cutoff,stance_gyro_dps=args.stance_gyro,
                                accel_magnitude_range=[args.accel_min,args.accel_max],
                                accel_variance_limit=args.accel_variance,
                                min_horizontal_displacement_m=args.min_displacement),
                caveats=['Not a clinical or validated FPA measurement.',
                         'PCA calibration requires an independently identified four-step straight calibration.',
                         'Foot trajectory is vulnerable to gyro drift, sensor misalignment and inaccurate ZUPTs.',
                         'Initial magnetometer-free yaw reference is arbitrary but local stride angles are invariant to common yaw offset.',
                         'Confidence and absolute accuracy require synchronized external ground truth.'])
    (out/'estimation_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    create_plots(out,t,proc,linear,proc['events'],paths,report,args.plot)
    print(f'Input: {len(t)} rows at {fs:.2f} Hz; stationary candidates: {len(proc["events"])}')
    print('Gyro bias (deg/s):',np.round(proc['bias'],4).tolist())
    print(f'Stride candidates: {len(rows)}; provisional FPA accepted: {len(accepted)}')
    if accepted: print('PROVISIONAL mean FPA (not validated):',round(float(np.mean(accepted)),2),'degrees')
    print('Files saved:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode',choices=['inspect','toe-axis','calibrate','estimate'])
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--side',choices=['left','right'],default='left')
    p.add_argument('--mount-forward',choices=['x','-x','y','-y','z','-z'],default=None,
                   help='Physical sensor axis pointing approximately toward toes, verified on shoe')
    p.add_argument('--toe-axis',type=Path,help='Measured 3D forward-axis JSON from toe-axis command')
    p.add_argument('--session',type=Path,help='FPA reader functional_session.json: uses recorded walking window')
    p.add_argument('--out',type=Path)
    p.add_argument('--output',type=Path,help='Calibration JSON output path')
    p.add_argument('--calibration',type=Path,help='Previously saved foot_calibration.json')
    p.add_argument('--start',type=float,help='Start of forward 4-step functional calibration in seconds')
    p.add_argument('--end',type=float,help='End of forward 4-step functional calibration in seconds')
    p.add_argument('--cutoff',type=float,default=15)
    p.add_argument('--bias-seconds',type=float,default=1.5)
    p.add_argument('--bias-max-gyro',type=float,default=8)
    p.add_argument('--score-window',type=float,default=.055)
    p.add_argument('--stance-gyro',type=float,default=50)
    p.add_argument('--accel-min',type=float,default=9)
    p.add_argument('--accel-max',type=float,default=11)
    p.add_argument('--accel-variance',type=float,default=0.5)
    p.add_argument('--bridge-ms',type=float,default=20)
    p.add_argument('--min-stance',type=float,default=.08)
    p.add_argument('--min-stride',type=float,default=.45)
    p.add_argument('--max-stride',type=float,default=3.2)
    p.add_argument('--min-displacement',type=float,default=.15)
    p.add_argument('--max-abs-fpa',type=float,default=60)
    p.add_argument('--max-gap',type=float,default=.025)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    if not args.input.is_file():p.error(f'Input CSV not found: {args.input}')
    if args.calibration and not args.calibration.is_file():p.error('Calibration JSON not found')
    if args.toe_axis and not args.toe_axis.is_file():p.error('Toe-axis JSON not found')
    if args.session and not args.session.is_file():p.error('Functional session JSON not found')
    if args.cutoff <=0 or args.bias_seconds<=0:p.error('Invalid processing settings')
    run(args)


if __name__=='__main__':main()
