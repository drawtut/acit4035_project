"""Phase 16: genuinely time-synchronous inter-foot relative heading.

Both foot orientations are sampled at ONE common host-monotonic instant t.
Only six raw measured axes, time stamps, calibrated shoe-forward axes and the
existing quaternion orientation timeline are used. All frozen filters,
footstep detection and reference-line FPA outputs remain unchanged.

Delta(t) = wrap( [yaw_L(t)-yaw_L(0)] - [yaw_R(t)-yaw_R(0)] ).
Positive: left physical shoe heading CCW relative to right after taped zero.
A robust single-value summary is the median delta during double support.

No fusion of one foot into the other's FPA, no differential drift cancellation,
no absolute global yaw, no sensor-native synchronization, no clinical claim.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

SYNC_FIELDS = ['time_s','left_heading_deg','right_heading_deg',
               'relative_heading_deg','left_stationary','right_stationary',
               'double_support','sample_quality']
SUPPORT_FIELDS = ['start_s','end_s','duration_s','median_relative_heading_deg',
                  'peak_to_peak_relative_heading_deg','samples']


def wrap_deg(x):
    return (np.asarray(x, dtype=float) + 180) % 360 - 180


def _side_orientation(data, stage, *, min_horiz=.60):
    info=stage.get('reference_line') or {}
    if not info.get('enabled') or info.get('baseline_s') is None:
        raise ValueError('Initial taped-reference orientation is unavailable')
    cal=data.get('calibration') or {}
    if not cal.get('usable') or cal.get('sensor_forward_unit') is None:
        raise ValueError('Independent calibrated shoe-forward axis unavailable')
    ori=data['orientation']
    t=np.asarray(ori['time_s'],float)
    q=np.asarray(ori['quaternion_xyzw'],float)
    axis=np.asarray(cal['sensor_forward_unit'],float)
    axis/=np.linalg.norm(axis)
    if t.ndim!=1 or q.shape!=(len(t),4) or len(t)<10 or np.any(np.diff(t)<=0):
        raise ValueError('Invalid monotonic orientation samples')
    v=Rotation.from_quat(q).apply(axis)
    h=np.linalg.norm(v[:,:2],axis=1)
    az=np.arctan2(v[:,1],v[:,0])
    base=np.deg2rad(float(info['baseline_heading_deg']))
    bstart,bend=info['baseline_s']
    initial=(t>=bstart)&(t<bend)
    if initial.sum()<30 or np.median(h[initial])<min_horiz:
        raise ValueError('Invalid horizontal initial-reference quality')
    if (float(info.get('baseline_yaw_spread_p95_deg',0))>3 or
        float(info.get('baseline_gyro_p90_dps',0))>12):
        raise ValueError('Initial taped-reference pose not sufficiently stationary')
    yaw=np.rad2deg(np.unwrap(np.angle(np.exp(1j*(az-base)))))
    mask=np.asarray(data['detector_result']['mask'],bool)
    src=np.asarray(data['filtered']['source_time_s'],float)
    if mask.shape!=t.shape or len(src)<10:raise ValueError('Missing raw stationary timing')
    offset=float(data.get('clock_offset_s',0))
    return dict(t=t+offset, yaw=yaw, horizontal=h, stationary=mask,
                raw_t=src+offset,baseline_end=float(bend+offset))


def _raw_support(grid, source, *, max_sample_distance_s=.026,
                 max_gap_s=.060):
    """Do not infer 'simultaneous' orientation through missing real USB samples."""
    idx=np.searchsorted(source,grid)
    left=np.clip(idx-1,0,len(source)-1)
    right=np.clip(idx,0,len(source)-1)
    nearest=np.minimum(abs(source[left]-grid),abs(source[right]-grid))
    bracket=(idx>0)&(idx<len(source))
    gap=source[right]-source[left]
    return bracket & (nearest<=max_sample_distance_s) & (gap<=max_gap_s)


def compute_synchronous(results,stage,*,sample_period_s=.04,
                        min_double_support_s=.12):
    """Calculate simultaneous inter-foot yaw and evidence-labeled double support.

    results[side].clock_offset_s MUST be sourced from a shared host monotonic
    time base. Invalid clocks or baseline never produce fabricated angles.
    Clock sync is host request/response midpoint, not hardware simultaneity.
    """
    summary={'method':'synchronized_relative_foot_heading_from_two_IMU_quaternions',
        'time_relationship':'same_shared_host_midpoint_time_interpolated',
        'not_sequential_landing_pair':True,
        'reference':'initial_shoes_physically_aligned_to_same_taped_zero',
        'not_absolute_yaw_or_drift_cancellation':True,
        'status':'UNAVAILABLE','reason':None,'samples':0,'double_support_samples':0,
        'double_support_periods':0,'median_interfoot_heading_deg':None,
        'median_double_support_heading_deg':None,
        'max_abs_moving_relative_heading_deg':None}
    result={'rows':[],'double_support_periods':[], 'summary':summary}
    if not all(results.get(s,{}).get('shared_host_clock_verified',False) for s in ('left','right')):
        summary['reason']='UNVERIFIED_SHARED_HOST_CLOCK';return result
    try:
        l=_side_orientation(results['left'],stage['left'])
        r=_side_orientation(results['right'],stage['right'])
        start=max(l['t'][0],r['t'][0]); end=min(l['t'][-1],r['t'][-1])
        if end-start<1.:raise ValueError('Less than one second of overlapping samples')
        if not 0.01<=sample_period_s<=.2:raise ValueError('Invalid sample_period_s')
        grid=np.arange(start,end+1e-8,sample_period_s)
        valid=(_raw_support(grid,l['raw_t'])&_raw_support(grid,r['raw_t']))
        valid &= (np.interp(grid,l['t'],l['horizontal'])>=.6)
        valid &= (np.interp(grid,r['t'],r['horizontal'])>=.6)
        left_yaw=np.interp(grid,l['t'],l['yaw'])
        right_yaw=np.interp(grid,r['t'],r['yaw'])
        delta=np.asarray(wrap_deg(left_yaw-right_yaw))
        ls=np.interp(grid,l['t'],l['stationary'].astype(float))>.95
        rs=np.interp(grid,r['t'],r['stationary'].astype(float))>.95
        double=valid&ls&rs
        for i in np.flatnonzero(valid):
            result['rows'].append(dict(time_s=round(float(grid[i]),6),
                left_heading_deg=round(float(wrap_deg(left_yaw[i])),5),
                right_heading_deg=round(float(wrap_deg(right_yaw[i])),5),
                relative_heading_deg=round(float(delta[i]),5),
                left_stationary=bool(ls[i]),right_stationary=bool(rs[i]),
                double_support=bool(double[i]),
                sample_quality='HOST_SYNCED_AND_RAW_SUPPORTED'))
        # The study starts with a long initial hold AND ends with a final
        # standing pose. Neither is gait double support. Delimit the actual
        # motion using independently detected contact times (not paper GT).
        contact_times=[float(row['contact_time_s'])+
                       float(results[side].get('clock_offset_s',0))
                       for side in ('left','right') for row in stage[side]['rows']
                       if row.get('contact_time_s') is not None]
        if len(contact_times)<2:
            raise ValueError('No walking window: fewer than two detected contacts')
        first_contact=min(contact_times);last_contact=max(contact_times)
        summary['walking_window_s']=[round(first_contact,5),round(last_contact,5)]
        walking=(grid>=first_contact)&(grid<=last_contact)
        # A strict simultaneous stationary period only qualifies when BOTH
        # detectors agree within the actual contact-defined walking period.

        activity=double&walking
        edges=np.diff(np.r_[False,activity,False].astype('int8'))
        starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
        for i,j in zip(starts,stops):
            duration=(j-i)*sample_period_s
            if duration+1e-9<min_double_support_s:continue
            angles=delta[i:j]
            unwrapped=np.unwrap(np.deg2rad(angles))
            median=float(wrap_deg(np.rad2deg(np.median(unwrapped))))
            result['double_support_periods'].append(dict(
                start_s=round(float(grid[i]),4),
                end_s=round(float(min(grid[j-1]+sample_period_s,end)),4),
                duration_s=round(float(duration),4),
                median_relative_heading_deg=round(median,5),
                peak_to_peak_relative_heading_deg=round(float(np.ptp(np.rad2deg(unwrapped))),5),
                samples=int(j-i)))
        active=np.flatnonzero(valid&walking)
        stable=[p['median_relative_heading_deg'] for p in result['double_support_periods']]
        summary.update(status='OK',reason=None,samples=int(valid.sum()),
            double_support_samples=int(activity.sum()),
            double_support_periods=len(stable),
            median_interfoot_heading_deg=round(float(wrap_deg(np.rad2deg(np.median(np.unwrap(np.deg2rad(delta[active])))))),5) if len(active) else None,
            median_abs_moving_relative_heading_deg=round(float(np.median(np.abs(delta[active]))),5) if len(active) else None,
            median_double_support_heading_deg=round(float(wrap_deg(np.rad2deg(np.median(np.unwrap(np.deg2rad(stable)))))),5) if stable else None,
            max_abs_moving_relative_heading_deg=round(float(np.max(np.abs(delta[active]))),5) if len(active) else None)
    except (ValueError,KeyError,IndexError,TypeError) as exc:
        summary['reason']=f'{type(exc).__name__}: {exc}'
    return result
