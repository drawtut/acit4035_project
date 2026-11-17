"""Stance-led experimental gait segmentation, shared between plotting and pipeline.

Uses fpa_processing.run() stationary masks and gyro magnitude; does not modify
sensor filtering or assert that a detected swing is ground-truth foot contact.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d

DEFAULTS = dict(
    merge_gap_max_s=0.14,
    merge_gap_max_peak_gyro_dps=80.0,
    reference_min_s=0.09,
    reference_offset_s=0.18,
    min_swing_duration_s=0.25,
    min_swing_peak_gyro_dps=90.0,
    min_swing_angular_excursion_deg=35.0,
    min_stride_s=0.60,
    max_stride_s=2.70,
)


def settings(config):
    out = dict(DEFAULTS)
    out.update(config.get('swing_segmentation', {}))
    return out


def _reference_index(t, interval, mode, params):
    i,j=interval
    min_n=max(2,round(float(params['reference_min_s'])/np.median(np.diff(t))))
    if j-i<min_n:
        return None
    offset=min(round(float(params['reference_offset_s'])/np.median(np.diff(t))), max(0,(j-i)//2))
    if mode=='initial':
        # Prefer end of long initial standing period, immediately before swing.
        return j-1-offset
    if mode=='final':
        # Prefer early stable samples after final settling, not middle of prolonged standing.
        return i+offset
    return (i+j-1)//2


def detect(result, walking_window=None):
    t=result['t'];g=result['gyro_magnitude']; envelope=result['impact_envelope']; cfg=settings(result['config'])
    rms_mode=bool(cfg.get('use_gyro_rms_for_swing',False))
    if rms_mode:
        fs=float(result['fs'])
        win=max(3,int(round(float(cfg.get('gyro_rms_window_ms',75.0))*fs/1000))|1)
        # Centered RMS is for OFFLINE tuning only; causal RMS for live use later.
        energy=np.sqrt(np.maximum(0,uniform_filter1d(g*g,size=win,mode='nearest')))
    else:
        energy=g
    original=[(s['start_index'],s['end_index_exclusive']) for s in result['stationary']]
    groups=[]
    for interval in original:
        if not groups:
            groups.append([*interval,1]);continue
        last=groups[-1]
        gap=(t[interval[0]]-t[last[1]-1])
        gap_peak=float(g[last[1]:interval[0]].max()) if interval[0]>last[1] else 0.
        if gap<=cfg['merge_gap_max_s'] and gap_peak<=cfg['merge_gap_max_peak_gyro_dps']:
            last[1]=interval[1]; last[2]+=1
        else:
            groups.append([*interval,1])
    stands=[]
    if walking_window:
        walking_start,walking_end=map(float,walking_window)
    else:
        walking_start,walking_end=0.,float(t[-1])
    for number,(i,j,parts) in enumerate(groups):
        # Select a nearby endpoint for long initial/final stationary intervals.
        kind='initial' if number==0 and t[i]<=walking_start+.10 else ('final' if number==len(groups)-1 and t[j-1]>=walking_end-.10 else 'intermediate')
        ref=_reference_index(t,(i,j),kind,cfg)
        stands.append(dict(stance_id=number+1,start_index=i,end_index_exclusive=j,
                           start_s=float(t[i]),end_s=float(t[j-1]),duration_s=float(t[j-1]-t[i]),
                           reference_index=ref,reference_s=float(t[ref]) if ref is not None else None,
                           merged_fragments=parts,kind=kind))
    swings=[]; strides=[]
    for number,(a,b) in enumerate(zip(stands,stands[1:]),1):
        start=a['end_index_exclusive'];end=b['start_index']
        if end<=start:continue
        segment=g[start:end]
        length=float(t[end-1]-t[start]);peak=float(np.max(segment))
        excursion=float(np.trapezoid(segment,t[start:end]))
        # A swing must cause a meaningful foot rotation; do not force expected counts.
        rms_peak=float(np.max(energy[start:end]))
        if rms_mode:
            # Simple, tunable RMS movement evidence replaces the TWO hard
            # peak/excursion criteria, for every swing including the first.
            duration_ok=length>=float(cfg.get('gyro_rms_min_swing_s',0.45))
            peak_ok=rms_peak>=float(cfg.get('gyro_rms_threshold_dps',40.0))
            excursion_ok=True
        else:
            duration_ok=length>=cfg['min_swing_duration_s']
            peak_ok=peak>=cfg['min_swing_peak_gyro_dps']
            excursion_ok=excursion>=cfg['min_swing_angular_excursion_deg']
        valid=duration_ok and peak_ok and excursion_ok
        impacts=[p for p in result['contacts'] if t[start]<=p['time_s']<=b['start_s']]
        # Impact is optional and merely a timing cue. Select latest plausible candidate.
        # Do not use it to create stride boundaries.
        landing_candidates=[p for p in impacts if p['time_s']>=t[start]+length*0.50]
        impact=max(landing_candidates,key=lambda p:p['prominence_mps2']) if landing_candidates else None
        reasons=[]
        if not duration_ok: reasons.append('short_movement')
        if not peak_ok: reasons.append('weak_gyro_rms' if rms_mode else 'weak_rotation')
        if not excursion_ok: reasons.append('small_integrated_rotation')
        if a['reference_index'] is None or b['reference_index'] is None: reasons.append('missing_stable_reference')
        end_of_walk=b['start_s']>walking_end or a['end_s']>walking_end
        starts_before_walk=t[start]<walking_start
        if end_of_walk:reasons.append('after_walking_window')
        if starts_before_walk:reasons.append('before_walking_window')
        # Special flag: last foot plant after a long swing may be a terminal stopping step.
        terminal=b['kind']=='final'
        if terminal:reasons.append('terminal_stopping_candidate')
        swings.append(dict(swing_id=number,from_stance_id=a['stance_id'],to_stance_id=b['stance_id'],
            start_s=float(t[start]),end_s=float(t[end-1]),duration_s=length,
            gyro_peak_dps=peak,gyro_rms_peak_dps=rms_peak,integrated_angular_motion_deg=excursion,
            impact_candidates=len(impacts),associated_impact_s=impact['time_s'] if impact else None,
            valid_swing=bool(valid),terminal_stopping_candidate=terminal,
            status='candidate_swing' if valid else 'rejected_movement',reason=';'.join(reasons) or 'movement_supported'))
        stride_duration=(b['reference_s']-a['reference_s']) if a['reference_s'] is not None and b['reference_s'] is not None else float('nan')
        duration_valid=np.isfinite(stride_duration) and cfg['min_stride_s']<=stride_duration<=cfg['max_stride_s']
        # Distinguish geometry-eligible from steady-walking eligibility.
        geometry_ok=valid and duration_valid and a['reference_index'] is not None and b['reference_index'] is not None
        rejection=[]
        if not valid:rejection.append('movement_not_swing')
        if not duration_valid:rejection.append('stride_duration_out_of_range')
        if a['reference_index'] is None or b['reference_index'] is None:rejection.append('no_reference')
        if terminal:rejection.append('terminal_stopping_candidate')
        if starts_before_walk:rejection.append('starts_before_walking_phase')
        if end_of_walk:rejection.append('ends_after_walking_phase')
        strides.append(dict(start_contact_s=impact['time_s'] if impact else float(t[start]),
            end_contact_s=float(t[end-1]),start_midstance_s=a['reference_s'],end_midstance_s=b['reference_s'],
            start_index=a['reference_index'] if a['reference_index'] is not None else start,
            end_index=b['reference_index'] if b['reference_index'] is not None else end-1,
            duration_s=float(stride_duration),accepted_for_reconstruction=bool(geometry_ok),
            steady_walking_candidate=bool(geometry_ok and not terminal and not starts_before_walk and not end_of_walk),
            reason=';'.join(rejection) if rejection else 'candidate_complete',
            swing_id=number,terminal_stopping_candidate=terminal))
    return dict(stands=stands,swings=swings,strides=strides,params=cfg)
