"""Phase 13: IMU-only multi-evidence foot contacts, no forced foot alternation.

A: strong swing + impact within 0.55 s before stable stance (confirmed candidate)
B: strong swing + stable stance without matching impact (provisional)
C: paired impacts + motion + short quiet gravity-consistent hold in an
   otherwise merged swing (review only).
All parameters are *event decision* rules. Original frozen filtering, calibration,
raw-six orientation and taped reference-line FPA are unchanged.

No ground-truth sequence, participant identity or paper angle is an input.
"""
from __future__ import annotations
import numpy as np
from fpa_combined_detector import detect as detect_combined

PARAMS = dict(
    impact_lookback_s=.55,
    stance_min_fraction=.60,
    stance_min_duration_s=.16,
    stance_max_median_gyro_dps=35.,
    min_same_foot_contact_spacing_s=.42,
    soft_min_earlier_gyro_dps=100.,
    soft_prior_peak_min_s=.30,
    soft_prior_peak_max_s=.75,
    soft_min_gyro_quiet_s=.12,
    soft_quiet_gyro_dps=80.,
    soft_quiet_window_start_s=.15,
    soft_quiet_window_end_s=.55,
    soft_min_gravity_fraction=.70,
    soft_gravity_tolerance_mps2=1.0,
    soft_min_swing_duration_s=1.7,
)


def detect(result, segmentation, *, include_soft=True, overrides=None):
    p=dict(PARAMS)
    if overrides:p.update(overrides)
    out=detect_combined(result,segmentation,overrides={
        'impact_before_stance_s':p['impact_lookback_s'],
        'first_swing_impact_before_stance_s':p['impact_lookback_s']})
    stands={int(s['stance_id']):s for s in segmentation['stands']}
    for e in out:
        e['confidence']='A_CONFIRMED' if e['status']=='accepted_candidate' else 'REJECTED'
        e['evidence_class']='swing_impact_stance' if e['status']=='accepted_candidate' else 'rejected'
        e['landing_orientation_index']=None
        e['temporal_uncertainty_s']=None
        if e['status']=='accepted_candidate':
            st=stands.get(int(e['to_stance_id']))
            e['landing_orientation_index']=st.get('reference_index') if st else None
            e['temporal_uncertainty_s']=.08
            continue
        # Only relax *missing impact*, never missing swing, stance, stability,
        # bad reference, or multiple other rejection reasons.
        if (e['reason']=='no_impact_near_swing_to_stance' and
            e['stance_start_s'] is not None and
            e['stationary_fraction']>=p['stance_min_fraction'] and
            np.isfinite(e['post_stance_median_gyro_dps']) and
            e['post_stance_median_gyro_dps']<=p['stance_max_median_gyro_dps']):
            st=stands.get(int(e['to_stance_id']))
            if (st is not None and st['duration_s']>=p['stance_min_duration_s']
                    and st.get('reference_index') is not None):
                e['status']='stance_supported_candidate'
                e['confidence']='B_PROVISIONAL'
                e['evidence_class']='swing_full_stance_without_matched_impact'
                e['reason']='PROVISIONAL_VALID_SWING_CONFIRMED_STANCE_NO_IMPACT'
                e['contact_time_s']=float(st['start_s'])
                e['contact_index']=int(st['start_index'])
                e['landing_orientation_index']=int(st['reference_index'])
                e['temporal_uncertainty_s']=.30
    if not include_soft or 'accel_magnitude' not in result:
        return out

    t=np.asarray(result['t'],float)
    gyro=np.asarray(result['gyro_magnitude'],float)
    accel=np.asarray(result['accel_magnitude'],float)
    peaks=result['contacts']
    if len(t)<30:return out
    # Exclude all already supported A/B events (not only confirmed ones), and
    # never insert an event within the same-foot refractory interval.
    accepted=[float(e['contact_time_s']) for e in out if
              e['confidence'].startswith(('A_','B_')) and e['contact_time_s'] is not None]
    generated=[]
    next_id=1+max((int(e['swing_id']) for e in out),default=0)
    # C applies ONLY to exceptionally long gyro activity between existing full
    # stances, where normal segmentation is known to merge separate strides.
    for sw in segmentation['swings']:
        if float(sw['duration_s'])<p['soft_min_swing_duration_s']:
            continue
        start=float(sw['start_s']);stop=float(sw['end_s'])
        for candidate in peaks:
            ct=float(candidate['time_s'])
            if not start+.2<=ct<=stop-.15:continue
            if any(abs(ct-v)<p['min_same_foot_contact_spacing_s'] for v in accepted+generated):continue
            # Independent preceding toe-off-like peak and gyroscopic excursion.
            preceding=[prev for prev in peaks if
                       p['soft_prior_peak_min_s']<=ct-float(prev['time_s'])<=p['soft_prior_peak_max_s']
                       and float(prev['prominence_mps2'])>float(candidate['prominence_mps2'])]
            if not preceding:continue
            moving=(t>=ct-.65)&(t<=ct-.12)
            if not np.any(moving) or float(np.max(gyro[moving]))<p['soft_min_earlier_gyro_dps']:
                continue
            next_window=(t>=ct+p['soft_quiet_window_start_s']) & (t<=ct+p['soft_quiet_window_end_s'])
            if int(np.count_nonzero(next_window))<10:continue
            # Continuous quiet run, not an aggregate count of isolated low
            # samples separated by more movement.
            quiet=next_window&(gyro<p['soft_quiet_gyro_dps'])
            indices=np.flatnonzero(quiet)
            if len(indices)<2:continue
            cuts=np.flatnonzero(np.diff(indices)>1)+1
            runs=np.split(indices,cuts)
            runs=[r for r in runs if len(r)>1 and t[r[-1]]-t[r[0]]>=p['soft_min_gyro_quiet_s']]
            if not runs:continue
            # Require MOST post-impact acceleration to be gravity-consistent,
            # and a continuous low-gyro run. Applying both gates per sample
            # would fragment otherwise convincing short stances.
            grav_ratio=float(np.mean(np.abs(accel[next_window]-9.80665)<p['soft_gravity_tolerance_mps2']))
            if grav_ratio<p['soft_min_gravity_fraction']:continue
            run=max(runs,key=len)
            ref_idx=int(run[len(run)//2])
            out.append(dict(swing_id=next_id,from_stance_id=int(sw['from_stance_id']),
                to_stance_id=int(sw['to_stance_id']),status='soft_stance_review',
                confidence='C_REVIEW_ONLY',evidence_class='paired_impact_motion_short_quiet',
                reason='REVIEW_UNCONFIRMED_SHORT_STANCE_INSIDE_MERGED_SWING',
                contact_time_s=ct,contact_index=int(candidate['index']),
                impact_rms_mps2=float(candidate['rms_mps2']),
                impact_prominence_mps2=float(candidate['prominence_mps2']),
                impact_candidates_near_transition=1,
                swing_start_s=start,swing_end_s=stop,
                swing_peak_gyro_dps=float(sw['gyro_peak_dps']),
                swing_duration_s=float(sw['duration_s']),
                swing_excursion_deg=float(sw['integrated_angular_motion_deg']),
                v21_rejection_details=str(sw.get('reason','')),
                stance_start_s=float(t[run[0]]),stance_end_s=float(t[run[-1]]),
                stationary_fraction=0.0,
                post_stance_median_gyro_dps=float(np.median(gyro[run])),
                landing_orientation_index=ref_idx,
                gravity_consistent_fraction=grav_ratio,
                short_quiet_duration_s=float(t[run[-1]]-t[run[0]]),
                temporal_uncertainty_s=.35))
            generated.append(ct)
            next_id+=1
    return sorted(out,key=lambda e:(e['contact_time_s'] if e['contact_time_s'] is not None else e['swing_end_s'],e['swing_id']))
