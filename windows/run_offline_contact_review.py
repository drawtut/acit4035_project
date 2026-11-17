#!/usr/bin/env python3
"""Conservative offline contact review for reference-line FPA sessions.

Does NOT rewrite the production detector, imply clinical ground truth, or turn
unconfirmed impact peaks into accepted footsteps.  Potential missed landings
are written as PROVISIONAL_REVIEW_ONLY with low-gyro heading estimates.

Use: python run_offline_contact_review.py participant.zip --aligned-start --output audit
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.spatial.transform import Rotation
from fpa_unified_pipeline import process_csv_pair
from fpa_reference_line import _horizontal_heading


def _session_folder(source, temp):
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix.lower() == '.zip':
        with ZipFile(source) as z:
            for info in z.infolist():
                member = Path(info.filename)
                if member.is_absolute() or '..' in member.parts or '\\' in info.filename:
                    raise ValueError(f'Unsafe path in archive: {info.filename!r}')
            z.extractall(temp)
        roots = list(Path(temp).rglob('left_raw.csv'))
    elif source.is_dir():
        roots = list(source.rglob('left_raw.csv'))
    else:
        raise ValueError('Provide a raw participant ZIP or extracted session directory')
    if len(roots) != 1 or not (roots[0].parent / 'right_raw.csv').is_file():
        raise ValueError('Exactly one pair of left_raw.csv and right_raw.csv is required')
    return roots[0].parent


def _toe_out_angles(data, side):
    ori = data['orientation']
    t = np.asarray(ori['time_s'], float)
    q = Rotation.from_quat(np.asarray(ori['quaternion_xyzw'], float))
    axis = np.asarray(data['calibration']['sensor_forward_unit'], float)
    az, horizontal = _horizontal_heading(q, axis)
    start, end = ori['initial_reference_s']
    reference = (t >= start) & (t < end)
    base = np.arctan2(np.mean(np.sin(az[reference])), np.mean(np.cos(az[reference])))
    delta = np.rad2deg(np.angle(np.exp(1j * (az - base))))
    if side == 'right':
        delta = -delta
    return t, np.asarray(ori['gyro_magnitude_dps'], float), delta, horizontal


def _round(v, d=3):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def analyze(source, output, *, aligned_start=False):
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = _session_folder(source, tmp)
        cal = folder / 'calibration'
        if not cal.is_dir():
            raise ValueError('Session calibration files required; do not silently substitute another shoe setup')
        results, stage = process_csv_pair(
            folder / 'left_raw.csv', folder / 'right_raw.csv', cal,
            reference_line_mode=aligned_start)

        # Both t axes are relative to each sensor's first sample. Express events
        # in one shared host monotonic axis for correct chronological sorting.
        stream_start = {}
        for side in ('left', 'right'):
            with (folder / f'{side}_raw.csv').open(newline='', encoding='utf-8') as f:
                row = next(csv.DictReader(f))
            stream_start[side] = float(row['timestamp_midpoint_monotonic_s'])
        first_global = min(stream_start.values())
        records = []
        for side in ('left', 'right'):
            data = results[side]
            detector = data['detector_result']
            t, gyro, heading, horizontal = _toe_out_angles(data, side)
            offset = stream_start[side] - first_global
            accepted = [float(r['contact_time_s']) for r in stage[side]['rows']]

            # The production engine's accepted steps are kept unchanged.
            for row in stage[side]['rows']:
                contact = float(row['contact_time_s'])
                records.append({
                    'foot': side, 'contact_time_s': _round(contact + offset, 4),
                    'contact_time_foot_s': _round(contact, 4),
                    'category': {'A_CONFIRMED':'A_PRODUCTION_CONTACT',
                        'B_PROVISIONAL':'B_STANCE_SUPPORTED',
                        'C_REVIEW_ONLY':'C_SOFT_STANCE_REVIEW'}.get(
                            row.get('detector_confidence'),'UNKNOWN_CANDIDATE'),
                    'reference_line_fpa_deg': _round(row.get('fpa_deg'), 3),
                    'fpa_status': row.get('fpa_status'),
                    'impact_prominence_mps2': None,
                    'post_impact_min_gyro_dps': None,
                    'quiet_samples': None,
                    'quiet_heading_p10_deg': None,
                    'quiet_heading_p90_deg': None,
                    'evidence': 'verified_swing_impact_stance_production',
                    'interpretation': 'provisional_angle_vs_initial_tape_line',
                })

            # A very specific early-step diagnostic for the older Phase 8 detector.
            # Phase 10 includes this timing rule already; never double-count it.
            # It still requires the independently established initial stance,
            # valid first swing, a real impact, and a *subsequent* confirmed stance.
            contacts = detector['contacts']
            initial_stances = {int(s['stance_id']): s for s in data['segmentation']['stands']}
            for event in data['detector_events']:
                if (int(event['swing_id']) != 1 or
                    event['status'] != 'needs_review' or
                    'no_impact_near_swing_to_stance' not in event['reason'] or
                    int(event['from_stance_id']) != 1 or
                    event['stance_start_s'] is None or
                    event['stationary_fraction'] < 0.95 or
                    initial_stances.get(1,{}).get('duration_s',0) < 1.0):
                    continue
                landing = initial_stances.get(int(event['to_stance_id']))
                if landing is None or landing.get('duration_s',0) < .16:
                    continue
                st = float(event['stance_start_s'])
                duration = float(event['swing_duration_s'])
                earliest = max(st-.50,
                    float(event['swing_start_s']) + .40*duration,
                    float(event['swing_start_s']) + .10)
                match = [c for c in contacts if earliest<=float(c['time_s'])<=st-.35
                         and not any(abs(float(c['time_s'])-a)<.4 for a in accepted)]
                if not match:continue
                candidate = max(match,key=lambda c:float(c['prominence_mps2']))
                ct = float(candidate['time_s'])
                idx = landing.get('reference_index')
                if idx is None or not 0 <= int(idx) < len(t):continue
                idx=int(idx)
                if float(horizontal[idx]) < .6:continue
                records.append({
                    'foot':side,'contact_time_s':_round(ct+offset,4),
                    'contact_time_foot_s':_round(ct,4),
                    'category':'POSSIBLE_FIRST_STEP_REVIEW',
                    'reference_line_fpa_deg':_round(float(heading[idx]),3) if aligned_start else None,
                    'fpa_status':'PROVISIONAL_CONFIRMED_STANCE_BUT_EXTENDED_IMPACT_TIMING' if aligned_start else 'NO_DECLARED_TAPED_START',
                    'impact_prominence_mps2':_round(float(candidate['prominence_mps2'])),
                    'post_impact_min_gyro_dps':None,
                    'quiet_samples':None,
                    'quiet_heading_p10_deg':None,
                    'quiet_heading_p90_deg':None,
                    'evidence':'initial_stationary_plus_first_swing_plus_late_confirmed_stance',
                    'interpretation':'older_detector_timing_exception_review_not_production',
                })

            # Review missed impacts only. Following movement strength and a short
            # low-gyro period must support them; do NOT call these accepted.
            for c in contacts:
                ct = float(c['time_s'])
                if any(abs(ct - a) < .40 for a in accepted) or any(
                    abs(ct - r['contact_time_foot_s']) < .40 for r in records
                    if r['foot'] == side and r['category'] == 'POSSIBLE_FIRST_STEP_REVIEW'):
                    continue
                later = (t >= ct + .10) & (t <= ct + .48)
                earlier_motion = (t >= ct - .65) & (t <= ct - .12)
                if not later.any() or not earlier_motion.any():
                    continue
                # A swing-like excursion and earlier contact candidate
                # (often toe-off) provide independent temporal evidence.
                if float(np.max(gyro[earlier_motion])) < 100.:
                    continue
                preceding = [p for p in contacts if .30 <= ct - float(p['time_s']) <= .75
                             and float(p['prominence_mps2']) > float(c['prominence_mps2'])]
                if not preceding:
                    continue
                quiet = later & (gyro < 30.) & (horizontal >= .6)
                if int(quiet.sum()) < 4:
                    continue
                if not aligned_start:
                    fpa = None
                    status = 'NO_DECLARED_TAPED_START'
                else:
                    fpa = float(np.median(heading[quiet]))
                    status = 'PROVISIONAL_NO_CONFIRMED_STANCE'
                records.append({
                    'foot': side, 'contact_time_s': _round(ct + offset, 4),
                    'contact_time_foot_s': _round(ct, 4),
                    'category': 'POSSIBLE_MISSED_LANDING_REVIEW',
                    'reference_line_fpa_deg': _round(fpa, 3),
                    'fpa_status': status,
                    'impact_prominence_mps2': _round(float(c['prominence_mps2'])),
                    'post_impact_min_gyro_dps': _round(float(np.min(gyro[later]))),
                    'quiet_samples': int(quiet.sum()),
                    'quiet_heading_p10_deg': _round(float(np.percentile(heading[quiet], 10))),
                    'quiet_heading_p90_deg': _round(float(np.percentile(heading[quiet], 90))),
                    'evidence': 'impact_plus_prior_swing_peak_plus_short_low_gyro_not_full_stationarity',
                    'interpretation': 'potential_contact_unverified_not_for_accuracy_MAE',
                })

        records.sort(key=lambda r:(r['contact_time_s'], r['foot']))
        path = out / 'contact_review.csv'
        fields = ['foot','contact_time_s','contact_time_foot_s','category',
                  'reference_line_fpa_deg','fpa_status','impact_prominence_mps2',
                  'post_impact_min_gyro_dps','quiet_samples','quiet_heading_p10_deg',
                  'quiet_heading_p90_deg','evidence','interpretation']
        with path.open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
        stats = {
            'source': Path(source).name,
            'aligned_start_asserted_by_operator': aligned_start,
            'production_accepted': {s:sum(r['foot']==s and r['category']=='A_PRODUCTION_CONTACT' for r in records)
                                    for s in ('left','right')},
            'possible_extra_for_review': {s:sum(r['foot']==s and r['category'] in ('POSSIBLE_MISSED_LANDING_REVIEW','POSSIBLE_FIRST_STEP_REVIEW') for r in records)
                                          for s in ('left','right')},
            'note': 'Phase13 A is strict impact+stance, B has confirmed stance without impact, C is soft-stance review only; B and C are NOT verified landings. Additional review events are separate.',
            'protocol': 'shoe_axes_aligned_to_tape_at_start_and_walking_progression_parallel_to_tape_assumed',
            'fpa': 'signed toe-out shoe heading vs initial reference, no ZUPT',
            'angle_measurement_at_unconfirmed_contact': 'Phase13 C uses orientation in a short low-gyro period; additional legacy review candidates use <30dps median within 0.10 to 0.48s after impact',
            'calibration': 'session_original',
        }
        (out/'contact_review_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
        return records,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',help='Participant session ZIP or extracted folder')
    p.add_argument('--aligned-start',action='store_true',help='Assert taped 0-degree start and straight lane')
    p.add_argument('--output',default='contact_review')
    args=p.parse_args()
    rows,summary=analyze(args.session,args.output,aligned_start=args.aligned_start)
    print('Accepted:',summary['production_accepted'])
    print('Possible missed, not accepted:',summary['possible_extra_for_review'])
    print('Saved',Path(args.output).resolve()/'contact_review.csv')
    for row in rows:
        print(f"{row['contact_time_s']:7.3f}s {row['foot']:5s} "
              f"{row['category']:34s} FPA={row['reference_line_fpa_deg']}")


if __name__=='__main__':
    main()
