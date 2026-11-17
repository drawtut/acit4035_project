"""Known-straight-walkway orientation reference, NOT a free-walking FPA estimator.

An EXPLICIT protocol flag asserts that both longitudinal shoe axes were
physically aligned to the same straight walking reference at the initial
stationary hold, AND walking progressed parallel to that reference. The
initial fixed yaw reference is taken from the existing raw-six orientation
quaternions and independently calibrated shoe-forward vector. It is never
reset at later stances. A reported walkway angle is only an anatomical FPA
under the external assumption of parallel foot progression.

No changes to frozen filtering or detector. No known
paper angles or fused IMU Euler/magnetometer signals are ingested.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

REFERENCE_SOURCE = 'aligned_shoes_at_initial_stationary_hold; straight_walkway_parallel_to_actual_progression_ASSUMED'


def _wrap_deg(angle):
    return float((float(angle)+180.0) % 360.0-180.0)


def _horizontal_heading(rotations, axis):
    vectors = rotations.apply(axis)
    horizontal = np.linalg.norm(vectors[:, :2], axis=1)
    bearings = np.arctan2(vectors[:, 1], vectors[:, 0])
    return bearings, horizontal


def add_reference_line_fpa(results, stage, *, enabled=False,
                           min_horizontal_fraction=0.6,
                           max_abs_angle_deg=90.0):
    """Annotate Stage4 row copies with walkway-relative landing shoe headings.

    Output requires only accepted contacts and the relative heading.
    But it DOES require an accepted IMU contact + steady initial reference.
    `enabled` must come from an explicit aligned-start research protocol.
    """
    for side in results:
        if side not in ('left','right'):
            raise ValueError('Only left and right foot sides supported')
        data=results[side]
        x=stage[side]
        rows=x['rows']
        for row in rows:
            row.update(reference_line_fpa_candidate_toe_out_deg=None,
                       reference_line_heading_change_ccw_deg=None,
                       reference_line_status='NOT_REQUESTED',
                       reference_line_end_stance_s=None,
                       reference_line_initial_zero_assumed=bool(enabled),
                       reference_line_requires_parallel_walking=True)
        metadata=dict(enabled=bool(enabled), reference=REFERENCE_SOURCE,
                      baseline_s=None,baseline_heading_deg=None,
                      baseline_yaw_spread_p95_deg=None,
                      baseline_gyro_p90_dps=None,
                      assumption='Not valid as anatomical FPA unless the walked path was parallel to the taped reference line',
                      validated_accuracy=False)
        x['reference_line']=metadata
        if not enabled:
            continue
        cal=data.get('calibration') or {}
        raw_axis=cal.get('sensor_forward_unit')
        if not cal.get('usable') or raw_axis is None:
            for row in rows: row['reference_line_status']='REJECTED_UNVERIFIED_SHOE_AXIS_CALIBRATION'
            continue
        axis=np.asarray(raw_axis,dtype=float)
        if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis)<0.5:
            for row in rows: row['reference_line_status']='REJECTED_INVALID_SHOE_AXIS'
            continue
        axis=axis/np.linalg.norm(axis)
        ori=data['orientation']
        t=np.asarray(ori['time_s'],dtype=float)
        q=np.asarray(ori['quaternion_xyzw'],dtype=float)
        if t.ndim!=1 or q.shape!=(len(t),4) or len(t)<50:
            for row in rows: row['reference_line_status']='REJECTED_MISSING_ORIENTATION_TIMELINE'
            continue
        r=np.asarray(ori['initial_reference_s'],dtype=float)
        if r.shape!=(2,) or not 0 <= r[0] < r[1]:
            for row in rows: row['reference_line_status']='REJECTED_NO_INITIAL_STILL_REFERENCE'
            continue
        ref=(t>=r[0])&(t<r[1]);metadata['baseline_s']=r.tolist()
        if int(ref.sum())<30 or (r[1]-r[0])<1.0:
            for row in rows: row['reference_line_status']='REJECTED_SHORT_INITIAL_REFERENCE'
            continue
        az,horiz=_horizontal_heading(Rotation.from_quat(q),axis)
        gyro=np.asarray(ori['gyro_magnitude_dps'],dtype=float)
        if gyro.shape!=(len(t),):
            for row in rows: row['reference_line_status']='REJECTED_MISSING_REFERENCE_GYRO'
            continue
        base=float(np.arctan2(np.mean(np.sin(az[ref])),np.mean(np.cos(az[ref]))))
        deviation=np.rad2deg(np.angle(np.exp(1j*(az[ref]-base))))
        spread=float(np.percentile(abs(deviation),95))
        p90=float(np.percentile(gyro[ref],90))
        metadata.update(baseline_heading_deg=float(np.rad2deg(base)),
                        baseline_yaw_spread_p95_deg=spread,
                        baseline_gyro_p90_dps=p90,
                        baseline_horizontal_fraction=float(np.median(horiz[ref])))
        if (not np.all(np.isfinite(az[ref])) or np.median(horiz[ref])<min_horizontal_fraction
                or spread>3.0 or p90>12.0):
            for row in rows: row['reference_line_status']='REJECTED_MOVING_OR_UNSTABLE_INITIAL_REFERENCE'
            continue
        links={int(link['swing_id']):link for link in data['accepted_links']}
        source_t=np.asarray(data.get('filtered',{}).get('source_time_s',t),dtype=float)
        for row in rows:
            link=links.get(int(row['swing_id']))
            if not link or not link.get('usable'):
                row['reference_line_status']='REJECTED_NO_ACCEPTED_LANDING_STANCE'
                continue
            idx=link['end_index']
            if idx is None or not 0 <= int(idx)<len(t):
                row['reference_line_status']='REJECTED_MISSING_LANDING_ORIENTATION'
                continue
            idx=int(idx)
            row['reference_line_end_stance_s']=float(t[idx])
            if horiz[idx]<min_horizontal_fraction:
                row['reference_line_status']='REJECTED_SHOE_AXIS_NEAR_VERTICAL_AT_LANDING'
                continue
            ccw=_wrap_deg(np.rad2deg(az[idx]-base))
            toe_out=ccw*(1.0 if side=='left' else -1.0)
            row['reference_line_heading_change_ccw_deg']=ccw
            if abs(toe_out)>max_abs_angle_deg:
                row['reference_line_status']='REJECTED_REFERENCE_LINE_ANGLE_OUTSIDE_FORWARD_WALKING_RANGE'
                continue
            # Verify the REAL serial timestamps. Interpolated grid alone can
            # conceal missing samples, so never treat it as evidence of quality.
            if len(source_t)>=2:
                subset=source_t[(source_t>=float(r[0]))&(source_t<=t[idx])]
                largest=float(np.max(np.diff(subset))) if len(subset)>=2 else float('inf')
            else:
                largest=float('inf')
            if largest>0.060:
                row['reference_line_status']='REJECTED_SEVERE_SAMPLING_GAP'
                continue
            row['reference_line_fpa_candidate_toe_out_deg']=float(toe_out)
            row['reference_line_status']=('PROVISIONAL_SAMPLING_GAP' if largest>0.030
                   else 'PROVISIONAL_WALKWAY_DIRECTION_ASSUMED')
    return stage
