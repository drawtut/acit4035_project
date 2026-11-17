"""ACIT4035 OFFLINE stages 1–3: reuse frozen candidate detector,
functional sensor-to-foot alignment and orientation contact references.

Only six measured IMU channels and their timestamps are used. No magnetic
measurements, fused Euler angles, video timing, manually labeled steps, or
known angle references. No anatomical FPA is reported at this stage.

IMPORTANT: PCA's horizontal forward sign is fundamentally ambiguous. A chosen
mount-forward sensor axis is a *hypothesis* that MUST be verified on the shoe.
No reconstructed displacement is used by the production pipeline.
"""
from __future__ import annotations

import hashlib
import numpy as np

from fpa_raw6_filters import filter_raw_six_axis
from fpa_timebase import prepare_timebase
from fpa_verified_sensor_axes import to_printed_axes
from fpa_raw6_orientation import estimate_six_axis_orientation
from fpa_segmentation_v21 import detect as detect_swings
from fpa_phase13_detector import detect as detect_multievidence
import fpa_processing as shared

MOUNT_AXES = {
    '+X': np.array([1., 0., 0.]), '-X': np.array([-1., 0., 0.]),
    '+Y': np.array([0., 1., 0.]), '-Y': np.array([0., -1., 0.]),
}


def accepted_event_links(t, detector_result, segmentation, combined_events):
    """One IMU-only link for each accepted step: preceding stance -> landing stance.

    Crucially this DOES NOT use independent legacy stride-pair associations.
    Rejected contacts cannot cause a trajectory to be reconstructed.
    """
    stands = {int(x['stance_id']): x for x in segmentation['stands']}
    links = []
    for event in combined_events:
        if event['status'] not in ('accepted_candidate','stance_supported_candidate','soft_stance_review'):
            continue
        a = stands.get(int(event['from_stance_id']))
        b = stands.get(int(event['to_stance_id']))
        if a is None or b is None:
            continue
        i, j = a['reference_index'], (event.get('landing_orientation_index') if event.get('landing_orientation_index') is not None else b['reference_index'])
        if i is None or j is None or int(j) <= int(i):
            links.append(dict(swing_id=int(event['swing_id']), step_time_s=event['contact_time_s'],
                              from_stance_id=int(a['stance_id']), to_stance_id=int(b['stance_id']),
                              usable=False, reason='missing_stable_reference',
                              start_index=None, end_index=None,
                              confidence=event.get('confidence','REJECTED')))
            continue
        i, j = int(i), int(j)
        if j >= len(t) or i < 0: continue
        quality=event.get('confidence','A_CONFIRMED')
        links.append(dict(swing_id=int(event['swing_id']),step_time_s=float(event['contact_time_s']),
                          from_stance_id=int(a['stance_id']),to_stance_id=int(b['stance_id']),
                          start_stance_s=float(a['reference_s']),end_stance_s=float(t[j]),
                          stance_start_s=float(event['stance_start_s']),
                          stance_end_s=float(event['stance_end_s']),
                          start_index=i,end_index=j,
                          swing_start_s=float(event['swing_start_s']),
                          swing_end_s=float(event['swing_end_s']),
                          usable=True,reason=event['reason'],
                          confidence=quality,evidence_class=event.get('evidence_class'),
                          temporal_uncertainty_s=event.get('temporal_uncertainty_s')))
    return links


def functional_foot_axis(filtered, orientation, links, side, mount_forward='+X',
                         minimum_swings=4, minimum_ratio=1.3,
                         max_calibration_swings=4):
    """PCA gyro rotations from IMU-confirmed swings + gravity level.

    The mounting-axis sign is a user-supplied *assumption*, not ground truth.
    Uses the same data for exploratory calibration and trajectories unless a
    separate compatible calibration recording is supplied later.
    """
    if mount_forward not in MOUNT_AXES:
        raise ValueError('Mount forward axis must be +X, -X, +Y, or -Y')
    t = filtered['time_s']
    gyro = np.asarray(filtered['gyro_filtered'], dtype=float)
    accel = np.asarray(filtered['accel_filtered'], dtype=float)
    selected = []
    used_ids = []
    for link in links:
        if not link['usable']:
            continue
        lo = np.searchsorted(t, link['swing_start_s'], side='left')
        hi = np.searchsorted(t, link['swing_end_s'], side='right')
        if hi <= lo:
            continue
        w = gyro[lo:hi]
        speed = np.linalg.norm(w,axis=1)
        # Avoid stationary noise and the most extreme impact-induced impulses.
        ok = (speed >= 35.) & (speed <= 260.)
        if np.count_nonzero(ok) >= 10:
            selected.append(w[ok]); used_ids.append(int(link['swing_id']))
            if len(selected) >= max_calibration_swings:
                break
    result = dict(schema='ACIT4035_RAW6_FUNCTIONAL_CALIBRATION_V1',side=side,
                  assumption_mount_forward_axis=mount_forward,
                  mounting_direction_physically_verified=False,
                  separate_calibration_trial=False,
                  usable=False, total_accepted_swings=len(links),
                  selection='first_four_IMU_accepted_swings_not_independently_verified_as_forward',
                  swings_used=len(used_ids), swing_ids=used_ids,
                  pca_eigenvalue_ratio=None, sensor_forward_unit=None,
                  sensor_up_reference=None, sensor_mediolateral_unit=None,
                  alignment_with_mount_axis=None,
                  warning='Cannot claim anatomical foot direction before verifying shoe mounting and independent calibration.')
    if len(selected) < minimum_swings:
        result['reason']='fewer_than_four_usable_accepted_swings'
        return result
    data=np.concatenate(selected,axis=0)
    if len(data) < 80:
        result['reason']='insufficient_gyro_samples_during_swings'
        return result
    # Principal motion axis is foot ML/pitch rotation. Center to remove bias.
    demeaned = data - np.mean(data,axis=0)
    _, singular, vt = np.linalg.svd(demeaned, full_matrices=False)
    ratio=float(singular[0]**2/max(singular[1]**2,1e-12))
    result['pca_eigenvalue_ratio']=ratio
    start,end=orientation['initial_reference_s']
    ref=(t>=start)&(t<end)
    up=np.median(accel[ref],axis=0)
    up_norm=float(np.linalg.norm(up))
    if up_norm<7.5 or up_norm>12.:
        result['reason']='implausible_stationary_gravity_magnitude'
        return result
    up/=up_norm
    ml=vt[0]-float(np.dot(vt[0],up))*up
    magnitude=float(np.linalg.norm(ml))
    if magnitude < .2:
        result['reason']='pca_axis_parallel_to_gravity'
        return result
    ml/=magnitude
    forward=np.cross(up,ml)
    forward/=np.linalg.norm(forward)
    mount=MOUNT_AXES[mount_forward]
    if np.dot(forward,mount)<0: forward=-forward;ml=-ml
    mount_alignment=float(np.dot(forward,mount))
    # Right-handed foot frame: X=forward, Y=lateral, Z=up.  All vectors
    # are expressed in sensor coordinates. Do NOT interpret X as true toe
    # direction until the mounting-axis sign assumption is physically verified.
    foot_lateral=-ml
    foot_to_sensor=np.column_stack([forward,foot_lateral,up])
    sensor_to_foot=foot_to_sensor.T
    result.update(sensor_forward_unit=forward.tolist(),sensor_up_reference=up.tolist(),
                  sensor_mediolateral_unit=ml.tolist(),
                  sensor_lateral_unit=foot_lateral.tolist(),
                  sensor_to_foot_rotation_matrix=sensor_to_foot.tolist(),
                  rotation_matrix_determinant=float(np.linalg.det(sensor_to_foot)),
                  alignment_with_mount_axis=mount_alignment)
    if ratio<minimum_ratio:
        result['reason']='ambiguous_pca_motion_axis'
    elif mount_alignment<.5:
        result['reason']='mount_axis_and_pca_forward_disagree'
    else:
        result['usable']=True
        result['reason']='PROVISIONAL_pca_axis_requires_physical_sign_verification'
    return result


def process_side(t, accel, gyro, *, side, detector_config, lab_config,
                 orientation_config, mount_forward='+X', external_calibration=None):
    """Stage 1 uses SAME frozen combined detector; stages 2–3 use filtered raw6."""
    if side not in ('left','right'):raise ValueError('Side must be left or right')
    # Fingerprint measured channels to prevent accidental same-trial calibration/evaluation.
    source_sha=hashlib.sha256(
        np.ascontiguousarray(np.column_stack([t,accel,gyro]),dtype='<f8').tobytes()).hexdigest()
    original_t=np.asarray(t,dtype=float).copy()
    t,accel,gyro,timebase=prepare_timebase(t,accel,gyro)
    r=shared.run(t,accel,gyro,detector_config)
    segments=detect_swings(r)
    events=detect_multievidence(r,segments)
    links=accepted_event_links(t,r,segments,events)
    # Keep the frozen detector on ORIGINAL reported raw6 data above.
    # Map ONLY the downstream FPA branch to physical printed axes (+Y toe).
    # The map is a rigid rotation, not a filter, and is applied once before
    # raw-channel filtering / gyro-bias / quaternion integration.
    printed_accel,printed_gyro=to_printed_axes(accel,gyro,installation=1)
    filtered=filter_raw_six_axis(t,printed_accel,printed_gyro,**lab_config)
    filtered['source_time_s']=original_t
    filtered['timebase_mode']=timebase['mode']
    orientation=estimate_six_axis_orientation(filtered,**orientation_config)
    # Functional calibration only uses fully confirmed A contacts, never
    # provisional/review steps. Evaluation can still include B/C contacts.
    calibration_links=[x for x in links if x.get('confidence')=='A_CONFIRMED']
    calibration=(functional_foot_axis(filtered,orientation,calibration_links,side,mount_forward)
                 if external_calibration is None else dict(external_calibration))
    if external_calibration is not None:
        if calibration.get('source_raw6_sha256') == source_sha:
            raise ValueError('Calibration and evaluation use the SAME raw recording; use independent walking trial')
        if not calibration.get('separate_calibration_trial') or calibration.get('side') != side:
            raise ValueError('External calibration side/source mismatch')
        if calibration.get('assumption_mount_forward_axis') != mount_forward:
            raise ValueError('External calibration mounting axis mismatch')
    contacts=[dict(side=side, swing_id=link['swing_id'],
                   contact_time_s=link['step_time_s'],
                   start_stance_s=link.get('start_stance_s'),
                   end_stance_s=link.get('end_stance_s'),
                   usable=bool(link.get('usable')),
                   confidence=link.get('confidence','A_CONFIRMED'),
                   reason=link.get('reason')) for link in links]
    return dict(side=side, detector_result=r,segmentation=segments,
                detector_events=events,accepted_links=links,
                filtered=filtered,orientation=orientation,calibration=calibration,
                contact_rows=contacts,
                raw6_axis_convention='VERTICAL_0x23_1_TO_PRINTED_Rx_NEG90_20261115',
                source_raw6_sha256=source_sha,timebase=timebase,
                imu_only=True,video_used=False,fused_angles_used=False)




def export_stage123(results, output_dir, *, detector_config, lab_config,
                    orientation_config, mount_axes):
    """Export contact, calibration, timebase and orientation diagnostics.

    NO velocity, trajectory, displacement or derived progression is calculated.
    Input raw CSVs are never altered.
    """
    import json
    from pathlib import Path
    import pandas as pd
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest={
        'schema':'ACIT4035_STAGE123_CONTACT_ORIENTATION_V2',
        'stage_1':'frozen raw6 impact/swing/stationary-contact detection',
        'stage_2':'independent functional shoe-forward axis calibration',
        'stage_3':'gyro integration and gravity-assisted shoe orientation timeline',
        'FPA_COMPUTED':False,
        'input_channels':['ax_mps2','ay_mps2','az_mps2','gx_dps','gy_dps','gz_dps','timestamps'],
        'detector_config':detector_config,'fpa_filter_config':lab_config,
        'sensor_coordinate_transform':{'applied_only_to_orientation_branch':True,
            'raw_accel_and_gyro_same_R':[[1,0,0],[0,0,1],[0,-1,0]],
            'physical_toes_axis':'+Y','requires_installation_register_0x23':1},
        'orientation_config':orientation_config,'mount_axes':mount_axes,
        'analysis':{},'video_used_for_estimator':False,
        'sensor_fused_euler_used':False,'magnetometer_used':False,
    }
    for side,x in results.items():
        pd.DataFrame(x['detector_events']).to_csv(out/f'{side}_all_detected_events.csv',index=False)
        pd.DataFrame(x['contact_rows']).to_csv(out/f'{side}_all_contact_candidates.csv',index=False)
        pd.DataFrame([r for r in x['contact_rows'] if r.get('confidence')=='A_CONFIRMED']).to_csv(
            out/f'{side}_accepted_footsteps.csv',index=False)
        (out/f'{side}_foot_axis_INDEPENDENT.json').write_text(
            json.dumps(x['calibration'],indent=2),encoding='utf-8')
        ori=x['orientation'];t=ori['time_s']
        pd.DataFrame(dict(time_s=t,
            relative_sensor_azimuth_deg=ori['relative_sensor_axis_azimuth_deg'],
            gyro_magnitude_dps=ori['gyro_magnitude_dps'],
            accelerometer_trusted=ori['trusted_tilt_correction'].astype(int),
            quaternion_x=ori['quaternion_xyzw'][:,0],
            quaternion_y=ori['quaternion_xyzw'][:,1],
            quaternion_z=ori['quaternion_xyzw'][:,2],
            quaternion_w=ori['quaternion_xyzw'][:,3])).to_csv(
                out/f'{side}_raw6_orientation.csv',index=False)
        manifest['analysis'][side]={
            'timebase':x.get('timebase'),
            'total_detector_events':len(x['detector_events']),
            'accepted_contact_events':sum(e['status']=='accepted_candidate' for e in x['detector_events']),
            'linked_accepted_steps':sum(x.get('confidence')=='A_CONFIRMED' for x in x['accepted_links']),
            'total_contact_candidates':len(x['accepted_links']),
            'stance_supported_candidates':sum(y.get('confidence')=='B_PROVISIONAL' for y in x['accepted_links']),
            'soft_stance_review_candidates':sum(y.get('confidence')=='C_REVIEW_ONLY' for y in x['accepted_links']),
            'foot_axis_calibration':x['calibration'],
        }
    (out/'offline_stage123_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def plot_stage123(results):
    """Plot detector contacts, gyroscope magnitude and shoe orientation only."""
    from matplotlib.figure import Figure
    fig=Figure(figsize=(13,8),dpi=100);axs=fig.subplots(2,2)
    for c,side in enumerate(('left','right')):
        data=results[side];ori=data['orientation'];t=ori['time_s']
        ax=axs[0,c]
        ax.plot(t,ori['relative_sensor_axis_azimuth_deg'],color='#187caa',lw=1)
        for link in data['accepted_links']:
            ax.axvline(link['step_time_s'],color='#238a63',alpha=.5,lw=.8)
        ax.set(title=side.upper()+' orientation + accepted contacts',xlabel='Time (s)',ylabel='Relative sensor heading (deg)')
        ax.grid(alpha=.2)
        ax=axs[1,c]
        ax.plot(t,ori['gyro_magnitude_dps'],color='#9059a6',lw=1)
        ax.set(xlabel='Time (s)',ylabel='Gyroscope magnitude (deg/s)')
        ax.grid(alpha=.2)
    fig.suptitle('ACIT4035 raw6 contact detection and orientation — no displacement reconstruction')
    fig.tight_layout(rect=[0,.0,1,.95]);return fig
