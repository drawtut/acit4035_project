"""Independent functional foot calibration for ACIT4035 (only raw6+t).

Pure routines usable by offline CLI and future live calibration state machine.
Does not use magnetic data, device Euler yaw, video, or reference walking angles.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'ACIT4035_INDEPENDENT_RAW6_FOOT_CALIBRATION_V1'
AXIS_CONVENTION = 'VERTICAL_0x23_1_TO_PRINTED_Rx_NEG90_20261115'


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def settings_fingerprint(detector_config, lab_config, orientation_config):
    """Reject settings drift between calibration and evaluation (conservative)."""
    obj = {'detector': detector_config, 'raw6_fpa_filters': lab_config,
           'orientation': orientation_config, 'axis_transform': AXIS_CONVENTION}
    return hashlib.sha256(_canonical(obj).encode('utf-8')).hexdigest()


def build_calibration(results, detector_config, lab_config, orientation_config,
                      mounts, *, hardware_installation=1,
                      forward_walk_confirmed=False, mounting_verified=False):
    if hardware_installation != 1:
        raise ValueError('This tested raw6 axis map requires hardware installation register 0x23=1')
    fingerprint = settings_fingerprint(detector_config,lab_config,orientation_config)
    out = {}
    for side in ('left','right'):
        data = results[side]
        cal = copy.deepcopy(data['calibration'])
        if not cal.get('usable') or int(cal.get('swings_used',0)) < 4:
            raise ValueError(f"{side} calibration not accepted: {cal.get('reason','unknown')}; "
                             'need >=4 usable accepted swings and PCA checks')
        if cal.get('assumption_mount_forward_axis') != mounts[side]:
            raise ValueError(f'{side} mounting axis mismatch')
        # The functional trial is assumed straight FORWARD by operator, not machine-measured.
        # PCA quality is a diagnostic and does not prove this assumption.
        cal['schema'] = SCHEMA
        cal['separate_calibration_trial'] = True
        cal['source_kind'] = 'separate_straight_walking_trial'
        cal['calibration_quality_status'] = 'PCA_PASS_STRAIGHT_WALK_UNVERIFIED'
        cal['forward_walk_confirmed_by_operator'] = bool(forward_walk_confirmed)
        cal['physical_mount_verified_by_operator'] = bool(mounting_verified)
        cal['mounting_direction_physically_verified'] = bool(mounting_verified)
        cal['hardware_installation_register_0x23'] = 1
        cal['sensor_axis_convention'] = AXIS_CONVENTION
        cal['settings_sha256'] = fingerprint
        # Filter/segmentation cutoffs are unchanged, but uniform-time processing
        # changes numerical results. Make this independent-method version explicit.
        cal['timebase_method'] = data.get('timebase',{}).get('schema')
        cal['timebase_mode_during_calibration'] = data.get('timebase',{}).get('mode')
        cal['detector_configuration_frozen'] = True
        cal['source_raw6_sha256'] = data['source_raw6_sha256']
        cal['used_measured_inputs']=['ax_mps2','ay_mps2','az_mps2',
                                     'gx_dps','gy_dps','gz_dps','IMU timestamps']
        cal['independent_accuracy_validated'] = False
        cal['warning'] = ('Separate functional calibration passed provisional PCA criteria. '
                          'Straight walking and shoe installation require operator confirmation; '
                          'independent anatomical FPA accuracy is unvalidated.')
        out[side] = cal
    return out


def save_pair(pair, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for side in ('left','right'):
        (folder/f'{side}_foot_calibration.json').write_text(
            json.dumps(pair[side],indent=2,allow_nan=False),encoding='utf-8')
    return folder


def load_pair(folder, detector_config, lab_config, orientation_config, mounts,
              *, require_operator_straight=True):
    folder=Path(folder)
    fingerprint=settings_fingerprint(detector_config,lab_config,orientation_config)
    out={}
    for side in ('left','right'):
        filename=folder/f'{side}_foot_calibration.json'
        cal=json.loads(filename.read_text(encoding='utf-8'))
        if cal.get('schema') != SCHEMA or not cal.get('separate_calibration_trial'):
            raise ValueError(f'{filename}: not an independent calibration file')
        if not cal.get('usable') or cal.get('settings_sha256') != fingerprint:
            raise ValueError(f'{filename}: invalid calibration or DIFFERENT detector/filter/orientation settings')
        # Prevent the production engine from silently applying legacy foot axes
        # learned using the old irregular-sample filter timing.
        if cal.get('timebase_method') != 'PHASE3_UNIFORM_RAW6_V1':
            raise ValueError(f'{filename}: calibrations must be regenerated with Phase 3 raw6 timebase processing')
        if cal.get('hardware_installation_register_0x23') != 1 or cal.get('sensor_axis_convention') != AXIS_CONVENTION:
            raise ValueError(f'{filename}: unsupported IMU installation/mapping')
        if cal.get('assumption_mount_forward_axis') != mounts[side]:
            raise ValueError(f'{filename}: mounting direction changed')
        if require_operator_straight and not cal.get('forward_walk_confirmed_by_operator'):
            raise ValueError(f'{filename}: straight-forward functional walk not confirmed; review and set flag in JSON')
        axis=np.asarray(cal.get('sensor_forward_unit'),dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-3:
            raise ValueError(f'{filename}: invalid foot-forward unit vector')
        out[side]=cal
    return out
