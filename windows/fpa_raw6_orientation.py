"""Offline raw-six-axis shoe orientation laboratory (research, not validated FPA).

Inputs: time_s, filtered accelerometer XYZ (m/s^2), filtered gyroscope XYZ
(deg/s), and a stationary initial reference window in IMU time. No video,
magnetometer, WITMOTION orientation, calibrated angle or step-label inputs.

Frames: q=[x,y,z,w] rotates SENSOR vectors to a gravity-leveled local WORLD
frame. World +Z aligns with measured stationary accelerometer specific force.
Global heading is unobservable: reference yaw is arbitrary. Heading is the
horizontal direction of a chosen SENSOR axis, NOT automatically a foot/toe axis.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d

GRAVITY = 9.80665
AXES = {
    '+X': np.array([1., 0., 0.]), '-X': np.array([-1., 0., 0.]),
    '+Y': np.array([0., 1., 0.]), '-Y': np.array([0., -1., 0.]),
}


def _unit_quat(q):
    return q / np.linalg.norm(q)


def _quat_mul(a, b):
    """Hamilton product, scipy xyzw format, world<-sensor convention."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])


def _quat_from_rotation_vector(v):
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return _unit_quat(np.array([.5*v[0], .5*v[1], .5*v[2], 1.]))
    return np.r_[v * (np.sin(theta/2) / theta), np.cos(theta/2)]


def _apply(q, v):
    """Fast quaternion rotation of 3-vector v (sensor -> world)."""
    qxyz = q[:3]
    uv = np.cross(qxyz, v)
    return v + 2.0*(q[3]*uv + np.cross(qxyz, uv))


def estimate_six_axis_orientation(filtered, *, forward_axis='+X',
                                  static_gyro_threshold_dps=12.,
                                  accel_gravity_tolerance_mps2=0.9,
                                  tilt_correction_tau_s=0.75):
    """Return quaternion integration with optional gated gravity correction.

    The entire timeline is reconstructed with actual IMU sample intervals.
    Initialization uses median accelerometer in the initial bias-reference
    interval (OFFLINE calibration). Tilt correction is ONLY admitted for low
    angular velocity, |a| approximately g, and low short-window acceleration
    variability. It never supplies absolute yaw; no absolute progression direction.
    """
    t = np.asarray(filtered['time_s'], dtype=float)
    acc = np.asarray(filtered['accel_filtered'], dtype=float)
    gyro_dps = np.asarray(filtered['gyro_filtered'], dtype=float)
    if t.ndim != 1 or len(t) < 30 or acc.shape != (len(t), 3) or gyro_dps.shape != (len(t), 3):
        raise ValueError('Expected IMU timestamps and three-axis accel/gyro signals')
    if not (np.isfinite(t).all() and np.isfinite(acc).all() and np.isfinite(gyro_dps).all()):
        raise ValueError('Non-finite IMU input')
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('IMU timestamps must be strictly increasing')
    if forward_axis not in AXES:
        raise ValueError('Sensor axis must be +X, -X, +Y or -Y')
    if not (1 <= static_gyro_threshold_dps <= 100):
        raise ValueError('Static gyro threshold must be between 1 and 100 degrees/s')
    if not (0.1 <= accel_gravity_tolerance_mps2 <= 3):
        raise ValueError('Gravity tolerance must be between 0.1 and 3 m/s²')
    if not (0.05 <= tilt_correction_tau_s <= 30):
        raise ValueError('Tilt correction tau must be between 0.05 and 30 s')
    ref = (t >= float(filtered['bias_start_s'])) & (t < float(filtered['bias_end_s']))
    if ref.sum() < 20:
        raise ValueError('Need >=20 initial reference samples')
    a_ref = np.median(acc[ref], axis=0)
    norm_ref = np.linalg.norm(a_ref)
    if not (7.5 < norm_ref < 12.):
        raise ValueError(f'Initial accel magnitude {norm_ref:.2f} m/s² is implausible for stationary reference')
    q_init = Rotation.align_vectors([[0, 0, 1]], [a_ref / norm_ref])[0].as_quat()
    initial_gyro = np.linalg.norm(gyro_dps[ref], axis=1)
    if np.percentile(initial_gyro, 90) > static_gyro_threshold_dps:
        raise ValueError('Reference interval contains substantial gyro motion. Select stationary IMU seconds.')

    magnitude = np.linalg.norm(acc, axis=1)
    gyro_magnitude = np.linalg.norm(gyro_dps, axis=1)
    # A second independent reliability indicator: short-window variance of
    # accel magnitude. This is NOT the step detector's stationary state.
    fs = 1.0 / np.median(dt)
    window = max(5, int(round(0.12 * fs)))
    mean_a = uniform_filter1d(magnitude, size=window, mode='nearest')
    var_a = np.maximum(0., uniform_filter1d(magnitude**2, size=window, mode='nearest')-mean_a**2)
    trusted = ((gyro_magnitude < static_gyro_threshold_dps) &
               (abs(magnitude-GRAVITY) < accel_gravity_tolerance_mps2) &
               (var_a < 0.25))
    # Where the initial window is very still, corrections can be trusted.
    q_gyro = np.empty((len(t), 4), dtype=float)
    q_tilt = np.empty_like(q_gyro)
    q_gyro[0] = q_init
    q_tilt[0] = q_init
    omega = np.deg2rad(gyro_dps)
    up = np.array([0., 0., 1.])
    for k in range(1, len(t)):
        delta = _quat_from_rotation_vector(0.5*(omega[k-1]+omega[k]) * dt[k-1])
        q_gyro[k] = _unit_quat(_quat_mul(q_gyro[k-1], delta))
        q = _unit_quat(_quat_mul(q_tilt[k-1], delta))
        if trusted[k]:
            # Accelerometer at rest measures specific force UP, so rotate its
            # WORLD direction toward +Z. Gravity cannot correct yaw.
            measured_up = _apply(q, acc[k] / max(magnitude[k], 1e-8))
            correction_world = np.cross(measured_up, up)
            alpha = 1. - np.exp(-dt[k-1] / tilt_correction_tau_s)
            correction = _quat_from_rotation_vector(alpha * correction_world)
            q = _unit_quat(_quat_mul(correction, q))
        q_tilt[k] = q

    rotations = Rotation.from_quat(q_tilt)
    raw_rotations = Rotation.from_quat(q_gyro)
    body_axis = AXES[forward_axis]
    forward = rotations.apply(body_axis)
    forward_raw = raw_rotations.apply(body_axis)
    horizontal_norm = np.linalg.norm(forward[:,:2], axis=1)
    if np.median(horizontal_norm[ref]) < 0.2:
        raise ValueError('Selected sensor axis is nearly vertical in standing reference; try a different axis')
    # Continuous sensor-axis azimuth; zero is the initial reference median.
    azimuth = np.rad2deg(np.unwrap(np.arctan2(forward[:,1],forward[:,0])))
    azimuth_raw = np.rad2deg(np.unwrap(np.arctan2(forward_raw[:,1],forward_raw[:,0])))
    reference_heading = float(np.median(azimuth[ref]))
    reference_heading_raw = float(np.median(azimuth_raw[ref]))
    heading_rel = azimuth - reference_heading
    heading_gyro = azimuth_raw - reference_heading_raw
    measured_up_world = rotations.apply(acc/np.maximum(magnitude[:,None],1e-8))
    up_world_gyro = raw_rotations.apply(acc/np.maximum(magnitude[:,None],1e-8))
    gravity_residual = np.rad2deg(np.arccos(np.clip(measured_up_world[:,2],-1,1)))
    gravity_residual_gyro = np.rad2deg(np.arccos(np.clip(up_world_gyro[:,2],-1,1)))
    relrot = rotations * Rotation.from_quat(q_init).inv()
    rotation_from_reference = np.rad2deg(relrot.magnitude())
    reliable_err = gravity_residual[trusted]
    return {
        'time_s':t, 'quaternion_xyzw':q_tilt, 'gyro_only_quaternion_xyzw':q_gyro,
        'relative_sensor_axis_azimuth_deg':heading_rel,
        'gyro_only_sensor_axis_azimuth_deg':heading_gyro,
        'gravity_alignment_residual_deg':gravity_residual,
        'gyro_only_gravity_residual_deg':gravity_residual_gyro,
        'rotation_magnitude_from_initial_deg':rotation_from_reference,
        'trusted_tilt_correction':trusted,
        'gyro_magnitude_dps':gyro_magnitude, 'accel_magnitude_mps2':magnitude,
        'initial_accel_mps2':a_ref, 'sensor_axis':forward_axis,
        'trusted_fraction':float(np.mean(trusted)),
        'median_trusted_tilt_residual_deg':float(np.median(reliable_err)) if len(reliable_err) else None,
        'initial_reference_s':[float(filtered['bias_start_s']),float(filtered['bias_end_s'])],
        'heading_is_fpa':False,
        'heading_warning':'Relative SENSOR-axis azimuth. Not anatomical foot direction, walking direction or FPA. Six-axis yaw drift remains.'
    }
