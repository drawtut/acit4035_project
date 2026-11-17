"""Six-channel coordinate map verified with 15 Nov 2026 physical pose tests.

Scope: WT901C485-E configured with installation register 0x23=1 (vertical)
for the left and right sensors tested. Printed physical +Y points toward toes.
Raw accel and gyro are mapped identically by a proper rotation; no device Euler,
magnetometer, quaternion filtering, angle smoothing or data-dependent sign flip.

Measured -> physical printed axis mapping:
    physical X = reported X
    physical Y = reported Z
    physical Z = -reported Y

This is R_x(-90deg) using column vectors. Only apply ONCE to the unmodified
six-channel stream used by the FPA orientation / PCA / integration branch.
The existing footstep detector runs on its original reported channels.
"""
from __future__ import annotations
import numpy as np

VERTICAL_TO_PRINTED = np.array([
    [1., 0., 0.],
    [0., 0., 1.],
    [0.,-1., 0.],
])


def to_printed_axes(accel, gyro, *, installation=1):
    """Return Nx3 arrays in the physical printed-axis frame, no filtering.

    Refuse unsupported installation modes rather than silently mis-mapping.
    """
    if installation != 1:
        raise ValueError('This mapping was physically verified ONLY for installation register 0x23=1')
    a=np.asarray(accel,dtype=float)
    w=np.asarray(gyro,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or a.shape!=w.shape:
        raise ValueError('Expected accel and gyro arrays of shape Nx3')
    return a @ VERTICAL_TO_PRINTED.T, w @ VERTICAL_TO_PRINTED.T
