"""Phase 3: safe, explicit raw-six-axis temporal regridding for offline filters.

The existing detector, Butterworth SOS filters and centered RMS operations are
sample-index-based. They *require* uniform sample spacing. This module takes
only timestamps and six measured channels, and interpolates them onto a
uniform grid before any such operations. Original CSV samples are NEVER
modified. Linear interpolation is a numerical timebase operation, NOT a
replacement for missing IMU measurements; gaps >30ms are NOT bridged and use
legacy processing with an explicit warning until acquisition can be repaired.

No FPA, quaternion, Euler angle, trajectory or derived channel is filtered.
"""
from __future__ import annotations
import numpy as np

VERSION='PHASE3_UNIFORM_RAW6_V1'
MAX_INTERPOLATED_GAP_S=0.030
MIN_GRID_HZ=65.0  # required to represent the frozen 24Hz impact passband
MAX_GRID_HZ=250.0


def prepare_timebase(t,accel,gyro, *, max_interpolated_gap_s=MAX_INTERPOLATED_GAP_S):
    t=np.asarray(t,dtype=float)
    a=np.asarray(accel,dtype=float)
    w=np.asarray(gyro,dtype=float)
    if len(t)<30 or t.ndim!=1 or a.shape!=(len(t),3) or w.shape!=(len(t),3):
        raise ValueError('Timebase expects t[N], accel[N,3] and gyro[N,3], N>=30')
    if not all(np.isfinite(x).all() for x in (t,a,w)):
        raise ValueError('Nonfinite raw6 data')
    dt=np.diff(t)
    if np.any(dt<=0):
        raise ValueError('Nonmonotonic source timestamps cannot be regridded')
    duration=float(t[-1]-t[0])
    median=float(np.median(dt)); mean=duration/(len(t)-1)
    effective=1.0/mean; median_hz=1.0/median
    max_gap=float(np.max(dt))
    diagnostics=dict(schema=VERSION,mode=None,source_samples=len(t),source_duration_s=duration,
        source_median_interval_ms=median*1000.,source_mean_interval_ms=mean*1000.,
        source_median_hz=median_hz,source_effective_hz=effective,
        source_max_gap_ms=max_gap*1000.,source_gaps_over_30ms=int(np.count_nonzero(dt>.030)),
        source_gaps_over_60ms=int(np.count_nonzero(dt>.060)),
        source_nonmonotonic=0,raw_channels_only=True,raw_csv_unchanged=True,
        interpolation='linear per raw channel; no other signals/angles/trajectory interpolated',
        grid_samples=None,grid_hz=None,warning=None)
    # Large gaps imply missing measurements: do NOT invent intermediary samples.
    # The old detector may still produce contacts, but the original guarded
    # Input sampling defects must remain flagged.
    if max_gap > max_interpolated_gap_s+1e-12:
        diagnostics.update(mode='LEGACY_UNSAFE_GAPS_NO_INTERPOLATION',grid_samples=len(t),
            grid_hz=median_hz,
            warning='Source contains >30ms gaps; no regridding performed, index-domain filtering is unvalidated. Fix acquisition.')
        return t,a,w,diagnostics
    # Preserve exact regular samples; avoid needless numerical modifications.
    if np.max(np.abs(dt-mean)) <= max(1e-7,.005*mean):
        diagnostics.update(mode='NATIVE_ALREADY_UNIFORM',grid_samples=len(t),grid_hz=effective)
        return t,a,w,diagnostics
    # Upsample only to a regular computational grid. Grid density is NOT a
    # claim of physical internal sensor updates or independent measurements.
    grid_target=min(MAX_GRID_HZ,max(MIN_GRID_HZ,effective,median_hz))
    n=max(30,int(round(duration*grid_target))+1)
    grid=np.linspace(t[0],t[-1],n,dtype=np.float64)
    output_a=np.column_stack([np.interp(grid,t,a[:,k]) for k in range(3)])
    output_g=np.column_stack([np.interp(grid,t,w[:,k]) for k in range(3)])
    diagnostics.update(mode='UNIFORM_INTERPOLATED_RAW6',grid_samples=n,
        grid_hz=(n-1)/duration,warning='Uniform interpolation cannot restore sensor updates between observed samples.')
    return grid,output_a,output_g,diagnostics


def source_gap_audit(source_t,begin_s,end_s,nominal_s):
    """Original timestamp-gap evidence over the reconstructed stride interval."""
    t=np.asarray(source_t,dtype=float)
    # Include intervals that intersect a stride, not only interior sample points.
    starts=t[:-1];ends=t[1:]
    use=(starts<end_s)&(ends>begin_s)
    dt=np.diff(t)[use]
    if not len(dt):
        return dict(source_max_sample_gap_ms=float('nan'),source_gaps_over_30ms=0,
            source_missing_excess_s=0.,source_missing_fraction=0.)
    large=dt>.030
    duration=max(float(end_s-begin_s),1e-9)
    excess=float(np.maximum(dt[large]-nominal_s,0).sum())
    return dict(source_max_sample_gap_ms=1000.*float(max(dt)),
        source_gaps_over_30ms=int(np.count_nonzero(large)),
        source_missing_excess_s=excess,source_missing_fraction=excess/duration)
