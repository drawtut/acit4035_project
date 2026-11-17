"""Experimental foot-pitch continuous relative phase (CRP) from paired raw-six-axis IMUs.

Read-only downstream analysis: original FPA, footstep detection, filters and
calibration remain unchanged. Foot pitch is recovered from each existing
six-axis quaternion and the calibrated forward shoe axis. Common shared-host
midpoint times are interpolated only when both ORIGINAL streams support them.
A zero-phase gait-band filter and Hilbert analytic phase yield one joint
inter-foot phase at each common timestamp. These are retrospective estimates,
NOT causal real-time angles or validated clinical/ground-truth CRP.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.spatial.transform import Rotation
from fpa_synchronous import _raw_support

CRP_FIELDS = [
    'time_s','left_foot_pitch_deg','right_foot_pitch_deg',
    'left_oscillation_deg','right_oscillation_deg',
    'left_hilbert_phase_deg','right_hilbert_phase_deg',
    'right_minus_left_crp_deg','abs_deviation_from_antiphase_deg',
    'valid_both_sensor_support','quality'
]


def _runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype('i1'))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def _side_pitch(data, side):
    orient=data['orientation']
    cal=data['calibration']
    forward=np.asarray(cal.get('sensor_forward_unit',[]),float)
    if (not cal.get('usable') or forward.shape!=(3,) or
            np.linalg.norm(forward)<.8):
        raise ValueError(f'{side} forward shoe axis calibration unavailable')
    forward=forward/np.linalg.norm(forward)
    t=np.asarray(orient['time_s'],float)+float(data.get('clock_offset_s',0))
    raw=np.asarray(data['filtered']['source_time_s'],float)+float(data.get('clock_offset_s',0))
    quat=np.asarray(orient['quaternion_xyzw'],float)
    if (len(t)<50 or len(raw)<50 or quat.shape!=(len(t),4) or
            np.any(np.diff(t)<=0) or np.any(np.diff(raw)<=0) or
            not np.all(np.isfinite(quat))):
        raise ValueError(f'{side} orientation/raw timebase invalid')
    axis_world=Rotation.from_quat(quat).apply(forward)
    # Vertical component of toe-direction, gravity-leveled, invariant to yaw.
    # Relative orientation has no magnetic heading and no Euler IMU input.
    pitch=np.rad2deg(np.arcsin(np.clip(axis_world[:,2],-1,1)))
    return t,raw,pitch


def _circular(deg):
    return (np.asarray(deg,float)+180.)%360.-180.


def compute_continuous_relative_phase(results, stage, *,
                                      sample_period_s=.02,
                                      bandpass_hz=(.35,1.25),
                                      min_contiguous_s=4.0,
                                      edge_trim_s=.65,
                                      min_pitch_std_deg=2.,
                                      min_envelope_deg=1.5):
    """Compute retrospective common-time CRP; never infer missing samples.

    Contact times only delimit the walking section, NEVER determine or force
    foot phase or the expected 180-degree separation. Thus this isn't PCI and
    cannot be validated by agreement with those same IMU contact detections.
    """
    summary=dict(method='exploratory_synchronous_foot_pitch_hilbert_crp',
        status='UNAVAILABLE',reason=None,not_clinically_validated=True,
        hardware_synchronized=False,
        time_relationship='shared_host_midpoint_interpolated_with_raw_gap_guards',
        pitch_source='gravity_leveled_six_axis_quaternions_and_frozen_calibrated_shoe_forward_axes',
        low_pass_from_existing_frozen_profile=True,
        downstream_gait_band_hz=list(bandpass_hz),
        zero_phase_offline_processing=True,
        sample_period_s=float(sample_period_s),
        edge_trim_s=float(edge_trim_s),
        ground_truth_used=False,imu_contacts_only_define_walking_window=True,
        reference_relation='anti_phase_180_degrees_for_consistently_defined_foot_pitch',
        samples=0,usable_duration_s=0.,valid_segments=0,
        A_left_contacts=0,A_right_contacts=0,
        circular_mean_relative_phase_deg=None,
        circular_resultant_length=None,
        circular_std_deg=None,
        mean_abs_antiphase_deviation_deg=None,
        median_abs_antiphase_deviation_deg=None,
        fraction_within_30deg_of_antiphase=None,
        confidence='NOT_VALIDATED_SHORT_BOUT',walking_window_s=None)
    out={'rows':[],'summary':summary}
    if not all(results.get(s,{}).get('shared_host_clock_verified',False)
               for s in ('left','right')):
        summary['reason']='UNVERIFIED_SHARED_HOST_CLOCK';return out
    try:
        if (not .01<=sample_period_s<=.1 or
            not .05<bandpass_hz[0]<bandpass_hz[1]<.5/sample_period_s or
            min_contiguous_s<=2 or edge_trim_s<.3 or
            2*edge_trim_s>=min_contiguous_s):
            raise ValueError('Invalid phase analysis settings')
        sides={side:_side_pitch(results[side],side) for side in ('left','right')}
        start=max(sides[side][0][0] for side in sides)
        end=min(sides[side][0][-1] for side in sides)
        if end-start<min_contiguous_s:
            raise ValueError('Insufficient overlapping recording duration')
        contacts=[]
        for side in ('left','right'):
            for row in stage[side]['rows']:
                if row.get('contact_time_s') is None:continue
                ts=float(row['contact_time_s'])+float(results[side].get('clock_offset_s',0))
                contacts.append(ts)
                if row.get('detector_confidence')=='A_CONFIRMED':
                    summary['A_'+side+'_contacts']+=1
        if len(contacts)<4:
            raise ValueError('Fewer than four IMU contact estimates; cannot define walking period')
        first,last=min(contacts),max(contacts)
        if last-first<3.:
            raise ValueError('Walking interval shorter than three seconds')
        summary['walking_window_s']=[round(first,4),round(last,4)]
        # Extend a short margin only for filtering; ALL reported samples remain
        # between first/last detected contacts and never inside initial hold.
        grid=np.arange(start,end+1e-9,sample_period_s)
        valid=np.ones(len(grid),bool)
        pitch={}
        for side,(t,raw,p) in sides.items():
            valid &= _raw_support(grid,raw)
            pitch[side]=np.interp(grid,t,p)
        in_analysis=(grid>=first-1.)&(grid<=last+1.)
        support=valid&in_analysis
        order=3
        sos=butter(order,bandpass_hz,btype='bandpass',fs=1./sample_period_s,output='sos')
        all_phase=[];all_errors=[]
        for i,j in _runs(support):
            if (j-i)*sample_period_s+1e-9<min_contiguous_s:continue
            series={side:pitch[side][i:j] for side in ('left','right')}
            oscillation={}
            analytic={}
            if any(np.std(v)<min_pitch_std_deg for v in series.values()):continue
            for side in ('left','right'):
                # Remove DC posture/yaw-free tilt; same gait-band filter BOTH sides.
                # Hilbert transform assumes sufficiently cyclic walking motion.
                x=sosfiltfilt(sos,series[side])
                x=x-(np.max(x)+np.min(x))/2.
                oscillation[side]=x
                analytic[side]=hilbert(x)
            amp={side:np.abs(analytic[side]) for side in ('left','right')}
            if any(np.median(amp[side])<min_envelope_deg for side in ('left','right')):continue
            pl=np.rad2deg(np.angle(analytic['left']))
            pr=np.rad2deg(np.angle(analytic['right']))
            delta=_circular(pr-pl)
            deviation=np.abs(_circular(delta-180.))
            good=(grid[i:j]>=first)&(grid[i:j]<=last)
            good &= ((np.arange(j-i)*sample_period_s)>=edge_trim_s)
            good &= (((j-i-1-np.arange(j-i))*sample_period_s)>=edge_trim_s)
            # Undefined analytic phase when one foot's oscillation is near-zero.
            for side in ('left','right'):
                good &= amp[side]>=max(min_envelope_deg,.25*np.median(amp[side]))
            for k in np.flatnonzero(good):
                out['rows'].append(dict(
                    time_s=round(float(grid[i+k]),5),
                    left_foot_pitch_deg=round(float(series['left'][k]),5),
                    right_foot_pitch_deg=round(float(series['right'][k]),5),
                    left_oscillation_deg=round(float(oscillation['left'][k]),5),
                    right_oscillation_deg=round(float(oscillation['right'][k]),5),
                    left_hilbert_phase_deg=round(float(pl[k]),5),
                    right_hilbert_phase_deg=round(float(pr[k]),5),
                    right_minus_left_crp_deg=round(float(delta[k]),5),
                    abs_deviation_from_antiphase_deg=round(float(deviation[k]),5),
                    valid_both_sensor_support=True,
                    quality='HOST_SYNCED_AND_RAW_SUPPORTED_OFFLINE_HILBERT'))
            if good.any():
                all_phase.extend(delta[good].tolist())
                all_errors.extend(deviation[good].tolist())
                summary['valid_segments']+=1
        if len(all_phase)<100:
            raise ValueError('Fewer than 100 trustworthy simultaneous walking phase samples')
        delta=np.asarray(all_phase)
        dev=np.asarray(all_errors)
        resultant=np.mean(np.exp(1j*np.deg2rad(delta)))
        # CRP is circular: arithmetic means around +/-180 are misleading.
        mean=np.rad2deg(np.angle(resultant))%360.
        concentration=float(abs(resultant))
        summary.update(status='OK_EXPLORATORY',reason=None,
            samples=len(delta),usable_duration_s=round(len(delta)*sample_period_s,4),
            circular_mean_relative_phase_deg=round(float(mean),4),
            circular_resultant_length=round(concentration,5),
            circular_std_deg=round(float(np.rad2deg(np.sqrt(max(0.,-2*np.log(max(concentration,1e-12)))))),4),
            mean_abs_antiphase_deviation_deg=round(float(np.mean(dev)),4),
            median_abs_antiphase_deviation_deg=round(float(np.median(dev)),4),
            fraction_within_30deg_of_antiphase=round(float(np.mean(dev<=30)),5),
            confidence='EXPLORATORY_FEW_STEPS_NO_INDEPENDENT_KINEMATIC_GT')
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        summary['reason']=f'{type(exc).__name__}: {exc}'
    return out
