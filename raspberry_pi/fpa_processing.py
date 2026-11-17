"""One source of truth for ACIT4035 offline FPA candidate events.

No fused yaw, no magnetometer, no FPA computed here. The results are
unvalidated candidate contacts and stationary regions; raw CSVs unchanged.
Both the GUI and offline estimator import this module and a shared JSON config.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import uniform_filter1d, binary_closing, binary_opening, gaussian_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt, find_peaks, savgol_filter, iirnotch, filtfilt

G = 9.80665
ACC = ('ax_mps2','ay_mps2','az_mps2')
GYRO = ('gx_dps','gy_dps','gz_dps')
CONFIG_PATH = Path(__file__).with_name('fpa_processing_config.json')


def load_config(path=None):
    cfg=json.loads(Path(path or CONFIG_PATH).read_text(encoding='utf-8'))
    if cfg.get('schema') != 'ACIT4035_SHARED_PROCESSING_V1':
        raise ValueError('Expected ACIT4035_SHARED_PROCESSING_V1 configuration')
    return cfg


def lowpass(x, hz, fs, order=4):
    if not 0.2 < hz < fs/2:
        raise ValueError(f'Invalid low-pass cutoff {hz} Hz for fs {fs:.2f}')
    return sosfiltfilt(butter(int(order),float(hz),btype='lowpass',fs=fs,output='sos'),x,axis=0)


def bandpass(x, low, high, fs, order=4):
    if not 0.2 < low < high < fs/2:
        raise ValueError(f'Invalid band-pass ({low}, {high}) Hz for fs {fs:.2f}')
    return sosfiltfilt(butter(int(order),[float(low),float(high)],btype='bandpass',fs=fs,output='sos'),x,axis=0)




def filter_impact(x,c,fs):
    """Apply the active impact channel's actual Studio filter settings.

    Supports the same scalar filter options as Filter Studio. Offline / zero-
    phase methods must not be treated as causal real-time implementations.
    """
    x=np.asarray(x,float)
    method=c.get('method','Butterworth band-pass')
    lo=float(c['low_hz']);hi=float(c['high_hz']);order=int(c['order'])
    nyquist=fs/2
    window=float(c.get('window_ms',75))
    if method=='Raw':return x.copy()
    if method=='Butterworth band-pass':
        return bandpass(x,min(lo,hi),max(lo,hi),fs,order)
    if method in ('Butterworth low-pass','Butterworth high-pass','Median + Butterworth'):
        sos=butter(order,np.clip(lo,.2,.9*nyquist),
            btype='highpass' if method=='Butterworth high-pass' else 'lowpass',
            fs=fs,output='sos')
        if method=='Median + Butterworth':
            n=window_samples(window,fs)
            x=median_filter(x,size=n,mode='nearest')
        return sosfiltfilt(sos,x)
    if method=='Savitzky-Golay':
        n=max(5,window_samples(window,fs))
        return savgol_filter(x,n,3,mode='interp')
    if method=='Gaussian':
        return gaussian_filter1d(x,max(.2,window*fs/1000/2.355),mode='nearest')
    if method=='Median':
        return median_filter(x,size=window_samples(window,fs),mode='nearest')
    if method=='EMA (causal)':
        alpha=1-np.exp(-2*np.pi*max(.2,lo)/fs)
        out=np.empty_like(x);out[0]=x[0]
        for i in range(1,len(out)):out[i]=alpha*x[i]+(1-alpha)*out[i-1]
        return out
    if method=='Notch':
        target=min(float(c.get('notch_hz',50)),.9*nyquist)
        b,a=iirnotch(target,15.,fs)
        return filtfilt(b,a,x)
    raise ValueError(f'Unknown impact filter method: {method}')

def window_samples(ms, fs):
    return max(3,int(round(ms/1000*fs))|1)


def intervals(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype('int8'))
    return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))


def estimate_bias(t,gyro,cfg):
    o=cfg['orientation']
    initial=np.asarray(t) <= float(o['bias_initial_window_s'])
    quiet=initial & (np.linalg.norm(gyro,axis=1) < float(o['bias_quiet_threshold_dps']))
    min_count=max(5,round(float(o['bias_min_quiet_ms'])/1000 / np.median(np.diff(t))))
    if quiet.sum()<min_count:
        return np.zeros(3), 'NO_STATIONARY_START: bias set to 0 (flagged)'
    return np.median(gyro[quiet],axis=0), 'early_quiet_samples'


def run(t, accel, gyro, config=None):
    cfg=config if config is not None else load_config()
    t=np.asarray(t,float); a=np.asarray(accel,float); w=np.asarray(gyro,float)
    if len(t)<50 or a.shape!=(len(t),3) or w.shape!=(len(t),3):
        raise ValueError('Require N>=50 timestamps and Nx3 accel/gyro')
    if not np.isfinite(t).all() or not np.isfinite(a).all() or not np.isfinite(w).all() or np.any(np.diff(t)<=0):
        raise ValueError('Nonfinite or nonmonotonic input')
    fs=1/np.median(np.diff(t))
    c=cfg['impact']; s=cfg['stationary']; o=cfg['orientation']
    bias,bias_source=estimate_bias(t,w,cfg)
    corrected=w-bias
    ori_gyro=lowpass(corrected,o['gyro_lowpass_hz'],fs,o['order'])
    detection_gyro=lowpass(corrected,s.get('gyro_lowpass_hz',s['detector_lowpass_hz']),fs,s['order'])
    detection_acc=lowpass(a,s.get('accel_lowpass_hz',s['detector_lowpass_hz']),fs,s['order'])
    gnorm=np.linalg.norm(detection_gyro,axis=1)
    anorm=np.linalg.norm(detection_acc,axis=1)
    var_window=window_samples(s['variance_window_ms'],fs)
    avar=np.maximum(0,uniform_filter1d(anorm**2,size=var_window,mode='nearest')-
                    uniform_filter1d(anorm,size=var_window,mode='nearest')**2)
    mask=(gnorm < s['gyro_threshold_dps']) & (np.abs(anorm-G)<s['gravity_tolerance_mps2']) & (avar<s['accel_variance_mps4'])
    gap=max(1,round(s['bridge_gap_ms']/1000*fs))
    minimum=max(1,round(s['min_duration_ms']/1000*fs))
    if gap>1: mask=binary_closing(mask,structure=np.ones(gap),border_value=0)
    if minimum>1: mask=binary_opening(mask,structure=np.ones(minimum),border_value=0)
    stands=[{'start_index':int(i),'end_index_exclusive':int(j),
             'start_s':float(t[i]),'end_s':float(t[j-1]),
             'duration_s':float(t[j-1]-t[i]),'mid_index':int((i+j-1)//2)} for i,j in intervals(mask)]
    axis_key=c['axis']
    if axis_key == 'Acceleration magnitude |a|':
        impact_source=np.linalg.norm(a,axis=1)
    else:
        impact_source=a[:,ACC.index(axis_key)]
    impact_filtered=filter_impact(impact_source,c,fs)
    envelope=np.sqrt(np.maximum(0,uniform_filter1d(impact_filtered**2,
                        size=window_samples(c['rms_window_ms'],fs),mode='nearest')))
    peaks,properties=find_peaks(envelope,prominence=float(c['prominence_mps2']),
                                distance=max(1,round(c['min_spacing_s']*fs)))
    contacts=[{'index':int(i),'time_s':float(t[i]),'rms_mps2':float(envelope[i]),
               'prominence_mps2':float(properties['prominences'][n])} for n,i in enumerate(peaks)]
    pairs=associate(t,contacts,stands,cfg['segmentation'])
    return dict(t=t,fs=fs,accel=a,gyro=w,bias=bias,bias_source=bias_source,
                orientation_gyro=ori_gyro,gyro_for_detector=detection_gyro,
                accel_for_detector=detection_acc,gyro_magnitude=gnorm,accel_magnitude=anorm,
                accel_variance=avar,mask=mask,stationary=stands,
                impact_bandpassed=impact_filtered,impact_envelope=envelope,
                contacts=contacts,pairs=pairs,config=cfg)


def associate(t,contacts,stands,params):
    """One-to-one contact -> subsequent *distinct* stationary interval.

    These are provisional linkages. Every unmatched contact and stationary interval
    remains in the exported report. No synthetic stride is invented to fill gaps.
    """
    seen=set(); pairs=[]
    lo=float(params['contact_to_stance_min_s']); hi=float(params['contact_to_stance_max_s'])
    for contact in contacts:
        ct=contact['time_s']; eligible=[]
        for si,s in enumerate(stands):
            if si in seen: continue
            # Prefer stance starting after contact; if contact already inside it,
            # treat the candidate as ambiguous and leave unmatched.
            delay=s['start_s']-ct
            if lo <= delay <= hi:
                eligible.append((delay,si))
        if not eligible:
            pairs.append(dict(**contact,stationary_index=None,stationary_start_s=None,
                              stationary_end_s=None,midstance_s=None,status='unmatched_contact'))
            continue
        _,si=min(eligible)
        seen.add(si); s=stands[si]
        pairs.append(dict(**contact,stationary_index=si,stationary_start_s=s['start_s'],
                          stationary_end_s=s['end_s'],midstance_s=float(t[s['mid_index']]),
                          status='matched_candidate'))
    return pairs


def stride_pairs(result):
    """Valid-length consecutive matched same-foot contacts, plus explicit rejects."""
    cfg=result['config']['segmentation']; matches=[x for x in result['pairs'] if x['status']=='matched_candidate']
    out=[]
    for p,q in zip(matches,matches[1:]):
        duration=q['midstance_s']-p['midstance_s']
        skipped= sum(p['time_s']<x['time_s']<q['time_s'] and x['status']!='matched_candidate' for x in result['pairs'])
        valid=cfg['min_stride_s']<=duration<=cfg['max_stride_s'] and skipped==0
        out.append(dict(start_contact_s=p['time_s'],end_contact_s=q['time_s'],
                        start_midstance_s=p['midstance_s'],end_midstance_s=q['midstance_s'],
                        start_index=result['stationary'][p['stationary_index']]['mid_index'],
                        end_index=result['stationary'][q['stationary_index']]['mid_index'],
                        duration_s=float(duration),accepted_for_reconstruction=bool(valid),
                        reason='candidate_complete' if valid else 'duration_out_of_range_or_unmatched_intermediate_contact'))
    return out


def rows(result,side):
    contacts=[dict(side=side,event='contact',**x) for x in result['pairs']]
    stationary=[dict(side=side,event='stationary_unmatched',**s) for j,s in enumerate(result['stationary'])
                if j not in {p['stationary_index'] for p in result['pairs'] if p['stationary_index'] is not None}]
    return contacts,stationary
