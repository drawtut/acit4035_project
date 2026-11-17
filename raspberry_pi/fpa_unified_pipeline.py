"""Single canonical raw6 -> contacts -> heading relative to initial taped line.

Used by offline Windows scripts and the online Windows / Raspberry Pi dashboard.
No movement-vector reconstruction, acceleration integration, or path azimuth.
"""
from __future__ import annotations
from pathlib import Path
import json
import csv
import numpy as np
from fpa_frozen_profile import frozen_settings
from fpa_calibration_engine import load_pair
from fpa_raw6_stride import process_side,export_stage123
from fpa_raw6_fpa import stage4_fpa,export_stage4
from fpa_synchronous import compute_synchronous
from fpa_continuous_relative_phase import compute_continuous_relative_phase, CRP_FIELDS
from run_offline_stage123 import load as load_raw_csv


def calibration_pair(calibration_dir):
    cfg,lab,orient,mounts=frozen_settings()
    return load_pair(calibration_dir,cfg,lab,orient,mounts)


def clip_to_shared_bilateral_window(raw, *, tail_difference_s=0.75):
    """Do not analyze right-only/left-only tail after the other sensor stops.

    Inputs use the same host monotonic clock; the original raw CSVs are never
    rewritten. Trim ONLY when one stream ends substantially earlier. Return
    conservative metadata explaining the cut so it cannot be mistaken for a
    full bilateral trial. This is *not* an imputation or reconnection process.
    """
    end={side:float(np.asarray(raw[side][0],float)[-1]) for side in ('left','right')}
    start={side:float(np.asarray(raw[side][0],float)[0]) for side in ('left','right')}
    cutoff=min(end.values())
    lost=(max(end.values())-cutoff)>tail_difference_s
    clipped={}
    dropped={}
    for side,(t,a,g) in raw.items():
        t=np.asarray(t,float);a=np.asarray(a,float);g=np.asarray(g,float)
        keep=(t<=cutoff+1e-6) if lost else np.ones(len(t),dtype=bool)
        dropped[side]=int(len(t)-int(keep.sum()))
        if keep.sum()<50:raise ValueError('Insufficient simultaneous bilateral data')
        clipped[side]=(t[keep],a[keep],g[keep])
    return clipped,dict(applied=bool(lost),cutoff_host_time_s=cutoff if lost else None,
        last_sample_host_s=end,first_sample_host_s=start,
        discarded_samples_after_other_sensor_stopped=dropped,
        reason='BILATERAL_TAIL_LOSS_EVALUATION_EXCLUDED' if lost else 'NO_EXCESSIVE_BILATERAL_TAIL_DIFFERENCE')


def process_arrays(raw,calibration_dir=None,*,calibration=None,mounting_verified=None,
                   reference_line_mode=False,shared_host_clock_verified=False):
    cfg,lab,orient,mounts=frozen_settings()
    cal=calibration if calibration is not None else calibration_pair(calibration_dir)
    raw,shared_window=clip_to_shared_bilateral_window(raw)
    results={}
    # Original serial midpoint timestamps, not the separately-zeroed orientation
    # times, define inter-sensor phase. A caller must explicitly attest to a
    # shared host clock; otherwise the synchronous metric is NOT computed.
    clock_origin=min(float(np.asarray(raw[side][0])[0]) for side in ('left','right'))
    for side in ('left','right'):
        t,a,g=raw[side];t=np.asarray(t,dtype=float).copy()
        initial_host=float(t[0])
        if len(t):t-=t[0]
        results[side]=process_side(t,np.asarray(a,dtype=float),np.asarray(g,dtype=float),
            side=side,detector_config=cfg,lab_config=lab,
            orientation_config=orient,mount_forward=mounts[side],
            external_calibration=cal[side])
        results[side]['clock_offset_s']=initial_host-clock_origin
        results[side]['shared_host_clock_verified']=bool(shared_host_clock_verified)
    for side in results:results[side]['bilateral_window']=shared_window
    stage=stage4_fpa(results,mounting_verified=mounting_verified,
                              reference_line_mode=reference_line_mode)
    sync=compute_synchronous(results,stage)
    crp=compute_continuous_relative_phase(results,stage)
    for side in results:
        results[side]['synchronous_relative']=sync
        results[side]['bilateral_crp']=crp
    return results,stage


def process_csv_pair(left_csv,right_csv,calibration_dir,*,mounting_verified=None,
                     reference_line_mode=False):
    raw={}
    has_shared_host_clock={}
    for side,path in [('left',left_csv),('right',right_csv)]:
        t,a,g=load_raw_csv(path)
        # The CSV host monotonic midpoint is a SHARED clock across USB ports;
        # load_raw_csv normally zeroes each channel independently.
        with open(path,newline='',encoding='utf-8') as file:
            first=next(csv.DictReader(file))
        has_shared_host_clock[side]='timestamp_midpoint_monotonic_s' in first
        if has_shared_host_clock[side]:
            t=np.asarray(t,float)+float(first['timestamp_midpoint_monotonic_s'])
        raw[side]=(t,a,g)
    return process_arrays(raw,calibration_dir,mounting_verified=mounting_verified,
                          reference_line_mode=reference_line_mode,
                          shared_host_clock_verified=all(has_shared_host_clock.values()))


def export_results(results,stage,output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg,lab,orient,mounts=frozen_settings()
    export_stage123(results,out,detector_config=cfg,lab_config=lab,
                    orientation_config=orient,mount_axes=mounts)
    manifest=export_stage4(stage,out,detector_config=cfg,lab_config=lab,
                         orientation_config=orient,mount_axes=mounts,
                         synchronous=results['left'].get('synchronous_relative'))
    # Supplementary CRP is derived from paired, software-time-aligned raw-six-axis data.
    # Keep all existing Stage 4 FPA and Phase 16 output schemas unchanged.
    crp=results['left'].get('bilateral_crp',{})
    crp_rows=crp.get('rows',[])
    with (out/'bilateral_continuous_relative_phase.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=CRP_FIELDS)
        writer.writeheader();writer.writerows(crp_rows)
    (out/'bilateral_continuous_relative_phase_summary.json').write_text(
        json.dumps(crp.get('summary',{}),indent=2),encoding='utf-8')
    manifest['continuous_relative_phase']=crp.get('summary',{})
    manifest['bilateral_window']=results['left'].get('bilateral_window')
    (out/'offline_stage4_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def events_for_dashboard(stage):
    out={}
    for side in ('left','right'):
        out[side]=[]
        for r in stage[side]['rows']:
            out[side].append(dict(step=int(r['swing_id']),
                time_s=round(float(r['contact_time_s']),3),
                fpa_deg=r['fpa_deg'],
                reference_line_fpa_deg=r['fpa_deg'],
                reference_line_status=r['fpa_status'],
                status=r['fpa_status'],
                detector_confidence=r.get('detector_confidence','A_CONFIRMED'),
                reason=r.get('reason'),
                detector_evidence=r.get('detector_evidence'),
                fpa_reference='landing_shoe_heading_vs_initial_aligned_walkway',
                late_assigned=True,
                fpa_start_stance_s=r.get('start_stance_s'),
                fpa_end_stance_s=r.get('end_stance_s')))
    return out


def report_summary(stage,manifest):
    out={}
    for side in ('left','right'):
        rows=stage[side]['rows']
        values=[float(r['fpa_deg']) for r in rows if r['fpa_deg'] is not None]
        out[side]={
            'angles':[{'time_s':r['contact_time_s'],
                       'candidate_deg':r['fpa_deg'],
                       'status':r['fpa_status']} for r in rows],
            'mean_candidate_deg':float(np.mean(values)) if values else None,
            'fpa_reference':'initial_taped_line_vs_landing_foot_heading',
            'initial_reference':stage[side].get('reference_line'),
            'accepted':manifest['results'][side]['accepted_imu_step_rows'],
            'provisional':manifest['results'][side].get('stance_supported_candidates',0),
            'review':manifest['results'][side].get('soft_stance_review_candidates',0)}
    return out


def source_audit():
    import fpa_processing,fpa_segmentation_v21,fpa_combined_detector,fpa_raw6_stride
    return {m.__name__:str(Path(m.__file__).resolve()) for m in
            [fpa_processing,fpa_segmentation_v21,fpa_combined_detector,fpa_raw6_stride]}
