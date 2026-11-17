"""Stage 4: initial-aligned walkway-referenced signed FPA, from raw-six IMU.

No velocity integration, displacement or path reconstruction is performed.
FPA is released ONLY for a participant explicitly documented as standing
stationary with both shoes aligned at zero degrees to a straight walking lane.
The user's walking progression must follow the lane for the reported quantity
also to be a genuine foot progression angle. Otherwise it is a shoe heading.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fpa_reference_line import add_reference_line_fpa
from fpa_differential import pair_landings, summarize_pairs, PAIR_FIELDS
from fpa_synchronous import SYNC_FIELDS, SUPPORT_FIELDS

FPA_SIGN_CONVENTION='toe-out positive for left and right; left CCW, right CW'


def stage4_fpa(results, *, mounting_verified=None, reference_line_mode=False):
    """One output row per IMU-supported contact, with explicit A/B/C quality."""
    mounting_verified=mounting_verified or {'left':False,'right':False}
    out={}
    for side,data in results.items():
        if side not in ('left','right'):raise ValueError('Only left and right feet supported')
        rows=[]
        for link in data['accepted_links']:
            rows.append(dict(side=side,swing_id=int(link['swing_id']),
                contact_time_s=float(link['step_time_s']),
                start_stance_s=link.get('start_stance_s'),
                end_stance_s=link.get('end_stance_s'),
                accepted_imu_footstep=(link.get('confidence')=='A_CONFIRMED'),
                detector_confidence=link.get('confidence','A_CONFIRMED'),
                detector_evidence=link.get('evidence_class'),
                contact_timing_uncertainty_s=link.get('temporal_uncertainty_s'),
                fpa_deg=None,fpa_candidate_toe_out_deg=None,
                fpa_status='NOT_EVALUATED',
                fpa_reference='initial_aligned_0deg_stance_to_landing_heading',
                fpa_sign_convention=FPA_SIGN_CONVENTION,
                independent_functional_calibration=bool(data['calibration'].get('separate_calibration_trial',False)),
                shoe_mounting_physically_verified=bool(mounting_verified.get(side,False)),
                stage4_source='raw6_quaternion_shoe_heading_at_landing',
                reason=link.get('reason')))
        out[side]={'rows':rows,'timebase':data.get('timebase'),
                   'mounting_physically_verified':bool(mounting_verified.get(side,False)),
                   'separate_functional_calibration':bool(data['calibration'].get('separate_calibration_trial',False))}
    add_reference_line_fpa(results,out,enabled=reference_line_mode)
    for side,entry in out.items():
        for row in entry['rows']:
            angle=row['reference_line_fpa_candidate_toe_out_deg']
            status=row['reference_line_status']
            row['fpa_candidate_toe_out_deg']=angle
            row['fpa_deg']=float(angle) if angle is not None else None
            row['fpa_status']=('REVIEW_ONLY_' if row['detector_confidence']=='C_REVIEW_ONLY' else
                               'STANCE_SUPPORTED_' if row['detector_confidence']=='B_PROVISIONAL' else '') + status
    return out


def export_stage4(stage4,output_dir,*,detector_config,lab_config,orientation_config,mount_axes,
                  synchronous=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'ACIT4035_STAGE4_REFERENCE_LINE_ONLY_V2',
        'algorithm':'landing calibrated shoe heading minus fixed initial aligned walkway heading',
        'requires_explicit_initial_taped_zero':True,
        'requires_walk_path_parallel_to_reference':True,
        'not_general_free_walking_FPA':True,
        'magnetometer_used':False,
        'input_measured_channels':['ax_mps2','ay_mps2','az_mps2','gx_dps','gy_dps','gz_dps','timestamps'],
        'frozen_detector_config':detector_config,
        'fpa_raw6_filter_config':lab_config,
        'orientation_config':orientation_config,
        'mount_axes':mount_axes,'clinically_validated':False,
        'sign_convention':FPA_SIGN_CONVENTION,'results':{},
        'initial_reference_metadata':{},'timebase_per_side':{s:d.get('timebase') for s,d in stage4.items()}}
    for side,data in stage4.items():
        rows=data['rows']
        pd.DataFrame(rows).to_csv(out/f'{side}_stage4_fpa_candidates.csv',index=False)
        valid=[float(r['fpa_deg']) for r in rows if r['fpa_deg'] is not None]
        counts={}
        for r in rows:counts[r['fpa_status']]=counts.get(r['fpa_status'],0)+1
        manifest['results'][side]={'accepted_imu_step_rows':sum(r['detector_confidence']=='A_CONFIRMED' for r in rows),
            'total_contact_candidates':len(rows),
            'stance_supported_candidates':sum(r['detector_confidence']=='B_PROVISIONAL' for r in rows),
            'soft_stance_review_candidates':sum(r['detector_confidence']=='C_REVIEW_ONLY' for r in rows),
            'candidate_fpa_rows':len(valid),
            'research_fpa_released_rows':len(valid),
            'research_fpa_mean_deg':float(np.mean(valid)) if valid else None,
            'research_fpa_std_deg':float(np.std(valid,ddof=1)) if len(valid)>1 else None,
            'status_counts':counts,
            'mount_forward_physically_verified':data['mounting_physically_verified'],
            'independent_functional_calibration_used':data['separate_functional_calibration'],
            'clinically_validated':False}
        manifest['initial_reference_metadata'][side]=data['reference_line']
    # Phase 15: derived bilateral angles, entirely downstream of Stage 4.
    pairs=pair_landings(stage4)
    pd.DataFrame(pairs,columns=PAIR_FIELDS).to_csv(out/'bilateral_differential_pairs.csv',index=False)
    manifest['bilateral_differential']=summarize_pairs(pairs)
    # Phase16 genuinely simultaneous measurement; legacy Phase15 files are
    # retained for historical reproduction, not shown as a sync measurement.
    if synchronous is not None:
        pd.DataFrame(synchronous['rows'],columns=SYNC_FIELDS).to_csv(
            out/'bilateral_synchronous_relative_heading.csv',index=False)
        pd.DataFrame(synchronous['double_support_periods'],columns=SUPPORT_FIELDS).to_csv(
            out/'bilateral_double_support_windows.csv',index=False)
        manifest['synchronous_relative_heading']=synchronous['summary']
    (out/'offline_stage4_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def plot_stage4(results,stage4):
    from matplotlib.figure import Figure
    fig=Figure(figsize=(12,8),dpi=100);axes=fig.subplots(3,1,sharex=True)
    for side,ax in zip(('left','right'),axes[:2]):
        rows=stage4[side]['rows']
        x=[r['contact_time_s'] for r in rows if r['fpa_deg'] is not None]
        y=[r['fpa_deg'] for r in rows if r['fpa_deg'] is not None]
        ax.scatter(x,y,label=side,color=('#317abd' if side=='left' else '#d77844'))
        for r in rows:
            if r['fpa_deg'] is None:ax.axvline(r['contact_time_s'],color='#b77b37',ls=':',lw=.7)
        ax.axhline(0,color='gray',ls='--',lw=.8)
        ax.set_ylabel(side.capitalize()+' FPA (deg)')
        ax.grid(alpha=.2)
    sync=results.get('left',{}).get('synchronous_relative',{})
    ax=axes[2]
    synced=sync.get('rows',[])
    if synced:
        ax.plot([v['time_s'] for v in synced], [v['relative_heading_deg'] for v in synced],
            label='Both IMUs, same host time',lw=1.6)
        held=[v for v in synced if v['double_support']]
        if held:
            ax.scatter([v['time_s'] for v in held], [v['relative_heading_deg'] for v in held],
                s=8,label='Both stationary')
        ax.legend(fontsize=8)
    else:
        ax.text(.03,.65,'Synchronous relative heading unavailable: '+str(sync.get('summary',{}).get('reason','missing shared host clock')),
                transform=ax.transAxes,fontsize=9)
    ax.set_ylabel('Simultaneous L−R (deg)')
    ax.axhline(0,color='gray',ls='--',lw=.8)
    ax.grid(alpha=.2)
    axes[-1].set_xlabel('Contact time (s)')
    fig.suptitle('Stage 4 FPA + synchronous two-IMU relative shoe heading')
    fig.tight_layout(rect=[0,0,1,.94]);return fig
