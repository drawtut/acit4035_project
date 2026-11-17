"""Run offline stages 1–3 on two real six-axis IMU recordings.

Requires the SAME fpa_processing_config.json used by Filter Studio's footstep
pipeline. Video and sensor-fused yaw/roll/pitch are neither loaded nor accessed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import fpa_processing as processing
from fpa_raw6_stride import process_side, export_stage123, plot_stage123

ACC = ['ax_mps2','ay_mps2','az_mps2']
GYR = ['gx_dps','gy_dps','gz_dps']

def load(path):
    df=pd.read_csv(path)
    col=next((key for key in ['timestamp_midpoint_monotonic_s','time_s','host_elapsed_s'] if key in df),None)
    if col is None:raise ValueError(f'{path}: missing IMU sample time')
    t=df[col].to_numpy(float,copy=True)
    t-=t[0]
    return t,df[ACC].to_numpy(float),df[GYR].to_numpy(float)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--left',required=True,type=Path)
    ap.add_argument('--right',required=True,type=Path)
    ap.add_argument('--config',type=Path,required=True,help='Your SAVED, FROZEN footstep detector config')
    ap.add_argument('--out',type=Path,default=Path('offline_step123_results'))
    ap.add_argument('--mount-left',choices=['+X','-X','+Y','-Y'],default='+Y',
                    help='UNVERIFIED hypothesized physical sensor axis toward toes')
    ap.add_argument('--mount-right',choices=['+X','-X','+Y','-Y'],default='+Y')
    ap.add_argument('--gyro-hz',type=float,default=10.)
    ap.add_argument('--accel-hz',type=float,default=5.)
    ap.add_argument('--bias-start',type=float,default=1.)
    ap.add_argument('--bias-end',type=float,default=4.)
    args=ap.parse_args()
    cfg=processing.load_config(args.config)
    lab=dict(gyro_hz=args.gyro_hz,accel_hz=args.accel_hz,order=4,bias_start_s=args.bias_start,
             bias_end_s=args.bias_end,correct_gyro_bias=True,filter_gyro=True,filter_accel=True)
    orient=dict(forward_axis='+X',tilt_correction_tau_s=.75,
                static_gyro_threshold_dps=12.,accel_gravity_tolerance_mps2=.9)
    mounts={'left':args.mount_left,'right':args.mount_right}
    results={}
    for side in ('left','right'):
        t,a,w=load(getattr(args,side))
        results[side]=process_side(t,a,w,side=side,detector_config=cfg,
                                   lab_config=lab,orientation_config=orient,
                                   mount_forward=mounts[side])
    manifest=export_stage123(results,args.out,detector_config=cfg,lab_config=lab,
                              orientation_config=orient,mount_axes=mounts)
    fig=plot_stage123(results)
    fig.savefig(args.out/'offline_stage123_diagnostics.png',dpi=140)
    for side in ('left','right'):
        s=manifest['analysis'][side]
        print(f"{side.upper()}: {s['accepted_contact_events']} accepted contact events; "
              f"{s['linked_accepted_steps']} linked contacts; "
              f"PCA status={s['foot_axis_calibration']['reason']}")
    print('Reports:',args.out.resolve())
    print('No validated FPA generated. Shoe forward-sign calibration remains unverified.')

if __name__=='__main__':main()
