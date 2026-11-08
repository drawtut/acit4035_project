#!/usr/bin/env python3
"""Exploratory, offline bilateral IMU stationary-interval filter/threshold grid.
Not ground truth or FPA. Does not modify raw CSV.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d, binary_closing, binary_opening
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G=9.80665
AX=['ax_mps2','ay_mps2','az_mps2']
GX=['gx_dps','gy_dps','gz_dps']
CUTOFFS=(10,15,20,30)
GYRO_THRESHOLDS=(20,35,50)
ACC_TOLS=(1.0,1.8)
VAR_THRESHOLDS=(0.5,0.8)
MIN_MS=(70,120)

def odd(n):
    n=max(3,int(round(n)))
    return n if n%2 else n+1

def get_intervals(mask,t):
    d=np.diff(np.r_[False,mask,False].astype(int)); starts=np.flatnonzero(d==1); ends=np.flatnonzero(d==-1)
    return [(int(a),int(b),float(t[b-1]-t[a])) for a,b in zip(starts,ends)]

def load(path,side,quiet_start,quiet_end):
    df=pd.read_csv(path)
    for col in AX+GX:
        if col not in df:raise ValueError(f'{side}: missing {col}')
    key='timestamp_midpoint_monotonic_s' if 'timestamp_midpoint_monotonic_s' in df else 'time_s'
    t=df[key].to_numpy(dtype=float,copy=True);t-=t[0]
    a=df[AX].to_numpy(float); w=df[GX].to_numpy(float)
    if len(t)<200 or not np.all(np.diff(t)>0) or not(np.isfinite(a).all() and np.isfinite(w).all()):
        raise ValueError(f'{side}: invalid data/timestamps')
    fs=1/np.median(np.diff(t));q=(t>=quiet_start)&(t<=quiet_end)
    if q.sum()<20:raise ValueError(f'{side}: quiet reference <20 samples; set --quiet-start / --quiet-end')
    bias=np.median(w[q],axis=0)
    return dict(t=t,a=a,w=w-bias,fs=fs,gyro_bias=bias,side=side,path=str(path))

def analyse(data,out,window_ms=110):
    side=data['side']; t,a,w,fs=(data[k] for k in ['t','a','w','fs'])
    records=[]; configs={}; phase={}
    for cutoff in CUTOFFS:
        sos=butter(4,cutoff,fs=fs,output='sos')
        af=sosfiltfilt(sos,a,axis=0); wf=sosfiltfilt(sos,w,axis=0)
        amag=np.linalg.norm(af,axis=1); gmag=np.linalg.norm(wf,axis=1)
        width=odd(window_ms*fs/1000)
        var=np.maximum(0,uniform_filter1d(amag**2,width)-uniform_filter1d(amag,width)**2)
        phase[cutoff]=(gmag,amag,var)
        for gt in GYRO_THRESHOLDS:
            for at in ACC_TOLS:
                for vt in VAR_THRESHOLDS:
                    for mm in MIN_MS:
                        mask=(gmag<gt)&(np.abs(amag-G)<at)&(var<vt)
                        mask=binary_closing(mask,structure=np.ones(max(1,round(.04*fs))),border_value=0)
                        mask=binary_opening(mask,structure=np.ones(max(2,round(mm*.001*fs))),border_value=0)
                        iv=get_intervals(mask,t)
                        durations=[x[2] for x in iv]
                        # Diagnostic only: these labels are NOT validation accuracy.
                        during_high=float(np.mean(mask & (gmag>min(gt*.8,40))))
                        key=f'butter{cutoff}_gyro{gt}_acc{at}_var{vt}_min{mm}'
                        records.append(dict(side=side,config=key,cutoff_hz=cutoff,gyro_threshold_dps=gt,
                                            accel_tolerance_mps2=at,accel_variance_threshold=vt,min_stance_ms=mm,
                                            interval_count=len(iv),coverage_percent=100*np.mean(mask),
                                            total_stationary_s=sum(durations),median_interval_s=np.median(durations) if durations else 0,
                                            short_interval_count=sum(d<.15 for d in durations),
                                            fraction_all_high_gyro_marked=during_high))
                        configs[key]=(mask,iv)
    pd.DataFrame(records).to_csv(out/f'{side}_grid_metrics.csv',index=False)
    # Grid overview at fixed accel and variance thresholds: compare cutoff × gyro threshold
    fig,axes=plt.subplots(len(CUTOFFS),len(GYRO_THRESHOLDS),figsize=(15,10),sharex=True,sharey=True)
    for i,cutoff in enumerate(CUTOFFS):
        for j,gt in enumerate(GYRO_THRESHOLDS):
            ax=axes[i,j]; k=f'butter{cutoff}_gyro{gt}_acc1.0_var0.5_min120';mask,iv=configs[k]
            gmag,amag,var=phase[cutoff]
            ax.plot(t,gmag,color='#356eaa',lw=.7)
            ax.axhline(gt,color='#c86444',lw=.8,ls='--')
            for start,stop,_ in iv:ax.axvspan(t[start],t[stop-1],color='#2e9470',alpha=.25,lw=0)
            ax.set_title(f'{cutoff} Hz / gyro < {gt}°/s | {len(iv)} intervals',fontsize=10)
            ax.set_ylim(0,min(450,max(gmag)*1.1));ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Angular speed (°/s)')
            if i==len(CUTOFFS)-1:ax.set_xlabel('Elapsed time (s)')
    fig.suptitle(f'{side.upper()} — filter/threshold grid | |a| within 1.0 m/s² of g | var<0.5 | min 120 ms\nGreen = stationary candidates, NOT validated ground contacts',fontsize=13)
    fig.tight_layout();fig.savefig(out/f'{side}_cutoff_threshold_grid.png',dpi=150);plt.close(fig)
    # See impact and detection features together for baseline 15Hz config
    key='butter15_gyro35_acc1.0_var0.5_min120';mask,iv=configs[key]
    fig,axes=plt.subplots(3,1,figsize=(14,8),sharex=True)
    gmag,amag,var=phase[15]
    axes[0].plot(t,gmag,lw=.8,label='15Hz gyro magnitude');axes[0].axhline(35,color='tab:red',ls='--',label='35 deg/s')
    axes[1].plot(t,amag,lw=.8,label='15Hz accel magnitude');axes[1].axhline(G,color='gray',ls='--',label='g')
    axes[2].plot(t,var,lw=.8,label='Local accel-magnitude variance');axes[2].axhline(.5,color='tab:red',ls='--',label='variance 0.5')
    for ax in axes:
        for x,y,_ in iv:ax.axvspan(t[x],t[y-1],color='#2e9470',alpha=.20)
        ax.legend(loc='upper right');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Seconds');fig.suptitle(f'{side.upper()} baseline: 15Hz, gyro 35, accel ±1.0, variance 0.5, min 120ms\nCandidate stationary intervals only')
    fig.tight_layout();fig.savefig(out/f'{side}_baseline_diagnostics.png',dpi=170);plt.close(fig)
    pd.DataFrame([dict(start_s=t[x],end_s=t[y-1],duration_s=d) for x,y,d in iv],columns=['start_s','end_s','duration_s']).to_csv(out/f'{side}_baseline_intervals.csv',index=False)
    return dict(side=side,samples=len(t),rate_hz=fs,gyro_bias_dps=data['gyro_bias'].tolist(),tested_configs=len(records),
                baseline_interval_count=len(iv),baseline_interval_seconds=sum(d for _,_,d in iv))

def main():
    p=argparse.ArgumentParser(description='Offline stationary detector grid search; no ground truth, no FPA')
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--outdir',type=Path,default=Path('fpa_stance_optimization_output'))
    p.add_argument('--quiet-start',type=float,default=0);p.add_argument('--quiet-end',type=float,default=.4)
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    result={}
    for side in ('left','right'):
        obj=load(getattr(args,side),side,args.quiet_start,args.quiet_end)
        result[side]=analyse(obj,args.outdir)
        print(f"{side.upper()}: {result[side]['samples']} samples, {result[side]['rate_hz']:.1f} Hz, {result[side]['tested_configs']} configurations; baseline {result[side]['baseline_interval_count']} candidate intervals")
    (args.outdir/'summary.json').write_text(json.dumps(result,indent=2))
    print('Saved graphs, metrics and intervals to:',args.outdir.resolve())
    print('IMPORTANT: No ground truth, so do not select an optimal setting solely from candidate counts.')
if __name__=='__main__':main()
