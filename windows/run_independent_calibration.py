"""Create frozen-compatible per-foot functional calibration from a DIFFERENT trial."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from fpa_frozen_profile import frozen_settings
from fpa_raw6_stride import process_side
from fpa_calibration_engine import build_calibration,save_pair
from run_offline_stage123 import load


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--left',type=Path,required=True)
    ap.add_argument('--right',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--confirm-straight-walk',action='store_true')
    ap.add_argument('--confirm-mount',action='store_true')
    a=ap.parse_args()
    if not a.confirm_straight_walk or not a.confirm_mount:
        ap.error('Both --confirm-straight-walk and --confirm-mount are required after physical inspection')
    cfg,lab,orient,mounts=frozen_settings()
    results={}
    for side in ('left','right'):
        t,acc,gyro=load(getattr(a,side))
        results[side]=process_side(t,acc,gyro,side=side,
            detector_config=cfg,lab_config=lab,orientation_config=orient,
            mount_forward=mounts[side])
    cal=build_calibration(results,cfg,lab,orient,mounts,
        forward_walk_confirmed=True,mounting_verified=True)
    save_pair(cal,a.out)
    report={'schema':'ACIT4035_REFERENCE_ONLY_FUNCTIONAL_CALIBRATION',
            'new_calibration_saved_at':str(a.out),
            'results':{side:{'swings_used':cal[side]['swings_used'],
                'pca_eigenvalue_ratio':cal[side]['pca_eigenvalue_ratio'],
                'quality':cal[side]['calibration_quality_status']} for side in ('left','right')}}
    (a.out/'calibration_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
