"""Windows offline Stage 4 using the same reference-line-only engine as Pi.

Example:
  py run_offline_stage4.py --left left_raw.csv --right right_raw.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
from fpa_unified_pipeline import process_csv_pair,export_results
from fpa_raw6_fpa import plot_stage4
from fpa_raw6_stride import plot_stage123


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--left',type=Path,required=True)
    ap.add_argument('--right',type=Path,required=True)
    ap.add_argument('--calibration-dir',type=Path,
        default=Path(__file__).resolve().parent/'calibration_frozen_v1')
    ap.add_argument('--out',type=Path,default=Path('offline_stage4_results'))
    args=ap.parse_args()
    data,stage=process_csv_pair(args.left,args.right,args.calibration_dir,
        reference_line_mode=True)
    manifest=export_results(data,stage,args.out)
    plot_stage123(data).savefig(args.out/'offline_stage123_diagnostics.png',dpi=125)
    plot_stage4(data,stage).savefig(args.out/'offline_stage4_fpa_diagnostics.png',dpi=125)
    for side in ('left','right'):
        res=manifest['results'][side]
        print(f"{side.upper()}: {res['accepted_imu_step_rows']} accepted contacts; "
              f"{res['candidate_fpa_rows']} reference-line FPA angles; "
              f"status={res['status_counts']}")
    print('Outputs:',args.out.resolve())

if __name__=='__main__':main()
