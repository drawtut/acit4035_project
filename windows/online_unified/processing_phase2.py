"""Phase 2 online preview runs outside the acquisition and HTTP processes."""
import numpy as np
from fpa_unified_pipeline import process_arrays,calibration_pair,events_for_dashboard

def compute_preview(points,calibration_dir,reference_line_mode=False):
    calibration=calibration_pair(calibration_dir)
    raw={}
    for side in ('left','right'):
        arr=np.asarray(points[side],dtype=float)
        if arr.ndim!=2 or arr.shape[1]!=7 or len(arr)<900:
            return {'events':{'left':[],'right':[]},'synchronous':None,'crp':None}
        raw[side]=(arr[:,0],arr[:,1:4],arr[:,4:7])
    results,stage=process_arrays(raw,calibration=calibration,
                            reference_line_mode=reference_line_mode,
                            shared_host_clock_verified=True)
    return {'events':events_for_dashboard(stage),
            'synchronous':results['left'].get('synchronous_relative'),
            'crp':results['left'].get('bilateral_crp')}
