"""Phase16: require actual shared-clock two-stream samples, not landing pairs."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fpa_synchronous import compute_synchronous


def make_synthetic():
    t=np.arange(0,8,.02)
    motion=np.clip((t-3)/.5,0,1)
    # Both sensors rotate at the same physical instant in opposing directions.
    yaw={'left':10*motion,'right':-5*motion}
    mask=np.zeros(len(t),dtype=bool)
    mask[t<2]=True
    mask[(t>4)&(t<4.4)]=True
    mask[(t>6)&(t<6.4)]=True
    results={};stage={}
    for side in ('left','right'):
        offset=0 if side=='left' else .004
        q=Rotation.from_euler('z',yaw[side][:,None],degrees=True).as_quat()
        results[side]=dict(
            clock_offset_s=offset,shared_host_clock_verified=True,
            calibration={'usable':True,'sensor_forward_unit':[1.,0.,0.]},
            orientation={'time_s':t,'quaternion_xyzw':q},
            detector_result={'mask':mask},
            filtered={'source_time_s':t})
        stage[side]=dict(reference_line=dict(enabled=True,baseline_s=[0.,2.],
            baseline_heading_deg=0.,baseline_yaw_spread_p95_deg=0.,baseline_gyro_p90_dps=0.),
            rows=[dict(contact_time_s=3.5),dict(contact_time_s=6.5)])
    return results,stage


class SynchronousRelativeTests(unittest.TestCase):
    def test_both_orientations_at_same_sample_not_consecutive_contacts(self):
        results,stage=make_synthetic()
        outcome=compute_synchronous(results,stage)
        self.assertEqual(outcome['summary']['status'],'OK')
        rows=outcome['rows']
        earlier=min(rows,key=lambda r:abs(r['time_s']-1.))
        later=min(rows,key=lambda r:abs(r['time_s']-5.))
        self.assertAlmostEqual(earlier['relative_heading_deg'],0,delta=.5)
        self.assertAlmostEqual(later['relative_heading_deg'],15,delta=.5)
        self.assertAlmostEqual(outcome['summary']['median_abs_moving_relative_heading_deg'],15,delta=.6)
        # Clock-offset handling must not accidentally read right(t + 4ms) as right(t=0).
        self.assertLess(abs(later['time_s']-5),.05)

    def test_no_verified_shared_clock_no_combined_measurement(self):
        results,stage=make_synthetic()
        results['right']['shared_host_clock_verified']=False
        out=compute_synchronous(results,stage)
        self.assertEqual(out['summary']['status'],'UNAVAILABLE')
        self.assertEqual(out['summary']['reason'],'UNVERIFIED_SHARED_HOST_CLOCK')
        self.assertEqual(out['rows'],[])

    def test_raw_usb_gap_blocks_interpolated_samples(self):
        results,stage=make_synthetic()
        results['right']['filtered']['source_time_s']=np.r_[np.arange(0,4,.02),np.arange(4.5,8,.02)]
        out=compute_synchronous(results,stage)
        self.assertEqual(out['summary']['status'],'OK')
        self.assertFalse(any(4.08<r['time_s']<4.44 for r in out['rows']))

    def test_no_gait_double_support_from_initial_or_terminal_hold(self):
        results,stage=make_synthetic()
        # synthetic overlapping still regions in mid-walk should be included
        out=compute_synchronous(results,stage)
        periods=out['double_support_periods']
        self.assertGreaterEqual(len(periods),2)
        self.assertTrue(all(x['start_s']>=3.5 for x in periods))
        self.assertTrue(all(x['end_s']<=6.5 for x in periods))

    def test_never_invents_double_support_when_only_one_foot_stationary(self):
        results,stage=make_synthetic()
        results['right']['detector_result']['mask'][:]=False
        out=compute_synchronous(results,stage)
        self.assertEqual(out['summary']['double_support_periods'],0)
        self.assertIsNone(out['summary']['median_double_support_heading_deg'])
        self.assertGreater(out['summary']['samples'],100)

if __name__=='__main__':unittest.main()
