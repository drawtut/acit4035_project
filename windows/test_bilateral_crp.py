"""Synthetic signal verification of the exploratory retrospective two-IMU CRP.

This verifies the math and gap rejection, NOT medical accuracy/ground truth.
"""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fpa_continuous_relative_phase import compute_continuous_relative_phase


def synthetic_trial(phase_difference_deg=180, verify_clock=True, drop_right=False):
    t = np.arange(0, 20, .01)
    results = {}
    for side,offset in [('left',0.),('right',phase_difference_deg)]:
        pitch_deg = 14*np.sin(2*np.pi*.72*t + np.deg2rad(offset))
        # Each calibrated shoe's +Y longitudinal direction tilts with X rotation.
        q = Rotation.from_rotvec(np.deg2rad(pitch_deg)[:,None]*np.array([1.,0.,0.])).as_quat()
        raw=t.copy()
        if drop_right and side=='right':raw=raw[(raw<6.4)|(raw>7.2)]
        results[side]={
            'shared_host_clock_verified':verify_clock,
            'clock_offset_s':0.,
            'orientation':{'time_s':t,'quaternion_xyzw':q},
            'calibration':{'usable':True,'sensor_forward_unit':[0,1,0]},
            'filtered':{'source_time_s':raw},
        }
    events=[]
    for x in np.arange(4,17,0.85):
        events.append({'contact_time_s':float(x),'detector_confidence':'A_CONFIRMED'})
    stage={'left':{'rows':events[::2]},'right':{'rows':events[1::2]}}
    return results,stage


class BilateralCRPTests(unittest.TestCase):
    def test_antiphase_motion_returns_180deg(self):
        result=compute_continuous_relative_phase(*synthetic_trial())
        s=result['summary']
        self.assertEqual(s['status'],'OK_EXPLORATORY')
        self.assertLess(abs(s['circular_mean_relative_phase_deg']-180),5)
        self.assertLess(s['mean_abs_antiphase_deviation_deg'],10)
        self.assertTrue(result['rows'])

    def test_inphase_motion_is_not_antiphase(self):
        result=compute_continuous_relative_phase(*synthetic_trial(0))
        s=result['summary']
        self.assertEqual(s['status'],'OK_EXPLORATORY')
        self.assertLess(min(s['circular_mean_relative_phase_deg'],360-s['circular_mean_relative_phase_deg']),5)
        self.assertGreater(s['mean_abs_antiphase_deviation_deg'],170)

    def test_missing_shared_host_clock_rejected(self):
        result=compute_continuous_relative_phase(*synthetic_trial(verify_clock=False))
        self.assertEqual(result['summary']['status'],'UNAVAILABLE')
        self.assertEqual(result['summary']['reason'],'UNVERIFIED_SHARED_HOST_CLOCK')
        self.assertEqual(result['rows'],[])

    def test_missing_raw_usb_samples_are_never_interpolated(self):
        result=compute_continuous_relative_phase(*synthetic_trial(drop_right=True))
        self.assertEqual(result['summary']['status'],'OK_EXPLORATORY')
        times=[r['time_s'] for r in result['rows']]
        self.assertFalse(any(6.38<=x<=7.22 for x in times))
        # A shorter pre-gap segment may be rejected by the 4s quality rule.
        self.assertTrue(any(x>8. for x in times))

    def test_no_step_window_no_phase_result(self):
        results,stage=synthetic_trial()
        stage['left']['rows']=[];stage['right']['rows']=[]
        out=compute_continuous_relative_phase(results,stage)
        self.assertEqual(out['summary']['status'],'UNAVAILABLE')
        self.assertEqual(out['rows'],[])


if __name__=='__main__':unittest.main()
