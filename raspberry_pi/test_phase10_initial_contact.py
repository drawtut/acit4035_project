"""Regression tests for cautious first-post-reference step timing recovery."""
import copy
import unittest
import numpy as np
from fpa_frozen_profile import frozen_settings
from fpa_combined_detector import detect


def fake_case():
    t=np.arange(0,7,0.01)
    # First impact is 0.45 s before next stance, second is also 0.45 s
    result=dict(t=t,mask=np.ones(len(t),bool),gyro_magnitude=np.ones(len(t))*5,
                impact_envelope=np.zeros(len(t)),contacts=[
                    dict(time_s=2.60,index=260,rms_mps2=2.,prominence_mps2=2.),
                    dict(time_s=4.55,index=455,rms_mps2=2.,prominence_mps2=2.)],
                config=frozen_settings()[0])
    stands=[dict(stance_id=1,start_s=0.,end_s=1.8,duration_s=1.8),
            dict(stance_id=2,start_s=3.05,end_s=4.0,duration_s=.95),
            dict(stance_id=3,start_s=5.0,end_s=6.,duration_s=1.)]
    swings=[dict(swing_id=1,from_stance_id=1,to_stance_id=2,
                 valid_swing=True,start_s=1.8,end_s=3.04,duration_s=1.24,
                 gyro_peak_dps=130.,integrated_angular_motion_deg=75.,reason='movement_supported'),
            dict(swing_id=2,from_stance_id=2,to_stance_id=3,
                 valid_swing=True,start_s=4.,end_s=4.99,duration_s=.99,
                 gyro_peak_dps=130.,integrated_angular_motion_deg=75.,reason='movement_supported')]
    return result,dict(stands=stands,swings=swings)

class Phase10Tests(unittest.TestCase):
    def test_first_extended_second_stays_strict(self):
        r,sg=fake_case();events=detect(r,sg)
        self.assertEqual(events[0]['status'],'accepted_candidate')
        self.assertIn('FIRST_STEP_EXTENDED_TIMING_WINDOW',events[0]['reason'])
        self.assertEqual(events[1]['status'],'needs_review')
        self.assertIn('no_impact',events[1]['reason'])
    def test_unaligned_short_initial_stance_not_recovered(self):
        r,sg=fake_case();sg['stands'][0]['duration_s']=.3
        self.assertEqual(detect(r,sg)[0]['status'],'needs_review')
    def test_original_combined_settings_without_override_unchanged(self):
        r,sg=fake_case();r['config']['combined_detection']['first_swing_impact_before_stance_s']=.35
        self.assertEqual(detect(r,sg)[0]['status'],'needs_review')
    def test_fpa_filter_and_impact_frozen(self):
        d,f,o,m=frozen_settings()
        self.assertEqual(d['impact']['low_hz'],11.)
        self.assertEqual(d['impact']['high_hz'],24.)
        self.assertEqual(d['stationary']['gyro_threshold_dps'],50.)
        self.assertEqual(f['accel_hz'],5.)
        self.assertEqual(f['gyro_hz'],10.)
