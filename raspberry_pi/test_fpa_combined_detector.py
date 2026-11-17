"""Run: py -m unittest test_fpa_combined_detector.py"""
import unittest
import numpy as np

from fpa_combined_detector import detect, fpa_for_event, export_rows


def sample(with_impact=True, with_stance=True, valid_swing=True):
    t=np.arange(0,3,.005)
    stationary=np.zeros(t.size,dtype=bool)
    stationary[t<=.90]=True
    if with_stance:
        stationary[(t>=2.05)&(t<=2.80)]=True
    gyro=np.full(t.size,120.)
    gyro[stationary]=2.
    contacts=[
        dict(index=240,time_s=1.20,rms_mps2=2.,prominence_mps2=1.2),
        dict(index=400,time_s=2.00,rms_mps2=4.,prominence_mps2=3.5),
        dict(index=404,time_s=2.02,rms_mps2=3.,prominence_mps2=2.0),
    ] if with_impact else []
    result=dict(t=t,mask=stationary,gyro_magnitude=gyro,
                impact_envelope=np.zeros(len(t)),contacts=contacts,config={})
    segmentation=dict(
        stands=[
            dict(stance_id=1,start_s=0.,end_s=.90,duration_s=.90),
            dict(stance_id=2,start_s=2.05,end_s=2.80,duration_s=.75)],
        swings=[dict(swing_id=1,from_stance_id=1,to_stance_id=2,
                     start_s=.905,end_s=2.045,duration_s=1.14,
                     gyro_peak_dps=220.,valid_swing=valid_swing)])
    return result,segmentation


def startup_sample(with_impact=True, correction_at_end=True):
    """First short footstep, later normal step, optional terminal adjustment."""
    t=np.arange(0,8,.005)
    mask=np.zeros(len(t),dtype=bool)
    for a,b in [(0,2.0),(2.40,3.25),(4.70,5.40),(6.85,7.8)]:
        mask[(t>=a)&(t<=b)]=True
    gyro=np.full(len(t),110.)
    gyro[mask]=3.
    contacts=[]
    for t_contact,level in [(2.35,2.2),(4.66,4.0),(6.82,2.2)]:
        if t_contact>3 or with_impact:
            contacts.append(dict(index=int(round(t_contact/.005)),time_s=t_contact,
                                 rms_mps2=level,prominence_mps2=level))
    result=dict(t=t,mask=mask,gyro_magnitude=gyro,
                impact_envelope=np.zeros(len(t)),contacts=contacts,config={})
    seg=dict(stands=[
        dict(stance_id=1,start_s=0.,end_s=2.,duration_s=2.),
        dict(stance_id=2,start_s=2.4,end_s=3.25,duration_s=.85),
        dict(stance_id=3,start_s=4.7,end_s=5.4,duration_s=.7),
        dict(stance_id=4,start_s=6.85,end_s=7.80,duration_s=.95)],
        swings=[
        dict(swing_id=1,from_stance_id=1,to_stance_id=2,
             start_s=2.005,end_s=2.395,duration_s=.39,
             gyro_peak_dps=150.,integrated_angular_motion_deg=25.,
             reason='small_integrated_rotation',valid_swing=False,
             terminal_stopping_candidate=False),
        dict(swing_id=2,from_stance_id=2,to_stance_id=3,
             start_s=3.255,end_s=4.695,duration_s=1.44,
             gyro_peak_dps=190.,integrated_angular_motion_deg=80.,
             reason='movement_supported',valid_swing=True,
             terminal_stopping_candidate=False),
        dict(swing_id=3,from_stance_id=3,to_stance_id=4,
             start_s=5.405,end_s=6.845,duration_s=1.44,
             gyro_peak_dps=110.,integrated_angular_motion_deg=30.,
             reason='small_integrated_rotation',valid_swing=False,
             terminal_stopping_candidate=correction_at_end)])
    return result,seg


class CombinedDetectorTests(unittest.TestCase):
    def test_one_peak_per_completed_swing(self):
        rows=detect(*sample())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'accepted_candidate')
        self.assertAlmostEqual(rows[0]['contact_time_s'],2.,delta=.03)

    def test_requires_impact(self):
        self.assertEqual(detect(*sample(with_impact=False))[0]['status'],'needs_review')

    def test_requires_stable_stance(self):
        self.assertEqual(detect(*sample(with_stance=False))[0]['status'],'needs_review')

    def test_requires_gyro_supported_swing(self):
        self.assertEqual(detect(*sample(valid_swing=False))[0]['status'],'needs_review')

    def test_heading_link_requires_matching_stance(self):
        event=detect(*sample())[0]
        good=[dict(stance_start_s=2.05,fpa_deg=20.1,heading_source='test',quality='test')]
        bad=[dict(stance_start_s=3.05,fpa_deg=42.1,heading_source='test',quality='test')]
        self.assertAlmostEqual(fpa_for_event(event,good)['fpa_deg'],20.1)
        self.assertIsNone(fpa_for_event(event,bad))
        self.assertAlmostEqual(export_rows([event],good,'left')[0]['fpa_deg'],20.1)

    def test_startup_rescue_when_explicitly_enabled(self):
        rows=detect(*startup_sample(),overrides={'startup_recovery_enabled':True})
        self.assertEqual(rows[0]['status'],'recovered_startup_candidate')
        self.assertEqual(rows[1]['status'],'accepted_candidate')
        self.assertEqual(rows[2]['status'],'needs_review')
        self.assertAlmostEqual(rows[0]['contact_time_s'],2.35,delta=.02)

    def test_end_repositioning_does_not_get_recovered(self):
        rows=detect(*startup_sample())
        self.assertEqual(rows[-1]['status'],'needs_review')

    def test_startup_without_impact_does_not_get_recovered(self):
        rows=detect(*startup_sample(with_impact=False))
        self.assertEqual(rows[0]['status'],'needs_review')

    def test_startup_disabled_by_config(self):
        rows=detect(*startup_sample(),overrides={'startup_recovery_enabled':False})
        self.assertEqual(rows[0]['status'],'needs_review')

    def test_early_short_swing_without_following_step_stays_rejected(self):
        result,seg=startup_sample()
        seg['swings']=seg['swings'][:1]
        rows=detect(result,seg)
        self.assertEqual(rows[0]['status'],'needs_review')


if __name__=='__main__':
    unittest.main()
