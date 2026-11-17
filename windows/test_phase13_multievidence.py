"""Phase13 rule-level tests (never use paper GT as a detector input)."""
import unittest
import numpy as np
from fpa_phase13_detector import detect, PARAMS
from fpa_frozen_profile import frozen_settings
from fpa_unified_pipeline import clip_to_shared_bilateral_window
from test_phase10_initial_contact import fake_case


def base_stances():
    return [dict(stance_id=1,start_index=0,start_s=0.0,end_s=1.80,duration_s=1.8,
                 reference_index=150),
            dict(stance_id=2,start_index=305,start_s=3.05,end_s=4.00,duration_s=.95,
                 reference_index=350),
            dict(stance_id=3,start_index=500,start_s=5.0,end_s=6.0,duration_s=1.0,
                 reference_index=550)]


class Phase13Tests(unittest.TestCase):
    def test_55_centisecond_impact_gate_recovers_second_step(self):
        r,sg=fake_case()
        sg['stands']=base_stances()
        events=detect(r,sg,include_soft=False)
        self.assertEqual([e['confidence'] for e in events],['A_CONFIRMED','A_CONFIRMED'])
        self.assertEqual(events[1]['contact_time_s'],4.55)

    def test_full_stance_without_impact_is_provisional(self):
        r,sg=fake_case()
        sg['stands']=base_stances()
        r['contacts']=[]
        events=detect(r,sg,include_soft=False)
        self.assertEqual([e['confidence'] for e in events],['B_PROVISIONAL','B_PROVISIONAL'])
        self.assertEqual(events[0]['landing_orientation_index'],350)
        self.assertEqual(events[0]['contact_time_s'],3.05)

    def test_invalid_swing_must_not_be_upgraded(self):
        r,sg=fake_case()
        sg['stands']=base_stances()
        r['contacts']=[]
        sg['swings'][0]['valid_swing']=False
        events=detect(r,sg,include_soft=False)
        self.assertEqual(events[0]['confidence'],'REJECTED')

    def test_weak_stance_must_not_be_upgraded(self):
        r,sg=fake_case()
        sg['stands']=base_stances()
        r['contacts']=[]
        r['mask'][:]=False
        events=detect(r,sg,include_soft=False)
        self.assertEqual(events[0]['confidence'],'REJECTED')

    def test_soft_recovery_only_inside_long_merged_swing(self):
        t=np.arange(0,7,.01)
        g=np.full(len(t),155.)
        a=np.full(len(t),9.80665)
        g[(t>2.65)&(t<2.93)]=12.
        r=dict(t=t,mask=np.zeros(len(t),bool),gyro_magnitude=g,
               accel_magnitude=a, impact_envelope=np.zeros(len(t)),
               contacts=[dict(index=i,time_s=ct,rms_mps2=prom,prominence_mps2=prom)
                         for i,ct,prom in [(200,2.,4.),(250,2.5,2.)]],
               config=frozen_settings()[0])
        sg=dict(stands=[dict(stance_id=1,start_index=0,start_s=0.,end_s=1.,duration_s=1.,reference_index=70),
                        dict(stance_id=2,start_index=600,start_s=6.,end_s=6.9,duration_s=.9,reference_index=650)],
                swings=[dict(swing_id=1,from_stance_id=1,to_stance_id=2,valid_swing=True,
                             start_s=1.5,end_s=5.5,duration_s=4.,gyro_peak_dps=155.,
                             integrated_angular_motion_deg=350.,reason='movement_supported')])
        events=detect(r,sg)
        soft=[e for e in events if e['confidence']=='C_REVIEW_ONLY']
        self.assertEqual(len(soft),1)
        self.assertAlmostEqual(soft[0]['contact_time_s'],2.5)
        self.assertGreater(soft[0]['landing_orientation_index'],250)
        self.assertEqual(soft[0]['reason'],'REVIEW_UNCONFIRMED_SHORT_STANCE_INSIDE_MERGED_SWING')
        sg['swings'][0]['duration_s']=1.0
        self.assertFalse(any(e['confidence']=='C_REVIEW_ONLY' for e in detect(r,sg)))

    def test_bilateral_sensor_tail_is_excluded_without_rewriting_raw(self):
        t1=np.arange(0,10,.01)+100.0
        t2=np.arange(0,21,.01)+100.0
        raw={'left':(t1,np.zeros((len(t1),3)),np.zeros((len(t1),3))),
             'right':(t2,np.zeros((len(t2),3)),np.zeros((len(t2),3)))}
        clipped,meta=clip_to_shared_bilateral_window(raw)
        self.assertTrue(meta['applied'])
        self.assertEqual(len(clipped['left'][0]),len(t1))
        self.assertEqual(len(clipped['right'][0]),len(t1))
        self.assertEqual(len(raw['right'][0]),len(t2)) # no mutation
        raw['right']=(t1,np.zeros((len(t1),3)),np.zeros((len(t1),3)))
        _,meta2=clip_to_shared_bilateral_window(raw)
        self.assertFalse(meta2['applied'])

    def test_soft_does_not_use_gt_or_expected_foot_count(self):
        self.assertFalse(any('ground_truth' in k or 'target_steps' in k for k in PARAMS))
        d,f,o,m=frozen_settings()
        self.assertEqual(d['impact']['low_hz'],11.)
        self.assertEqual(d['impact']['high_hz'],24.)
        self.assertEqual(d['stationary']['gyro_threshold_dps'],50.)
        self.assertEqual(f['accel_hz'],5.)
        self.assertEqual(f['gyro_hz'],10.)

    def test_no_extra_on_short_swing_without_stance(self):
        r,sg=fake_case()
        r['contacts']=[]
        sg['swings'][0]['duration_s']=.4
        sg['swings'][0]['valid_swing']=False
        x=detect(r,sg)
        self.assertEqual(x[0]['confidence'],'REJECTED')

if __name__=='__main__':unittest.main()
