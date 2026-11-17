import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fpa_calibration_engine import build_calibration,save_pair,load_pair

CFG={'impact':{'low_hz':10}}
LAB={'gyro_hz':10.,'accel_hz':5.}
ORI={'tilt_correction_tau_s':.75}
MOUNT={'left':'+Y','right':'+Y'}

def example(side):
    return {'calibration':{'usable':True,'side':side,'separate_calibration_trial':False,
      'assumption_mount_forward_axis':'+Y','swings_used':4,'total_accepted_swings':8,
      'sensor_forward_unit':[0.,1.,0.], 'pca_eigenvalue_ratio':4.,
      'alignment_with_mount_axis':.9},'timebase':{'schema':'PHASE3_UNIFORM_RAW6_V1','mode':'NATIVE_ALREADY_UNIFORM'},'source_raw6_sha256':('1' if side=='left' else '2')*64}

class SeparateCalibrationTests(unittest.TestCase):
    def make(self, **kw):
        return build_calibration({'left':example('left'),'right':example('right')},CFG,LAB,ORI,MOUNT,**kw)
    def test_roundtrip_and_flags(self):
        cal=self.make(forward_walk_confirmed=True,mounting_verified=True)
        with tempfile.TemporaryDirectory() as d:
            save_pair(cal,d);restored=load_pair(d,CFG,LAB,ORI,MOUNT)
        self.assertEqual(restored['right']['source_raw6_sha256'],'2'*64)
        self.assertTrue(restored['left']['separate_calibration_trial'])
    def test_requires_straight_walk_confirmation(self):
        cal=self.make()
        with tempfile.TemporaryDirectory() as d:
            save_pair(cal,d)
            with self.assertRaisesRegex(ValueError,'straight-forward functional walk'):
                load_pair(d,CFG,LAB,ORI,MOUNT)
    def test_filter_mismatch_rejected(self):
        cal=self.make(forward_walk_confirmed=True)
        with tempfile.TemporaryDirectory() as d:
            save_pair(cal,d)
            with self.assertRaisesRegex(ValueError,'DIFFERENT'):
                load_pair(d,CFG,{'gyro_hz':20},ORI,MOUNT)
    def test_minimum_swings(self):
        r={'left':example('left'),'right':example('right')}
        r['right']['calibration']['swings_used']=3
        with self.assertRaisesRegex(ValueError,'need >=4'):
            build_calibration(r,CFG,LAB,ORI,MOUNT)
    def test_mount_mismatch_rejected(self):
        cal=self.make(forward_walk_confirmed=True)
        with tempfile.TemporaryDirectory() as d:
            save_pair(cal,d)
            with self.assertRaisesRegex(ValueError,'mounting direction changed'):
                load_pair(d,CFG,LAB,ORI,{'left':'-Y','right':'+Y'})

if __name__=='__main__':unittest.main()
