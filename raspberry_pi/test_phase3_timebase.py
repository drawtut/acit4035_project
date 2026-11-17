"""Phase3 test time grid and preservation of true acquisition gap evidence."""
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fpa_timebase import prepare_timebase,source_gap_audit

class TimebaseTests(unittest.TestCase):
    def test_known_sine_and_uniform_grid(self):
        rng=np.random.default_rng(7)
        dt=np.clip(.010+rng.normal(0,.002,1500),.003,.020)
        t=np.r_[0,np.cumsum(dt)]
        a=np.column_stack((np.sin(2*np.pi*2*t),np.cos(2*np.pi*3*t),np.ones(len(t))*9.8))
        w=np.column_stack((np.sin(2*np.pi*2*t),np.zeros(len(t)),np.zeros(len(t))))
        a0=a.copy();g0=w.copy();t0=t.copy()
        u,aa,ww,info=prepare_timebase(t,a,w)
        self.assertEqual(info['mode'],'UNIFORM_INTERPOLATED_RAW6')
        self.assertTrue(np.allclose(np.diff(u),np.diff(u)[0],rtol=0,atol=1e-10))
        self.assertTrue(np.array_equal(a,a0) and np.array_equal(w,g0) and np.array_equal(t,t0))
        self.assertLess(np.sqrt(np.mean((aa[:,0]-np.sin(2*np.pi*2*u))**2)),.02)
        self.assertEqual(aa.shape,(len(u),3))
        self.assertEqual(ww.shape,(len(u),3))

    def test_perfect_grid_preserved(self):
        t=np.linspace(0,10,1001);a=np.ones((len(t),3));g=np.zeros_like(a)
        u,aa,ww,m=prepare_timebase(t,a,g)
        self.assertEqual(m['mode'],'NATIVE_ALREADY_UNIFORM')
        self.assertTrue(np.array_equal(u,t))
        self.assertTrue(np.array_equal(aa,a))

    def test_missing_interval_not_interpolated(self):
        t=np.r_[np.linspace(0,3,301),np.linspace(3.085,6,290)]
        a=np.ones((len(t),3));g=np.ones_like(a)
        u,aa,ww,m=prepare_timebase(t,a,g)
        self.assertEqual(m['mode'],'LEGACY_UNSAFE_GAPS_NO_INTERPOLATION')
        self.assertEqual(len(u),len(t))
        self.assertGreater(m['source_max_gap_ms'],80)
        self.assertEqual(m['source_gaps_over_60ms'],1)
        self.assertIn('unvalidated',m['warning'])

    def test_source_gaps_retained_after_regridding(self):
        src=np.array([0,.01,.02,.03,.08,.09,.1])
        audit=source_gap_audit(src,.015,.09,.01)
        self.assertEqual(audit['source_gaps_over_30ms'],1)
        self.assertAlmostEqual(audit['source_max_sample_gap_ms'],50.)
        self.assertGreater(audit['source_missing_fraction'],0)

    def test_no_invented_sensor_channels(self):
        t=np.linspace(0,1,100);t[50:]+=.018
        a=np.random.default_rng(3).normal(size=(100,3));w=np.zeros_like(a)
        grid,aa,gg,meta=prepare_timebase(t,a,w)
        self.assertEqual(aa.shape[1],3)
        self.assertEqual(gg.shape[1],3)
        self.assertTrue(meta['raw_channels_only'])
        self.assertEqual(meta['source_samples'],100)

if __name__ == '__main__':unittest.main()
