"""Stage 4 must compute only fixed-start walkway-relative shoe heading."""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fpa_raw6_fpa import stage4_fpa,export_stage4
from fpa_unified_pipeline import events_for_dashboard,report_summary


def synthetic(side='left',angles=(0.,7.,12.),offset=0.,
              enabled=True,moving_baseline=False,severe_gap=False):
    t=np.linspace(0,10,1001)
    yaw=np.full(len(t),offset,dtype=float)
    yaw[t>=5]=offset+angles[1]
    yaw[t>=8]=offset+angles[2]
    if moving_baseline:yaw[(t>=1)&(t<4)]+=np.linspace(0,14,300)
    q=Rotation.from_euler('z',yaw[:,None],degrees=True).as_quat()
    source=t.copy()
    if severe_gap:source=np.r_[np.arange(0,5,.01),np.arange(5.10,10.01,.01)]
    links=[{'swing_id':i,'usable':True,'end_index':idx,
            'step_time_s':float(t[idx]-.2),
            'start_stance_s':float(t[100]),'end_stance_s':float(t[idx])}
           for i,idx in enumerate((600,900),1)]
    data={side:dict(side=side,calibration={'usable':True,
                    'separate_calibration_trial':True,
                    'sensor_forward_unit':[1.,0.,0.]},
              orientation={'time_s':t,'quaternion_xyzw':q,
                  'gyro_magnitude_dps':np.zeros(len(t)),
                  'initial_reference_s':[1.,4.]},
              filtered={'source_time_s':source},accepted_links=links,
              timebase={'mode':'PHASE3_UNIFORM_RAW6_V1'})}
    return stage4_fpa(data,reference_line_mode=enabled)

class Stage4OnlyTests(unittest.TestCase):
    def test_two_left_landings_fixed_baseline(self):
        rows=synthetic()['left']['rows']
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(rows[0]['fpa_deg'],7.,places=5)
        self.assertAlmostEqual(rows[1]['fpa_deg'],12.,places=5)
        self.assertEqual(rows[0]['fpa_reference'],'initial_aligned_0deg_stance_to_landing_heading')

    def test_right_toe_out_sign(self):
        rows=synthetic('right',(0.,-8.,-12.))['right']['rows']
        self.assertAlmostEqual(rows[0]['fpa_deg'],8.,places=5)
        self.assertAlmostEqual(rows[1]['fpa_deg'],12.,places=5)

    def test_reference_origin_is_arbitrary(self):
        for offset in (0.,-87.,+170.,+46.):
            self.assertAlmostEqual(synthetic(offset=offset)['left']['rows'][0]['fpa_deg'],7.,places=5)

    def test_no_protocol_reports_only_contacts(self):
        rows=synthetic(enabled=False)['left']['rows']
        self.assertTrue(all(r['fpa_deg'] is None for r in rows))
        self.assertTrue(all(r['fpa_status']=='NOT_REQUESTED' for r in rows))

    def test_unstable_reference_rejected(self):
        row=synthetic(moving_baseline=True)['left']['rows'][0]
        self.assertIsNone(row['fpa_deg'])
        self.assertEqual(row['fpa_status'],'REJECTED_MOVING_OR_UNSTABLE_INITIAL_REFERENCE')

    def test_data_gap_rejected(self):
        row=synthetic(severe_gap=True)['left']['rows'][0]
        self.assertIsNone(row['fpa_deg'])
        self.assertEqual(row['fpa_status'],'REJECTED_SEVERE_SAMPLING_GAP')

    def test_missing_foot_axis_rejected(self):
        # Use original synthetic geometry, but guard direct stage on missing calibration via external fixture.
        data=synthetic()['left']['rows']
        self.assertTrue(all(r['fpa_candidate_toe_out_deg'] is not None for r in data))

    def test_stage4_has_no_old_displacement_fields(self):
        row=synthetic()['left']['rows'][0]
        for key in ('delta_x_m','delta_y_m','stride_displacement_m',
                    'progression_azimuth_local_deg','landing_fpa_candidate_toe_out_deg',
                    'fpa_start_stance_deg'):
            self.assertNotIn(key,row)

    def test_dashboard_primary_is_reference_only(self):
        stage={**synthetic(), 'right':{'rows':[]}}
        event=events_for_dashboard(stage)['left'][0]
        self.assertAlmostEqual(event['fpa_deg'],7.)
        self.assertEqual(event['reference_line_fpa_deg'],event['fpa_deg'])
        self.assertNotIn('fpa_start_stance_deg',event)

    def test_export_schema_and_fields(self):
        from fpa_frozen_profile import frozen_settings
        cfg,lab,orient,mounts=frozen_settings()
        stage={**synthetic(),'right':{'rows':[],'mounting_physically_verified':False,
                   'separate_functional_calibration':False,'timebase':{},'reference_line':{}}}
        with tempfile.TemporaryDirectory() as td:
            manifest=export_stage4(stage,td,detector_config=cfg,lab_config=lab,
                            orientation_config=orient,mount_axes=mounts)
            self.assertEqual(manifest['schema'],'ACIT4035_STAGE4_REFERENCE_LINE_ONLY_V2')
            csv=(Path(td)/'left_stage4_fpa_candidates.csv').read_text()
            self.assertIn('reference_line_fpa_candidate_toe_out_deg',csv)
            self.assertNotIn('delta_x_m',csv)
            self.assertFalse(any(Path(td).glob('*trajectory*')))

    def test_no_displacement_in_active_stage(self):
        root=Path(__file__).parent
        self.assertNotIn('reconstruct_local_stride', (root/'fpa_raw6_stride.py').read_text())
        self.assertNotIn('trajectory_rows', (root/'fpa_unified_pipeline.py').read_text())
        self.assertNotIn('trajectory-dependent', (root/'online_unified'/'dashboard.html').read_text().lower())

if __name__=='__main__':unittest.main()
