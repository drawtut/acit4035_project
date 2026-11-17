"""Phase15 differential is a READ-ONLY derivative of Phase14 Stage4 contacts."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from fpa_differential import pair_landings, summarize_pairs, wrap_deg
from fpa_frozen_profile import frozen_settings
from fpa_raw6_fpa import export_stage4


def row(t, fpa, q='A_CONFIRMED'):
    return dict(contact_time_s=float(t),fpa_deg=fpa,detector_confidence=q)


def mock_stage(left,right):
    return {side:dict(rows=values,mounting_physically_verified=False,
                      separate_functional_calibration=True,
                      reference_line={'reference':0},timebase={})
            for side,values in (('left',left),('right',right))}


class Phase15BilateralTests(unittest.TestCase):
    def test_positive_toe_out_relative_heading_and_asymmetry(self):
        pair=pair_landings(mock_stage([row(6.8,5)],[row(6.1,10)]))[0]
        self.assertEqual(pair['interfoot_divergence_deg'],15)
        self.assertEqual(pair['toe_out_asymmetry_deg'],-5)
        self.assertEqual(pair['pair_confidence'],'A_CONFIRMED')
        self.assertEqual((pair['first_foot'],pair['second_foot']),('right','left'))

    def test_negative_toe_in_and_wrapped_angles(self):
        pair=pair_landings(mock_stage([row(5.,-12)],[row(5.7,4)]))[0]
        self.assertEqual(pair['interfoot_divergence_deg'],-8)
        self.assertEqual(pair['toe_out_asymmetry_deg'],-16)
        self.assertEqual(wrap_deg(200),-160)

    def test_pairing_skips_duplicate_side_without_forced_alternation(self):
        pairs=pair_landings(mock_stage([row(6.,5),row(6.4,6)],[row(6.7,9)]))
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['left_fpa_deg'],6)

    def test_no_pairs_across_large_gap_or_missing_fpa(self):
        self.assertEqual(pair_landings(mock_stage([row(6.,3)],[row(8.1,5)])),[])
        self.assertEqual(pair_landings(mock_stage([row(6.,None)],[row(6.7,5)])),[])
        self.assertEqual(pair_landings(mock_stage([row(6.,float('nan'))],[row(6.7,5)])),[])
        self.assertEqual(pair_landings(mock_stage([row(6.,5)],[row(6.7,float('inf'))])),[])

    def test_contact_never_used_twice(self):
        pairs=pair_landings(mock_stage([row(6.,3),row(7.2,4)],[row(6.6,5),row(7.8,6)]))
        self.assertEqual(len(pairs),2)
        self.assertEqual([p['pair_index'] for p in pairs],[1,2])
        self.assertEqual(pairs[0]['interfoot_divergence_deg'],8)
        self.assertEqual(pairs[1]['interfoot_divergence_deg'],10)

    def test_review_quality_propagation_and_confirmed_only_means(self):
        p=pair_landings(mock_stage([row(6.,3),row(8.,9)],[row(6.7,5),row(8.7,11,'C_REVIEW_ONLY')]))
        self.assertEqual([x['pair_confidence'] for x in p],['A_CONFIRMED','C_REVIEW_ONLY'])
        s=summarize_pairs(p)
        self.assertEqual((s['confirmed_pair_count'],s['review_only_pair_count']),(1,1))
        self.assertEqual(s['mean_interfoot_divergence_deg_confirmed_only'],8)
        self.assertEqual(s['mean_interfoot_divergence_deg_all_candidates'],14)

    def test_dashboard_event_format_is_accepted(self):
        p=pair_landings({'left':[dict(time_s=5.0,fpa_deg=8.,detector_confidence='A_CONFIRMED')],
                         'right':[dict(time_s=5.6,fpa_deg=3.,detector_confidence='B_PROVISIONAL')]})
        self.assertEqual(p[0]['pair_confidence'],'B_PROVISIONAL')

    def test_export_adds_csv_manifest_without_changing_original_angles(self):
        stage=mock_stage([row(6.,3)],[row(6.6,9,'B_PROVISIONAL')])
        for side in stage:
            stage[side]['rows'][0].update(fpa_status='REFERENCE_LINE',
                start_stance_s=0,independent_functional_calibration=True)
        before=[stage[s]['rows'][0]['fpa_deg'] for s in ('left','right')]
        cfg,lab,orient,mounts=frozen_settings()
        with tempfile.TemporaryDirectory() as tmp:
            manifest=export_stage4(stage,tmp,detector_config=cfg,lab_config=lab,
                                   orientation_config=orient,mount_axes=mounts)
            with open(Path(tmp)/'bilateral_differential_pairs.csv',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertAlmostEqual(float(rows[0]['interfoot_divergence_deg']),12)
            self.assertEqual(manifest['bilateral_differential']['stance_supported_pair_count'],1)
            self.assertIn('bilateral_differential',json.loads((Path(tmp)/'offline_stage4_manifest.json').read_text()))
        self.assertEqual(before,[stage[s]['rows'][0]['fpa_deg'] for s in ('left','right')])


if __name__ == '__main__':unittest.main()
