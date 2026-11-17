"""Headless workflow checks. Run: py test_reader_workflow.py"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Stub only serial/acquisition at import: no port I/O in tests.
serial = types.ModuleType('serial')
serial.SerialException = type('SerialException', (Exception,), {})
serial.Serial = object
sys.modules.setdefault('serial', serial)
imu_reader = types.ModuleType('imu_reader')
imu_reader.LEFT_PORT = 'COM5'
imu_reader.RIGHT_PORT = 'COM4'
imu_reader.MODBUS_ADDRESS = 0x50
sys.modules.setdefault('imu_reader', imu_reader)
acq = types.ModuleType('imu_acquisition')
acq.FIELDS = ['side', 'timestamp_midpoint_monotonic_s']
acq.REQUEST_TIMEOUT_S = 0.1
for name in ('read_modbus_registers', 'decode_sample', 'try_baud', 'wait_until'):
    setattr(acq, name, lambda *a, **k: None)
sys.modules.setdefault('imu_acquisition', acq)
spec = importlib.util.spec_from_file_location('fpa_reader', Path(__file__).with_name('fpa_reader.py'))
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

class Dummy:
    pass

class Tests(unittest.TestCase):
    def test_manifest_hash_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'toe.json'
            p.write_text(json.dumps({'schema':'ACIT4035_TOE_AXIS_V1','side':'left'}))
            entry = {'path':str(p), 'sha256':reader.sha256_file(p)}
            self.assertEqual(reader.verified_calibration_entry(entry,'left','ACIT4035_TOE_AXIS_V1'),p)
            self.assertIsNone(reader.verified_calibration_entry(entry,'right','ACIT4035_TOE_AXIS_V1'))
            p.write_text('tampered')
            self.assertIsNone(reader.verified_calibration_entry(entry,'left','ACIT4035_TOE_AXIS_V1'))

    def test_functional_calibration_uses_distinct_toe_files(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            manifest={'schema':'ACIT4035_READER_CAL_MANIFEST_V1','toe':{},'foot':{}}
            for side in ('left','right'):
                toe = d / f'toe_axis_{side}.json'
                toe.write_text(json.dumps({'schema':'ACIT4035_TOE_AXIS_V1','side':side}))
                manifest['toe'][side]={'path':str(toe),'sha256':reader.sha256_file(toe)}
            reader.MANIFEST=d/'manifest.json'
            o = Dummy()
            o.sides=['left','right'];o.run_dir=d;o.calibration_manifest=manifest
            seen=[]
            def fake_run(cmd, **kw):
                self.assertIn('--toe-axis',cmd)
                side=cmd[cmd.index('--side')+1]
                toe=cmd[cmd.index('--toe-axis')+1]
                self.assertEqual(Path(toe).name,f'toe_axis_{side}.json')
                output=Path(cmd[cmd.index('--output')+1])
                output.write_text(json.dumps({'schema':'ACIT4035_FPA_CALIBRATION_V2','side':side,
                                              'toe_pca_agreement_deg':5.0}))
                seen.append(side)
                return types.SimpleNamespace(returncode=0,stdout='ok',stderr='')
            original_is_file = Path.is_file
            def fake_is_file(path):
                return True if Path(path).name == 'fpa_estimation.py' else original_is_file(path)
            with patch.object(reader.subprocess, 'run',fake_run), patch.object(reader.messagebox,'showinfo'), patch.object(Path,'is_file',fake_is_file):
                reader.FPAReader.run_offline_calibration(o, {})
            self.assertEqual(seen,['left','right'])
            self.assertEqual(set(manifest['foot']),{'left','right'})
            self.assertTrue((d/'manifest.json').exists())

    def test_toe_lift_processing_invalidates_old_functional_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            reader.MANIFEST=d/'manifest.json'
            o=Dummy();o.run_dir=d
            o.calibration_manifest={'schema':'ACIT4035_READER_CAL_MANIFEST_V1',
                'toe':{},'foot':{'left':{'path':'old.json','sha256':'invalid'}}}
            side='left'; seen=[]
            def fake_run(cmd, **kw):
                self.assertEqual(cmd[cmd.index('--side')+1],side)
                self.assertEqual(cmd[2],'toe-axis')
                out=Path(cmd[cmd.index('--output')+1])
                out.write_text(json.dumps({'schema':'ACIT4035_TOE_AXIS_V1',
                    'side':side,'minimum_direction_cosine':0.999}))
                seen.append('ok')
                return types.SimpleNamespace(returncode=0,stdout='done',stderr='')
            original_is_file=Path.is_file
            with patch.object(reader.subprocess,'run',fake_run), \
                 patch.object(reader.messagebox,'showinfo'), \
                 patch.object(Path,'is_file',lambda p: True if Path(p).name == 'fpa_estimation.py' else original_is_file(p)):
                reader.FPAReader.run_toe_processing(o,side)
            self.assertEqual(seen,['ok'])
            self.assertNotIn('left',o.calibration_manifest['foot'])
            self.assertIn('left',o.calibration_manifest['toe'])

    def test_guided_trial_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            manifest={'foot':{}}
            for side in ('left','right'):
                p=d/f'foot_calibration_{side}.json'
                p.write_text(json.dumps({'schema':'ACIT4035_FPA_CALIBRATION_V2','side':side}))
                manifest['foot'][side]={'path':str(p),'sha256':reader.sha256_file(p)}
            o=Dummy();o.sides=['left','right'];o.run_dir=d;o.calibration_manifest=manifest
            o.simple_straight_trial=False
            o.trial_t0=100.;o.trial_move_start=105.;o.trial_move_end=115.
            report={'duration_s':20.}
            reader.FPAReader.write_trial_metadata(o,report)
            meta=json.loads((d/'trial_session.json').read_text())
            self.assertEqual(meta['walking_window_s'],[5.,15.])
            self.assertTrue(all(meta['calibration'][s] for s in o.sides))
            self.assertIsNone(meta['video_offset_s'])

if __name__=='__main__': unittest.main()
