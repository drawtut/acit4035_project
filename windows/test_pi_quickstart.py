"""Self-contained fast-deployment checks that do not require attached sensors."""
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from online_unified import fpa_online as server

class QuickStartTest(unittest.TestCase):
    def test_core_runtime_imports(self):
        from fpa_unified_pipeline import calibration_pair, source_audit
        cal=calibration_pair(Path(__file__).parent/'online_unified/calibration_frozen_v1')
        self.assertEqual(set(cal), {'left','right'})
        self.assertEqual(len(source_audit()), 4)
    def test_demo_root_status_and_export_error(self):
        with TestClient(server.app) as client:
            r=client.get('/')
            self.assertEqual(r.status_code,200)
            self.assertIn('Download session ZIP',r.text)
            self.assertEqual(client.get('/api/status').status_code,200)
            self.assertEqual(client.get('/api/download_latest').status_code,503)
    def test_download_latest_includes_raw_data_and_frozen_calibration(self):
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile
        import io, threading
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'P01_walk'
            folder.mkdir()
            (folder/'left_raw.csv').write_text('t,ax\n0,1\n')
            (folder/'right_raw.csv').write_text('t,ax\n0,1\n')
            class Stub:
                lock=threading.RLock()
                recording=False
                processing=False
                session=folder
                def boot(self): pass
                def shutdown(self): pass
            with patch.object(server,'controller',Stub()):
                with TestClient(server.app) as client:
                    response=client.get('/api/download_latest')
                    self.assertEqual(response.status_code,200)
                    with ZipFile(io.BytesIO(response.content)) as z:
                        self.assertIn('P01_walk/left_raw.csv',z.namelist())
                        self.assertIn('P01_walk/right_raw.csv',z.namelist())
                        self.assertIn('P01_walk/calibration/left_foot_calibration.json',z.namelist())
                        self.assertIn('P01_walk/calibration/right_foot_calibration.json',z.namelist())
    def test_preflight_missing_bilateral_samples(self):
        class FakeAcquisition:
            def start(self): pass
            def stop(self): pass
            def begin_session(self): pass
            def snapshot(self,session=False): return {}
        class Args:
            left=right=None;baud=230400;hz=100;timing='precise'
        instance=server.Controller(Args())
        instance.acquisition=FakeAcquisition()
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            from unittest.mock import patch
            with patch.object(server,'HERE',Path(tmp)):
                instance.start('P01')
                instance.stop()
                instance.analyze()
            self.assertIn('Not enough sensor data',instance.last_error)

if __name__=='__main__': unittest.main()
