"""Phase 14 regression checks for the automatic taped reference and review-only panel."""
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from fastapi.testclient import TestClient
from online_unified import fpa_online as server


class Phase14UIAndProtocolTests(unittest.TestCase):
    def test_dashboard_has_separate_review_panel_but_no_alignment_checkbox(self):
        with TestClient(server.app) as client:
            page=client.get('/').text
        self.assertIn('id="reviewpanel"',page)
        self.assertIn('id="reviewledger"',page)
        self.assertIn("r.detector_confidence==='C_REVIEW_ONLY'",page)
        self.assertIn('reference-line mode is automatic',page.lower())
        self.assertNotIn('id="reference_line_mode"',page)
        self.assertNotIn("$('reference_line_mode')",page)
        self.assertIn("post('start',{label:$('label').value})",page)
        self.assertIn('No review-only candidates',page)

    def test_new_session_is_aligned_by_default_without_checkbox(self):
        class FakeAcquisition:
            def begin_session(self):pass
            def snapshot(self,session=False):return {}
        class Args:
            left=None;right=None;baud=230400;hz=150;timing='precise'
        instance=server.Controller(Args())
        instance.acquisition=FakeAcquisition()
        with TemporaryDirectory() as d:
            with patch.object(server,'HERE',Path(d)):
                location=Path(instance.start('PTEST'))
                protocol=json.loads((location/'protocol.json').read_text())
                self.assertIs(protocol['reference_line_mode'],True)
                self.assertEqual(protocol['initial_left_right_shoe_heading_deg'],0)
                self.assertTrue(instance.status()['reference_line_mode'])
                instance.stop()
                summary=json.loads((location/'summary.json').read_text())
                self.assertIs(summary['reference_line_mode'],True)
                instance.live_stop.set()
                if instance.live_thread:instance.live_thread.join(timeout=1)
                if instance._pool:instance._pool.shutdown(wait=False,cancel_futures=True)

    def test_legacy_api_false_cannot_disable_reference_mode(self):
        payload=server.StartRequest(label='PTEST',reference_line_mode=False)
        self.assertEqual(payload.model_dump(),{'label':'PTEST'})

    def test_offline_stage4_uses_automatic_reference(self):
        script=(Path(__file__).parent/'run_offline_stage4.py').read_text()
        self.assertIn('reference_line_mode=True',script)
        self.assertNotIn("args.aligned_start",script)


if __name__=='__main__': unittest.main()
