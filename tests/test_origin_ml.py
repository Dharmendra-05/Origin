"""
Unit tests for Origin-ML Data Lineage & Provenance Tracker module.
"""

import unittest
import os
import tempfile
from origin_ml.tracker import ProvenanceTracker
from origin_ml.audit import ProvenanceAuditor


class TestOriginML(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db = tmp.name
        tmp.close()
        self.tracker = ProvenanceTracker(db_path=self.tmp_db)
        self.auditor = ProvenanceAuditor(self.tracker)

    def tearDown(self):
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)

    def test_log_and_verify_experiment(self):
        dataset_content = "feature_1,feature_2,label\n1.0,2.0,1\n3.0,4.0,0"
        prompt = "System prompt for classification"
        
        record = self.tracker.log_experiment(
            experiment_id="exp_001",
            model_name="origin-classifier-v1",
            raw_dataset_text=dataset_content,
            prompt_template=prompt,
            hyperparams={"learning_rate": 0.001, "epochs": 10},
            metrics={"accuracy": 0.95, "f1_score": 0.94}
        )

        self.assertEqual(record.experiment_id, "exp_001")
        self.assertIsNotNone(record.dataset_hash)

        # Audit verification pass
        audit_res = self.auditor.verify_experiment("exp_001", dataset_content, prompt)
        self.assertTrue(audit_res["verified"])
        self.assertTrue(audit_res["dataset_match"])
        self.assertTrue(audit_res["prompt_match"])

        # Audit verification fail on tampered data
        tampered_dataset = "feature_1,feature_2,label\n1.0,2.0,0"
        failed_audit = self.auditor.verify_experiment("exp_001", tampered_dataset, prompt)
        self.assertFalse(failed_audit["verified"])
        self.assertFalse(failed_audit["dataset_match"])


if __name__ == "__main__":
    unittest.main()
