"""
Provenance Auditor module verifying dataset integrity and experiment lineage.
"""

import json
import sqlite3
from typing import Optional, Dict, Any
from origin_ml.tracker import ProvenanceTracker, ExperimentRecord


class ProvenanceAuditor:
    """Audits and verifies recorded model provenance against raw datasets."""

    def __init__(self, tracker: ProvenanceTracker):
        self.tracker = tracker

    def verify_experiment(
        self,
        experiment_id: str,
        raw_dataset_text: str,
        prompt_template: str
    ) -> Dict[str, Any]:
        """Audits an experiment against candidate dataset and prompt text."""
        with sqlite3.connect(self.tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT experiment_id, model_name, dataset_hash, prompt_hash, hyperparameters, metrics, timestamp FROM experiment_provenance WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()

        if not row:
            return {"verified": False, "reason": f"Experiment ID '{experiment_id}' not found in database."}

        stored_dataset_hash = row[2]
        stored_prompt_hash = row[3]

        current_dataset_hash = self.tracker.compute_hash(raw_dataset_text)
        current_prompt_hash = self.tracker.compute_hash(prompt_template)

        dataset_match = (stored_dataset_hash == current_dataset_hash)
        prompt_match = (stored_prompt_hash == current_prompt_hash)

        is_valid = dataset_match and prompt_match

        return {
            "verified": is_valid,
            "experiment_id": experiment_id,
            "model_name": row[1],
            "dataset_match": dataset_match,
            "prompt_match": prompt_match,
            "stored_metrics": json.loads(row[5])
        }
