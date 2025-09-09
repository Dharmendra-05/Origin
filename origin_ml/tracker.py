"""
Provenance Tracker module logging datasets, prompt templates, and hyperparameter telemetry.
"""

import hashlib
import json
import sqlite3
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ExperimentRecord(BaseModel):
    """Container for experiment metadata and lineage hashes."""
    experiment_id: str
    model_name: str
    dataset_hash: str
    prompt_hash: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float = Field(default_factory=time.time)


class ProvenanceTracker:
    """Tracks and logs machine learning experiment provenance into SQLite."""

    def __init__(self, db_path: str = "origin_provenance.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes SQLite provenance database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_provenance (
                    experiment_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)

    def compute_hash(self, content: str) -> str:
        """Computes SHA-256 hash of raw data or prompts."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def log_experiment(
        self,
        experiment_id: str,
        model_name: str,
        raw_dataset_text: str,
        prompt_template: str,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> ExperimentRecord:
        """Logs an ML training experiment record with verifiable dataset/prompt hashes."""
        dataset_hash = self.compute_hash(raw_dataset_text)
        prompt_hash = self.compute_hash(prompt_template)
        
        record = ExperimentRecord(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_hash=dataset_hash,
            prompt_hash=prompt_hash,
            hyperparameters=hyperparams,
            metrics=metrics
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_provenance 
                (experiment_id, model_name, dataset_hash, prompt_hash, hyperparameters, metrics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.model_name,
                    record.dataset_hash,
                    record.prompt_hash,
                    json.dumps(record.hyperparameters),
                    json.dumps(record.metrics),
                    record.timestamp
                )
            )

        return record
