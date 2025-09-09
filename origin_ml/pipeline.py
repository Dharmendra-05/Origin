"""
Pipeline Orchestrator for chaining RAG retrieval, ML inference, and provenance logging.
Coordinates multi-stage workflows with automatic lineage tracking at each step.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class StageStatus(Enum):
    """Execution status for each pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Represents a single stage in the ML pipeline."""
    name: str
    handler: Callable[..., Dict[str, Any]]
    status: StageStatus = StageStatus.PENDING
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineOrchestrator:
    """
    Orchestrates multi-stage ML pipelines with automatic
    provenance capture at each transition.

    Stages are executed in sequence. Each stage receives the accumulated
    context from all prior stages and appends its own outputs.
    """

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.stages: List[PipelineStage] = []
        self.context: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self._created_at = time.time()

    def add_stage(
        self,
        name: str,
        handler: Callable[..., Dict[str, Any]],
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None
    ) -> "PipelineOrchestrator":
        """Registers a new stage in the pipeline. Returns self for chaining."""
        stage = PipelineStage(
            name=name,
            handler=handler,
            input_keys=input_keys or [],
            output_keys=output_keys or []
        )
        self.stages.append(stage)
        return self

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the full pipeline sequentially.

        Args:
            initial_context: Seed data passed to the first stage.

        Returns:
            Final accumulated context after all stages complete.
        """
        self.context = initial_context.copy() if initial_context else {}
        pipeline_start = time.time()

        for stage in self.stages:
            stage_input = {k: self.context[k] for k in stage.input_keys if k in self.context}
            stage.status = StageStatus.RUNNING

            t0 = time.time()
            try:
                result = stage.handler(**stage_input) if stage_input else stage.handler()
                stage.duration_ms = (time.time() - t0) * 1000
                stage.status = StageStatus.COMPLETED
                stage.result = result

                if isinstance(result, dict):
                    self.context.update(result)

            except Exception as exc:
                stage.duration_ms = (time.time() - t0) * 1000
                stage.status = StageStatus.FAILED
                stage.error = str(exc)
                self._log_event(stage, "FAILED")
                raise RuntimeError(
                    f"Pipeline '{self.pipeline_id}' failed at stage '{stage.name}': {exc}"
                ) from exc

            self._log_event(stage, "OK")

        total_ms = (time.time() - pipeline_start) * 1000
        self.context["_pipeline_duration_ms"] = total_ms
        return self.context

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all stage executions."""
        return {
            "pipeline_id": self.pipeline_id,
            "total_stages": len(self.stages),
            "stages": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration_ms": round(s.duration_ms, 2),
                    "error": s.error
                }
                for s in self.stages
            ],
            "execution_log": self.execution_log
        }

    def _log_event(self, stage: PipelineStage, outcome: str):
        """Appends a timestamped event to the execution log."""
        self.execution_log.append({
            "timestamp": time.time(),
            "stage": stage.name,
            "outcome": outcome,
            "duration_ms": round(stage.duration_ms, 2)
        })
