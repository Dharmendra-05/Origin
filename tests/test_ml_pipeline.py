"""
Unit tests for the PipelineOrchestrator multi-stage execution engine.
"""

import unittest
from origin_ml.pipeline import PipelineOrchestrator, StageStatus


class TestPipelineOrchestrator(unittest.TestCase):

    def test_empty_pipeline_returns_context(self):
        pipe = PipelineOrchestrator("test-empty")
        result = pipe.run({"seed": 42})
        self.assertEqual(result["seed"], 42)
        self.assertIn("_pipeline_duration_ms", result)

    def test_single_stage_execution(self):
        def greet():
            return {"message": "hello"}

        pipe = PipelineOrchestrator("test-single")
        pipe.add_stage("greeter", greet, output_keys=["message"])
        result = pipe.run()

        self.assertEqual(result["message"], "hello")
        self.assertEqual(pipe.stages[0].status, StageStatus.COMPLETED)

    def test_chained_stages_accumulate_context(self):
        def stage_a():
            return {"value": 10}

        def stage_b(value=0):
            return {"doubled": value * 2}

        pipe = PipelineOrchestrator("test-chain")
        pipe.add_stage("produce", stage_a)
        pipe.add_stage("double", stage_b, input_keys=["value"])
        result = pipe.run()

        self.assertEqual(result["value"], 10)
        self.assertEqual(result["doubled"], 20)

    def test_failing_stage_raises_runtime_error(self):
        def bad_stage():
            raise ValueError("boom")

        pipe = PipelineOrchestrator("test-fail")
        pipe.add_stage("explode", bad_stage)

        with self.assertRaises(RuntimeError) as ctx:
            pipe.run()

        self.assertIn("boom", str(ctx.exception))
        self.assertEqual(pipe.stages[0].status, StageStatus.FAILED)

    def test_get_summary_records_all_stages(self):
        def noop():
            return {}

        pipe = PipelineOrchestrator("test-summary")
        pipe.add_stage("step1", noop)
        pipe.add_stage("step2", noop)
        pipe.run()

        summary = pipe.get_summary()
        self.assertEqual(summary["pipeline_id"], "test-summary")
        self.assertEqual(summary["total_stages"], 2)
        self.assertTrue(all(s["status"] == "completed" for s in summary["stages"]))

    def test_fluent_api_chaining(self):
        pipe = (
            PipelineOrchestrator("test-fluent")
            .add_stage("a", lambda: {"x": 1})
            .add_stage("b", lambda: {"y": 2})
        )
        result = pipe.run()
        self.assertEqual(result["x"], 1)
        self.assertEqual(result["y"], 2)


if __name__ == "__main__":
    unittest.main()
