"""Tests for the SANGRE BoundaryGuard preflight gate."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agents.boundary_guard import (
    ALLOW,
    AUDIT_REQUIRED,
    STOP_RUN,
    BoundaryGuard,
)


class TestBoundaryGuardCodeSynthesis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.guard = BoundaryGuard(
            log_path=os.path.join(self.tmp.name, "log.jsonl"),
            manifest_path=os.path.join(self.tmp.name, "manifest.json"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_synthesis_without_stated_gain_is_blocked(self):
        """guard_010: a tool with no declared gain must not reach disk."""
        decision = self.guard.preflight_code_synthesis(
            tool_name="evo_tool_123", write_target="extensions/evo_tool_123.py"
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.decision, AUDIT_REQUIRED)
        self.assertIn("guard_010_complexity_without_gain", decision.triggered_guards)

    def test_synthesis_with_gain_and_criterion_is_allowed(self):
        decision = self.guard.preflight_code_synthesis(
            tool_name="cache_warmer",
            write_target="extensions/cache_warmer.py",
            measurable_gain="Cuts pipeline cold-start from 7s to under 1s",
            validation_criterion="Second render of same prompt completes in <1s",
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.triggered_guards, [])

    def test_missing_rollback_manifest_stops_the_run(self):
        """guard_013 is critical, so it outranks guard_010's audit_required."""
        decision = self.guard.preflight_code_synthesis(
            tool_name="t",
            write_target="extensions/t.py",
            rollback_manifest=False,
        )
        self.assertEqual(decision.decision, STOP_RUN)
        self.assertIn("guard_013_no_ingestion_without_rewind", decision.triggered_guards)

    def test_every_decision_is_logged(self):
        self.guard.preflight_code_synthesis(tool_name="a", write_target="extensions/a.py")
        self.guard.preflight_code_synthesis(
            tool_name="b",
            write_target="extensions/b.py",
            measurable_gain="g",
            validation_criterion="c",
        )
        with open(self.guard.log_path) as f:
            entries = [json.loads(line) for line in f]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["decision"], AUDIT_REQUIRED)
        self.assertEqual(entries[1]["decision"], ALLOW)


class TestBoundaryGuardResultClaims(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.guard = BoundaryGuard(
            log_path=os.path.join(self.tmp.name, "log.jsonl"),
            manifest_path=os.path.join(self.tmp.name, "manifest.json"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_unmeasured_speedup_claim_is_flagged(self):
        """guard_005: the old hardcoded '3.4x throughput speedup' must not pass."""
        decision = self.guard.preflight_result_claim(
            {"status": "evolved", "efficiency_gain": "3.4x throughput speedup"},
            measured_keys=[],
        )
        self.assertFalse(decision.allowed)
        self.assertIn("guard_005_experiment_as_proof", decision.triggered_guards)

    def test_measured_value_is_permitted(self):
        decision = self.guard.preflight_result_claim(
            {"inference_time": "2.1x faster than baseline"},
            measured_keys=["inference_time"],
        )
        self.assertTrue(decision.allowed)

    def test_plain_report_passes(self):
        decision = self.guard.preflight_result_claim(
            {"status": "evolved", "goal": "reduce queue latency"}, measured_keys=[]
        )
        self.assertTrue(decision.allowed)


class TestRollback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.guard = BoundaryGuard(
            log_path=os.path.join(self.tmp.name, "log.jsonl"),
            manifest_path=os.path.join(self.tmp.name, "manifest.json"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_rollback_removes_recorded_files(self):
        created = os.path.join(self.tmp.name, "generated.py")
        with open(created, "w") as f:
            f.write("def run(data): return data\n")

        self.guard.record_rollback_manifest(run_id="run_1", created_files=[created])
        self.assertTrue(os.path.exists(created))

        result = self.guard.rollback("run_1")
        self.assertTrue(result["success"])
        self.assertEqual(result["removed"], [created])
        self.assertFalse(os.path.exists(created))

    def test_rollback_without_manifest_reports_failure(self):
        result = self.guard.rollback("nonexistent")
        self.assertFalse(result["success"])


class TestCoderAgentIntegration(unittest.TestCase):
    def test_coder_agent_does_not_write_when_guard_blocks(self):
        """The gate must sit in front of the write, not after it."""
        from agents.coder_agent import CoderAgent

        with tempfile.TemporaryDirectory() as tmp:
            guard = BoundaryGuard(
                log_path=os.path.join(tmp, "log.jsonl"),
                manifest_path=os.path.join(tmp, "manifest.json"),
            )
            ext_dir = os.path.join(tmp, "extensions")
            agent = CoderAgent(extensions_dir=ext_dir, boundary_guard=guard)

            with patch.object(agent.gemma, "generate") as mock_gen:
                result = agent.synthesize_tool(task_description="do a thing", tool_name="ungated")
                mock_gen.assert_not_called()

            self.assertFalse(result["success"])
            self.assertTrue(result["blocked_by_guard"])
            self.assertFalse(os.path.exists(os.path.join(ext_dir, "ungated.py")))


if __name__ == "__main__":
    unittest.main()
