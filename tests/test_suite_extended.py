"""
Extended Unit Test Suite for Master Spine and Advanced Gemma Agents.
"""

import os
import pytest
from agents import (
    MasterSpineCoordinator,
    get_spine,
    VideoAgent,
    CoderAgent,
    AutoHealer,
    DebateEngine,
    RAGAgent,
    GemmaAgent,
)


@pytest.fixture
def spine():
    """Get MasterSpineCoordinator singleton."""
    return get_spine()


class TestMasterSpineCoordinator:
    """Tests for Master Execution Spine locking and barriers."""

    def test_spine_singleton(self, spine):
        """Test singleton pattern."""
        spine2 = get_spine()
        assert spine is spine2

    def test_acquire_and_release_lock(self, spine):
        """Test lock acquisition and release."""
        res = spine.acquire_lock("video", "task_123")
        assert res is True
        status = spine.get_status()
        assert "video" in status["active_locks"]

        rel = spine.release_lock("video", "task_123")
        assert rel is True
        status2 = spine.get_status()
        assert "video" not in status2["active_locks"]

    def test_spine_guarded_decorator(self, spine):
        """Test spine_guarded decorator execution."""

        @spine.spine_guarded("test_component")
        def sample_work(x: int) -> int:
            return x * 2

        assert sample_work(5) == 10


class TestQuetzalVideoAgent:
    """Tests for VideoAgent and Quetzal Diffusion Engine."""

    def test_create_clip(self, tmp_path):
        """Test video clip generation."""
        out_dir = str(tmp_path / "videos")
        agent = VideoAgent(output_dir=out_dir)
        res = agent.create_clip(
            prompt="Cybernetic Quetzal flying over city",
            width=256,
            height=256,
            num_frames=4,
            fps=4,
            style="quetzal_diffusion",
        )
        assert res["success"] is True
        assert "clip_id" in res
        assert os.path.exists(res["file_path"])


class TestCoderAgent:
    """Tests for CoderAgent tool synthesizer."""

    def test_synthesize_tool(self, tmp_path):
        """Test tool code synthesis."""
        ext_dir = str(tmp_path / "extensions")
        agent = CoderAgent(extensions_dir=ext_dir)
        res = agent.synthesize_tool(
            "Process payload dictionary",
            tool_name="test_tool",
            measurable_gain="Normalizes payload keys in one pass instead of three",
            validation_criterion="run({'a': 1}) returns a dict containing key 'a'",
        )
        assert res["success"] is True
        assert os.path.exists(res["filepath"])

    def test_execute_tool(self, tmp_path):
        """Test dynamic execution of synthesized tool."""
        ext_dir = str(tmp_path / "extensions")
        agent = CoderAgent(extensions_dir=ext_dir)
        synth = agent.synthesize_tool(
            "Process payload dictionary",
            tool_name="sample_exec_tool",
            measurable_gain="Normalizes payload keys in one pass instead of three",
            validation_criterion="run({'a': 1}) returns a dict containing key 'a'",
        )
        assert synth["success"] is True

        exec_res = agent.execute_tool("sample_exec_tool", {"test_key": "test_val"})
        assert exec_res["success"] is True


class TestAutoHealer:
    """Tests for AutoHealer exception repair."""

    def test_execute_with_repair_success(self):
        """Test execution when function succeeds."""
        healer = AutoHealer()

        def happy_func(data):
            return data["val"] * 2

        res = healer.execute_with_repair(happy_func, {"val": 5})
        assert res["success"] is True
        assert res["result"] == 10

    def test_execute_with_repair_auto_fix(self):
        """Test automatic payload repairing on exception."""
        healer = AutoHealer()

        def strict_func(data):
            # Requires numeric val
            if isinstance(data.get("val"), str):
                raise ValueError("val must be numeric")
            return float(data["val"]) * 3

        res = healer.execute_with_repair(strict_func, {"val": "10.0"})
        assert res["success"] is True
        assert res["repaired"] is True
        assert res["result"] == 30.0


class TestDebateEngine:
    """Tests for Multi-Agent Debate Engine."""

    def test_run_debate(self):
        """Test debate engine consensus round."""
        engine = DebateEngine()
        res = engine.run_debate(
            topic="Deploying edge agent on Apple Silicon",
            proposal={"hardware": "Apple M5 Pro", "ram": "24GB"},
            rounds=1,
        )
        assert res["success"] is True
        assert "final_verdict" in res
        assert "transcript" in res


class TestRAGAgent:
    """Tests for Local Edge RAG Engine."""

    def test_rag_ingest_and_query(self):
        rag = RAGAgent()
        chunks = rag.ingest_text("doc1", "Gemma is Google open weights model built on Apple Silicon GPU architecture.")
        assert chunks > 0

        res = rag.query("What model is Gemma?")
        assert res["success"] is True
        assert res["context_chunks_retrieved"] > 0


class TestAutonomousEvolutionEngine:
    """Tests for Continuous Self-Improvement Engine."""

    def test_evolution_cycle(self, tmp_path):
        from agents.boundary_guard import BoundaryGuard
        from agents.coder_agent import CoderAgent
        from agents.evolution_engine import AutonomousEvolutionEngine

        # Keep synthesized tools out of the repo's real extensions/ directory.
        guard = BoundaryGuard(
            log_path=str(tmp_path / "log.jsonl"),
            manifest_path=str(tmp_path / "manifest.json"),
        )
        coder = CoderAgent(extensions_dir=str(tmp_path / "extensions"), boundary_guard=guard)
        engine = AutonomousEvolutionEngine(coder_agent=coder)
        result = engine.run_evolution_cycle("Self-test optimization")
        assert result["success"] is True
        assert "cycle_id" in result
        assert result["total_cycles_executed"] >= 1

    def test_evolution_cycle_blocks_ungated_tool_write(self, tmp_path):
        """A cycle that declares no gain must not leave a tool behind."""
        from agents.boundary_guard import BoundaryGuard
        from agents.coder_agent import CoderAgent
        from agents.evolution_engine import AutonomousEvolutionEngine

        ext_dir = tmp_path / "extensions"
        guard = BoundaryGuard(
            log_path=str(tmp_path / "log.jsonl"),
            manifest_path=str(tmp_path / "manifest.json"),
        )
        coder = CoderAgent(extensions_dir=str(ext_dir), boundary_guard=guard)
        engine = AutonomousEvolutionEngine(coder_agent=coder)
        result = engine.run_evolution_cycle("Self-test optimization")

        assert result["evolution_report"]["status"] == "blocked"
        assert list(ext_dir.glob("*.py")) == []


class TestSelfOptimizingVisualTrainer:
    """Tests for Reverse-Engineering Visual Trainer."""

    def test_trainer_extract_and_train(self):
        from agents.trainer_agent import SelfOptimizingVisualTrainer
        trainer = SelfOptimizingVisualTrainer()
        ref_path = "static/videos/diffused_472cd5f623.png"
        meta = trainer.extract_image_metadata(ref_path)
        assert meta["success"] is True

        res = trainer.train_image_match_loop(
            reference_path=ref_path,
            user_description="Test reference image",
            max_iterations=1,
            target_score=50.0,
        )
        assert res["success"] is True

        # The score must be a real measurement against the reference, not a
        # value derived from the iteration or step count.
        record = res["iterations"][0]
        assert record["metric"] is not None
        assert -1.0 <= record["ssim"] <= 1.0
        assert -1.0 <= record["histogram_correlation"] <= 1.0
        assert record["score"] == res["achieved_score"]


class TestCleanHouse:
    """Tests for CleanHouse Pre-Flight Cleanup Utility."""

    def test_cleanhouse_sweep(self):
        from unittest.mock import patch
        from scripts import cleanhouse as ch

        # Stub the listener lookup so the sweep never terminates a real dev
        # server that happens to be running on these ports during the test run.
        with patch.object(ch, "find_processes_on_port", return_value=[]):
            report = ch.cleanhouse(ports=[5005, 8000])
        assert report["success"] is True
        assert "recommended_port" in report
        assert report["cleaned_processes"] == []

    def test_cleanhouse_terminates_stale_listener(self):
        from unittest.mock import MagicMock, patch
        from scripts import cleanhouse as ch

        fake_proc = MagicMock()
        fake_proc.name.return_value = "Python"
        fake_proc.username.return_value = "dev"

        with patch.object(ch, "find_processes_on_port", return_value=[4242]), \
             patch.object(ch.psutil, "Process", return_value=fake_proc), \
             patch.object(ch, "is_port_in_use", return_value=False):
            report = ch.cleanhouse(ports=[5005])

        assert [p["pid"] for p in report["cleaned_processes"]] == [4242]
        fake_proc.send_signal.assert_called_once()

    def test_cleanhouse_skips_system_processes(self):
        from unittest.mock import MagicMock, patch
        from scripts import cleanhouse as ch

        fake_proc = MagicMock()
        fake_proc.name.return_value = "ControlCenter"
        fake_proc.username.return_value = "dev"

        with patch.object(ch, "find_processes_on_port", return_value=[623]), \
             patch.object(ch.psutil, "Process", return_value=fake_proc), \
             patch.object(ch, "is_port_in_use", return_value=True):
            report = ch.cleanhouse(ports=[5000])

        assert report["cleaned_processes"] == []
        assert [p["pid"] for p in report["skipped_system_processes"]] == [623]
        fake_proc.send_signal.assert_not_called()
        # A port still occupied by a protected process must not be advertised.
        assert 5000 not in report["available_ports"]


class TestModelFetcher:
    """Tests for Model Pre-Fetcher Configuration."""

    def test_model_targets_config(self):
        from scripts.download_models import MODEL_TARGETS
        assert isinstance(MODEL_TARGETS, list)
        assert len(MODEL_TARGETS) >= 3
        model_ids = [m[0] for m in MODEL_TARGETS]
        assert "runwayml/stable-diffusion-v1-5" in model_ids


class TestEdgeDiffusionOptimizer:
    """Tests for EdgeDiffusionOptimizer Agent."""

    def test_edge_optimizer_benchmark(self):
        from agents.edge_optimizer import EdgeDiffusionOptimizer
        opt = EdgeDiffusionOptimizer()
        res = opt.run_self_optimization_loop(max_profiles=1)
        assert res["success"] is True
        assert "optimal_configuration" in res
