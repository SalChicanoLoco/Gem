"""
AutonomousEvolutionEngine - Continuous Self-Improvement & Evolution Loop (agents/evolution_engine.py)

Ties together AutoHealer, Gemma Meta-Orchestrator, CoderAgent tool synthesis, and DebateEngine
into an autonomous continuous self-improvement loop for SenaAIgent.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from .boundary_guard import get_boundary_guard
from .gemma_agent import GemmaAgent
from .coder_agent import CoderAgent
from .auto_healer import AutoHealer
from .debate_engine import DebateEngine
from .spine import get_spine

logger = logging.getLogger(__name__)


class AutonomousEvolutionEngine:
    """
    Continuous Self-Improvement and Evolution Engine for SenaAIgent Pre-AI OS.
    """

    def __init__(
        self,
        gemma_agent: Optional[GemmaAgent] = None,
        coder_agent: Optional[CoderAgent] = None,
        auto_healer: Optional[AutoHealer] = None,
        debate_engine: Optional[DebateEngine] = None,
    ):
        """Initialize AutonomousEvolutionEngine."""
        self.gemma = gemma_agent or GemmaAgent()
        self.coder = coder_agent or CoderAgent(gemma_agent=self.gemma)
        self.healer = auto_healer or AutoHealer(gemma_agent=self.gemma)
        self.debate = debate_engine or DebateEngine(gemma_agent=self.gemma)
        self.guard = get_boundary_guard()
        self.spine = get_spine()
        self.evolution_history: List[Dict[str, Any]] = []

    def run_evolution_cycle(
        self,
        target_goal: str = "Optimize task throughput and repair code bottlenecks",
        measurable_gain: Optional[str] = None,
        validation_criterion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a full multi-stage autonomous self-improvement cycle.

        Args:
            target_goal: Strategic goal for the self-improvement cycle.
            measurable_gain: Measurable behavioral gain the synthesized tool should
                provide. Required by guard_010 before any tool is written.
            validation_criterion: How to tell whether the synthesized tool works.

        Returns:
            Dictionary detailing evolution actions and strategy patches. Stages
            that were blocked by a boundary guard are reported as blocked.
        """
        cycle_id = f"evo_{int(time.time())}"
        logger.info(f"Starting Autonomous Evolution Cycle {cycle_id} for goal: {target_goal}")

        # Stage 1: Self-Diagnostic & Optimization Strategy
        queue_status = {"pending": 2, "processing": 1, "completed": len(self.evolution_history)}
        load_metrics = {"load_score": 35.0, "queue_length": 2}
        opt_result = self.gemma.self_optimize_workflow(queue_status, load_metrics)

        # Stage 2: Tool Synthesis & Capability Extension.
        # The gain and validation criterion are required by guard_010; without
        # them the BoundaryGuard blocks the write before it reaches extensions/.
        synthesis_result = self.coder.synthesize_tool(
            task_description=f"Auto-evolved helper for: {target_goal}",
            tool_name=f"evo_tool_{cycle_id[-6:]}",
            measurable_gain=measurable_gain,
            validation_criterion=validation_criterion,
        )

        # Stage 3: Multi-Agent Dialectic Critique & Strategy Alignment
        debate_result = self.debate.run_debate(
            topic=f"Optimal execution strategy for {target_goal}",
            proposal={"goal": target_goal, "optimization_strategy": opt_result},
            rounds=2,
        )

        # Stage 4: Auto-Healer Health Check Verification
        health_check_payload = {"val": "100.0", "target": target_goal}
        heal_result = self.healer.execute_with_repair(
            task_func=lambda d: float(d["val"]) * 2.5,
            payload=health_check_payload,
        )

        record = {
            "cycle_id": cycle_id,
            "timestamp": time.time(),
            "target_goal": target_goal,
            "stages": {
                "gemma_optimization": opt_result.get("recommendations"),
                "tool_synthesized": synthesis_result.get("filename"),
                "tool_blocked_by_guard": synthesis_result.get("triggered_guards"),
                "debate_consensus": debate_result.get("consensus_verdict"),
                "auto_healer_status": heal_result.get("success"),
            },
            "status": "evolved" if synthesis_result.get("success") else "blocked",
        }

        # guard_005: never report a gain this cycle did not measure.
        claim_check = self.guard.preflight_result_claim(record, measured_keys=[])
        if not claim_check.allowed:
            logger.warning("Evolution report claim blocked: %s", claim_check.decision_rationale)
            record["unverified_claims"] = claim_check.triggered_guards

        self.evolution_history.append(record)
        return {
            "success": True,
            "cycle_id": cycle_id,
            "evolution_report": record,
            "total_cycles_executed": len(self.evolution_history),
        }
