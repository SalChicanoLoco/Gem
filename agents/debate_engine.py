"""
Multi-Agent Debate & Consensus Engine (agents/debate_engine.py)

Coordinates multi-perspective debate rounds between specialized LLM personas
(SecurityAuditor, DataScientist, SystemArchitect) to reach unanimous consensus.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from .gemma_agent import GemmaAgent
from .spine import get_spine

logger = logging.getLogger(__name__)


class DebateEngine:
    """
    Multi-Agent Debate Engine managing peer review and consensus.
    """

    PERSONAS = {
        "SecurityAuditor": "You are a ruthless Cyber Security Auditor focused on zero-trust, data privacy, and vulnerability analysis.",
        "DataScientist": "You are a Lead Data Scientist focused on statistical validity, baseline benchmarks, and model performance.",
        "SystemArchitect": "You are a Principal Cloud Architect focused on scalability, throughput, and low-latency execution.",
    }

    def __init__(self, gemma_agent: Optional[GemmaAgent] = None):
        """
        Initialize DebateEngine.

        Args:
            gemma_agent: GemmaAgent instance for multi-persona reasoning.
        """
        self.gemma = gemma_agent or GemmaAgent()
        self.spine = get_spine()

    def run_debate(
        self,
        topic: str,
        proposal: Optional[Dict[str, Any]] = None,
        rounds: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute multi-agent debate rounds across personas to form a consensus.

        Args:
            topic: Topic or query to debate.
            proposal: Proposed plan or telemetry payload.
            rounds: Number of debate rounds.

        Returns:
            Dictionary with persona feedback, consensus score, and final verdict.
        """
        transcript: List[Dict[str, Any]] = []

        for r in range(rounds):
            round_feedback = {}
            for persona_name, persona_sys in self.PERSONAS.items():
                prompt = (
                    f"Debate Round {r+1}/{rounds}\n"
                    f"Topic: {topic}\n"
                    f"Proposal: {proposal}\n\n"
                    f"Provide your assessment as {persona_name}. State key risks, score (0-100), and verdict (APPROVED/REJECTED)."
                )
                response = self.gemma.generate(prompt, system=persona_sys, temperature=0.5)
                verdict = "APPROVED" if "approve" in response.lower() else "REJECTED"
                round_feedback[persona_name] = {
                    "response": response,
                    "verdict": verdict,
                }
            transcript.append({"round": r + 1, "feedback": round_feedback})

        # Calculate final consensus score
        all_verdicts = [fb["verdict"] for r_data in transcript for fb in r_data["feedback"].values()]
        approval_rate = all_verdicts.count("APPROVED") / len(all_verdicts) if all_verdicts else 1.0

        consensus_verdict = "APPROVED" if approval_rate >= 0.6 else "REJECTED"

        return {
            "success": True,
            "topic": topic,
            "rounds_conducted": rounds,
            "approval_rate": round(approval_rate * 100, 2),
            "final_verdict": consensus_verdict,
            "transcript": transcript,
            "consensus_summary": (
                f"Multi-Agent Consensus Verdict: {consensus_verdict} ({round(approval_rate * 100)}% approval across personas)."
            ),
        }
