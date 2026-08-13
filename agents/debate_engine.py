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
                round_feedback[persona_name] = {
                    "response": response,
                    "verdict": self._read_verdict(response),
                }
            transcript.append({"round": r + 1, "feedback": round_feedback})

        all_verdicts = [fb["verdict"] for r_data in transcript for fb in r_data["feedback"].values()]
        decided = [v for v in all_verdicts if v in ("APPROVED", "REJECTED")]

        # No debate means no consensus. Defaulting an empty debate to 1.0 reported
        # unanimous approval for a run in which nobody was asked anything.
        if not decided:
            return {
                "success": False,
                "topic": topic,
                "rounds_conducted": rounds,
                "error": "no persona returned a readable verdict; there is no consensus to report",
                "transcript": transcript,
            }

        approval_rate = decided.count("APPROVED") / len(decided)
        consensus_verdict = "APPROVED" if approval_rate >= 0.6 else "REJECTED"
        unreadable = len(all_verdicts) - len(decided)

        return {
            "success": True,
            "topic": topic,
            "rounds_conducted": rounds,
            "approval_rate": round(approval_rate * 100, 2),
            "verdicts_counted": len(decided),
            "verdicts_unreadable": unreadable,
            "final_verdict": consensus_verdict,
            "transcript": transcript,
            "consensus_summary": (
                f"{consensus_verdict} on {round(approval_rate * 100)}% approval across "
                f"{len(decided)} readable verdict(s)"
                + (f"; {unreadable} response(s) stated no verdict." if unreadable else ".")
            ),
        }

    @staticmethod
    def _read_verdict(response: str) -> str:
        """
        Read a persona's verdict from its reply.

        A substring test for "approve" was previously enough to count a verdict as
        APPROVED, so "I cannot approve this" was recorded as approval. Negations
        are checked first, and a reply that states neither is UNCLEAR rather than
        being silently counted as a rejection.
        """
        text = (response or "").lower()
        rejected_markers = ("rejected", "reject", "not approve", "cannot approve",
                            "can't approve", "do not approve", "disapprove")
        if any(marker in text for marker in rejected_markers):
            return "REJECTED"
        if "approved" in text or "approve" in text:
            return "APPROVED"
        return "UNCLEAR"
