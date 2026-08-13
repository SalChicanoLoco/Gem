"""
AutoHealer - Autonomous Task Error Repair Engine (agents/auto_healer.py)

Intercepts task failures and runtime exceptions, diagnoses root cause via Gemma,
generates corrected parameters/payloads, and re-executes automatically.
"""

import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional
from .gemma_agent import GemmaAgent, GemmaUnavailable
from .spine import get_spine

logger = logging.getLogger(__name__)


class AutoHealer:
    """
    Autonomous error interceptor and task self-repair engine.
    """

    def __init__(self, gemma_agent: Optional[GemmaAgent] = None):
        """
        Initialize AutoHealer.

        Args:
            gemma_agent: GemmaAgent instance for root cause diagnosis.
        """
        self.gemma = gemma_agent or GemmaAgent()
        self.spine = get_spine()
        self._repair_history: list[dict[str, Any]] = []

    def execute_with_repair(
        self,
        task_func: Callable[..., Any],
        payload: Dict[str, Any],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute a function with automatic LLM diagnosis and repair retries.

        Args:
            task_func: Function to execute.
            payload: Input dictionary argument.
            max_retries: Maximum repair attempt count.

        Returns:
            Dictionary with execution result and repair logs.
        """
        current_payload = dict(payload)

        for attempt in range(max_retries + 1):
            try:
                result = task_func(current_payload)
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "repaired": attempt > 0,
                    "result": result,
                    "final_payload": current_payload,
                }
            except Exception as e:
                err_msg = str(e)
                tb = traceback.format_exc()
                logger.warning(f"AutoHealer caught task exception (attempt {attempt+1}/{max_retries+1}): {err_msg}")

                if attempt == max_retries:
                    return {
                        "success": False,
                        "error": err_msg,
                        "traceback": tb,
                        "attempts": attempt + 1,
                    }

                # Diagnose and repair payload via Gemma
                repaired_payload = self.diagnose_and_repair(err_msg, tb, current_payload)
                current_payload = repaired_payload
                self._repair_history.append({
                    "timestamp": time.time(),
                    "attempt": attempt + 1,
                    "error": err_msg,
                    "repaired_payload": repaired_payload,
                })

        return {"success": False, "error": "Max retries exceeded"}

    def diagnose_and_repair(
        self, error_msg: str, error_traceback: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Gemma to analyze exception traceback and fix payload values.
        """
        prompt = (
            f"An error occurred during task execution:\n"
            f"Error: {error_msg}\n"
            f"Payload: {payload}\n\n"
            f"Analyze the error and return a corrected payload in JSON format."
        )

        repaired = dict(payload)
        parsed = None
        try:
            response = self.gemma.generate(prompt, temperature=0.1)
            parsed = GemmaAgent._extract_json(response)
        except GemmaUnavailable as e:
            # This runs inside the caller's except block. Letting the model's
            # unavailability propagate would replace the original task error with
            # an unrelated one, so fall through to the heuristic repair below.
            logger.warning("No model available to diagnose the failure (%s); applying heuristics only.", e)

        if parsed and isinstance(parsed, dict):
            for k, v in parsed.items():
                if k in payload or not isinstance(v, (dict, list)):
                    repaired[k] = v

        # Apply heuristic type conversion for string numbers across all keys
        for k, v in list(repaired.items()):
            if isinstance(v, str):
                cleaned = v.strip()
                if "." in cleaned and cleaned.replace(".", "", 1).isdigit():
                    try:
                        repaired[k] = float(cleaned)
                    except ValueError:
                        pass
                elif cleaned.isdigit():
                    try:
                        repaired[k] = int(cleaned)
                    except ValueError:
                        pass
        return repaired
