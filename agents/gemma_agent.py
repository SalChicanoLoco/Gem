"""
GemmaAgent - Local Gemma LLM intelligence wrapper for SenaAIgent.

Provides real-time local inference (via Ollama REST API or local PyTorch/MLX engine),
structured reasoning, telemetry analysis, self-reflective workflow optimization,
and multi-agent prompt synthesis running on Apple Silicon hardware.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

# How many messages of a conversation to keep. Two per exchange, so this is 30
# turns; gemma2:2b has a small context and older turns fall out of it anyway.
MAX_HISTORY_MESSAGES = 60


class GemmaUnavailable(RuntimeError):
    """
    Raised when the local model cannot be reached.

    This exists so callers cannot mistake a failure for an answer. The previous
    behaviour was to return keyword-matched prose that read exactly like a real
    response — including invented sensor readings — which made an offline model
    indistinguishable from a working one.
    """


class ConversationStore:
    """
    In-memory chat history, keyed by conversation id.

    Ollama's chat API is stateless: multi-turn context exists only if the caller
    resends prior messages. Without this, every message was an isolated one-shot
    and the model could not refer to anything said earlier.
    """

    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES):
        self.max_messages = max_messages
        self._conversations: Dict[str, List[Dict[str, str]]] = {}

    def get(self, conversation_id: str) -> List[Dict[str, str]]:
        return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, role: str, content: str) -> None:
        history = self._conversations.setdefault(conversation_id, [])
        history.append({"role": role, "content": content})
        if len(history) > self.max_messages:
            del history[: len(history) - self.max_messages]

    def reset(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    def ids(self) -> List[str]:
        return list(self._conversations)


class GemmaAgent:
    """
    Agent responsible for local Gemma LLM reasoning, telemetry analysis,
    and self-optimizing orchestrator operations.
    """

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        ollama_host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """
        Initialize GemmaAgent.

        Args:
            model_name: Name of the Gemma model (e.g. gemma2:2b, gemma2:9b, gemma3).
            ollama_host: Base URL for local Ollama server.
            timeout: Request timeout in seconds. Generous on purpose: the old 4s
                budget expired on any answer longer than a sentence, and the
                timeout was then reported as a model response.
        """
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout
        self.conversations = ConversationStore()
        self.num_ctx: Optional[int] = None
        self._context_probed = False
        self.system_prompt = (
            "You are Gemma, an advanced local AI reasoning engine running inside SenaAIgent Pre-AI OS. "
            "Your task is to analyze telemetry, optimize task queues, synthesize creative prompts, "
            "and perform self-reflective agentic reasoning."
        )

    def context_length(self) -> Optional[int]:
        """
        The model's own context window, asked of Ollama and cached.

        Ollama does not use the full window unless told to. Measured here with
        gemma2:2b, which declares 8192: with no num_ctx the prompt was truncated
        at about 2050 tokens no matter how much was sent, and a fact placed at the
        start of a longer prompt was silently dropped. Passing num_ctx doubled the
        accepted prompt. So the window is asked for explicitly rather than left to
        a default that quietly discards the beginning of a conversation.

        $GEMMA_NUM_CTX overrides, which matters because a larger window costs
        memory per loaded model.
        """
        if self._context_probed:
            return self.num_ctx
        self._context_probed = True

        override = os.environ.get("GEMMA_NUM_CTX")
        if override:
            try:
                self.num_ctx = int(override)
                return self.num_ctx
            except ValueError:
                logger.warning("Ignoring non-numeric GEMMA_NUM_CTX=%r", override)

        try:
            resp = requests.post(f"{self.ollama_host}/api/show",
                                 json={"model": self.model_name}, timeout=5.0)
            if resp.status_code == 200:
                info = resp.json().get("model_info") or {}
                for key, value in info.items():
                    if key.endswith(".context_length") and isinstance(value, int):
                        self.num_ctx = value
                        logger.info("Model %s declares a %d-token context; requesting it explicitly.",
                                    self.model_name, value)
                        break
        except Exception as e:
            logger.warning("Could not read the context length for %s: %s", self.model_name, e)
        return self.num_ctx

    def _options(self, temperature: float) -> Dict[str, Any]:
        """Sampling options, including the full context window when known."""
        options: Dict[str, Any] = {"temperature": temperature}
        ctx = self.context_length()
        if ctx:
            options["num_ctx"] = ctx
        return options

    def is_available(self) -> bool:
        """
        Check if local Ollama Gemma service is available and auto-detect installed model.

        Returns:
            True if service is running and model is loaded, False otherwise.
        """
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=1.0)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                if models:
                    # Auto-select downloaded model if requested model is missing
                    if not any(self.model_name in m for m in models):
                        self.model_name = models[0]
                    return True
            return False
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get status metadata for Gemma agent.

        Returns:
            Dictionary with connectivity, model name, and engine information.
        """
        available = self.is_available()
        return {
            "agent_id": "gemma",
            "model": self.model_name,
            "ollama_host": self.ollama_host,
            "status": "online" if available else "fallback_mode",
            "available": available,
            "hardware_acceleration": "Apple Silicon Metal (MPS)",
        }

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Generate text response from Gemma model.

        Args:
            prompt: User or task prompt.
            system: Optional custom system prompt.
            temperature: Sampling temperature.

        Returns:
            Generated response string.

        Raises:
            GemmaUnavailable: if the local model cannot be reached or errors.
        """
        sys_msg = system or self.system_prompt
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": sys_msg,
                "stream": False,
                "options": self._options(temperature),
            }
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            raise GemmaUnavailable(
                f"Ollama returned HTTP {resp.status_code} for model {self.model_name}: {resp.text[:200]}"
            )
        except GemmaUnavailable:
            raise
        except Exception as e:
            raise GemmaUnavailable(
                f"Local Gemma inference failed against {self.ollama_host}: {e}. "
                f"Start it with `ollama serve` and ensure `{self.model_name}` is pulled."
            ) from e

    def chat(
        self,
        message: str,
        conversation_id: str = "default",
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Hold a multi-turn conversation, carrying prior turns as context.

        Uses Ollama's /api/chat with the accumulated message list, which is what
        makes a follow-up like "and why?" resolvable. The user message is recorded
        before the call and the reply after, so a failed turn does not leave a
        dangling prompt in the history.

        Raises:
            GemmaUnavailable: if the local model cannot be reached.
        """
        history = self.conversations.get(conversation_id)
        messages = [{"role": "system", "content": system or self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": self._options(temperature),
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise GemmaUnavailable(
                    f"Ollama returned HTTP {resp.status_code} for model {self.model_name}: {resp.text[:200]}"
                )
            reply = (resp.json().get("message") or {}).get("content", "").strip()
        except GemmaUnavailable:
            raise
        except Exception as e:
            raise GemmaUnavailable(
                f"Local Gemma chat failed against {self.ollama_host}: {e}. "
                f"Start it with `ollama serve` and ensure `{self.model_name}` is pulled."
            ) from e

        self.conversations.append(conversation_id, "user", message)
        self.conversations.append(conversation_id, "assistant", reply)
        return {
            "success": True,
            "model": self.model_name,
            "conversation_id": conversation_id,
            "message": message,
            "response": reply,
            "turns": len(self.conversations.get(conversation_id)) // 2,
        }

    def chat_stream(
        self,
        message: str,
        conversation_id: str = "default",
        system: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Stream a conversational reply token by token as the model produces it.

        Yields text chunks. The previous streaming endpoint generated the whole
        answer first and then released it word by word on a timer, which looked
        live but was not; nothing reached the client any sooner.

        Raises:
            GemmaUnavailable: if the local model cannot be reached.
        """
        history = self.conversations.get(conversation_id)
        messages = [{"role": "system", "content": system or self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": self._options(temperature),
                },
                timeout=self.timeout,
                stream=True,
            )
            if resp.status_code != 200:
                raise GemmaUnavailable(
                    f"Ollama returned HTTP {resp.status_code} for model {self.model_name}"
                )
        except GemmaUnavailable:
            raise
        except Exception as e:
            raise GemmaUnavailable(
                f"Local Gemma chat stream failed against {self.ollama_host}: {e}. "
                f"Start it with `ollama serve` and ensure `{self.model_name}` is pulled."
            ) from e

        parts: List[str] = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                parts.append(piece)
                yield piece
            if chunk.get("done"):
                break

        self.conversations.append(conversation_id, "user", message)
        self.conversations.append(conversation_id, "assistant", "".join(parts))

    def analyze_telemetry(self, telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deep LLM analysis on environmental/system telemetry data.

        Args:
            telemetry_data: Dictionary containing key parameters (e.g. pH, temperature).

        Returns:
            Dictionary containing LLM analytical assessment, score adjustment, and advice.
        """
        prompt = (
            f"Analyze the following telemetry parameters and provide a concise risk assessment:\n"
            f"{json.dumps(telemetry_data, indent=2)}\n\n"
            f"Respond with JSON format containing: 'assessment', 'risk_level' (LOW/MEDIUM/HIGH/CRITICAL), "
            f"and 'actionable_steps' (list of strings)."
        )

        try:
            response = self.generate(prompt, temperature=0.3)
        except GemmaUnavailable as e:
            return {"success": False, "llm_engine": "Gemma Local", "error": str(e)}

        parsed = self._extract_json(response)

        if parsed and "assessment" in parsed:
            return {
                "success": True,
                "llm_engine": "Gemma Local",
                "analysis": parsed,
                "raw_response": response,
            }

        # The model answered but not as JSON. Hand back what it said; do not
        # invent a risk level it never assessed.
        return {
            "success": True,
            "llm_engine": "Gemma Local",
            "parsed": False,
            "analysis": {
                "assessment": response[:500],
                "risk_level": None,
                "actionable_steps": [],
            },
            "raw_response": response,
        }

    def self_optimize_workflow(
        self, queue_status: Dict[str, Any], load_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform self-reflective optimization on Orchestrator task queues.

        Args:
            queue_status: Current queue metrics from OrchestratorAgent.
            load_metrics: Current load metrics.

        Returns:
            Dictionary with optimization recommendations, priority adjustments, and code patches.
        """
        prompt = (
            f"Review current system load and task queue status:\n"
            f"Queue Status: {json.dumps(queue_status)}\n"
            f"Load Metrics: {json.dumps(load_metrics)}\n\n"
            f"Provide an autonomous optimization strategy. Return JSON with keys:\n"
            f"- 'bottleneck_detected': boolean\n"
            f"- 'recommended_worker_threads': int\n"
            f"- 'priority_reassignments': dict\n"
            f"- 'rationale': string\n"
        )

        try:
            response = self.generate(prompt, temperature=0.2)
        except GemmaUnavailable as e:
            return {"success": False, "optimizer": "Gemma Autonomous Meta-Orchestrator", "error": str(e)}

        parsed = self._extract_json(response)

        if parsed and "rationale" in parsed:
            return {
                "success": True,
                "optimizer": "Gemma Autonomous Meta-Orchestrator",
                "recommendations": parsed,
            }

        # Arithmetic on the metrics we were handed is a real derivation; the
        # rationale is labelled as such rather than attributed to the model.
        return {
            "success": True,
            "optimizer": "Threshold heuristic (model reply was not JSON)",
            "parsed": False,
            "recommendations": {
                "bottleneck_detected": load_metrics.get("load_score", 0) > 80,
                "recommended_worker_threads": min(8, max(2, load_metrics.get("queue_length", 1) + 2)),
                "priority_reassignments": {},
                "rationale": "Derived from load_score and queue_length thresholds, not from the model.",
            },
            "raw_response": response,
        }

    def synthesize_aesthetic_prompt(self, base_concept: str, style: str) -> str:
        """
        Synthesize detailed multi-modal artistic prompt for Image/Art agents.

        Args:
            base_concept: Base description or theme.
            style: Artistic style keyword.

        Returns:
            Enhanced detailed prompt string.
        """
        prompt = (
            f"Expand the following concept into a vivid, highly detailed image prompt suitable for AI art generation:\n"
            f"Concept: '{base_concept}'\n"
            f"Style: '{style}'\n\n"
            f"Write only the final enhanced prompt description without preamble."
        )

        try:
            enhanced = self.generate(prompt, temperature=0.8)
        except GemmaUnavailable:
            # A template is a legitimate answer here, unlike an invented one; the
            # caller gets a usable prompt and the log says it was not synthesized.
            logger.warning("Gemma unavailable; using template prompt for %r.", base_concept)
            return f"A vibrant {style} masterpiece of {base_concept}, ultra-detailed, 8k"
        return enhanced if len(enhanced) > 10 else f"A vibrant {style} masterpiece of {base_concept}, ultra-detailed, 8k"

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to extract JSON block from model response text."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception:
            return None

