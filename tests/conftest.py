"""
Shared test configuration.

The suite must not depend on a local Ollama being up. Before the chat rewrite it
appeared not to: with the server down, requests to localhost:11434 were refused
instantly and the agent handed back canned prose, so tests passed in seconds and
looked isolated. They were not — they were exercising the fallback. With a real
model reachable the same tests make real inference calls, and the suite went from
about 29s to 155s with results that vary by machine and by whether the user
happens to have `ollama serve` running.

So GemmaAgent's network methods are stubbed by default, everywhere. A module that
genuinely needs the real implementation opts out:

    pytestmark = pytest.mark.real_gemma

The stub below is a test double, which is a different thing from what was removed
from the agent: it lives in the test tree, it is obviously fake to anyone reading
a failure, and no production path can reach it.
"""

import pytest
from unittest.mock import patch

from agents.gemma_agent import GemmaAgent

STUB_REPLY = "[stubbed model reply for tests]"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_gemma: exercise the real GemmaAgent instead of the offline stub",
    )


@pytest.fixture(autouse=True)
def stub_gemma(request):
    """
    Replace GemmaAgent's model calls with deterministic local stubs.

    Callers that parse the reply are unaffected: CoderAgent._extract_code already
    falls back to a valid ``run()`` definition when a response contains no code
    block, and the JSON-parsing wrappers already handle a non-JSON reply.
    """
    if "real_gemma" in request.keywords:
        yield
        return

    def _generate(self, prompt, system=None, temperature=0.7):
        return STUB_REPLY

    def _chat(self, message, conversation_id="default", system=None, temperature=0.7):
        self.conversations.append(conversation_id, "user", message)
        self.conversations.append(conversation_id, "assistant", STUB_REPLY)
        return {
            "success": True,
            "model": self.model_name,
            "conversation_id": conversation_id,
            "message": message,
            "response": STUB_REPLY,
            "turns": len(self.conversations.get(conversation_id)) // 2,
        }

    def _chat_stream(self, message, conversation_id="default", system=None, temperature=0.7):
        for piece in STUB_REPLY.split(" "):
            yield piece + " "
        self.conversations.append(conversation_id, "user", message)
        self.conversations.append(conversation_id, "assistant", STUB_REPLY)

    with patch.object(GemmaAgent, "generate", _generate), \
         patch.object(GemmaAgent, "chat", _chat), \
         patch.object(GemmaAgent, "chat_stream", _chat_stream), \
         patch.object(GemmaAgent, "is_available", lambda self: True):
        yield
