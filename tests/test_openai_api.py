"""
Tests for OpenAI-compatible gateway endpoints in api/app.py.
"""

import json
import unittest
from api.app import create_app


class TestOpenAIGateway(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_list_models(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("object"), "list")
        self.assertTrue(len(data.get("data", [])) >= 2)

    def test_chat_completions(self):
        payload = {
            "model": "gemma2:2b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ]
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("object"), "chat.completion")
        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("content", data["choices"][0]["message"])

    def test_coder_api_endpoints(self):
        # Test code analysis
        payload = {
            "action": "analyze",
            "code": "def foo(x):\n    return x * 2"
        }
        response = self.client.post("/api/coder", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))
        self.assertIn("foo", data.get("functions", []))

    def test_diffusion_api_endpoint(self):
        payload = {
            "prompt": "Test prompt",
            "width": 128,
            "height": 128
        }
        response = self.client.post("/api/image/diffusion", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get("success"))


if __name__ == "__main__":
    unittest.main()
