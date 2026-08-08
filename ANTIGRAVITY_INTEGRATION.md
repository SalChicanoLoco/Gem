# Antigravity IDE & Local Model Integration Guide

This guide details how to connect **Antigravity IDE** (or any OpenAI-compatible client, VSCode extension, or Cursor setup) to **SenaAIgent Pre-AI OS** running locally on Apple Silicon.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       Antigravity IDE                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ OpenAI v1 REST Requests
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            SenaAIgent Local Gateway (Port 5000)             │
│   • POST /v1/chat/completions                               │
│   • GET  /v1/models                                         │
│   • POST /api/coder                                         │
│   • POST /api/image/diffusion                               │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Engine Layer                    │
│   • Local Ollama Gemma 2 (gemma2:2b)                       │
│   • PyTorch Metal Performance Shaders (MPS / Neural)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Starting the SenaAIgent Local Server

1. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```
2. **Start the SenaAIgent API gateway**:
   ```bash
   python start_api.py
   ```
   *The server will start on `http://localhost:5000`.*

---

## 2. Connecting Antigravity IDE / External Tools

Use the following settings in Antigravity IDE or your custom LLM provider configuration:

| Setting | Value |
|---------|-------|
| **Base URL / Endpoint** | `http://localhost:5005/v1` *(or 5000)* |
| **API Key** | `senaai-local` *(any non-empty string)* |
| **Default Model** | `gemma2:2b` |
| **Coding Engine Model** | `senaai-coder` |
| **Diffusion Model** | `senaai-diffusion` |

---

## 3. Testing the Endpoints via `curl`

### List Available Models
```bash
curl http://localhost:5000/v1/models
```

### Chat & Code Completions (OpenAI Format)
```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "messages": [
      {"role": "system", "content": "You are a kick-ass coding assistant."},
      {"role": "user", "content": "Write a Python function to check for prime numbers."}
    ]
  }'
```

### Kick-Ass Coding Agent API
```bash
# Code Synthesis
curl -X POST http://localhost:5000/api/coder \
  -H "Content-Type: application/json" \
  -d '{
    "action": "generate",
    "specification": "Create a fast LRU cache class with TTL support"
  }'

# AST Codebase Analysis
curl -X POST http://localhost:5000/api/coder \
  -H "Content-Type: application/json" \
  -d '{
    "action": "analyze",
    "code": "class DataStore:\n  def save(self, data): pass"
  }'
```

### Hardware-Accelerated Local Image Diffusion (MPS / Neural Engine)
```bash
curl -X POST http://localhost:5000/api/image/diffusion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cybernetic falcon flying over futuristic Tokyo at sunset",
    "width": 512,
    "height": 512
  }'
```

---

## 4. Hardware Acceleration Details

- **Gemma 2 LLM Inference**: Runs natively via Ollama on Apple Silicon Metal GPU unified memory.
- **PyTorch Image Diffusion**: Powered by `torch.device("mps")` (Metal Performance Shaders) using 16-bit floating point precision (`torch.float16`).
