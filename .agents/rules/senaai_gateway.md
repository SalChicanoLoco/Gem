# SenaAIgent & Local Gemma Integration Rule

Whenever assisting in this workspace, Antigravity is connected directly to the local **SenaAIgent Pre-AI OS & Gemma Engine** running on port 5005.

---

## Active Local Gateways

- **OpenAI v1 Gateway**: `http://localhost:5005/v1/chat/completions`
- **OpenAI Models List**: `http://localhost:5005/v1/models`
- **Coding Agent API**: `http://localhost:5005/api/coder`
- **PyTorch MPS Diffusion API**: `http://localhost:5005/api/image/diffusion`
- **Autonomous Evolution API**: `http://localhost:5005/api/evolution`
- **Photorealistic Control Center UI**: `http://localhost:5005/dashboard`

---

## Hardware Engine

- **LLM Engine**: Ollama `gemma2:2b` running live on Apple Silicon Metal GPU unified memory.
- **Image/Video Engine**: PyTorch `torch.device("mps")` hardware acceleration on Mac Neural Engine.
