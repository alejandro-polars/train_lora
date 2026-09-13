# Local Fine-Tuning & GGUF Quantization Pipeline for Hardware-Constrained Environments

An end-to-end Machine Learning pipeline designed to fine-tune Large Language Models (LLMs) using **Unsloth (QLoRA)** and compile them into standalone, deployment-ready **GGUF binaries** for local, low-latency execution (Ollama / llama.cpp).

This architecture focuses on maximizing compute efficiency and minimizing VRAM footprint during both training and local deployment.

## 🚀 Technical Highlights

- **Memory Optimization:** Leveraged Unsloth's `FastLanguageModel` with strict 4-bit base model quantization (`load_in_4bit=True`) and custom sequence length gating (2048 tokens) to prevent out-of-memory (OOM) errors.
- **Parameter-Efficient Fine-Tuning (PEFT):** Implemented targeted LoRA matrix adaptation across core attention and MLP modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with low rank ($r=16$) to optimize tensor updates.
- **Structured Chat Template Enforcement:** Automated role mapping (`system`, `user`, `assistant`) utilizing native Llama-3 structural token templates to ensure conversational formatting integrity.
- **Edge Deployment Compilation:** Native compilation of post-training adapters directly into balanced 4-bit GGUF format (`q4_k_m`) for highly optimized local CPU/GPU tensor execution loops.

## 🛠️ Environment Setup & Installation

This project utilizes the `uv` package manager for high-speed dependency resolution. 

```bash
# Install the fast package installer
pip install uv

# Install Unsloth with auto torch backend detection
uv pip install unsloth --torch-backend=auto

# Install auxiliary training dependencies without redundant overheads
pip install --no-deps xformers trl peft loralib datasets transformers accelerate
```

## 📊 Pipeline Structure

1. **Architecture Ingestion:** Loads `Llama-3-8B-Instruct` under extreme 4-bit quantization constraints.
2. **PEFT Layer Injection:** Wraps the base model with lightweight LoRA adapters to lock 99% of original parameters.
3. **Data Sanitization:** Pre-tokenizes local structured dialogs (`dataset.jsonl`) into tensor formats using native role-template schemas.
4. **Supervised Fine-Tuning (SFT):** Executes the training loop using custom gradient accumulation to simulate large batch processing with tiny memory footprint.
5. **Binary Quantization:** Exports the fine-tuned parameters directly into an operational `.gguf` file ready for production.

## 📜 Disclaimer
This repository acts as an experimental sandbox for resource-constrained fine-tuning workflows, hardware stress-testing, and local alignment validation.
