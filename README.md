# AEGIS v2
 
**A fine-tuned LLM for embedded systems, drone firmware, and robotics code generation — verified against real compilers, not just heuristics.**
 
> ⚙️ **Status: Training in progress.** Benchmarks and model weights will be added once the current run completes.
 
[![License](https://img.shields.io/badge/license-PolyForm--Noncommercial--1.0.0-blue)](LICENSE)
[![Base Model](https://img.shields.io/badge/base-DeepSeek--R1--Distill--Qwen--32B-blue)]()
[![Method](https://img.shields.io/badge/finetuning-QLoRA%20%2B%20DPO-orange)]()
 
---
 
## Overview
 
AEGIS v2 is the successor to [AEGIS v1](https://github.com/xerxes-28/AEGIS), scaling up both dataset size and verification rigor. Where v1 proved the concept on a 7B model and ~5K samples, v2 targets production-grade reliability for real embedded targets.
 
The core idea: most code-generation models are trained to produce code that *looks* right. AEGIS is trained to produce code that **compiles** — its preference data is built directly from compiler pass/fail signals, not synthetic judgments.
 
## Key Features
 
- **Base model:** DeepSeek-R1-Distill-Qwen-32B (4-bit quantized)
- **Fine-tuning:** QLoRA (rank 64, alpha 128) → DPO alignment
- **Compiler-in-the-loop verification:** `arm-none-eabi-gcc`, `avr-gcc`, `xtensa-esp32-elf-gcc`
- **DPO negative pairs sourced from real compiler failures** — not synthetic rejections
- **Hybrid RAG pipeline:** Qdrant + BGE-M3 + BM25, for grounding generations in real datasheets/docs
- **Dataset:** 340K–420K+ samples across Arduino, AVR, ESP32, STM32, robotics/control domains
- **Domains covered:** embedded C/C++, drone firmware, robotics control code
## Architecture
 
```
                    ┌─────────────────────┐
                    │   User Query /       │
                    │   Code Request       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Hybrid RAG Layer     │
                    │  Qdrant + BGE-M3      │
                    │  + BM25               │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  AEGIS v2 (QLoRA)     │
                    │  DeepSeek-R1-Distill  │
                    │  -Qwen-32B            │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Compiler Verification│
                    │  arm-gcc / avr-gcc /  │
                    │  xtensa-esp32-gcc     │
                    └──────────┬───────────┘
                               │
                 Pass ─────────┼───────── Fail
                    │                       │
              Return code           Feed back as DPO
                                     negative pair
```
 
## Training Details
 
| Component | Detail |
|---|---|
| Base model | DeepSeek-R1-Distill-Qwen-32B-bnb-4bit |
| Fine-tuning method | QLoRA (r=64, α=128) + DPO |
| Dataset size | 340K–420K+ samples |
| Negative pair source | Real compiler failures |
| Compute | Colab Pro, 80GB A100 |
| RAG stack | Qdrant + BGE-M3 + BM25 |
 
## Results
 
_Benchmarks pending — will be published once the current training run completes._
 
## Installation
 
```bash
git clone https://github.com/xerxes-28/AEGIS-v2.git
cd AEGIS-v2
pip install -r requirements.txt
```
 
## Usage
 
```python
# Placeholder — update once inference script is finalized
from aegis import AegisModel
 
model = AegisModel.load("xerxes-28/aegis-v2")
response = model.generate("Write ESP32 firmware to read a BME280 sensor over I2C")
print(response)
```
 
## Relationship to AEGIS v1
 
AEGIS v1 (QLoRA fine-tuned Qwen2.5-Coder-7B on ~5K Alpaca-format samples) proved the approach on a smaller scale across Arduino, ESP32, AVR, and STM32. AEGIS v2 expands this substantially:
 
| | v1 | v2 |
|---|---|---|
| Base model | Qwen2.5-Coder-7B | DeepSeek-R1-Distill-Qwen-32B |
| Dataset size | ~5K samples | 340K–420K+ samples |
| Alignment | QLoRA only | QLoRA + DPO |
| Verification | — | Compiler-in-the-loop |
| RAG | — | Qdrant + BGE-M3 + BM25 |
 
## Team
 
AEGIS v2 is developed by a six-person team:
 
- **Charles (xerxes-28)** — Lead architect
- **Yoganand (A-47)** — Co-lead, daily QC
- **Gopi** — RAG / math / hardware
- **Jaswanth** — Scraping pipeline
- **Dinesh** — Robotics/control datasets
- **Krishna** — Electronic components dataset used in RAG

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0 — free to use, modify, and share for any noncommercial purpose. Commercial use is not permitted without a separate agreement.

Dataset construction followed a strict licensing policy: MIT/BSD/Apache-licensed sources were used directly for training; GPLv3 sources were used for pattern extraction only, not direct inclusion.

## Citation
 
```bibtex
@misc{aegis2026,
  title={AEGIS v2: Compiler-Verified Code Generation for Embedded Systems and Robotics},
  author={Charles and Yoganand and GP and RS and AD and Krishna},
  year={2026},
  howpublished={\url{https://github.com/xerxes-28/AEGIS-v2}}
}
```
 
