# LLM from Scratch

This repository provides an **educational walkthrough** of building a **GPT-style language model from scratch** using PyTorch.

The project focuses on understanding **model architecture, training dynamics, and text generation**, starting from raw text data and progressing toward a tiny GPT-2–like model that is pretrained and fine-tuned on real datasets.

---

## Contents
The core of the repository consists of **step-by-step notebooks**, designed to be read in order:

- **00_tiny_gpt_from_scratch.ipynb** - Minimal char-level GPT-like model ("Tiny Shakespeare" dataset) to build intuition

- **01_working_with_text_data.ipynb** - Text processing, tokenization (tiktoken), batching, and dataset setup

- **02_attention_mechanisms.ipynb** - Attention mechanisms in practice (Self-attention, Causal-attention, Multi-head attention)

- **03_gpt_model_from_scratch.ipynb** - Building a decoder-only Transformer (GPT-2 architecture)

- **04_pretraining_on_unlabeled_data.ipynb** - Language-model pretraining on raw text ("The Verdict" dataset)

- **05_finetuning_for_classification.ipynb** - Task-specific fine-tuning using the pretrained model

- **06_finetuning_to_follow_instructions.ipynb** - Instruction-style fine-tuning using the pretrained model

Supporting files:
- `utils.py` – shared model and training components used across notebooks  
- `gpt_download.py` – download and load pretrained GPT-2 weights  
- `ollama_evaluate.py` – lightweight evaluation using Ollama  
- `app.py` – simple Chainlit interface for interactive generation  
- `datasets/` – datasets used throughout the project  
- `gpt2_paper.pdf` – original GPT-2 paper for reference  

---

## Main Takeaways
By working through this repository, you will gain an understanding of:

- Decoder-only Transformer design
- Self-attention and causal masking in practice
- Tokenization trade-offs (character-level vs. Byte Pair Encoding)
- How language models are trained end-to-end
- Pretraining vs fine-tuning workflows
- Instruction-following behavior
- Text generation and sampling strategies

The emphasis is on **clarity and intuition**, not scale or performance.

---

## Practical Considerations
- Models are intentionally **small** and computationally inexpensive
- Training is suitable for **CPU or single-GPU setups**
- Code prioritizes **readability over optimization**
- Utilities and interfaces are included for exploration, not deployment
- Future extensions may include script-based training or inference

---

## References / Inspired By
- *Attention Is All You Need* 
- *Language Models are Unsupervised Multitask Learners* (GPT-2)  
- Andrej Karpathy’s **minGPT** and **nanoGPT**  
- Sebastian Raschka's book *Build a Large Language Model (from Scratch)*

---

## Educational Use Only
This repository is intended **solely for learning and experimentation**.

It is not designed, tested, or optimized for real-world or production deployment.

---

