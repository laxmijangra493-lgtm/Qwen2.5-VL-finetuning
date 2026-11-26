<img width="736" height="736" alt="image" src="https://github.com/user-attachments/assets/b5583bdf-2b62-4db6-bea2-48c27b07ec64" />


# Handwritten Document Transcription 
with Qwen2.5-VL-3B (4-Bit QLoRA) | trains on Google colab's T4 GPU

Fine-tuning **Qwen2.5-VL-3B-Instruct** using **4-bit QLoRA** on the [`radhakrishnadeshpande/gnhk-pagelevel`](https://huggingface.co/datasets/radhakrishnadeshpande/gnhk-pagelevel) dataset to transcribe handwritten notebook pages into clean, structured text on a single 15 GB NVIDIA T4 GPU (Google Colab).

---

## Overview

Vision-Language Models (VLMs) can read complex page layouts, headings, and lists without relying on brittle multi-stage OCR pipelines. However, fine-tuning multimodal models on free-tier hardware (15 GB GPU VRAM and ~12.7 GB CPU RAM) often triggers out-of-memory crashes during image decoding and batch collation.

This repository provides a memory-safe training and inference workflow designed specifically for resource-constrained environments:
- **4-Bit NF4 Quantization (`bitsandbytes`)** with `float16` compute tailored for T4 GPUs.
- **LoRA Adapters (`peft`)** targeting attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Lazy Dataset Loading & Dynamic Image Resizing** (`max_pixels = 512 * 28 * 28`) to prevent uncompressed `PIL.Image` RAM spikes.
- **Assistant-Only Loss Masking** so gradients update exclusively on target transcription tokens rather than prompt or image tokens.

---
