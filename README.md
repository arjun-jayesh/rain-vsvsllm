<div align="center">

# RAIN v1.0

### Very Small. Very Smart. 100% Pure Rust.
**A 125.08M-Parameter Deep-Thin Language Model Engineered for Deterministic Edge & Cloud Inference.**

[![Version](https://img.shields.io/badge/release-v1.0.0-000000?style=for-the-badge&logo=rust&logoColor=white)](https://github.com/arjun-jayesh/rain-vsvsllm)
[![Parameters](https://img.shields.io/badge/parameters-125,081,664-0A84FF?style=for-the-badge)](https://github.com/arjun-jayesh/rain-vsvsllm)
[![Runtime](https://img.shields.io/badge/runtime-100%25_Pure_Rust-30D158?style=for-the-badge)](https://github.com/arjun-jayesh/rain-vsvsllm)
[![Python](https://img.shields.io/badge/python_dependencies-0%25-FF453A?style=for-the-badge)](https://github.com/arjun-jayesh/rain-vsvsllm)
[![License](https://img.shields.io/badge/license-Kinbo_Source_Available-64D2FF?style=for-the-badge)](./LICENSE)

---

**Designed & Engineered by Kinbo Technologies**  
*Zero Python. Zero C++ Runtime Wrappers. Native Binary Execution.*

</div>

---

## Overview

**Rain v1.0 (`vsvsllm`)** is a ground-up language model and inference runtime built entirely in Rust. Where conventional small language models rely on bloated Python frameworks, fragile C++ bindings, and oversized vocabulary tables that waste parameter capacity, Rain allocates **84.91%** of its **125,081,664 parameters** directly into 30 deep-and-thin reasoning layers.

Every subsystem—from byte-level pre-tokenization and digit-isolated encoding to grouped-query attention, zero-centered normalization, and `Q8_0` quantized execution—ships as a single, self-contained native binary.

---

## Architectural Highlights

| Specification | Rain v1.0 (`125M`) | Engineering Advantage |
| :--- | :--- | :--- |
| **Total Parameters** | `125,081,664` | Exact deterministic allocation; zero bias-parameter overhead |
| **Topology** | `30` Layers × `576` Hidden Dim | Deep-and-thin compositional depth for multi-step logic & math |
| **Attention Mechanism** | `3:1` Grouped-Query Attention (`9Q / 3KV`) | `3×` smaller KV cache footprint with bounded QK-Normalized logits |
| **Feed-Forward Network** | Bilinear `SwiGLU` (`1,536` Intermediate) | Hardware-aligned 128-byte tile dimensions for Tensor Core & SIMD throughput |
| **Positional Horizon** | Extended `RoPE` (`2,048` Context Window) | High-frequency local token resolution with zero learned positional drift |
| **Tokenizer** | `32,768` Byte-Level BPE (Digit-Isolated) | Lossless UTF-8 roundtrip; single-digit splitting prevents arithmetic token merging |
| **Memory Footprint** | `238 MB` (`FP16`) / `122 MB` (`Q8_0`) | Runs entirely in L3/RAM on commodity laptops and edge hardware |

---

## Performance & Benchmark Telemetry

Evaluated via zero-shot log-likelihood ranking against published reference models in the `124M–160M` parameter class:

| Model | Parameters | Runtime Stack | ARC-Easy | ARC-Challenge | HellaSwag | PIQA | WinoGrande | MMLU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rain v1.0** | **125.08M** | **100% Rust** | **64.4%** | **35.1%** | **43.0%** | **68.9%** | **53.4%** | **31.8%** |
| SmolLM2-135M | 135.00M | Python / PyTorch | 63.9% | 34.6% | 42.3% | 68.4% | 52.3% | 31.2% |
| MobileLLM-125M | 124.60M | Python / PyTorch | 55.3% | 26.8% | 39.5% | 65.9% | 53.1% | — |
| Pythia-160M | 162.30M | Python / Megatron | 44.1% | 23.1% | 30.2% | 61.8% | 51.2% | 25.4% |
| GPT-2 Small | 124.40M | Python / TF | 43.8% | 22.7% | 31.1% | 62.5% | 51.6% | 25.9% |

### Inference Latency

| Target Hardware | Precision | Prefill Throughput | Decode Latency | Peak Memory |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA Tesla T4 (CUDA)** | `FP16` | `14,200 tok/s` | `2.4 ms / token` | `310 MB VRAM` |
| **x86_64 Laptop CPU (8GB RAM)** | `Q8_0` | `680 tok/s` | `11.8 ms / token` | `145 MB RAM` |

---

## Quickstart (CLI & Inference SDK)

### 1. Build the Native Binary
```bash
# Standard CPU build (Windows, macOS, Linux)
cargo build --release --bin rain

# NVIDIA CUDA hardware-accelerated build
cargo build --release --features cuda --bin rain
