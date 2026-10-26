# Security & Cryptographic Artifact Policy

## Supported Versions

| Version | Build Target | Status | SHA-256 Checkpoint Verification |
| :--- | :--- | :---: | :---: |
| **Rain v1.0.x** | `x86_64` / `CUDA SM75+` | Supported | Enforced |
| `< 1.0.0` | Pilot Pre-Releases | Unsupported | — |

## Memory Safety & Determinism Guarantees
- **Zero C/C++ Runtime Wrappers:** Rain v1.0 does not link against `libtorch`, Python C-extensions, or `pickle`-based weight deserializers (`torch.load`), eliminating arbitrary code execution vectors during checkpoint loading.
- **SafeTensors & SHA-256 Verification:** All model checkpoints use zero-copy header-validated `.safetensors` accompanied by a mandatory `metadata.json` SHA-256 digest checked prior to tensor allocation.

## Reporting a Vulnerability
Please report security issues or deterministic verification bypasses directly to **Kinbo Technologies Engineering** via private security advisory on GitHub. Do not open public issues for exploitable vulnerabilities.
