# Remediation Summary: llama_3_1_70b_bnb_4bit-causal_lm-pytorch-llama_3_1_70b_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_1_70b_bnb_4bit/causal_lm/pytorch-llama_3_1_70b_bnb_4bit-single_device-inference]

## Result
XFAIL — LLaMA 3.1 70B in BF16 (~140 GB) exceeds single-device TT DRAM (~96 GB on p150b); BNB 4-bit format is CUDA-specific and cannot run natively on TT hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
2026-04-23 22:02:40.546 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```

Raised in tt-metal at `assert.hpp:104` when the TT runtime attempted to use an ethernet core connected to a remote MMIO device during model execution on silicon. In the local reproduce environment (bitsandbytes not installed), the test fails earlier with:
```
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

## Root cause
Two overlapping issues, both rooted in hardware capacity:

1. **Missing `bitsandbytes` requirement**: The loader at `llama_3_1_70b_bnb_4bit/causal_lm/pytorch/loader.py` had no `requirements.txt`, so `bitsandbytes` was not guaranteed to be installed in the test venv. Without it, `AutoModelForCausalLM.from_pretrained` raises `ImportError` before reaching the device.

2. **Hardware capacity ceiling**: `unsloth/Meta-Llama-3.1-70B-bnb-4bit` is a 70-billion-parameter model stored in BitsAndBytes 4-bit format (~35 GB on disk). After dequantization to BF16 the weights occupy ~140 GB — far beyond the ~96 GB DRAM available on a single TT p150b device. The BNB 4-bit format (`Params4bit`) is CUDA-specific and cannot be executed natively on TT hardware; the model must be dequantized to BF16 before use, which makes it too large.

The `TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device` error fires in tt-metal when the runtime tries to use ethernet fabric during an operation on an oversized model — the runtime attempts to route through eth cores that connect to a remote MMIO device, which is unsupported in single-device mode.

## Fix
- **`tt_forge_models` (loader)**: Added `llama_3_1_70b_bnb_4bit/causal_lm/pytorch/requirements.txt` with `bitsandbytes>=0.46.1` so the dependency is declared and the import error is surfaced clearly.
- **`tt-xla` (test config)**: Added `KNOWN_FAILURE_XFAIL` entry for `llama_3_1_70b_bnb_4bit/causal_lm/pytorch-llama_3_1_70b_bnb_4bit-single_device-inference` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. No compiler fix is possible; the model exceeds single-device hardware capacity.

## Verification
- pytest exit: XFAIL (1 xfailed, 185.23s) — confirmed locally with ImportError path; original silicon error is hardware capacity
- Hardware:    not-run (model exceeds device DRAM; eth-core error on silicon is secondary to OOM)
- Duration:    185.23s (local, ImportError path)
- Tier A attempts: N/A

## Files changed
- `llama_3_1_70b_bnb_4bit/causal_lm/pytorch/requirements.txt` (new, in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 46c0d2160f1afe95dcf9f6dd0c9584d4686eb07f |
| tt-forge-models | 24d7f51bbb03cc1c1dc06729259a49623d20f273 |
