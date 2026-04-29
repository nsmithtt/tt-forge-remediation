# Remediation Summary: bartowski_kwaipilot_kat_dev_72b_exp_gguf-causal_lm-pytorch-KAT_Dev_72B_Exp_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_kwaipilot_kat_dev_72b_exp_gguf/causal_lm/pytorch-KAT_Dev_72B_Exp_GGUF-single_device-inference]

## Result
XFAIL — 72B model dequantizes from Q4_K_M GGUF to BF16 at load time (~144 GB), exceeding single-device DRAM (p150b: 24 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-missing-requirements-and-72b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

## Root cause
Two coexisting issues:

1. **Loader bug**: `bartowski_kwaipilot_kat_dev_72b_exp_gguf/causal_lm/pytorch/` had no `requirements.txt`. `AutoModelForCausalLM.from_pretrained` with `gguf_file=` requires the `gguf` Python package (≥0.10.0). Without it, transformers raises `ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.` The `RequirementsManager` only installs per-model requirements when a `requirements.txt` is present; with no file, gguf is never installed.

2. **Hardware-class ceiling**: Even after fixing the gguf ImportError, `Kwaipilot_KAT-Dev-72B-Exp-Q4_K_M.gguf` stores ~72B parameters in Q4_K_M quantization (~40 GB on disk). HuggingFace's GGUF loader dequantizes all weights to BF16 at load time: 72B params × 2 bytes = ~144 GB. This far exceeds all single TT device DRAM (n150: 12 GB, p150b: 24 GB). The model is a genuine hardware capacity ceiling, not a compiler bug.

## Fix
1. Created `bartowski_kwaipilot_kat_dev_72b_exp_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0` in `tt-forge-models` on branch `remediation/bartowski_kwaipilot_kat_dev_72b_exp_gguf-causal_lm-pytorch-KAT_Dev_72B_Exp_GGUF-single_device-inference`.

2. Added `KNOWN_FAILURE_XFAIL` entry in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` with reason explaining the hardware capacity ceiling (144 GB BF16 >> 24 GB p150b DRAM).

## Verification
- pytest exit: not-run
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `bartowski_kwaipilot_kat_dev_72b_exp_gguf/causal_lm/pytorch/requirements.txt` (new, in tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry added, in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4b2d3c07e3c4f3555fb8e5ab488a6b3e86810e41 |
| tt-forge-models | bff23c42ac5d557d4138ce0c198e227ad1df95b3 |
