# Remediation Summary: karnak_gguf-causal_lm-pytorch-karnak-i1-gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[karnak_gguf/causal_lm/pytorch-Karnak-i1-GGUF-single_device-inference]

## Result
XFAIL — Karnak is a 40.7B-parameter Qwen3-MoE model; at BF16 it requires 81 GB DRAM, which exceeds the p150b single-device capacity of 34 GB.

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
The Karnak model (`mradermacher/Karnak-i1-GGUF`, `Karnak.i1-IQ2_XXS.gguf`) is based on
Qwen3-30B-A3B-Instruct-2507 (general.architecture: `qwen3moe`). The GGUF file is 11 GB on disk
(IQ2_XXS ≈ 2.06 bits/weight). Enumerating all tensor shapes from the GGUF metadata gives 40.7B
total parameters; dequantized to BF16 the model occupies 81.3 GB. The p150b device has 34 GB
DRAM. The test therefore cannot complete: CPU dequantization of the 11 GB file timed out at 600 s
in local reproduction, and transferring the resulting 81 GB model to device would OOM regardless.
This is a hardware-class ceiling, not a compiler bug.

A secondary loader issue was also found: the loader directory was missing `requirements.txt`
(`gguf>=0.10.0`), which would cause an ImportError in clean environments before reaching the
timeout. This was fixed in the tt_forge_models remediation commit.

## Fix
- `tt_forge_models` (`karnak_gguf/causal_lm/pytorch/requirements.txt`): added `gguf>=0.10.0`
- `tt-xla` (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`): added
  `KNOWN_FAILURE_XFAIL` entry with hardware-capacity reason.

No compiler-stack changes were made.

## Verification
- pytest exit: TIMEOUT (model dequantization exceeded 600 s on CPU; device load would OOM)
- Hardware:    not-run (timed out before reaching device)
- Duration:    >600s (model loading only)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/karnak_gguf/causal_lm/pytorch/requirements.txt` (added)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0aec8350c72b8219eb0c882da3bb61758fac87b6 |
| tt-forge-models | ca88897b1a2b9bdf7ebedcf2ed779c6f6e858767 |
