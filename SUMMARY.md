# Remediation Summary: dinerburger_qwen_3_5_27b_gguf-causal_lm-pytorch-27B_Q8_0_XXL-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dinerburger_qwen_3_5_27b_gguf/causal_lm/pytorch-27B_Q8_0_XXL-single_device-inference]

## Result
XFAIL — 27B Q8_0 GGUF (~35 GB on disk, ~54 GB bfloat16) exceeds n150 12 GB DRAM; hardware-class failure

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-27b-model-exceeds-n150-dram

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
Two layered issues prevent this test from passing:

1. **Primary — hardware capacity:** The `dinerburger/Qwen3.5-27B-GGUF` `Qwen3.5-27B.Q8_0_XXL.gguf` file is 35 GB on disk. The transformers GGUF loader dequantizes weights to bfloat16, producing a ~54 GB in-memory model. The n150 device has 12 GB DRAM. The model cannot fit on the device by a factor of ~4.5×. The CI timeout was triggered by the test attempting to download the uncached 35 GB file before any compilation occurred.

2. **Secondary — loader bug:** The GGUF file identifies its architecture as `qwen35`, but `transformers.integrations.ggml.GGUF_CONFIG_MAPPING` has no `qwen35` entry (only `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe`). This would cause `AutoConfig.from_pretrained` to raise `"GGUF model with architecture qwen35 is not supported yet"` even after the download completes. Fixing this arch registration would additionally require implementing GGUF tensor name mappings for Qwen3.5's hybrid GatedDeltaNet + full-attention layers (a Tier B new-infrastructure task: no existing converter covers the hybrid architecture).

## Fix
Marked `dinerburger_qwen_3_5_27b_gguf/causal_lm/pytorch-27B_Q8_0_XXL-single_device-inference` as `KNOWN_FAILURE_XFAIL` in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

The secondary loader bug (GGUF arch `qwen35` not registered + missing GatedDeltaNet tensor mappings) is Tier B new-infrastructure and is not fixed here.

## Verification
- pytest exit: not-run (hardware capacity confirmed analytically; file size 35 GB from `huggingface_hub.get_paths_info`, n150 DRAM 12 GB)
- Hardware:    n150
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 983a05a9e3ee54012fa45db81e5e7da6c9e7cf96 |
| tt-forge-models | d44b5a37543e89fa06fb71b56d930b76eedb3593 |
