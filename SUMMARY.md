# Remediation Summary: dazipe_qwen3_next_80b_a3b_instruct_gptq_int4a16-causal_lm-pytorch-Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dazipe_qwen3_next_80b_a3b_instruct_gptq_int4a16/causal_lm/pytorch-Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16-single_device-inference]

## Result
XFAIL — Hardware capacity ceiling: 80B model dequantizes to ~160 GB BF16 which exceeds single-device p150b DRAM (24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gptq-int4a16-80b-exceeds-single-device-dram

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
The `dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16` model is an 80B parameter Qwen3-Next MoE hybrid (SSM + full attention) quantized to 4-bit using the `compressed-tensors` format. The quantized checkpoint requires ~40 GB of storage; dequantizing to BF16 for inference requires ~160 GB (80B × 2 bytes), which vastly exceeds the 24 GB DRAM of the single p150b device.

On CI, if `compressed-tensors` was installed by a prior test in the same session, `from_pretrained` would proceed past the import check and attempt to download the ~40 GB quantized weights, causing the job-level timeout. Locally (without `compressed-tensors` installed), the failure reproduces as an immediate `ImportError: compressed_tensors is not installed`. Both failure modes confirm this model cannot reach the device.

The model config is `model_type: qwen3_next`, 48 layers, 512 experts (10 active), `full_attention_interval=4` (12 full-attention layers, 36 linear-attention). Even if the compressed-tensors import and download were resolved, the dequantized weights (160 GB) would still exceed device DRAM by an order of magnitude.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla. No loader or compiler changes were made.

## Verification
- pytest exit: XFAIL (1 xfailed in 20.08s — test correctly skipped without attempting model load)
- Hardware:    not-run
- Duration:    20.08s (collection + xfail, no model download attempted)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f66af2771dbfacc992d06f42d1dff940a28a008a |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
