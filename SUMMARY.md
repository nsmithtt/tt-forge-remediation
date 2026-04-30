# Remediation Summary: jan_v2_vl_max_gguf/image_to_text/pytorch-jan_v2_vl_max_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jan_v2_vl_max_gguf/image_to_text/pytorch-jan_v2_vl_max_gguf-single_device-inference]

## Result
XFAIL — model is Qwen3-VL-30B-A3B (128 experts, ~30B total params); dequantizes to ~60 GB BF16, far exceeding n150 (12 GB) and p150b (32 GB) single-device DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen3-vl-30b-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Jan v2 VL Max is based on Qwen3-VL-30B-A3B, a 128-expert MoE vision-language
model with ~30 billion total parameters. When dequantized to BF16 for TT
hardware inference, the model requires approximately 60 GB of DRAM (30B × 2
bytes/param). This exceeds the capacity of every single-device configuration:
n150 (12 GB), n300 (24 GB), and p150b Blackhole (32 GB). The GGUF file itself
is 18 GB (Q4_K_M quantization), which also exceeds n150/n300 capacity.

The INTERNAL: Error code: 13 (kInternal) error manifests during tt-mlir
compilation after the model is traced by torch/XLA. The runtime does not emit a
clean OOM message; instead the compilation fails internally when the allocator
cannot satisfy the model's DRAM requirements.

A secondary issue is present: the loader on the hf-bringup-41 branch uses
`AutoModelForImageTextToText.from_config(config)` with random weights (the
comment notes "For compile-only environments this is acceptable") rather than
loading the actual GGUF weights. This workaround was necessary because
transformers 5.x lacks native GGUF support for the `qwen3vlmoe` architecture
(not registered in GGUF_CONFIG_MAPPING). A proper GGUF registration patch was
developed in a parallel remediation (bartowski_browser_use_bu_30b_a3b_preview_gguf,
commit 74b671b2a3) and could be applied here once hardware capacity permits.
However the hardware capacity ceiling means neither the random-weight nor the
properly-loaded model can run on any current single device.

Additionally, Qwen3VLMoe uses a Python for-loop over experts (similar to
Qwen3MoE and Jamba) that would also trigger a Tier B MoE-segfault bug during
XLA tracing even if hardware capacity were sufficient. Fixing this would require
setting `model.config._experts_implementation = "batched_mm"` analogously to
the Qwen3MoE fix.

## Fix
Added KNOWN_FAILURE_XFAIL entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in
`tt-xla` on branch
`remediation/jan-v2-vl-max-gguf-image-to-text-pytorch-jan-v2-vl-max-gguf-single-device-inference`:

```yaml
jan_v2_vl_max_gguf/image_to_text/pytorch-jan_v2_vl_max_gguf-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "Hardware capacity: Jan v2 VL Max (Qwen3-VL-30B-A3B, 128 experts x 30B total params) dequantizes to ~60 GB BF16, far exceeding n150 (12 GB) and p150b (32 GB) single-device DRAM."
```

## Verification
- pytest exit: FAIL (test was killed after 31+ minutes during model loading; full 30B from_config load exhausts CI budget before reaching the INTERNAL error)
- Hardware:    n150
- Duration:    not-run (killed at 31:07 during model loading)
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | f4d0e7ade4f7847c060c5cb7a37259f38b416e2c |
| tt-forge-models | ebcfe743a1f2fd8b850014c4554bf931b137e40b |
