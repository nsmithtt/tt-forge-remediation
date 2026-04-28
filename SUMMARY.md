# Remediation Summary: hulu_med-image_to_text-pytorch-30A3-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[hulu_med/image_to_text/pytorch-30A3-single_device-inference]

## Result
XFAIL — Hulu-Med-30A3 (~30B params, ~60 GB BF16) exceeds single-device DRAM (n150: 12 GB, p150b: 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-30b-moe-exceeds-single-device-dram

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
Hulu-Med-30A3 is a medical VLM fine-tuned on Qwen3-VL-30B-A3B (30B total parameters,
3B active via MoE routing). When loaded in BF16 format: 30B × 2 bytes ≈ 60 GB. This
far exceeds both n150 (12 GB DRAM) and p150b (24 GB DRAM). The failure manifests as
INTERNAL: Error code: 13 (hardware OOM) inside the vision encoder's
`fast_pos_embed_interpolate()` at `grid_thw.tolist()`, which is the first graph break
where the compiled TT subgraph must materialize a result back to Python. The model
loads successfully from HuggingFace (no loader bugs) but cannot execute on any
single-device TT hardware. This is the same capacity class as other 30B+ MoE models
already confirmed as hardware-class failures.

## Fix
No code fix. Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in
`tt-xla` on branch
`remediation/hulu_med-image_to_text-pytorch-30A3-single_device-inference`.

## Verification
- pytest exit: FAIL (reproduced — INTERNAL: Error code: 13 confirmed)
- Hardware:    n150
- Duration:    347.51s
- Tier A attempts: N/A

## Files changed
- tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b843edc9e193a4f406cb4adec3846a5c57e06669 |
| tt-forge-models | f44eb7ec0b6d2193454439d78a44229bbb806143 |
