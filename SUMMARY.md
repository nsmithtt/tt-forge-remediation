# Remediation Summary: ernie-causal_lm-pytorch-21B_A3B_PT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ernie/causal_lm/pytorch-21B_A3B_PT-single_device-inference]

## Result
XFAIL — ERNIE 4.5 21B-A3B at bfloat16 (~42 GB) exceeds single-device DRAM on all current TT hardware (n150: 12 GB, n300: 32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
ernie-21b-a3b-bfloat16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_FATAL: Out of Memory: Not enough space to allocate 1038090240 B DRAM buffer across 8 banks,
where each bank needs to store 129761280 B, but bank size is 4273390016 B
(allocated: 4182683456 B, free: 90706560 B, largest free block: 87848384 B)
(assert.hpp:104)
```
The runtime error surfaces as:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
The original reported failure (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__
attribute`) was a misleading warning printed as the last output line. The true loader bug was
`NotImplementedError: "histogram_cpu" not implemented for 'Int'` from `torch.histc` in
`grouped_mm_experts_forward`, which uses integer top-k indices to compute per-expert token counts.
XLA/TT does not support `histc` on integer tensors. The alternative `batched_mm` path would
produce a 3D gather tensor of shape [S, 3072, 2560] where each row is 15.7 MB — far exceeding
the 1.5 MB L1 limit.

The loader fix (static per-expert masked matmul, registered via `ALL_EXPERTS_FUNCTIONS`) resolved
the XLA dispatch issue. After the fix the model compiled and ran on silicon. However, ERNIE 4.5
21B-A3B at bfloat16 requires approximately 42 GB of DRAM to hold all parameters (21B × 2 bytes),
which exceeds the largest available single-device DRAM (n300: 32 GB). The OOM occurred during
runtime weight loading (`LoadCachedOp` → `to_device`) after 31.4 GB of 32 GB was already
allocated.

## Fix
**Loader fix** (`ernie/causal_lm/pytorch/loader.py` in `tt_forge_models`):
- Registered `_tt_static_ernie_moe_forward` in `ALL_EXPERTS_FUNCTIONS["tt_static_ernie_moe"]`
  at module import time (before `from_pretrained` so the dispatch table is ready).
- After `from_pretrained` completes, set `model.config._experts_implementation = "tt_static_ernie_moe"`.
  The setter bypasses the `_check_and_adjust_experts_implementation` validator (which only accepts
  "eager", "grouped_mm", "batched_mm"), so setting it post-load avoids the ValueError.
- The static forward loops over all 64 experts using Python int indices (dynamo unrolls to
  static `F.linear` calls), computing a masked accumulation instead of histogram dispatch or
  3D gather.

**XFAIL** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in `tt-xla`):
- Added `status: KNOWN_FAILURE_XFAIL` for `ernie/causal_lm/pytorch-21B_A3B_PT-single_device-inference`
  because the model's DRAM footprint (~42 GB BF16) exceeds all single-device TT hardware.

## Verification
- pytest exit: FAIL (DRAM OOM, INTERNAL Error code 13)
- Hardware:    n300
- Duration:    40m32s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/ernie/causal_lm/pytorch/loader.py` — static MoE forward
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 952ccf8320473a186f9cc864b65ed149e1883ee4 |
| tt-forge-models | 0b31be0567884d17b7477cd974961abb5b34715f |
