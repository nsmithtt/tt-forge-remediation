# Remediation Summary: gpt_oss-causal_lm-pytorch-20B_BF16_abliterated-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss/causal_lm/pytorch-20B_BF16_abliterated-single_device-inference]

## Result
XFAIL — 20B BF16 model (~40 GB) exceeds single-device DRAM (~32 GB on p150b / 12 GB on n150); hardware capacity ceiling

## Stack layer
tt-xla, hardware-class

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Value out of range (expected to be in range of [-88, 87], but got -127)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_3, 2, -127, 9223372036854775807), kwargs = {})
Original traceback:
  File ".../transformers/cache_utils.py", line 208, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

## Root cause
Two issues, in order of discovery:

**Issue 1 (tt-xla, Tier A, fixed):** `SlidingWindowCache.update()` computes the KV slice start as `-self.sliding_window + 1` (i.e. `-127` for `sliding_window=128`). When the actual sequence length is 88, the valid negative range is `[-88, 87]`. PyTorch eager silently clamps out-of-range negative indices; the XLA lazy backend (`torch/csrc/lazy/core/helpers.cpp`) validates them strictly and raises `"Value out of range"`. The fix is to pre-clamp `start` and `end` before dispatching `aten.slice.Tensor` in `TorchFunctionOverride.__torch_function__`.

**Issue 2 (hardware-class):** After the slice fix allows compilation to complete, hardware execution fails with DRAM OOM. The 20B BF16 model has ~40 GB of weights; the device has ~32 GB of DRAM on Blackhole p150b (8 banks × ~4 GB each). The allocator reports: "Not enough space to allocate 1061683200 B DRAM buffer across 8 banks (allocated: 4111190848 B, free: 162199168 B, largest free block: 66355200 B)". This is purely hardware capacity — the model is larger than device DRAM.

## Fix
**Issue 1 (tt-xla, `python_package/tt_torch/torch_overrides.py`):**
Added a handler for `func is torch.ops.aten.slice.Tensor` in `TorchFunctionOverride.__torch_function__`. When `args[2]` (start) or `args[3]` (end) is an integer less than `-size` for the sliced dimension, clamp it to `-size` before dispatching. This matches PyTorch eager's silent clamping behavior.

Branch: `remediation/gpt_oss-causal_lm-pytorch-20B_BF16_abliterated-single_device-inference` in tt-xla.
Commit: `5b32b1769` — "Clamp out-of-range negative aten.slice start/end in TorchFunctionOverride"

**Issue 2 (tt-xla test config, hardware-class XFAIL):**
Added `KNOWN_FAILURE_XFAIL` entry for `gpt_oss/causal_lm/pytorch-20B_BF16_abliterated-single_device-inference` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.
Commit: `75aa04b46` — "Mark gpt_oss/causal_lm/pytorch-20B_BF16_abliterated as KNOWN_FAILURE_XFAIL"

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — Issue 1 is Tier A (fixed); Issue 2 is hardware-class XFAIL (not a bug).

## Verification
- pytest exit: FAIL (OOM on hardware after slice fix; xfail config added but hardware cannot pass)
- Hardware:    blackhole-p150b
- Duration:    ~244s (hardware OOM run); ~332s (second reproduction confirming original error)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — clamp OOB negative aten.slice start/end
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 75aa04b463ea49f628773f635d9a49612983b54f |
| tt-forge-models | 2b2f3a3cd6521be0f0b0847b3416230ac55fb964 |
