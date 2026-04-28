# Remediation Summary: gemma3-multimodal-mlx-community-gemma-3-4b-it-qat-bf16-single-device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3/multimodal/pytorch-mlx-community/gemma-3-4b-it-qat-bf16-single_device-inference]

## Result
FAIL — model runs through but PCC = 0.7455 vs required 0.99; root cause is a compiler bug producing incorrect outputs relative to CPU golden reference

## Stack layer
tt-xla

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
The reported failure was:

> The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

On reproduction with the configured branch (`arch-c-36-tt-xla-dev/nsmith/hf-bringup-19`), the image processor warning is still emitted but no longer fatal; the test instead crashes with:

```
RuntimeError: Value out of range (expected to be in range of [-277, 276], but got -1023)
While executing %slice_7 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807), kwargs = {})
Original traceback:
  transformers/cache_utils.py:214, in DynamicSlidingWindowLayer.update:
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

After applying the slice fix, the model runs to completion but fails PCC:

```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7455990988734212. Required: pcc=0.99.
```

## Root cause

Two bugs were found:

**Bug 1 (fixed — Tier A):** XLA's `aten.slice.Tensor` implementation validates that `start` is in `[-dim_size, dim_size-1]` and raises `ValueError` when `start < -dim_size`. PyTorch silently clamps out-of-bounds negative starts to `-dim_size` (effectively 0). Gemma3's `DynamicSlidingWindowLayer` uses `full_value_states[:, :, -sliding_window+1:, :]`; with `sliding_window=1024` and `seq_len=277`, this produces `start=-1023` on a 277-element dimension, triggering the XLA error. The fix adds a guard in `TorchFunctionOverride.__torch_function__` to clamp `start` to `-dim_size` before dispatch, restoring PyTorch semantics.

**Bug 2 (not fixed — Tier B):** After the slice fix, the model executes to completion but achieves only `pcc=0.7455` vs the required `pcc=0.99`. This is well below any BF16-vs-FP32 accumulation floor (~0.95). Model weight loading was verified correct: only the expected `patch_embedding.weight` shows a MISMATCH (NHWC→NCHW layout difference), which is correctly repaired by `_fix_mlx_patch_embedding`. The PCC gap therefore comes from TT hardware/compiler producing outputs that diverge significantly from the CPU reference for some operation(s) in the Gemma3 multimodal pipeline. Per-layer comparison is needed to identify the specific layer; this requires more invasive debugging tooling than is available in a single-report session.

## Fix
**Bug 1:** `python_package/tt_torch/torch_overrides.py` in `tt-xla` — added a guard in `TorchFunctionOverride.__torch_function__` that intercepts `torch.ops.aten.slice.Tensor` calls where `start < -dim_size` and clamps `start` to `-dim_size` before dispatching.

Remediation branch: `remediation/gemma3-multimodal-mlx-community-gemma-3-4b-it-qat-bf16-single-device-inference`
Commit: `ba8b8db950882e50c1f06931ccf5b04e64d3425e`

**Bug 2 (proposed fix location):** Unknown; per-layer CPU vs TT comparison required. Likely in the attention or FFN layers of the Gemma3 language model or vision encoder. The `tt-xla` compiler frontend or `tt-mlir` lowering is the most probable layer.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

The test was reached by applying a Tier A fix (slice bounds clamping). The subsequent PCC failure (Bug 2) is Tier B:

**Indicator:** `internal-error-unknown-mechanism` — the root cause layer and specific failing operation are unknown without per-layer comparison tooling. A second Tier A fix cannot be attempted without first diagnosing which operation produces incorrect results. Per the skill rules, chaining Tier A fixes in a single report is not permitted; the first fix is committed and the second bug is filed as FAIL.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    298.41s (0:04:58) — post slice-fix run
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — clamp out-of-bounds negative slice start in `TorchFunctionOverride`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ba8b8db950882e50c1f06931ccf5b04e64d3425e |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
