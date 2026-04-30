# Remediation Summary: docling_layout_heron-pytorch-docling_layout_heron_101-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[docling_layout_heron/pytorch-docling_layout_heron_101-single_device-inference]

## Result
FAIL — after fixing INT32 reduction precision, a second bug surfaces: BF16 overflow in logits (TT gives ±Inf, CPU BF16 reference gives NaN), pcc=nan

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
rtdetrv2-bf16-logit-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure (misleading): `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

Actual first failure (pre-fix):
```
ValueError: Make sure to align the spatial shapes with the sequence length of the encoder hidden states
```
Inside `RTDetrV2MultiscaleDeformableAttention.forward()`, the check:
```python
torch_compilable_check(
    (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length, ...)
```
computed `sum([6400, 1600, 400])` = 8384 (wrong) instead of 8400, because
`createReductionOpOperandsWorkarounds` was casting INT32 reduction inputs to BFloat16,
and BF16 (7 mantissa bits) rounds 8400 → 8384.

Second failure (post-fix, Tier B):
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.
```
Duration: 385.29s (0:06:25). TT logits: 1454 +inf, 918 -inf out of 5100 elements (46% overflow).
CPU BF16 reference logits: has_nan=True. CPU FP32 logits: finite, mean=-5.68, std=1.09.

## Root cause

**First bug (fixed — Tier A):** `TTNNWorkaroundsPass.cpp::createReductionOpOperandsWorkarounds`
unconditionally cast all non-F32/BF16 reduction inputs to BFloat16, including INT32. BF16 has
only 7 mantissa bits, so it exactly represents integers only up to ~256; values like 8400
(sum of spatial shape products for a 640×640 image) round to 8384, causing the
`torch_compilable_check` to throw ValueError.

**Second bug (unfixed — Tier B):** After the INT32 fix allows the model to run to
completion, the output logits contain ±Inf on TT (46% of elements) and NaN on CPU BF16.
CPU FP32 is numerically stable (std=1.09), but BF16 is not. The overflow originates
somewhere in the RT-DETRv2 decoder (likely the deformable cross-attention or FFN layers),
and the root cause — which specific op first overflows and why — has not been isolated.
The TT BF16 and CPU BF16 overflow patterns differ (Inf vs NaN), indicating TT's BF16
matmul accumulation diverges from CPU in a way that causes overflow at different
points. Isolating the first overflowing op would require probing intermediate activations
across 6 decoder layers, which is multi-layer diagnostic work (cross-cutting).

## Fix

**First bug (applied):** In `tt-mlir/lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`,
`createReductionOpOperandsWorkarounds`: when the input element type is an integer type
with width ≥ 32 bits, use `DataType::Float32` instead of `DataType::BFloat16` as the
workaround dtype. Float32 exactly represents integers up to 2^24 (well above any
spatial dimension sum in practice). Committed as `71d866c52` on branch
`remediation/docling_layout_heron-pytorch-docling_layout_heron_101-single_device-inference`
in tt-mlir.

**Second bug (proposed fix):** Identify which specific op in the decoder produces the
first BF16 overflow, then determine whether (a) TT's TTNN kernel for that op has a
bug (Tier A fix in one file) or (b) the model requires FP32 precision end-to-end
(cross-cutting Tier B). The diagnostic would probe layer-by-layer activations in the
6-layer RT-DETRv2 decoder.

## Tier B justification

Tier B indicator: **cross-cutting** (or **internal-error-unknown-mechanism** pending
diagnosis). The overflow propagates across 6 decoder layers; which specific op causes
the first Inf has not been determined. Fixing it may require either a targeted kernel
fix (if one op is faulty, Tier A) or preserving FP32 accumulation across all decoder
matmuls (cross-cutting, Tier B). Without isolating the first overflowing op, the
fix scope is unknown.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    385.29s (0:06:25) — after INT32 reduction fix
- Tier A attempts: 1 (INT32 reduction fix applied and verified; second bug not attempted)

## Files changed
- `tt-mlir/lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` — use Float32 (not BF16) workaround for 32-bit integer reduction inputs

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 71d866c52a1e97069ed7bbc537ac2585d61d0dd2 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5cf81503309b8c476a57bf9e4eb9667e215e3f12 |
