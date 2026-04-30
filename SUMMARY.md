# Remediation Summary: docling-pytorch-Egret_XLarge-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[docling/pytorch-Egret_XLarge-single_device-inference]

## Result
FAIL — pcc=-0.048 after fixing torch_compilable_check false-positive; catastrophically low PCC indicates a compiler-stack bug with unknown root cause

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
dfine-egret-deformable-attn-pcc-near-zero

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (loader layer):
```
ValueError: Make sure to align the spatial shapes with the sequence length of the encoder hidden states
```
Raised by `torch_compilable_check` inside `DFineMultiscaleDeformableAttention.forward` at
`transformers/models/d_fine/modeling_d_fine.py:266`.

After loader fix: pytest exits FAIL with pcc=-0.048 (required: 0.99).

The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
is the last line of pytest output — a harmless SWIG deprecation warning, not the actual error.

## Root cause

**Loader bug (fixed):** DFine's `torch_compilable_check` is a `torch.compile` compile-time
assertion: under `torch.compile` it is a no-op at runtime. Under TorchXLA on TT hardware, the
condition tensor `(spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length` is
evaluated on-device. TT hardware computes int64 `.sum()` in bfloat16; the actual sum
6400+1600+400=8400 rounds to 8384 (bfloat16 ULP=64 in the range 8192–16384), producing a false
False condition and a spurious `ValueError`.

**Compiler-stack bug (unfixed, Tier B):** After bypassing the false-positive check, the model
runs on silicon and produces pcc=-0.048 — catastrophically near zero (same pattern as
Egret_Large which also produced pcc=0.023 after the identical fix). The DFine decoder (AIFI
self-attention, DFineIntegral, DFineGate, DFineLQE, deformable cross-attention with
`grid_sample`) is producing near-zero outputs on TT silicon. Root cause requires expert
instrumentation of intermediate tensors across the decoder stack to identify which op first
diverges.

## Fix

**Loader fix** (`tt-xla/third_party/tt_forge_models/docling/pytorch/loader.py`):

Added `_patch_dfine_compilable_check()` static method to `ModelLoader` that replaces
`torch_compilable_check` in the `transformers.models.d_fine.modeling_d_fine` module namespace
with a wrapper that skips the check for non-CPU tensors (matching `torch.compile` behavior).
Called from `load_model()` before `from_pretrained()` for DFine variants.

Commit: `9533ac284bd204d0e47fe6479568b22b56bfe351` on `remediation/docling-Egret_XLarge` branch
of tt_forge_models.

**Proposed fix for PCC failure:** Systematic bisection of DFine decoder intermediate activations
to identify the first layer producing near-zero outputs on TT silicon. This is the same unfixed
bug as Egret_Large (`dfine-egret-deformable-attn-pcc-near-zero`). Likely candidates are the
deformable cross-attention sampling/aggregation or the DFineIntegral softmax+linear projection.
Fix would live in tt-mlir lowering or tt-metal kernel(s) for the identified op.

## Tier B justification

**Indicator: internal-error-unknown-mechanism**

The root cause of the near-zero PCC is not identified. Reproducing it requires running the full
DFine decoder on TT silicon and extracting intermediate tensors op-by-op. This is diagnostic
work that must precede any fix attempt, not a scoped one-file change. The same bug was filed
for Egret_Large and remains unresolved.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    529.85s (0:08:49) — second run with loader fix; 417.33s for first run (loader bug)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/docling/pytorch/loader.py` — added `_patch_dfine_compilable_check()`, called from `load_model()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9533ac284bd204d0e47fe6479568b22b56bfe351 |
