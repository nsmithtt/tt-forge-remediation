# Remediation Summary: docling-pytorch-Egret_Large-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[docling/pytorch-Egret_Large-single_device-inference]

## Result
FAIL — pcc=0.023 after fixing torch_compilable_check false-positive; catastrophically low PCC indicates a compiler-stack bug with unknown root cause

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

After loader fix: pytest exits FAIL with pcc=0.023 (required: 0.99).

## Root cause

**Loader bug (fixed):** DFine's `torch_compilable_check` is a `torch.compile` compile-time
assertion: under `torch.compile` it is a no-op at runtime. Under TorchXLA on TT hardware, the
condition tensor `(spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length` is
evaluated on-device. TT hardware computes int64 `.sum()` in bfloat16; the actual sum
6400+1600+400=8400 rounds to 8384 (bfloat16 ULP=64 in the range 8192–16384), producing a false
False condition and a spurious `ValueError`.

**Compiler-stack bug (unfixed, Tier B):** After bypassing the false-positive check, the model
runs on silicon and produces pcc=0.023 — catastrophically near zero. Isolated checks showed:
- `F.grid_sample` (used in DFine deformable sampling): pcc=1.0 — not the cause.
- `topk` minor bfloat16 issues (298/300 correct for large tensors) — insufficient to cause 97.7%
  error.
The exact failing op in DFine's decoder (AIFI self-attention, DFineIntegral, DFineGate, DFineLQE,
deformable cross-attention with `grid_sample`) is not identified. Root cause requires expert
instrumentation of intermediate tensors across the decoder stack.

## Fix

**Loader fix** (`tt-xla/third_party/tt_forge_models/docling/pytorch/loader.py`):

Added `_patch_dfine_compilable_check()` static method to `ModelLoader` that replaces
`torch_compilable_check` in the `transformers.models.d_fine.modeling_d_fine` module namespace with
a wrapper that skips the check for non-CPU tensors (matching `torch.compile` behavior). Called
from `load_model()` before `from_pretrained()` for DFine variants.

Commit: `ba9229b375213cc0f52ec2774f042101dad7c413` on `remediation/docling-Egret_Large` branch of
tt_forge_models.

**Proposed fix for PCC failure:** Systematic bisection of DFine decoder intermediate activations
to identify the first layer producing near-zero outputs on TT silicon. Likely candidates are the
deformable cross-attention sampling/aggregation or the DFineIntegral softmax+linear projection.
Fix would live in tt-mlir lowering or tt-metal kernel(s) for the identified op.

## Tier B justification

**Indicator: internal-error-unknown-mechanism**

The root cause of the near-zero PCC is not identified. Reproducing it requires running the full
DFine decoder on TT silicon and extracting intermediate tensors op-by-op. This is diagnostic work
that must precede any fix attempt, not a scoped one-file change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    293.13s (first run, loader bug), 420.74s (second run, pcc failure)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/docling/pytorch/loader.py` — added `_patch_dfine_compilable_check()`, called from `load_model()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c33722cb922870bfb595dc65af8e934e625364ca |
| tt-forge-models | ba9229b375213cc0f52ec2774f042101dad7c413 |
