# Remediation Summary: efficientvit/pytorch-EfficientViT_B2.r224_in1k-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[efficientvit/pytorch-EfficientViT_B2.r224_in1k-single_device-inference]

## Result
FAIL — PCC 0.85 on TT silicon; root-caused to tt-mlir not preserving float32 precision for LiteMLA attention matmuls (Tier B, same bug as EfficientViT_L2)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
E   AttributeError: module 'spacy' has no attribute 'Language'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
After loader fix:
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8504920668808637. Required: pcc=0.99.
```

## Root cause

Two bugs were present.

**Bug 1 (loader, fixed):** `huspacy/pytorch/loader.py` imported `spacy` at module level. The test runner inserts `models_root` (`third_party/tt_forge_models/`) at `sys.path[0]`, so `import spacy` resolves to the `tt_forge_models/spacy/` namespace package (a model directory, not the real spaCy library). This namespace package has no `Language` attribute. When EfficientViT's `load_inputs` later called `load_dataset("huggingface/cats-image")`, the `datasets` dill helper (in `datasets/utils/_dill.py:42`) checked `issubclass(obj_type, spacy.Language)` and crashed with `AttributeError: module 'spacy' has no attribute 'Language'`.

**Bug 2 (tt-mlir, unfixed):** `EfficientViT_B2` uses `LiteMLA` (Lightweight Multi-Scale Linear Attention) in its attention blocks. The `LiteMLA._attn` method explicitly upcasts inputs to float32 before the key-value and query-output matmuls:

```python
def _attn(self, q, k, v):
    dtype = v.dtype
    q, k, v = q.float(), k.float(), v.float()
    kv = k.transpose(-1, -2) @ v
    out = q @ kv
    out = out[..., :-1] / (out[..., -1:] + self.eps)
    return out.to(dtype)
```

This float32 upcast is essential for numerical stability: the normalisation denominator `out[..., -1:]` (the sum of attention weights) can be very small, and single-precision matmul accuracy is required to avoid large relative errors in the subsequent division. The XLA/StableHLO graph correctly contains float32 dot operations, but tt-mlir silently downcasts these to bf16 during lowering, producing PCC of 0.85. This is the same bug documented for EfficientViT_L2 (PCC 0.70 there, higher here because B2 is shallower with fewer LiteMLA blocks accumulating error).

## Fix

**Loader fix (applied):** Moved `import spacy` from module level to inside `huspacy/pytorch/loader.py::_load_nlp()`, so the import only runs when the HuSpaCy model is actually loaded — after the real spaCy package (not the namespace package) is resolved from the standard library path.

**Compiler fix (proposed, not applied):** The `tt-mlir` StableHLO-to-TTIR lowering pass should not silently downcast `f32` dot/matmul operations to `bf16`. Options:
1. Emit genuine float32 matmuls where hardware supports them (Wormhole supports fp32 compute).
2. Preserve fp32 accumulation via `fp32_dest_acc_en` or equivalent.
3. At minimum, emit a diagnostic when an explicitly-typed f32 op is silently converted to bf16.

## Tier B justification
cross-cutting — preserving f32 precision through every lowering pass touches multiple files across the StableHLO→TTIR→TTNN pipeline and cannot be scoped to a single named function.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    62.54s (1:02)
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` — moved `import spacy` inside `_load_nlp()` (remediation branch: `remediation/efficientvit-pytorch-EfficientViT_B2.r224_in1k-single_device-inference` in tt-forge-models at commit `19dfd59d18`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 19dfd59d18450e5e169392f08915fbbd81daf034 |
