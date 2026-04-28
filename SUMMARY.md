# Remediation Summary: efficientvit/pytorch-EfficientViT_L2.r384_in1k-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[efficientvit/pytorch-EfficientViT_L2.r384_in1k-single_device-inference]

## Result
FAIL — PCC 0.70 on TT silicon; root-caused to tt-mlir not preserving float32 precision for matmuls in LiteMLA._attn

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7023129554054035. Required: pcc=0.99.

(Original branch reported pcc=0.8674345152594858 against required pcc=0.95.)

## Root cause

**Layer: tt-mlir (compiler core)**

`EfficientViT_L2` uses `LiteMLA` (Lightweight Multi-Scale Linear Attention) as its attention mechanism. The `LiteMLA._attn` method explicitly upcasts inputs to float32 before computing the key-value and query-output matmuls, then casts back to the original dtype:

```python
def _attn(self, q, k, v):
    dtype = v.dtype
    q, k, v = q.float(), k.float(), v.float()
    kv = k.transpose(-1, -2) @ v    # precision-critical: k·v product
    out = q @ kv
    out = out[..., :-1] / (out[..., -1:] + self.eps)  # linear-attn normalisation
    return out.to(dtype)
```

This float32 upcast is essential for numerical stability: the normalisation denominator `out[..., -1:]` (sum of attention weights) can be very small, and precision in the preceding matmuls determines whether the denominator is accurate enough to avoid large errors in the division.

Verification: simulating the `_attn` with bfloat16 matmuls instead of float32 produces a PCC of ~0.60 in isolation (consistent with the observed full-model PCC of 0.70 once the remaining float32-correct operations partially compensate).

The XLA/StableHLO graph correctly contains float32 dot operations (`%dot.89 = f32[...] dot(f32[...], f32[...])`) generated from the explicit `.float()` casts. A standalone test of an isolated float32 matmul path on TT silicon (via `xm.xla_device()`) gives PCC 0.9945, confirming the hardware itself supports float32 matmuls. The precision loss therefore occurs during the **tt-mlir lowering of the StableHLO graph**: the compiler appears to downcast the f32 dot operations to bf16 for hardware efficiency, discarding the precision the model explicitly requested.

There is a secondary loader-layer bug that was also present and blocking the test from running at all: `huspacy/pytorch/loader.py` imported `spacy` at module level. Because the test runner adds `models_root` (the `tt_forge_models` directory) to `sys.path[0]`, Python resolves `import spacy` to the `tt_forge_models/spacy/` namespace package (which has no `Language` attribute) instead of the real spaCy library. This pollutes `sys.modules["spacy"]` during test discovery, and later when EfficientViT calls `load_dataset(...)`, the `datasets` dill helper checks `if "spacy" in sys.modules` and then crashes with `AttributeError: module 'spacy' has no attribute 'Language'`. This was fixed by moving the `import spacy` inside `_load_nlp()` so it is only imported when the HuSpaCy model actually runs.

## Fix

**Loader fix (applied):** Moved `import spacy` from module-level to inside `huspacy/pytorch/loader.py::_load_nlp()`, preventing the namespace-package pollution that caused EfficientViT's `load_dataset` call to crash.

**Compiler fix (proposed, not applied):** In `tt-mlir`, the StableHLO-to-TTIR lowering pass should not silently downcast `f32` dot/matmul operations to `bf16`. Options:
1. Emit genuine float32 matmuls where the hardware supports them (Wormhole supports fp32 compute).
2. If fp32 matmuls are unsupported, preserve fp32 accumulation at the output of each tile (i.e., use `fp32_dest_acc_en` or equivalent).
3. At minimum, emit a diagnostic when an explicitly-typed f32 op is being silently converted to bf16.

## Verification
pytest exit: FAILED — pcc=0.7023129554054035, required pcc=0.99  
Wall-clock duration: 96.57 s  
Hardware: Wormhole N150 (arch-c-36)

## Files changed
- `huspacy/pytorch/loader.py` — moved `import spacy` inside `_load_nlp()` to prevent sys.modules namespace-package pollution

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8fc888af179805af603b7accbaa91698e284f3ba |
| tt-forge-models | 5ed51013bee0fd405a686928a268c4ddf4ed946c |
