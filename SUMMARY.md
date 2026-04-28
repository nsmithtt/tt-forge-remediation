# Remediation Summary: botnet/pytorch-Botnet26t_256.c1_in1k-single_device-inference

## Skill version
16

## Test
tests/runner/test_models.py::test_all_models_torch[botnet/pytorch-Botnet26t_256.c1_in1k-single_device-inference]

## Result
FAIL — reshape→expand→permute(0,3,1,4,2) produces incorrect output in a fused TT graph (compiler bug in tt-mlir/tt-metal)

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8813457208132631. Required: pcc=0.95.

## Root cause
**Compiler stack layer: tt-mlir / tt-metal**

The BotNet model's `BottleneckAttn` module uses relative position embeddings
(`PosEmbedRel`) that call `rel_logits_1d` twice — once for the width dimension
and once for the height dimension.  The height-direction call ends with:

```python
x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1).permute(0, 3, 1, 4, 2)
```

This `reshape → expand → permute(0,3,1,4,2)` sequence gives the wrong result
when compiled as a single StableHLO/TT graph (PCC ≈ 0.05–0.07 vs the CPU
bfloat16 reference).  The width-direction call uses `permute(0,1,3,2,4)` and
computes correctly (PCC ≈ 1.0).

Executing each op individually with an `xm.mark_step()` between them gives the
correct answer, confirming the error is in how the fused graph handles
`stablehlo.broadcast_in_dim` (expand) followed by `stablehlo.transpose` with
the non-trivial permutation `(0,3,1,4,2)`.  This looks like an incorrect
lowering or fusion of broadcast + non-contiguous transpose in the tt-mlir
compiler.

**Minimal reproducer (bfloat16, single xm.mark_step):**
```python
import torch, torch_xla.core.xla_model as xm
device = xm.xla_device()
x = torch.randn(64, 16, 16, dtype=torch.bfloat16).to(device)
result = x.reshape(4, 16, 1, 16, 16).expand(-1, -1, 16, -1, -1).permute(0, 3, 1, 4, 2)
xm.mark_step()
# result.cpu() does not match the CPU reference (PCC ≈ 0.05)
```

A secondary loader bug was also fixed: `botnet/pytorch/loader.py` imported
`from datasets import load_dataset` at module level, which (a) prevented test
collection when `datasets` was not installed, and (b) triggered
`AttributeError: module 'spacy' has no attribute 'Language'` at runtime
because the `spacy/` directory inside `tt_forge_models` is treated as a
namespace package when `tt_forge_models` is added to `sys.path`, causing
`datasets._dill` to fail during fingerprinting.  This was fixed by removing
the `datasets` import and delegating image loading to `VisionPreprocessor`
(which already handles `image=None` via a default COCO URL).

## Fix
**Loader fix (in tt_forge_models, applied):** Removed the top-level
`from datasets import load_dataset` import and simplified `load_inputs` to
pass `image=None` directly to `VisionPreprocessor.preprocess`, which already
handles `None` by fetching a default COCO image URL.  This is not a forbidden
workaround — it removes an incorrect dependency that was masking the real
failure.

**Compiler fix (not yet applied, required in tt-mlir/tt-metal):** The
lowering of `stablehlo.broadcast_in_dim` (XLA `expand`) followed by
`stablehlo.transpose` with permutation `(0,3,1,4,2)` on a tensor that was
previously reshaped to include a size-1 broadcast dimension must be corrected.
The fix should be scoped to tt-mlir's lowering or fusion pass for these
three ops in sequence.

## Verification
Test fails on TT silicon with PCC=0.8813457208132631 (required 0.99 in this
environment, 0.95 in CI).  The loader fix alone is not sufficient to make
the test pass because the PCC failure persists after the fix.

Wall-clock duration of failing run: 105s (n150)

## Files changed
- `tt_forge_models/botnet/pytorch/loader.py` — removed `from datasets import load_dataset`, simplified `load_inputs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 399122604bb1b2929a23f67a462e0b3a18254a30 |
| tt-forge-models | 18cec2ec25f3fb0dddcf2ce44a73e45e27b79bde |
