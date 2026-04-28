# Remediation Summary: ben2-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ben2/pytorch-Base-single_device-inference]

## Result
FAIL — PCC=nan on TT silicon: BEN2's Swin Transformer + MCLM cross-attention produces constant/NaN output; root cause op requires per-op diagnosis

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ben2-swin-wrong-tt-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

The DeprecationWarning was the terminal symptom. The actual failure chain was:

1. `ImportError: cannot import name 'BEN_Base' from 'ben2' (.../tt_forge_models/ben2/__init__.py)`
   — The `tt_forge_models/ben2/` model directory shadowed the real `ben2` pip package
     because the dynamic loader inserts `models_root` at `sys.path[0]`.
   — Additionally, `ben2` had no `requirements.txt` so the package was never installed.

2. `AttributeError: module 'spacy' has no attribute 'Language'` (from `datasets._dill`)
   — `huspacy/pytorch/loader.py` had a module-level `import spacy` that ran at
     collection time. `tt_forge_models/spacy/` is a namespace package that shadowed
     the real spacy library (which is not installed), installing the empty stub in
     `sys.modules['spacy']`. `datasets._dill` then crashed during `load_dataset`.

3. `RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`
   — The loader ignored `dtype_override=bfloat16` for the model but applied it to
     inputs, causing dtype mismatch in Conv2d.

4. `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != BF16`
   — After applying `dtype_override` to the model: BEN2's `BasicLayer.forward`
     creates the Swin window-attention mask with `torch.zeros(..., device=x.device)`
     (no `dtype` argument), producing float32 regardless of input dtype. When the
     model is bfloat16, this float32 mask is added to bfloat16 attention scores,
     promoting to float32. The subsequent `attn @ v` (float32 @ bfloat16) fails.

5. After running in float32 (both model and inputs): test runs to completion on
   TT silicon but evaluator reports `pcc=nan (invalid value)`. CPU reference output
   is valid (std=0.47, no NaN). TT output is constant or NaN — a compiler-stack
   bug causing wrong computation for BEN2's architecture (Swin Transformer backbone +
   MCLM multi-field cross-attention).

## Root cause
Three sequential loader bugs were fixed (bugs 1–4 above). The remaining silicon
failure (bug 5) is in the compiler stack (tt-mlir / tt-metal). BEN2 uses a Swin
Transformer backbone feeding a custom multi-scale cross-attention decoder (MCLM).
When compiled and run on TT hardware in float32, the model executes without error
but produces a constant/NaN output tensor, yielding pcc=nan. The specific failing
operation could not be determined without per-op instrumentation; candidates include:

- `F.adaptive_avg_pool2d` in MCLM (used to pool `[1,128,32,32]` → 3 different
  spatial scales; if this returns zeros, the cross-attention key/value is zero).
- The Swin window-attention mask (`img_mask = torch.zeros(..., device=x.device)`)
  created dynamically in `BasicLayer.forward` with potential in-place fill issues
  under TorchDynamo tracing.
- `torch.cumsum` in `PositionEmbeddingSine.__call__` on a boolean tensor, combined
  with subsequent sin/cos — may lower incorrectly.

CPU reference produces valid, non-constant output (min=0, max=1, std=0.47).

## Fix
Loader fixes committed to `remediation/ben2-pytorch-Base-single_device-inference`
in `tt_forge_models`:

1. **`ben2/requirements.txt`** (new file): adds
   `git+https://github.com/PramaLLC/BEN2.git` — the `ben2` package is not on PyPI.

2. **`ben2/pytorch/loader.py`** — `load_model`: temporarily removes `models_root`
   from `sys.path` before `from ben2 import BEN_Base`, then restores it. This
   bypasses the `tt_forge_models/ben2/` namespace shadow.

3. **`ben2/pytorch/loader.py`** — both `load_model` and `load_inputs`: do NOT apply
   `dtype_override`, keeping the model and inputs in native float32. BEN2 has a
   bfloat16 incompatibility: `BasicLayer.forward` creates float32 temporaries
   regardless of model dtype.

4. **`huspacy/pytorch/loader.py`**: moved `import spacy` from module level into
   `_load_nlp()` to prevent spacy namespace pollution during pytest collection.

The silicon PCC=nan failure is a compiler-stack bug that requires further diagnosis.
No fix is attempted; the test is left as FAIL.

## Tier B justification
Indicator: `internal-error-unknown-mechanism`

The model executes on TT silicon without a runtime error (no TT_THROW, no INTERNAL
Error code 13, no timeout) but produces wrong output (pcc=nan). The specific
operation causing the wrong computation is unknown without per-op debugging. BEN2's
architecture combines a Swin Transformer (with dynamic window-attention masks), a
custom 5-input multi-field cross-attention (MCLM with adaptive pooling and
PositionEmbeddingSine), and a multi-scale decoder — multiple candidate failure
points across a complex graph. Identifying and fixing the specific bug would require
per-op comparison tooling applied to a compiled, silicon-run graph, which is
diagnosis-first work beyond a single-file Tier A fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    429.31s (0:07:09) including compilation + 2 inference runs of ~1.24s each
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/ben2/requirements.txt` (new)
- `tt_forge_models/ben2/pytorch/loader.py`
- `tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8146127e4f37fcfa84267996c8e6407736e9e2ce |
