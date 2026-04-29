# Remediation Summary: beitv2-pytorch-Base_Patch16_224-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[beitv2/pytorch-Base_Patch16_224-single_device-inference]

## Result
FAIL — loader fix applied (spacy namespace bug); test now runs on silicon but fails PCC (0.29 vs 0.99) — Tier B compiler-stack bug remaining

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
beitv2-pcc-below-threshold-vit-precision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (from collection):
```
AttributeError: module 'spacy' has no attribute 'Language'
```
Reported as: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

After loader fix, failure is:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.29145954291819903. Required: pcc=0.99.
```

## Root cause
**Bug 1 (fixed — loader):** `huspacy/pytorch/loader.py` had a top-level `import spacy`. The `setup_test_discovery` path adds `tt_forge_models/` to `sys.path`. A directory named `spacy/` exists inside `tt_forge_models/` (the spacy model loader directory). Python 3 treats it as a namespace package, so `import spacy` during test collection resolves to this stub rather than the real spacy library, putting a broken object in `sys.modules['spacy']`. The `datasets` library's `_dill.py` checks `if 'spacy' in sys.modules` then accesses `spacy.Language`, which raises `AttributeError` on the namespace stub. Fix: move `import spacy` inside `_load_nlp()` so it is deferred to model-load time.

**Bug 2 (unfixed — compiler):** After the loader fix, the model compiles and runs on silicon but produces PCC = 0.29 against CPU reference. BEiT v1 (same architecture family, `timm`) has the same known issue in the test config (`assert_pcc: false # Fell to <0.3 after experimental compile`). PCC = 0.29 is far outside bfloat16 accumulation range for a classification model of this size, indicating a real numerical correctness bug in the compiler stack. The specific op or pass causing the divergence has not been identified.

## Fix
**Bug 1 (applied):**
- File: `huspacy/pytorch/loader.py`
- Repo: `tt_forge_models`
- Branch: `remediation/beitv2-pytorch-Base_Patch16_224-single_device-inference`
- Commit: `9c88f15c58556bbfcbb17e5ea5dc438227781a91`
- Change: removed top-level `import spacy`; added `import spacy` inside `_load_nlp()` method body.

**Bug 2 (proposed fix):** Investigate which op produces divergent output by comparing tt-device vs CPU op-by-op on BEiT v2. This is likely the same regression as BEiT v1. Fix would live in tt-mlir (lowering pass) or tt-metal (kernel).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

PCC = 0.29 on a vision transformer indicates a significant numerical correctness failure in the compiled execution path. The diverging op has not been isolated; diagnosis (not a straightforward single-function fix) must precede any fix. BEiT v1 exhibits identical PCC collapse and has remained unresolved in the test config.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    36.95s
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (tt_forge_models) — removed top-level `import spacy`, added lazy import inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9c88f15c58556bbfcbb17e5ea5dc438227781a91 |
