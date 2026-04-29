# Remediation Summary: beit_image_orientation_fixer/pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[beit_image_orientation_fixer/pytorch-Base-single_device-inference]

## Result
FAIL — loader fix applied (spacy namespace collision); test runs on silicon but fails PCC (0.40 vs 0.99) — same Tier B compiler-stack bug as BEiT and BEiTv2

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
beit-pcc-0186-deterministic-compiler-regression

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Actual failure after loader fix:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.39958979358908814. Required: pcc=0.99.
```

## Root cause

**Issue 1 (FIXED — loader layer):** `huspacy/pytorch/loader.py` imported `spacy` at module level. During pytest collection, all model loaders are imported. The `dynamic_loader.py` adds `tt_forge_models/` to `sys.path`, and a directory named `spacy/` exists inside `tt_forge_models/`. Python 3 treats it as a namespace package, so the module-level `import spacy` resolves to the stub rather than the real SWIG-wrapped spacy library, installing a broken object in `sys.modules['spacy']`. When `beit_image_orientation_fixer.load_inputs` calls `load_dataset(...)`, `datasets/utils/_dill.py:42` checks `if issubclass(obj_type, spacy.Language)` — this raises `AttributeError: module 'spacy' has no attribute 'Language'`. Fix: move `import spacy` inside `_load_nlp()` so it is deferred to model-load time, after pytest collection finishes and spacy has not been imported from the stub.

**Issue 2 (NOT FIXED — compiler stack, Tier B):** After the loader fix, the model compiles and runs on TT silicon but produces PCC = 0.399 against CPU reference. `amaye15/Beit-Base-Image-Orientation-Fixer` is a fine-tuned BEiT-Base model (12 layers, 768 hidden, 197 tokens from 224×224 image). PCC = 0.4 is far outside any bfloat16 accumulation range for a 12-layer classifier — this is the same deterministic numerical correctness bug seen in:

- `beit/pytorch-Base-single_device-inference`: `assert_pcc: false # Fell to <0.3 after experimental compile` in test config
- `beit/pytorch-BEiT_Large_Patch16_384_IN22K_FT_IN22K_IN1K-single_device-inference`: PCC=0.186 (report dated today, fingerprint `beit-pcc-0186-deterministic-compiler-regression`)
- `beitv2/pytorch-Base_Patch16_224-single_device-inference`: PCC=0.29

All BEiT family models produce deeply wrong output (PCC < 0.5) on TT silicon. The underlying op producing the divergence has not been isolated across any of these reports. The mechanism is unknown — diagnosis must precede any fix.

## Fix
**Issue 1 (applied — loader):**
- File: `huspacy/pytorch/loader.py`
- Repo: `tt_forge_models`
- Branch: `remediation/beit_image_orientation_fixer-pytorch-Base-single_device-inference`
- Commit: `c013c465b3328ba618349bf35f0658fb8be6216d`
- Change: removed top-level `import spacy`; added `import spacy` inside `_load_nlp()` method body.

**Issue 2 (proposed fix):** Isolate the diverging op in BEiT attention by running tt-device vs CPU op-by-op on the 12-layer BEiT-Base. Candidate ops: SDPA with 197 non-tile-aligned sequence length (197 % 32 = 5), attention masking in TTNN, or layer norm lowering. The `shouldUseDecode()` guard in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` now requires `key_seq_len % 32 == 0` for SDPA decode, but BEiT encoder uses `query_seq_len = 197` (not 1), so decode path shouldn't apply. The bug likely lives in the prefill SDPA kernel or matmul lowering for the 197-token sequence. Fix would be in tt-mlir or tt-metal.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

PCC = 0.399 for a 12-layer classifier is a deep numerical correctness failure. The diverging op has not been isolated across three prior BEiT reports. The root cause mechanism is unknown; diagnosis (per-op comparison across 12 transformer layers) must precede any targeted fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    118.15s (1:58)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py` — moved `import spacy` from module scope to inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ab2a310eb17272d9745bfce46d2ee70458239f27 |
| tt-forge-models | c013c465b3328ba618349bf35f0658fb8be6216d |
