# Remediation Summary: cafe_aesthetic-pytorch-Default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cafe_aesthetic/pytorch-Default-single_device-inference]

## Result
FAIL — BeitForImageClassification produces PCC=-1.0 on TT hardware; same bug as beit/pytorch-Base-single_device-inference (assert_pcc: false)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
beit-wrong-output-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (loader):
  AttributeError: module 'spacy' has no attribute 'Language'

After loader fix, second failure (compiler):
  AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=-1.0. Required: pcc=0.99.

## Root cause

**Stage 1 — loader bug (fixed):** `huspacy/pytorch/loader.py` had a top-level `import spacy` executed during test collection via `TorchDynamicLoader.setup_test_discovery`. Because `models_root` (`third_party/tt_forge_models`) is inserted at the front of `sys.path` during loader discovery, Python resolved the bare `import spacy` to the `tt_forge_models/spacy/` directory (a namespace package with no `__init__.py`), not the real spaCy NLP library. This poisoned `sys.modules['spacy']` with a bare namespace package lacking a `Language` attribute. Later, when `cafe_aesthetic/pytorch/loader.py` called `load_dataset("huggingface/cats-image", ...)`, the `datasets/_dill.py` serialization path checked `"spacy" in sys.modules` (True), imported it, and crashed on `spacy.Language`.

Fix: move `import spacy` inside `_load_nlp()` so it runs only when the huspacy model is actually loaded (which never happens for this test run).

**Stage 2 — compiler bug (unfixed):** After fixing the loader, the test compiles and runs `cafeai/cafe_aesthetic` (a `BeitForImageClassification` model, architecture identical to `beit/pytorch-Base`). The TT device produces outputs with PCC=-1.0 relative to CPU. For a 2-class classifier, PCC=-1.0 is exact and indicates the TT output values are the logical reverse of the CPU output (either negated or swapped), not random noise. This is the same bug already noted in the test config for `beit/pytorch-Base-single_device-inference` (`assert_pcc: false # Fell to <0.3 after experimental compile`).

The BEiT architecture uses learned relative position bias in its attention layers, indexed via a complex gather over a 2D offset table. The incorrect PCC strongly suggests the compiler either miscompiles the gather/indexing op used for relative position bias, or miscompiles the attention softmax or output projection in a systematic way (sign flip or transposition). The root cause within tt-mlir is unknown and requires diagnosis.

## Fix

**Loader fix (committed):** `huspacy/pytorch/loader.py` — removed top-level `import spacy`; added lazy `import spacy` inside `ModelLoader._load_nlp()`.

Remediation branch: `remediation/cafe_aesthetic-pytorch-Default-single_device-inference` in `tt-forge-models`.

**Compiler bug (proposed, not implemented):** The wrong PCC for BEiT models is in the tt-mlir lowering of BEiT attention. The fix would need to identify which specific op (likely relative position bias gather, attention matmul, or output projection) is being lowered incorrectly, then patch the corresponding lowering pass in tt-mlir.

## Tier B justification

Indicator: `internal-error-unknown-mechanism`

The BEiT PCC failure is a systematic numerical error (PCC=-1.0 for 2-class output, consistent with a sign flip or value swap) rather than a missing op or a single bounded fix. The same failure already exists for `beit/pytorch-Base` with no documented root cause in the codebase. Diagnosing which specific attention sub-operation (relative position bias indexing, QKV projection, attention masking, output projection) is incorrect requires running bisection experiments on the compiler pipeline — a diagnosis-first effort that crosses the Tier A single-attempt boundary.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    44.43s (with loader fix applied, fails at PCC check)
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (tt-forge-models): moved `import spacy` from module level to inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4def0f02cec4e748f538628029c6977f971012a1 |
| tt-forge-models | 73a68a19f8b9db665f7db6e22c47297725bcf396 |
