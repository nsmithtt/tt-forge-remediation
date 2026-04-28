# Remediation Summary: efficientnet_ln-pytorch-Test_Efficientnet_Ln.r160_in1k-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[efficientnet_ln/pytorch-Test_Efficientnet_Ln.r160_in1k-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-package-shadows-real-spacy

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8477511245842417. Required: pcc=0.95.

During reproduction, the test failed earlier with:
E   AttributeError: module 'spacy' has no attribute 'Language'

(The PCC failure in CI was the downstream symptom of bad inputs from the spacy crash; see Root cause.)

## Root cause

The `tt_forge_models/spacy/` model directory creates a Python namespace package
when `models_root` (`third_party/tt_forge_models/`) is added to `sys.path` by
`DynamicLoader.setup_models_path()`. The `huspacy/pytorch/loader.py` had a
top-level `import spacy` which, during test collection, resolved to this fake
namespace package instead of the real spaCy NLP library.

This put a broken `spacy` module (with no `Language` attribute) into
`sys.modules['spacy']`. Later, when the `efficientnet_ln` test called
`load_dataset("huggingface/cats-image")`, the `datasets` library's dill
serializer checked `if "spacy" in sys.modules` and then tried to access
`spacy.Language`, raising `AttributeError` which caused dataset loading to fail
with corrupted/zero inputs — producing the PCC collapse seen in CI.

The `spacy/es_core_news_md/pytorch/loader.py` already used a lazy `import spacy`
(inside a method), which is the correct pattern. Only `huspacy` used a
module-level import.

## Fix

In `tt_forge_models/huspacy/pytorch/loader.py`:
- Removed the top-level `import spacy`.
- Added `import spacy` inside the `_load_nlp()` method, which is the only
  method that uses the `spacy` library at runtime.

This prevents the `spacy` namespace package from being registered in
`sys.modules` during test collection, eliminating the interference with
`datasets`' dill serializer in all model tests that load HuggingFace datasets.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    66.61s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4eff379d3171022bede2f05b7860bc692c98e26d |
| tt-forge-models | d96d7c4709a9a2f566df139c81db712b2c4d8828 |
