# Remediation Summary: dolma3_fasttext_quality_classifier/sequence_classification/pytorch-dolma3-fasttext-quality-classifier-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dolma3_fasttext_quality_classifier/sequence_classification/pytorch-dolma3-fasttext-quality-classifier-single_device-inference]

## Result
SILICON_PASS — lazy import + requirements.txt fixed namespace shadowing of fasttext

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
fasttext-namespace-package-shadowing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: module 'fasttext' has no attribute 'load_model'

(The ticket listed the symptom as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`, which is a trailing warning printed after the test summary; the actual pytest failure is the AttributeError above.)

## Root cause
`tt_forge_models/fasttext/` is a model directory (no `__init__.py`) that Python 3 treats as a namespace package when `models_root` is inserted at `sys.path[0]` by the dynamic loader. The dolma3 loader did `import fasttext` at module level (during test collection), which resolved to this empty namespace package instead of the real pip package. Since `fasttext-wheel` was also not declared in a `requirements.txt`, the real package was never installed, and `fasttext.load_model()` raised `AttributeError`.

## Fix
Two changes in `tt_forge_models` on branch `remediation/dolma3-fasttext-quality-classifier`:

1. **`dolma3_fasttext_quality_classifier/sequence_classification/pytorch/requirements.txt`** (new file): declares `fasttext-wheel>=0.9.2` so `RequirementsManager` installs the real package before the test runs.

2. **`dolma3_fasttext_quality_classifier/sequence_classification/pytorch/loader.py`**: removed top-level `import fasttext`; moved it to a lazy `import fasttext` inside `_load_fasttext_model()`. This ensures the import runs after `RequirementsManager` has installed the package (and purged the stale namespace-package entry from `sys.modules`), so Python finds the real `site-packages/fasttext/` regular package rather than the namespace package.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    44.44s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/dolma3_fasttext_quality_classifier/sequence_classification/pytorch/requirements.txt` (new)
- `tt_forge_models/dolma3_fasttext_quality_classifier/sequence_classification/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 52ef5716e04a8d59dfb827e389504dc372f51afb |
