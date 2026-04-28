# Remediation Summary: ashishupadhyay_nsfw_detection-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ashishupadhyay_nsfw_detection/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
huspacy-spacy-namespace-module-level-import

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: module 'spacy' has no attribute 'Language'

(reported as: sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute — the DeprecationWarning is a pytest session footer unrelated to the actual failure)

## Root cause
`tt_forge_models/spacy/` exists as a directory (model namespace for spacy/es_core_news_md) and, because `dynamic_loader.py` prepends `tt_forge_models/` to `sys.path`, it becomes a Python namespace package that shadows the real `spacy` library. `huspacy/pytorch/loader.py` had `import spacy` at module level; during pytest collection every loader is imported, so this runs and installs the empty stub namespace package into `sys.modules['spacy']`. Later, when `ashishupadhyay_nsfw_detection/loader.py:load_inputs()` calls `load_dataset("huggingface/cats-image")`, `datasets/utils/_dill.py:39` checks `if "spacy" in sys.modules:` (True — the stub), then does `spacy.Language` (AttributeError — stub has no attributes). The real spacy is not installed in the venv, so if no stub were in `sys.modules`, the datasets code's guard would correctly skip the spacy path.

## Fix
`huspacy/pytorch/loader.py` in `tt_forge_models` — moved `import spacy` from module level into the `_load_nlp()` method so it is not executed during test collection. This prevents the namespace-package stub from being placed into `sys.modules['spacy']`, causing `datasets._dill` to correctly skip its spacy-specific pickling path.

Branch: `remediation/ashishupadhyay_nsfw_detection-pytorch-Base-single_device-inference` in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    50.03s
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (tt_forge_models): remove top-level `import spacy`; add lazy `import spacy` inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 98cca52217feb36387f62bd133fe623c0611d3cf |
| tt-forge-models | f4e699e29d05426e82bda8735829706a20ab257d |
