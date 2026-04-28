# Remediation Summary: aimv2_pytorch-Large_Patch14_224_Apple_PT_Dist-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aimv2/pytorch-Large_Patch14_224_Apple_PT_Dist-single_device-inference]

## Result
SILICON_PASS — loader fix: move top-level `import spacy` in huspacy loader inside `_load_nlp()` to prevent spacy namespace package from polluting sys.modules

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-collision-huspacy-module-level-import

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

Full error (underlying cause):
```
AttributeError: module 'spacy' has no attribute 'Language'
```
in `datasets/utils/_dill.py:42` during `load_dataset("huggingface/cats-image")` call inside `aimv2/pytorch/loader.py:load_inputs`.

## Root cause
`huspacy/pytorch/loader.py` imported `spacy` at the module level (line 14). During pytest collection, `dynamic_loader.py` adds `tt_forge_models/` to `sys.path`, and `tt_forge_models/spacy/` exists as a directory (the spaCy model loader subtree). Python treats it as a namespace package and installs it as `sys.modules['spacy']`, shadowing the real spacy library.

When the aimv2 loader subsequently calls `load_dataset("huggingface/cats-image")`, the datasets library fingerprints the call arguments via `datasets/utils/_dill.py`. That file checks `if "spacy" in sys.modules: import spacy; if issubclass(obj_type, spacy.Language):`. Since the stub namespace package is in `sys.modules['spacy']`, the import succeeds but `spacy.Language` does not exist on the stub, raising `AttributeError`.

The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` shown as the reported failure is an unrelated SWIG warning that appears as the last stderr line of the pytest process, which the failure-capture infrastructure used as the error summary.

## Fix
In `tt-forge-models/huspacy/pytorch/loader.py`:
- Removed top-level `import spacy` (line 14)
- Added `import spacy` inside `_load_nlp()` method (deferred until model loading)

This prevents the `tt_forge_models/spacy/` namespace package from entering `sys.modules` during collection, so `datasets._dill` no longer finds a stub spacy.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    56.61s
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/huspacy/pytorch/loader.py` — move `import spacy` inside `_load_nlp()` to prevent namespace collision at collection time

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 2fabb4f65b47cc889fe658718f4fa11a10c10720 |
