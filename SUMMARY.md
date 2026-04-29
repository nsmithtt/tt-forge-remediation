# Remediation Summary: clip_vit_base_patch32_resisc45-pytorch-Base_Patch32_RESISC45-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[clip_vit_base_patch32_resisc45/pytorch-Base_Patch32_RESISC45-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
huspacy-module-level-spacy-import-namespace-collision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: in save
    if issubclass(obj_type, spacy.Language):
                            ^^^^^^^^^^^^^^
AttributeError: module 'spacy' has no attribute 'Language'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Root cause
During pytest test collection, `test_models.py` imports all model loaders via
`TorchDynamicLoader.setup_test_discovery()`. The `huspacy/pytorch/loader.py`
had a top-level `import spacy` statement (line 14). Because `setup_models_path`
adds `models_root` (the `tt_forge_models` directory) to `sys.path`, and
`tt_forge_models/spacy/` is a directory on that path (the spaCy model loader
family), Python's namespace package machinery registers
`sys.modules["spacy"]` pointing to that directory instead of the real spaCy
library. Later, when `clip_vit_base_patch32_resisc45/pytorch/loader.py` calls
`load_dataset("huggingface/cats-image")`, the `datasets._dill.Pickler.save()`
method checks `if "spacy" in sys.modules` — finds it True — then tries
`spacy.Language` on the namespace package, which has no such attribute,
producing the `AttributeError`.

## Fix
Moved the top-level `import spacy` inside `_load_nlp()` in
`huspacy/pytorch/loader.py` so it is only executed when the HuSpaCy model
is actually loaded, not during test collection. This prevents the namespace
package from being registered in `sys.modules["spacy"]` and eliminates the
collision.

File changed: `huspacy/pytorch/loader.py` in `tt-forge-models`.
Remediation branch: `remediation/clip-vit-base-patch32-resisc45-pytorch-Base-Patch32-RESISC45-single-device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    43.70s
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a40837460e5c6457030616271c7978bade6c53c9 |
| tt-forge-models | cfca91dff66ca2b1555b39b1e92d9ae214d749c1 |
