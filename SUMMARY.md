# Remediation Summary: biomedclip/pytorch-ViT_Base_Patch16_224-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[biomedclip/pytorch-ViT_Base_Patch16_224-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-package-sys-modules-pollution

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: module 'spacy' has no attribute 'Language'
```

Reported as: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

Full traceback:
```
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: if issubclass(obj_type, spacy.Language):
E   AttributeError: module 'spacy' has no attribute 'Language'
```

## Root cause
During test discovery, `setup_models_path` adds `models_root` (`tt_forge_models/`) to `sys.path`. The `tt_forge_models/` directory contains a `spacy/` subdirectory (the tt-forge spaCy model family). With `models_root` in `sys.path`, Python treats `tt_forge_models/spacy/` as a namespace package importable as `import spacy`.

`huspacy/pytorch/loader.py` had a top-level `import spacy`, which executed during test discovery via `import_model_loader`. Python found the `tt_forge_models/spacy/` namespace package (not the real spaCy library), creating a module object with no attributes — including no `Language`. This partial module was placed in `sys.modules['spacy']`.

When the biomedclip loader later called `load_dataset("huggingface/cats-image")`, the `datasets` library's `_dill.py` checks `if "spacy" in sys.modules` and then calls `issubclass(obj_type, spacy.Language)`. Since `sys.modules['spacy']` was the empty namespace package (not the real spaCy), this crashed with `AttributeError: module 'spacy' has no attribute 'Language'`.

## Fix
In `huspacy/pytorch/loader.py` (repo: tt-forge-models), moved `import spacy` from the module top level into the `_load_nlp()` method where it is actually used. This makes the import lazy: it only runs when the huspacy model is actually loaded, not during test discovery. The namespace package is therefore never placed in `sys.modules['spacy']` during discovery, and the biomedclip `load_dataset` call proceeds normally.

File changed: `huspacy/pytorch/loader.py`
Branch: `remediation/biomedclip-pytorch-ViT_Base_Patch16_224-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    47.17s
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4026d39a6a412b8d5aa2948a3a511e51efc31792 |
