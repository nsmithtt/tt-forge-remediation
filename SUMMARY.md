# Remediation Summary: bioclip_2-pytorch-ViT_L_14-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bioclip_2/pytorch-ViT_L_14-single_device-inference]

## Result
SILICON_PASS — loader bug fixed: huspacy top-level spacy import shadowed by tt_forge_models/spacy/ namespace package

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
huspacy-top-level-spacy-import-namespace-package-conflict

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
E   AttributeError: module 'spacy' has no attribute 'Language'
FAILED tests/runner/test_models.py::test_all_models_torch[bioclip_2/pytorch-ViT_L_14-single_device-inference]
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Root cause
The `huspacy/pytorch/loader.py` in `tt_forge_models` imports `spacy` at the top
level (`import spacy` outside any function). During pytest collection,
`dynamic_loader.py` adds `third_party/tt_forge_models` to `sys.path` so that
model-relative imports work. At that point, Python's import system resolves
`import spacy` to the `tt_forge_models/spacy/` directory — a namespace package
(no `__init__.py`) that holds spaCy model loaders, not the real spaCy NLP
library. This namespace stub has no `Language` attribute and gets registered as
`sys.modules['spacy']`.

Later, when `bioclip_2`'s `load_inputs` calls `datasets.load_dataset(...)`, the
HuggingFace `datasets` library's `_dill.py` serializer checks
`if "spacy" in sys.modules` (True, because the stub is present), then tries
`spacy.Language` — raising `AttributeError` because the stub exposes none.

The bioclip_2 model itself is correct; the root cause is the `huspacy` loader's
top-level import polluting `sys.modules` during collection.

## Fix
`huspacy/pytorch/loader.py` in `tt_forge_models` (branch
`remediation/bioclip_2-pytorch-ViT_L_14-single_device-inference`):
- Removed top-level `import spacy`
- Moved `import spacy` inside `_load_nlp()` so it only executes when the
  HuSpaCy model is actually loaded, not at collection time

This prevents the `tt_forge_models/spacy/` namespace package from entering
`sys.modules` during test collection, allowing `datasets.load_dataset` in
`bioclip_2`'s `load_inputs` to serialize correctly.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    129.38s (0:02:09)
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` in tt_forge_models (remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c0e63293e6eeb91e5fe738a3ad0a7cf7ff97203e |
| tt-forge-models | c698ff8cd228b164ca205134afe9ab91ee3c6d25 |
