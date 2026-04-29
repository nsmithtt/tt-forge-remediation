# Remediation Summary: bioclip_2-pytorch-ViT_H_14_2_5-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bioclip_2/pytorch-ViT_H_14_2_5-single_device-inference]

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
FAILED tests/runner/test_models.py::test_all_models_torch[bioclip_2/pytorch-ViT_H_14_2_5-single_device-inference]
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Root cause
The `huspacy/pytorch/loader.py` in `tt_forge_models` imports `spacy` at the top
level (`import spacy` outside any function). During pytest collection, the dynamic
loader imports every discovered loader module, including `huspacy`. At that point,
Python's import system resolves `import spacy` to the `tt_forge_models/spacy/`
directory on `sys.path`, which is a namespace package (no `__init__.py`) for
spaCy model loaders — not the real `spacy` NLP library. This namespace stub has
no `Language` attribute and gets registered in `sys.modules['spacy']`.

Later, when `bioclip_2`'s `load_inputs` calls `datasets.load_dataset(...)`, the
HuggingFace `datasets` library's `_dill.py` serializer checks
`if "spacy" in sys.modules` (true, because the stub is there), then imports and
dereferences `spacy.Language` — raising `AttributeError` because the stub has no
such attribute.

The bioclip_2 model itself is not the source of the bug; the root cause is the
`huspacy` loader's top-level import polluting `sys.modules` during collection.

## Fix
`huspacy/pytorch/loader.py` in `tt_forge_models` (branch
`remediation/bioclip_2-pytorch-ViT_H_14_2_5-single_device-inference`):
- Removed top-level `import spacy`
- Moved `import spacy` inside `_load_nlp()` so it only executes when the
  HuSpaCy model is actually loaded, not at collection time

This prevents the `tt_forge_models/spacy/` namespace package from entering
`sys.modules` during test collection, allowing `datasets.load_dataset` in
`bioclip_2`'s `load_inputs` to serialize correctly.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    98.00s (0:01:38)
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` in tt_forge_models (remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f5d66bd19911d511a4a48cb1009b6fc5a07e30cb |
| tt-forge-models | e9db04d3b431e26d0afbcc278bbd11e4a93470d9 |
