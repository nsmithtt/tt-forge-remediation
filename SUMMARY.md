# Remediation Summary: datacomp_clip-pytorch-ViT_B_16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[datacomp_clip/pytorch-ViT_B_16-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
huspacy-top-level-spacy-import-shadows-namespace-package

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
third_party/tt_forge_models/datacomp_clip/pytorch/loader.py:112: in load_inputs
    dataset = load_dataset("huggingface/cats-image")["test"]
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: in save
    if issubclass(obj_type, spacy.Language):
AttributeError: module 'spacy' has no attribute 'Language'
```

## Root cause
The test infrastructure (dynamic_loader.py) adds `third_party/tt_forge_models` to
`sys.path` during model discovery. This directory contains a `spacy/` subdirectory
(holding the `spacy/es_core_news_md` model), which Python treats as a namespace
package named `spacy`. The `huspacy/pytorch/loader.py` had a top-level `import spacy`
statement. When this loader was imported during test collection, Python resolved
`spacy` to the namespace package `tt_forge_models/spacy/` rather than the real spacy
library, adding the incomplete stub to `sys.modules['spacy']`. Later, when
`datacomp_clip` called `load_dataset("huggingface/cats-image")`, the
`datasets/_dill.py` Pickler checked `if "spacy" in sys.modules`, found the stub,
imported it, and tried `spacy.Language` — which does not exist on the namespace
package, raising `AttributeError`.

## Fix
`tt_forge_models/huspacy/pytorch/loader.py`: Moved `import spacy` from module
top-level into the `_load_nlp()` method (where it is used). This prevents spacy from
being imported during test collection, so the namespace package is never added to
`sys.modules` and `datasets._dill` no longer encounters the stub.

Branch: `remediation/datacomp_clip-pytorch-ViT_B_16-single_device-inference`
Commit: db41c0530cc7c58107793da6b1d721b9fb91e101

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    86.42s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7bb6947a33b52bcd932a458709f93be9c74c9f00 |
| tt-forge-models | db41c0530cc7c58107793da6b1d721b9fb91e101 |
