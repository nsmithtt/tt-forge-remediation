# Remediation Summary: efficientnet_gn-pytorch-Test_Efficientnet_Gn.r160_in1k-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[efficientnet_gn/pytorch-Test_Efficientnet_Gn.r160_in1k-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-collision-load-dataset

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
(The failure message reported was `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` — this is the last line printed by pytest after the actual failure, which is the spacy AttributeError.)

## Root cause
`huspacy/pytorch/loader.py` had a module-level `import spacy` at line 14. The `dynamic_loader.py` adds `tt_forge_models/` to `sys.path` at collection time, which makes `tt_forge_models/spacy/` (the Spanish spaCy model loader directory) a namespace package that shadows the real `spacy` library. When any model loader calls `load_dataset(...)`, `datasets/_dill.py` tries `issubclass(obj_type, spacy.Language)` during pickling and fails because the stub `spacy` module has no `Language` attribute. This is a loader bug.

## Fix
In `tt-xla/third_party/tt_forge_models` (`tenstorrent/tt-forge-models`), branch `remediation/efficientnet_gn-pytorch-Test_Efficientnet_Gn.r160_in1k-single_device-inference`:

- `huspacy/pytorch/loader.py`: removed top-level `import spacy`; moved the import inside `_load_nlp()` so it only executes at model-load time, not during pytest collection.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    76.83s
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (in tenstorrent/tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a24d956a605d75bcc28695686f2639b08e808ab8 |
| tt-forge-models | 5aaf310792d364b4c338300359452002ef940f94 |
