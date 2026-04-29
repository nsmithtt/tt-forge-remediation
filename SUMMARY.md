# Remediation Summary: convnextv2/pytorch-Nano_FCMAE_FT_IN22K_IN1K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[convnextv2/pytorch-Nano_FCMAE_FT_IN22K_IN1K-single_device-inference]

## Result
SILICON_PASS — lazy huspacy spacy import eliminates namespace package shadowing

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
huspacy-spacy-namespace-package-shadowing

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
(reported as `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`)

Inside `venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42` during
`load_dataset("huggingface/cats-image")` in the convnextv2 loader's `load_inputs`.

## Root cause
`dynamic_loader.setup_models_path()` adds `models_root` (the `tt_forge_models`
directory) to `sys.path` at position 0 so loader relative imports work. Because
Python 3 supports namespace packages, any subdirectory of `models_root` is now
importable as a top-level module name. `tt_forge_models/spacy/` exists as a
model-loader directory; since it has no `__init__.py`, Python creates it as a
namespace package when any code does `import spacy`.

The `huspacy/pytorch/loader.py` had a module-level `import spacy` statement.
During test collection, the dynamic loader imports all loader files to discover
variants. When the huspacy loader is exec'd, its `import spacy` triggers
namespace-package creation: `sys.modules['spacy']` is set to the stub namespace
package (which has no `Language` attribute) instead of raising `ImportError`.

Later, when `datasets._dill` processes the `load_dataset` hash for convnextv2's
input, it checks `if "spacy" in sys.modules:` (True — the stub is there) and
then accesses `spacy.Language`, producing `AttributeError`.

## Fix
In `tt_forge_models`, moved the module-level `import spacy` in
`huspacy/pytorch/loader.py` inside the `_load_nlp()` method (deferred until
the model is actually loaded). This prevents the namespace package from being
created during test collection.

File changed: `huspacy/pytorch/loader.py`
Commit: `db41c0530cc7c58107793da6b1d721b9fb91e101`
Branch: `remediation/convnextv2-pytorch-Nano_FCMAE_FT_IN22K_IN1K-single_device-inference`
Repo: `tenstorrent/tt-forge-models`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    88.63s
- Tier A attempts: N/A

## Files changed
- `huspacy/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3e81863feea42339be0d01e899cda6385ae2fed2 |
| tt-forge-models | db41c0530cc7c58107793da6b1d721b9fb91e101 |
