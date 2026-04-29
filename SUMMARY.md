# Remediation Summary: convnext-pytorch-Base_CLIP_LAION2B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[convnext/pytorch-Base_CLIP_LAION2B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-shadows-real-package

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: module 'spacy' has no attribute 'Language'

(surfaced as: sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute)

Full traceback:
```
third_party/tt_forge_models/convnext/pytorch/loader.py:119: in load_inputs
    dataset = load_dataset("huggingface/cats-image", split="test")
venv/lib/python3.12/site-packages/datasets/utils/_dill.py:42: in save
    if issubclass(obj_type, spacy.Language):
AttributeError: module 'spacy' has no attribute 'Language'
```

## Root cause
Two loader bugs, both in the loader layer:

1. **spacy namespace shadowing** (`tests/runner/utils/dynamic_loader.py`):
   `setup_models_path` added `models_root` (the `tt_forge_models/` directory) to
   `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`,
   Python creates a namespace package for `spacy` from it. This shadows the real spaCy
   library, so when `datasets._dill.save()` checks `if issubclass(obj_type, spacy.Language)`
   it finds a broken namespace package and raises `AttributeError`. Relative imports in
   loaders work via `__package__` and the manually-registered `tt_forge_models` module
   (with `__path__ = [models_root]`) — `sys.path` insertion is not needed.

2. **dtype mismatch in convnext load_model** (`convnext/pytorch/loader.py`):
   `timm.create_model` returns a float32 model. When the framework calls
   `load_model(dtype_override=torch.bfloat16)`, model weights (including biases) stay
   float32 while inputs are cast to bfloat16, causing:
   `RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`.
   The model must be cast to `dtype_override` after `model.eval()`.

## Fix
**Fix 1** — `tt-xla: tests/runner/utils/dynamic_loader.py`
Removed `sys.path.insert(0, models_root)` from `setup_models_path`. Added comment
explaining that relative imports work via `__package__` + `tt_forge_models.__path__`
and that adding `models_root` to `sys.path` causes namespace package pollution.

**Fix 2** — `tt-forge-models: convnext/pytorch/loader.py`
Added `model = model.to(dtype_override)` after `model.eval()` in `load_model`, so
the timm model is cast to the requested dtype before being stored and returned.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    88.38s
- Tier A attempts: N/A

## Files changed
- `tt-xla: tests/runner/utils/dynamic_loader.py`
- `tt-forge-models: convnext/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 87015a15d5b04a62f068a8f83704ced54936794d |
| tt-forge-models | e81930d7077db1e8ea664c71dbca3c2ccbe980b9 |
