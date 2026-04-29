# Remediation Summary: convnext-pytorch-Tiny_DINOv3_LVD1689M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[convnext/pytorch-Tiny_DINOv3_LVD1689M-single_device-inference]

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
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Underlying error surfaced during reproduction:
```
AttributeError: module 'spacy' has no attribute 'Language'
```

After fixing the spacy shadowing, a second loader bug surfaced:
```
RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
```

## Root cause

### Bug 1 — `spacy` namespace package shadowing (loader / tt-xla)

`DynamicLoader.setup_models_path` in `tests/runner/utils/dynamic_loader.py`
added `models_root` (the `tt_forge_models/` directory) to `sys.path` with the
comment "so relative imports work". Because `tt_forge_models/spacy/` is a
directory without `__init__.py`, Python creates a namespace package named
`spacy` from it when `sys.path` is searched during model discovery. The real
`spacy` package is never imported, so `sys.modules['spacy']` holds the empty
namespace. When `datasets._dill.save()` later checks `if issubclass(obj_type,
spacy.Language)`, it raises `AttributeError: module 'spacy' has no attribute
'Language'`.

Adding `models_root` to `sys.path` is not needed: relative imports inside
loaders work via `__package__` + the manually-registered `tt_forge_models`
namespace module (which already has `__path__ = [models_root]`).

### Bug 2 — `load_model` ignores `dtype_override` for TIMM models (loader / tt_forge_models)

`convnext/pytorch/loader.py::ModelLoader.load_model` accepts `dtype_override`
as a keyword argument but never uses it. `timm.create_model` returns a float32
model. The test framework calls `load_model(dtype_override=torch.bfloat16)` and
separately calls `load_inputs(dtype_override=torch.bfloat16)`, which casts the
input tensor to bfloat16. When the model's float32 bias is convolved against a
bfloat16 input in `torch.nn.Conv2d`, PyTorch raises:
`RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`.

## Fix

### Fix 1 — `tt-xla` `tests/runner/utils/dynamic_loader.py`

Removed the `if models_root not in sys.path: sys.path.insert(0, models_root)`
block from `setup_models_path`. Added a comment explaining why `models_root`
must NOT be on `sys.path`.

Commit: `7e791825c230e72671c00b1ba8b3c24325358956` on branch
`remediation/convnext-pytorch-Tiny_DINOv3_LVD1689M-single_device-inference`
in `tenstorrent/tt-xla`.

### Fix 2 — `tt_forge_models` `convnext/pytorch/loader.py`

Added `if dtype_override is not None: model = model.to(dtype_override)` after
`model.eval()` in `load_model`, matching the pattern used by other timm-based
loaders (e.g. `repvit/pytorch/loader.py`).

Commit: `aa4c7461d4e62c73cabff378addae2bbe2396e78` on branch
`remediation/convnext-pytorch-Tiny_DINOv3_LVD1689M-single_device-inference`
in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    54.17s
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`
- `tt_forge_models/convnext/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6ec1cb3893e7f14f2e4631871ab0cc1a33095923 |
| tt-forge-models | aa4c7461d4e62c73cabff378addae2bbe2396e78 |
