# Remediation Summary: gigapath-feature_extraction-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gigapath/feature_extraction/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gigapath-gated-repo-loader, spacy-namespace-shadows-real-package

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two cascading loader-layer failures:

1. GatedRepoError when loading the model:
```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url https://huggingface.co/prov-gigapath/prov-gigapath/resolve/main/config.json.
Access to model prov-gigapath/prov-gigapath is restricted and you are not in the authorized list.
```

2. After fix 1, AttributeError in datasets._dill due to spacy namespace shadowing:
```
AttributeError: module 'spacy' has no attribute 'Language'
```
(from datasets/utils/_dill.py:42: `if issubclass(obj_type, spacy.Language)`)

The originally-reported failure message (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`) is a harmless pytest warning at end of output, not the actual failure.

## Root cause
**Bug 1 (gigapath-gated-repo-loader):** The loader used `hf_hub:prov-gigapath/prov-gigapath` as the timm model name. Even with `pretrained=True`, timm tries to download `config.json` from HuggingFace when given an `hf_hub:` prefix, which fails with 403 for this gated medical-imaging repo. The equivalent timm architecture `vit_giant_patch14_dinov2` (giant ViT with DINOv2 pretraining at patch 14) works without gated access.

**Bug 2 (spacy-namespace-shadows-real-package):** `DynamicLoader.setup_models_path()` inserted `models_root` (the `tt_forge_models/` directory) into `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python creates a namespace package named `spacy` from it. When any loader that imports `spacy` (e.g. huspacy) runs during test discovery, this fake namespace package gets registered in `sys.modules['spacy']` — without `Language`. Later, `datasets._dill.save()` checks `if "spacy" in sys.modules` and then `issubclass(obj_type, spacy.Language)`, which AttributeErrors. The `sys.path` insertion was unnecessary: relative imports in loaders work via the `tt_forge_models` namespace package registered immediately below in `setup_models_path`.

## Fix
**Fix 1** — in `tt_forge_models`, `gigapath/feature_extraction/pytorch/loader.py`:
Changed `pretrained_model_name` from `"hf_hub:prov-gigapath/prov-gigapath"` to `"vit_giant_patch14_dinov2"` (commit `cd4ad095bb` on the remediation branch, already present at tip of `worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-start65-2`).

**Fix 2** — in `tt-xla`, `tests/runner/utils/dynamic_loader.py`:
Removed the 4-line block that called `sys.path.insert(0, models_root)` from `setup_models_path()`. The tt_forge_models namespace package registered below handles all relative imports without needing models_root in sys.path. (commit `2e69516ef`)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    133.05s (0:02:13)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gigapath/feature_extraction/pytorch/loader.py` — use `vit_giant_patch14_dinov2` timm name
- `tt-xla/tests/runner/utils/dynamic_loader.py` — remove `sys.path.insert(0, models_root)`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2e69516efc58fc05e08af06242aeaea8bad06a04 |
| tt-forge-models | e8db7752cf3b443bef1da30a8397d008c2dcdf71 |
