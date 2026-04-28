# Remediation Summary: hotdog_not_hotdog-pytorch-Default-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[hotdog_not_hotdog/pytorch-Default-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-use-fast-default

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Additionally, the loader used `load_dataset("huggingface/cats-image")` which triggered a spacy namespace collision: `tt_forge_models/spacy/` directory shadowed the real spacy package (namespace package via sys.path), causing `AttributeError: module 'spacy' has no attribute 'Language'` in `datasets._dill`.

## Root cause
Two loader bugs in `hotdog_not_hotdog/pytorch/loader.py`:

1. **transformers 5.x ViTImageProcessor default change**: In transformers 5.2.0, `AutoImageProcessor.from_pretrained` for ViT-based models loads a fast processor by default instead of the slow `ViTImageProcessor`. The fast processor may produce slightly different outputs. Fix: pass `use_fast=False`.

2. **spacy namespace collision via load_dataset**: The `tt_forge_models/spacy/` model directory (no `__init__.py`) becomes a namespace package when `tt_forge_models/` is added to `sys.path` by `dynamic_loader.py`. The real `spacy` package is shadowed. When `load_dataset("huggingface/cats-image")` calls `datasets._dill` which imports `spacy` and checks `spacy.Language`, it crashes. Fix: replace `load_dataset` with `PIL.Image.new` — the model only needs a sample RGB image.

## Fix
In `hotdog_not_hotdog/pytorch/loader.py`:
- `AutoImageProcessor.from_pretrained(pretrained_model_name)` → `AutoImageProcessor.from_pretrained(pretrained_model_name, use_fast=False)`
- Removed `from datasets import load_dataset` import
- Added `from PIL import Image` import
- Replaced `dataset = load_dataset("huggingface/cats-image")["test"]; image = dataset[0]["image"]` with `image = Image.new("RGB", (224, 224), color=(128, 128, 128))`

Branch: `remediation/hotdog_not_hotdog-pytorch-Default-single_device-inference` in `tenstorrent/tt-forge-models`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    46.96s
- Tier A attempts: N/A

## Files changed
- `hotdog_not_hotdog/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 132466719f20c7d89c226a13b18fc3ec85555962 |
| tt-forge-models | b6ade9c7e895e47fa8bb94d93ad7d1c3a0aafddf |
