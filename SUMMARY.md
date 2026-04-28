# Remediation Summary: depth_anything_vits14-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[depth_anything_vits14/pytorch-Base-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed: use_fast=False for DPTImageProcessor and replace load_dataset with PIL.Image.new

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
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader bugs in `depth_anything_vits14/pytorch/loader.py`:

1. **DPTImageProcessor use_fast default (primary failure)**: In transformers 5.x, `AutoImageProcessor.from_pretrained` now defaults to the fast processor variant for `DPTImageProcessor`. The loader called `AutoImageProcessor.from_pretrained(pretrained_model_name)` without `use_fast=False`, triggering the breaking-change warning/error.

2. **spacy namespace package shadowing load_dataset**: The `tt_forge_models` repo has a `spacy/` top-level directory (a model-category directory for spacy-based models) which Python treats as a namespace package when `third_party/tt_forge_models` is on `sys.path`. The `datasets` library imports `spacy` during serialization and accesses `spacy.Language`; the namespace package has no such attribute, so `AttributeError: module 'spacy' has no attribute 'Language'` is raised inside `load_dataset("huggingface/cats-image")`. The fix is to replace the `load_dataset` call with `PIL.Image.new("RGB", (518, 518))`, matching the pattern used by the `depth_anything_3` loader.

## Fix
In `tt-forge-models` on branch `remediation/depth_anything_vits14-pytorch-Base-single_device-inference`:

- `depth_anything_vits14/pytorch/loader.py`:
  - Commit `77a9ee641a` (pre-existing): switched model name from `LiheYoung/depth_anything_vits14` to `LiheYoung/depth-anything-small-hf` and added `use_fast=False` to `AutoImageProcessor.from_pretrained`.
  - Commit `fcd0ba3cc2`: replaced `from datasets import load_dataset` + `load_dataset("huggingface/cats-image")["test"]` with `from PIL import Image` + `Image.new("RGB", (518, 518))`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    129.11s (0:02:09)
- Tier A attempts: N/A

## Files changed
- `depth_anything_vits14/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 60352d66e790ac875136ff95044835f9d5181f48 |
| tt-forge-models | fcd0ba3cc27c5f61b39419f32245a70ca33eee9b |
