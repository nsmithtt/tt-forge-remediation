# Remediation Summary: donut/document_question_answering/pytorch-donut_base_docvqa-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[donut/document_question_answering/pytorch-donut_base_docvqa-single_device-inference]

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
The image processor of type `DonutImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader-layer bugs caused the failure:

**Bug 1 (404 broken URL):** `load_inputs()` called `hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")`, but the file no longer exists at that path in the dataset (404). This was already fixed in the `ip-172-31-22-17-tt-xla-dev/ubuntu/hf-bringup-3` base branch by switching to `load_dataset("hf-internal-testing/fixtures_docvqa", split="test")` to load the image from the structured dataset instead.

**Bug 2 (reported, transformers 5.x breaking change):** `DonutProcessor.from_pretrained()` in transformers 5.x now loads `DonutImageProcessor` as a fast image processor by default, even when the checkpoint was saved with the slow processor. This produces subtly different pixel values and could cause PCC failures. The fix is to pass `use_fast=False`.

## Fix
**Fix 1 (pre-existing in base branch):** Replaced `hf_hub_download` with `load_dataset("hf-internal-testing/fixtures_docvqa", split="test")` and used `ds[0]["image"].convert("RGB")` to get the input image in `donut/document_question_answering/pytorch/loader.py`.

**Fix 2:** Added `use_fast=False` to `DonutProcessor.from_pretrained()` in `donut/document_question_answering/pytorch/loader.py` to use the slow image processor as the checkpoint expects.

Both changes are in the `tt_forge_models` repo on branch `remediation/donut-document_question_answering-pytorch-donut_base_docvqa-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    167.76s (0:02:47)
- Tier A attempts: N/A

## Files changed
- `donut/document_question_answering/pytorch/loader.py` — added `use_fast=False` to `DonutProcessor.from_pretrained()`; replaced broken `hf_hub_download` with `load_dataset`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ae01394727a19d3fda39b4b0cded40023cc8eade |
