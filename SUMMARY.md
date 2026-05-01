# Remediation Summary: llava-pytorch-1_5_13B-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llava/pytorch-1.5_13B-single_device-inference]

## Result
XFAIL — LLaVA 1.5 13B (~26 GB BF16) exceeds single p150b device DRAM capacity; OOM during execution

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-oom-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original loader bug is:
```
The image processor of type CLIPImageProcessor is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and
may produce slightly different outputs. To continue using the slow processor, instantiate
this class with use_fast=False.
```

After applying the loader fix (use_fast=False + PIL.Image for sample loading), the test
reaches silicon and then hits:
```
TT_FATAL: Out of Memory: Not enough space to allocate 12499025920 B DRAM buffer across
8 banks, where each bank needs to store 1562378240 B, but bank size is 4273390016 B
(allocated: 3376500672 B, free: 896889344 B, largest free block: 773021760 B)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Two issues:

1. **Loader bug (transformers 5.x)**: AutoProcessor.from_pretrained without use_fast=False
   loads CLIPImageProcessor as the fast variant by default, causing a breaking-change warning.
   Additionally, load_dataset("huggingface/cats-image") crashes due to the spacy namespace
   collision in tt_forge_models (see feedback_load_dataset_spacy_dill.md memory entry).

2. **Hardware capacity ceiling**: LLaVA 1.5 13B has ~13B parameters; at BF16 this is ~26 GB.
   The single p150b device has ~34 GB total DRAM (8 banks x ~4.27 GB each), but the model
   attempts to allocate a 12.5 GB contiguous buffer (1.56 GB per bank) while only 773 MB
   is the largest free block per bank. The model is too large for single-device inference
   on p150b.

## Fix
Two changes applied:

1. **tt-forge-models** (remediation/llava-pytorch-1_5_13B-single_device-inference, commit c1df323790):
   - llava/pytorch/loader.py: Add use_fast=False to AutoProcessor.from_pretrained
   - llava/pytorch/loader.py: Replace load_dataset("huggingface/cats-image") with
     get_file(self.sample_image) + PIL.Image.open to avoid the spacy/dill crash

2. **tt-xla** (remediation/llava-pytorch-1_5_13B-single-device-inference, commit 7ed71c4f0):
   - tests/runner/test_config/torch/test_config_inference_single_device.yaml:
     Add llava/pytorch-1.5_13B-single_device-inference as KNOWN_FAILURE_XFAIL with OOM reason

## Verification
- pytest exit: FAIL (OOM TT_FATAL -> INTERNAL Error code: 13) — test runs but hits hardware ceiling
- Hardware: blackhole-p150b
- Duration: 214.21s (0:03:34) before OOM
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/llava/pytorch/loader.py (in tt-forge-models submodule)
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7ed71c4f0bbbc43b3b5d3a3c5ff85f6bd2094655 |
| tt-forge-models | c1df32379024e3427d235144c91226909114f6c8 |
