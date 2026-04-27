# Remediation Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[asid_captioner/pytorch-7B-single_device-inference]`

## tt-forge-models Branch
`ip-172-31-22-17-tt-xla-dev/ubuntu/hf-bringup-7`

## Failure
The test failed with:
```
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor
by default, even if the model checkpoint was saved with a slow processor. This is a
breaking change and may produce slightly different outputs. To continue using the slow
processor, instantiate this class with `use_fast=False`.
```

## Root Cause
In transformers 5.x, when loading a `Qwen2_5OmniProcessor` (or any processor that
includes a `Qwen2VLImageProcessor` sub-processor), the image processor now defaults
to loading the fast variant (`Qwen2VLImageProcessorFast`) if not explicitly set. The
fast processor uses lanczos interpolation which is not supported in all environments
and produces different outputs from the slow processor.

The `asid_captioner/pytorch/loader.py` on the hf-bringup-7 branch (which had already
been fixed to use `Qwen2_5OmniThinkerForConditionalGeneration` + `Qwen2_5OmniProcessor`
instead of the incorrect `Qwen2VLForConditionalGeneration` + `AutoProcessor`) was still
calling `Qwen2_5OmniProcessor.from_pretrained()` without `use_fast=False`.

## Fix Applied
**Repository**: `tt-forge-models`
**Branch**: `ip-172-31-22-17-tt-xla-dev/ubuntu/hf-bringup-7`
**Commit**: `db8c004bf9`

Added `use_fast=False` to `Qwen2_5OmniProcessor.from_pretrained()` in
`asid_captioner/pytorch/loader.py`:

```python
self.processor = Qwen2_5OmniProcessor.from_pretrained(
    self._variant_config.pretrained_model_name,
    use_fast=False,
    **processor_kwargs,
)
```

This is the same fix pattern applied to other models (adavar, align, etc.) affected
by the same transformers 5.x breaking change.

## Verification
The fix was verified on a CPU-only machine (no Tenstorrent hardware):

1. **Before fix**: Running the test reproduced the exact warning message:
   "The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast
   processor by default..."

2. **After fix**: The warning is completely absent. The processor config files
   (4 files) were fetched and loaded cleanly without any fast processor warning.
   The test subsequently crashed at `get_xla_device_arch()` due to the absence of
   Tenstorrent hardware — this is expected behavior on a CPU-only machine and
   unrelated to the fix.

The fix resolves the reported failure. On a machine with Tenstorrent hardware, the
test should proceed past the processor loading stage without the fast processor error.
