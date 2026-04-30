# Remediation Summary: instruct_pix2pix/pytorch-InstructPix2Pix-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[instruct_pix2pix/pytorch-InstructPix2Pix-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
diffusers-pipeline-not-nn-module

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: assert isinstance(self._model, torch.nn.Module)
(tests/infra/testers/single_chip/model/torch_model_tester.py:113)

Secondary failures (sequential, each unblocked by fixing the previous):
1. AttributeError: 'StableDiffusionInstructPix2PixPipeline' object has no attribute 'encode_prompt'. Did you mean: '_encode_prompt'?
2. RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same

## Root cause
Three loader bugs in `instruct_pix2pix/pytorch/`:

1. **load_model() returned the full pipeline instead of the UNet.**
   `loader.py` docstring says "Returns torch.nn.Module (UNet)" but the code returned
   `self.pipeline` — a `StableDiffusionInstructPix2PixPipeline`, which is not an
   `nn.Module`. The test framework checks `isinstance(model, torch.nn.Module)` and
   raises `AssertionError`.

2. **encode_prompt API renamed in diffusers ≥0.26.**
   `model_utils.py` called `pipe.encode_prompt(...)` (the old public method that
   returned a 2-tuple). In diffusers 0.37.1 (installed), this was renamed to
   `_encode_prompt` and its signature/return semantics changed: it now returns a
   single tensor (concatenated when `do_classifier_free_guidance=True`) rather than
   a `(prompt_embeds, negative_prompt_embeds)` tuple.

3. **Preprocessed image not cast to VAE dtype.**
   After `pipeline.to(dtype_override=bfloat16)`, the VAE weights are bfloat16 but
   `pipe.image_processor.preprocess()` returns float32. Passing the float32 image to
   `pipe.vae.encode()` raised `RuntimeError: Input type (float) and bias type
   (c10::BFloat16) should be the same`.

## Fix
All changes in `tt_forge_models/instruct_pix2pix/pytorch/`:

**`loader.py`**: Changed `return self.pipeline` → `return self.pipeline.unet` in
`load_model()` so the returned object is the `torch.nn.Module` the framework expects.

**`src/model_utils.py`** (two fixes):
- Replaced `pipe.encode_prompt(prompt=..., ..., do_classifier_free_guidance=...)` with
  two separate calls to `pipe._encode_prompt(..., do_classifier_free_guidance=False)`
  to get positive and negative embeddings independently, preserving the 3-condition
  CFG concatenation order `[neg, neg, pos]` used by InstructPix2Pix.
- Added `image = image.to(dtype=pipe.vae.dtype)` after `image_processor.preprocess()`
  so the image tensor matches the VAE's bfloat16 dtype.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    383.74s (0:06:23)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/instruct_pix2pix/pytorch/loader.py`
- `tt_forge_models/instruct_pix2pix/pytorch/src/model_utils.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c46f4080014ff1f7743107d7d34339f528485db5 |
| tt-forge-models | fc67a07893d20c873ea22c9f22fa191a73c1861b |
