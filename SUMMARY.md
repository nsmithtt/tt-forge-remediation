# Remediation Summary: controlnet_depth_sd3-pytorch-ControlNet_Depth_SD3_5_Large-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[controlnet_depth_sd3/pytorch-ControlNet_Depth_SD3.5_Large-single_device-inference]

## Result
FAIL — the base model `stabilityai/stable-diffusion-3.5-large` is a gated HuggingFace repository; the account `nsmithtt` has not accepted the Stability AI Community License Agreement

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
hf-gated-model-license-not-accepted

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
huggingface_hub.errors.GatedRepoError: 403 Client Error.

Cannot access gated repo for url https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/model_index.json.
Access to model stabilityai/stable-diffusion-3.5-large is restricted and you are not in the authorized list. Visit https://huggingface.co/stabilityai/stable-diffusion-3.5-large to ask for access.

(The originally-reported failure string `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is the last line of pytest's stderr output in all failure scenarios and is not the root cause.)

## Root cause
The loader (`third_party/tt_forge_models/controlnet_depth_sd3/pytorch/loader.py`) calls `StableDiffusion3ControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", ...)` to load the SD3.5 base model. That repository has `gated: auto` policy on HuggingFace, meaning every user must accept the Stability AI Community License Agreement on the model page before downloading. The CI account `nsmithtt` has a valid HF token set in `HF_TOKEN` but has not accepted the license, so every download attempt returns HTTP 403. There is no programmatic way to accept the HuggingFace gated-model license; it requires a one-time manual browser action on https://huggingface.co/stabilityai/stable-diffusion-3.5-large. This is an environment configuration issue, not a compiler-stack bug.

## Fix
Human action required: the `nsmithtt` HuggingFace account must visit https://huggingface.co/stabilityai/stable-diffusion-3.5-large, fill in the access form (Name, Email, Country), and click "Agree" to accept the Stability AI Community License Agreement. Once access is granted, re-run the test to see whether it passes or reveals a compiler-stack failure.

No code changes were made.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    32.84s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
