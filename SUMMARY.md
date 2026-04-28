# Remediation Summary: glm_4_6v_nvfp4-conditional_generation-pytorch-glm_4_6v_nvfp4-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_6v_nvfp4/conditional_generation/pytorch-glm_4_6v_nvfp4-single_device-inference]

## Result
FAIL — model checkpoint uses static per-token activation quantization calibrated for 256-token sequences; image-text input produces 6046 tokens causing a size mismatch in compressed_tensors forward pass

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
compressed-tensors-static-act-quant-seqlen-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
The image processor of type `Glm46VImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

Failure after loader fixes (current):
```
RuntimeError: The size of tensor a (6046) must match the size of tensor b (256) at non-singleton dimension 1
```

in `compressed_tensors/quantization/lifecycle/forward_helpers.py:194: scaled = x / scale`
during `forward_quantize(self, input, "input", scheme.input_activations)` for `q_proj` in
decoder layer 0.

## Root cause
Two loader bugs were present and fixed:

1. **Transformers 5.x image processor breaking change**: `AutoProcessor.from_pretrained` for
   `Glm46VImageProcessor` now defaults to the fast processor, but the checkpoint was saved with
   the slow processor. Fixed by passing `use_fast=False`.

2. **Missing `compressed-tensors` requirement**: The model uses NVFP4 quantization in
   `compressed-tensors` format, but the package was not listed in requirements.txt. Fixed by
   adding `requirements.txt` with `compressed-tensors`.

After these two fixes, the test runs further but fails during the CPU golden reference
computation. The model's quantization config (`quant_method: compressed-tensors`,
`strategy: tensor_group`, `group_size: 16`, `dynamic: False` for input activations) uses
**static per-token activation quantization**. The precomputed activation scale tensor has 256
entries at the sequence dimension (calibrated on short text-only sequences of ≤256 tokens).
When the image processor expands the image into tokens, the full input sequence length becomes
6046 tokens. The static scale tensor shape `(1, 256, ...)` cannot broadcast against the
activation shape `(1, 6046, ...)`, causing the `RuntimeError` in `_quantize_dequantize`.

The model's quantization calibration was performed on text-only sequences and does not account
for the image token expansion needed for vision-language inference. This is a bug in the model
checkpoint's quantization calibration: `GadflyII/GLM-4.6V-NVFP4` cannot perform vision-language
inference with its current static activation quantization scheme.

## Fix
**Applied loader fixes** (committed to
`remediation/glm_4_6v_nvfp4-conditional_generation-pytorch-glm_4_6v_nvfp4-single_device-inference`
in tt_forge_models and tt-xla):

- `glm_4_6v_nvfp4/conditional_generation/pytorch/loader.py`: Added `use_fast=False` to
  `AutoProcessor.from_pretrained` call.
- `glm_4_6v_nvfp4/conditional_generation/pytorch/requirements.txt`: Created with
  `compressed-tensors` entry.

**Remaining failure** (not fixable in the loader):

The static activation quantization calibration in the model checkpoint is incompatible with
image inputs. Fixing this would require either:
- Re-calibrating the model's activation quantization with image-text examples (requires
  re-running the quantization pipeline, not a code fix).
- Disabling activation quantization at inference time (would suppress a quantization step,
  changing model semantics — a forbidden workaround).
- Changing input shapes so the sequence length matches the calibrated 256 tokens (forbidden).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    2312.66s (0:38:32)
- Tier A attempts: N/A

## Files changed
- `glm_4_6v_nvfp4/conditional_generation/pytorch/loader.py` (tt_forge_models)
- `glm_4_6v_nvfp4/conditional_generation/pytorch/requirements.txt` (tt_forge_models, new file)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7e4148924df58e1b3ecaffe1f611873c38154823 |
| tt-forge-models | 32e3b656e0df1d1ffcf090dd33a4f790725bee0b |
