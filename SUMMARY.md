# Remediation Summary: flux1_kontext_dev-pytorch-Kontext_Dev-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux1_kontext_dev/pytorch-Kontext_Dev-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
flux1-kontext-dev-fp8-weights-bf16-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.95.

## Root cause
The loader used `AlekseyCalvin/Flux_Kontext_Dev_fp8_scaled_diffusers` whose transformer weights are stored as `float8_e4m3fn`. When loaded with `torch_dtype=bfloat16`, the FP8 bit-patterns are cast directly to BF16 producing values up to ±416. In a 3072-dimensional linear layer, the dot-product magnitude can reach 3072 × 416 ≈ 1.28 M, which overflows BF16 max (65504). Overflow propagates as `inf`; subsequent `inf − inf` (residual connections, attention softmax denominators) produces `nan`. The CPU baseline forward pass was therefore all-NaN, making PCC undefined (NaN) regardless of what TT silicon produced.

The previous model choice `Comfy-Org/flux1-kontext-dev_ComfyUI` returned HTTP 404 (ComfyUI-format repo has no `model_index.json`), and `black-forest-labs/FLUX.1-Kontext-dev` is gated and inaccessible without an accepted-license token.

## Fix
Changed `pretrained_model_name` in `flux1_kontext_dev/pytorch/loader.py` from `AlekseyCalvin/Flux_Kontext_Dev_fp8_scaled_diffusers` to `fuliucansheng/FLUX.1-Kontext-dev-diffusers`. The `fuliucansheng` repo is a public, non-gated diffusers conversion of the same model with native `bfloat16` weights (observed range ≈ ±5), producing valid finite CPU outputs.

Branch: `remediation/flux1_kontext_dev-pytorch-Kontext_Dev-single_device-inference` in `tenstorrent/tt-forge-models`.

File changed: `flux1_kontext_dev/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    608.96s (0:10:08)
- Tier A attempts: N/A

## Files changed
- `flux1_kontext_dev/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 04172358511c87c971cb329802a1232ea0a33457 |
