# Remediation Summary: olm_ocr-image_text_generation-pytorch-olmOCR-7B-0825-FP8-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[olm_ocr/image_text_generation/pytorch-olmOCR-7B-0825-FP8-single_device-inference]

## Result
FAIL — Conv3d patch embedding CB allocation (1745920 B) exceeds max L1 size (1572864 B) on Wormhole; Tier B tt-metal bug

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-l1-cb-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.

After fixing the loader (4 fixes), the test fails with:

    CRITICAL TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
    grow to 1745920 B which is beyond max L1 size of 1572864 B (assert.hpp:104)
    ...
    tt::runtime::ttnn::operations::conv::run(...Conv3dOp...)
    RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Four loader bugs were fixed. The terminal failure is a Tier B compiler/runtime bug.

**Loader bug 1:** `compressed-tensors` package not in `requirements.txt`. The FP8 model
(`allenai/olmOCR-7B-0825-FP8`) uses compressed-tensors format and transformers raises
`ImportError` without it.

**Loader bug 2:** `use_fast=False` missing from `AutoProcessor.from_pretrained`. transformers
5.x loads `Qwen2VLImageProcessor` as a fast processor by default — a breaking change. Adding
`use_fast=False` restores the expected slow processor.

**Loader bug 3:** FP8 weights not dequantized on load. The model has
`quantization_status: "compressed"` (float8_e4m3fn weights). TT does not support the FP8
matmul path. Fix: set `run_compressed=False` in `quantization_config` before `from_pretrained`
so compressed-tensors dequantizes weights to bfloat16. Then remove instance-level `forward`
overrides that compressed-tensors leaves on quantized modules (they shadow TT-XLA's
`__torch_function__`).

**Loader bug 4:** `Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb`, `get_window_index`,
`Qwen2_5_VLModel.get_image_features`, and `get_rope_index` call `.tolist()` on tensors placed
on TT device by the test runner. TT device does not support eager D2H readback for these
tensor ops. Fix: patch all four methods to move metadata tensors to CPU before `.tolist()` calls.

**Terminal bug (Tier B):** `Qwen2_5_VisionPatchEmbed` uses `Conv3d(in_channels=3,
embed_dim=1280, kernel_size=(2,14,14))`. The kernel weight is ~2.87 MB BF16, exceeding
tt-metal's 1.5 MB L1 CB allocation limit. tt-metal raises TT_THROW at runtime:
`circular buffers grow to 1745920 B > max L1 1572864 B`. Same bug as dolphin_v2 and Qwen3-VL.

## Fix
Loader fixes in `tt-forge-models`, branch
`remediation/olm_ocr-image_text_generation-pytorch-olmOCR-7B-0825-FP8-single_device-inference`:

1. `olm_ocr/image_text_generation/pytorch/requirements.txt` (new file): `compressed-tensors`
2. `olm_ocr/image_text_generation/pytorch/loader.py`:
   - `use_fast=False` in `AutoProcessor.from_pretrained`
   - `run_compressed=False` in quantization_config + remove instance-level `forward` overrides
   - `_patch_qwen2_5_vl_tolist()` patches 4 Qwen2.5-VL methods to call `.cpu()` before `.tolist()`

**Proposed fix for terminal Tier B bug:** Fix tt-metal `Conv3dDeviceOperation` CB allocation
to shard the kernel tensor across cores when it exceeds L1, or implement a guard that falls
back to DRAM allocation. Lives in `tt-metal` convolution kernel program factory. Same fix
would apply to `dolphin_v2` and all Qwen2.5-VL-7B / Qwen3-VL-7B models.

## Tier B justification
`cross-cutting`: Fixing the Conv3d L1 CB overflow requires changing the tt-metal
`Conv3dDeviceOperation` CB allocation strategy to support kernel tensors larger than 1.5 MB L1.
This affects all models using large Conv3d patch embeddings (Qwen2.5-VL, Qwen3-VL, dolphin_v2,
MAI-UI-8B). The fix must be coordinated with the cb-allocation sharding logic in tt-metal and
validated against existing Conv3d users.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    115.21s (to Conv3d failure)
- Tier A attempts: N/A

## Files changed
- `olm_ocr/image_text_generation/pytorch/requirements.txt` (new)
- `olm_ocr/image_text_generation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b  |
| tt-xla          | f692db768  |
| tt-forge-models | 72dfe600da |
