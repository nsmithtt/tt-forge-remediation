# Remediation Summary: ernie_4_5_vl-pytorch-28B_A3B_PT-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[ernie_4_5_vl/pytorch-28B_A3B_PT-single_device-inference]

## Result
XFAIL — 28B MoE model (~56 GB bfloat16) exceeds single-device DRAM (24 GB max); compilation also fails at MoE dispatch due to torch.nonzero() data-dependent shape

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Ernie4_5_VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
The reported failure was a loader-layer bug (transformers 5.x changed the default for image processors to `use_fast=True`, but the checkpoint was saved with the slow processor). After fixing that and three additional loader bugs (see Fix section), the CPU forward pass passes successfully.

The test cannot reach TT silicon for two independent reasons:

1. **Hardware capacity**: The 28B MoE model requires approximately 56 GB of bfloat16 DRAM for weights alone, which exceeds the 24 GB single-device maximum of supported hardware (n150, p150b/wormhole). This is an intrinsic hardware-class ceiling, not a compiler bug.

2. **Compilation blocker** (secondary): The MoE dispatch path uses `moe_use_aux_free=True` correction-bias update, which calls `expert_num_local[experts_type_mask[i]]` (boolean indexing → `torch.nonzero()`). TorchXLA's fake-tensor / meta-tensor propagation cannot handle `torch.nonzero()` because its output shape is data-dependent. This would need to be resolved even for smaller models, but is moot here due to (1).

## Fix
Four loader bugs fixed in `third_party/tt_forge_models/ernie_4_5_vl/pytorch/loader.py`:

1. **`use_fast=False`** in `_load_processor()` — transformers 5.x FutureWarning that errors during processor loading when the checkpoint was saved with the slow processor.

2. **Normalize uint8 image patches** in `load_inputs()` — the processor returns patches under key `"images"` as `torch.uint8` shape `[num_patches, C*patch_h*patch_w]`; `vision_forward`'s else-branch (when `image_preprocess=None`) asserts bfloat16. Added `_normalize_images()` to rescale and per-channel normalize to bfloat16 before calling forward.

3. **Recompute `inv_freq` on meta device** in `load_model()` — `low_cpu_mem_usage=True` loads weights through the meta device; `VisionRotaryEmbedding.inv_freq` is a plain tensor attribute (not `register_buffer`) and is not materialized by `from_pretrained`. Iterate all modules and recompute any `inv_freq` still on the meta device.

4. **Pad `token_type_ids`** in `load_inputs()` — the model's forward asserts `token_type_ids.shape[1] == input_ids.shape[1] + 1`, but the processor returns equal-length tensors. Append a copy of the last token type to satisfy the assertion.

Test config updated in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` to mark the test as `KNOWN_FAILURE_XFAIL` with an explanation of the hardware-capacity ceiling.

## Verification
- pytest exit: FAIL (not-run on silicon — hardware-class XFAIL)
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/ernie_4_5_vl/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 74d632d424137f787a9506e9f421c903d896c77d |
| tt-forge-models | 3a23ff0851ac92411677c722260493e3dbe36e9e |
