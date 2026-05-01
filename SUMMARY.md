# Remediation Summary: boreal_qwen_image-text_to_image-pytorch-small-discrete-low-rank-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[boreal_qwen_image/text_to_image/pytorch-small-discrete-low-rank-single_device-inference]

## Result
XFAIL — Qwen-Image transformer (~27B params, 54 GB on disk) exceeds p150b single-device DRAM capacity (~34 GB available); model weights alone fill device memory leaving <36 MB for inference activation buffers

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen-image-transformer-dram-oom

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: TT_FATAL @ bank_manager.cpp:439: false
info: Out of Memory: Not enough space to allocate 75497472 B DRAM buffer across 8 banks, where each bank needs to store 9437184 B, but bank size is 4273390016 B (allocated: 4268823936 B, free: 4566080 B, largest free block: 2359296 B)

Original reproducing failure (loader bug, before fix):
RuntimeError: The size of tensor a (4096) must match the size of tensor b (3584) at non-singleton dimension 2

## Root cause
Two bugs were found:

1. **Loader bug (fixed):** `load_inputs()` in the boreal_qwen_image loader hardcoded `text_dim = 4096` with an incorrect comment claiming this is the `joint_attention_dim` from config. The actual model config has `joint_attention_dim = 3584` (matching the Qwen2-7B hidden size). This caused a shape mismatch in `txt_norm` during the CPU forward pass (`encoder_hidden_states` shape `(1, 32, 4096)` vs `txt_norm.weight` shape `(3584,)`).

2. **Hardware capacity ceiling (XFAIL):** After fixing the loader bug, the test successfully runs on CPU but OOMs during TT device inference. The Qwen-Image transformer has 60 layers, 24 attention heads with head dim 128, joint_attention_dim=3584, resulting in approximately 27B parameters (model cache on disk is 54 GB at BF16). The p150b device has ~34 GB DRAM (8 banks × 4.27 GB). After weight upload, only 36 MB of free DRAM remains across all banks, insufficient for the 75 MB activation buffer requested during `tilize`.

## Fix
1. **tt_forge_models** (remediation branch `remediation/boreal_qwen_image-text_to_image-pytorch-small-discrete-low-rank-single_device-inference`):
   - `boreal_qwen_image/text_to_image/pytorch/loader.py`: Changed `text_dim = 4096` → `text_dim = 3584` to match the actual `joint_attention_dim` from the model config.

2. **tt-xla** (remediation branch `remediation/boreal_qwen_image-text_to_image-pytorch-small-discrete-low-rank-single_device-inference`):
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for `boreal_qwen_image/text_to_image/pytorch-small-discrete-low-rank-single_device-inference` with OOM reason.
   - Bumped `third_party/tt_forge_models` submodule pointer to the loader fix commit.

## Verification
- pytest exit: XFAIL (1 xfailed, 7 warnings in 460.47s)
- Hardware:    blackhole-p150b
- Duration:    460.47s (0:07:40)
- Tier A attempts: N/A

## Files changed
- `boreal_qwen_image/text_to_image/pytorch/loader.py` (in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 692833caa0d43661f7bb3d04b18bd3fda4b3a7e8 |
| tt-forge-models | d8b99c7f28b8fd9c81eafdae4f7f1d0ad66a0f31 |
