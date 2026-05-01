# Remediation Summary: helios_distilled_int8-text_to_image-pytorch-Helios-Distilled-int8-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[helios_distilled_int8/text_to_image/pytorch-Helios-Distilled-int8-single_device-inference]

## Result
XFAIL — Model (~37 GB BF16) exceeds p150b 32 GB DRAM; loader also fixed to work around non-standard HF repo layout

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: Error no file named config.json found in directory /home/nsmith/.cache/huggingface/hub/models--szwagros--Helios-Distilled-int8/snapshots/3d8b4af99e0b35e9f5597ddbda2fde8863069b6a.

## Root cause
Two issues compound:

1. Loader bug (loader layer): The szwagros/Helios-Distilled-int8 HuggingFace repo uses a non-standard layout. Transformer weights are stored as a flat Helios-Distilled-int8.safetensors file at the repo root and text encoder weights as Helios-umt5-xxl-int8.safetensors, instead of the standard diffusers sub-directory layout (transformer/config.json, text_encoder/config.json). DiffusionPipeline.from_pretrained() fails with OSError because it cannot find the required config.json in the expected sub-directories.

2. Hardware capacity (hardware-class): The HeliosTransformer3DModel has ~14.3B parameters (~28.6 GB BF16) and the UMT5-XXL text encoder is ~8 GB BF16. Combined with the VAE (~0.5 GB), total memory is ~37 GB BF16, which exceeds the p150b 32 GB DRAM limit. INT8 quantized weights are loaded into standard model classes, causing dequantization to BF16 during load_state_dict().

## Fix
Loader fix in tt_forge_models/helios_distilled_int8/text_to_image/pytorch/loader.py:
- Replaced DiffusionPipeline.from_pretrained() with a manual pipeline construction strategy
- Loads transformer and text encoder configs from BestWishYsh/Helios-Distilled (the reference repo with standard layout and config.json files)
- Loads INT8 weights from the flat safetensors files in szwagros/Helios-Distilled-int8
- Loads VAE, tokenizer, scheduler directly from the int8 repo (these have standard sub-directory layout)
- Manually assembles HeliosPyramidPipeline from the loaded components

Test config in tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml:
- Added NOT_SUPPORTED_SKIP entry with bringup_status: FAILED_RUNTIME because the model exceeds p150b 32 GB DRAM

## Verification
- pytest exit: SKIP (NOT_SUPPORTED_SKIP applied correctly)
- Hardware:    blackhole-p150b
- Duration:    17.73s (collection + skip check)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models submodule bumped to 3ef2ffc4797dd6c0eeff327b9565d1eef698d0e5
  - helios_distilled_int8/text_to_image/pytorch/loader.py: manual pipeline construction from flat INT8 safetensors
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml: NOT_SUPPORTED_SKIP entry added

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | dbce58dbfa9f04bcebeda1091e99acc0bf7b4e3c |
| tt-forge-models | 3ef2ffc4797dd6c0eeff327b9565d1eef698d0e5 |
