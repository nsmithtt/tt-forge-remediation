# Remediation Summary: longcat_flash_lite_gguf-causal_lm-pytorch-LongCat_Flash_Lite_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longcat_flash_lite_gguf/causal_lm/pytorch-LongCat_Flash_Lite_GGUF-single_device-inference]

## Result
XFAIL — LongCat-Flash-Lite is 68.5B parameters; BF16 dequantization requires ~137 GB which exceeds p150b DRAM capacity (96 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture longcat-flash-ngram is not supported yet.
```
(CI failure message `raise NotImplementedError(` corresponds to the second failure mode in `get_gguf_hf_weights_map` when the arch is registered in `GGUF_SUPPORTED_ARCHITECTURES` but absent from `gguf.MODEL_ARCH_NAMES`.)

## Root cause
LongCat-Flash-Lite is a 68.5B parameter Mixture-of-Experts model (InquiringMinds-AI/LongCat-Flash-Lite-GGUF). At BF16 precision, the full model requires approximately 137 GB (68.5B × 2 bytes), which exceeds the 96 GB DRAM capacity of the p150b single device. Even if all loader bugs were fixed, the model cannot fit on a single p150b.

There is also a secondary loader-level bug: the GGUF architecture string `longcat-flash-ngram` is not registered in transformers' `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_CONFIG_MAPPING`. While `LongcatFlash` is a native transformers 5.x model (`model_type: longcat_flash`), the GGUF tensor block structure is non-standard — pairs of GGUF blocks (`blk.2N` and `blk.2N+1`) map to a single HuggingFace `model.layers.N` with dual sublayers (`self_attn.0` and `self_attn.1`). Additionally, `longcat-flash-ngram` is absent from `gguf.MODEL_ARCH_NAMES` in the gguf library (v0.18.0), so `get_gguf_hf_weights_map` raises `NotImplementedError`. A proper fix would require a custom `TensorProcessor` with paired-block remapping logic — but this is moot given the hardware ceiling.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry for the test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla on branch `remediation/longcat_flash_lite_gguf-causal_lm-pytorch-LongCat_Flash_Lite_GGUF-single_device-inference`.

The secondary loader bug (missing GGUF arch registration + non-standard block mapping) remains unfixed because it does not affect the outcome: even with a working loader, the model exceeds single-device DRAM.

## Verification
- pytest exit: XFAIL (1 xfailed, 6 warnings in 39.89s)
- Hardware:    blackhole-p150b
- Duration:    39.89s
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d18df1ff2d50b8e72a32a4e7897edc962169654f |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
