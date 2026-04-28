# Remediation Summary: aes_sedai_nemotron_3_super_120b_a12b_gguf-causal_lm-pytorch-AesSedai_3_Super_120B_A12B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aes_sedai_nemotron_3_super_120b_a12b_gguf/causal_lm/pytorch-AesSedai_3_Super_120B_A12B_GGUF-single_device-inference]

## Result
XFAIL — 120B MoE model at Q4_K_M (~67.5 GB) far exceeds single-device n150 DRAM (12 GB); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-120b-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: GGUF model with architecture nemotron_h_moe is not supported yet.

Note: the originally reported error `NotImplementedError: "histogram_cpu" not implemented for 'Int'` was not reproduced. The current failure (with transformers 5.2.0) is earlier — at GGUF loading — because `nemotron_h_moe` is not in `GGUF_CONFIG_MAPPING`. Even if the GGUF loading were fixed, the model is too large for any single Tenstorrent device.

## Root cause
NVIDIA Nemotron Super 120B A12B is a Mamba2-SSM hybrid (Nemotron-H) MoE model. The GGUF file uses the architecture identifier `nemotron_h_moe`, which is absent from `transformers.modeling_gguf_pytorch_utils.GGUF_CONFIG_MAPPING` in transformers 5.2.0 (there is no `NemotronH` or `NemotronHMoe` config class in this version). The model cannot be loaded at all.

More fundamentally, the full model at Q4_K_M quantization spans 3 shards totalling ~67.5 GB (120B params × ~4.5 bits/weight). The n150 device has ~12 GB DRAM; even n300 has only ~24 GB. This model cannot be held in single-device memory by any path, making the test hardware-class.

The companion loaders for the 9B Nemotron-H GGUF variants (`nemotron_nano_9b_v2_heretic_i1_gguf`, `open_nemo_9b_i1_gguf`, `nemotron_nano_9b_v2_japanese_gguf`) all document this limitation and fall back to non-GGUF safetensors sources. No safetensors alternative is available for this 120B model.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla with reason explaining both the GGUF loading failure and the underlying hardware capacity ceiling.

## Verification
- pytest exit: XFAIL
- Hardware:    n150
- Duration:    48.86s
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 91d0a4970ed9534a407ba4ab931e0388b44d62f9 |
| tt-forge-models | cae9ccbc67a318736f656bee9a9ea776eb73e69c |
