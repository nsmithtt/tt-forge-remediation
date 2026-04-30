# Remediation Summary: devstral_small_2_mlx_8bit-causal_lm-pytorch-2512_MLX_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[devstral_small_2_mlx_8bit/causal_lm/pytorch-2512_MLX_8bit-single_device-inference]

## Result
XFAIL — 24B parameter model in BF16 (~48 GB) exceeds single-device DRAM (~32 GB); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-24b-bf16-oom-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'Mistral3Model' object has no attribute 'layers'

(After loader fix: RuntimeError: TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196976832 B, free: 76413184 B, largest free block: 37030336 B))

## Root cause
Two bugs were found in sequence:

1. **Loader bug (AttributeError)**: `load_shard_spec` in `devstral_small_2_mlx_8bit/causal_lm/pytorch/loader.py` iterates `model.model.layers`, but the model loaded by `AutoModelForImageTextToText` is `Mistral3ForConditionalGeneration`. Its structure is `model.model` = `Mistral3Model` (multimodal container with `vision_tower`, `multi_modal_projector`, `language_model`) — there is no `.layers` directly on `Mistral3Model`. The transformer layers live at `model.model.language_model.layers`.

2. **Hardware capacity ceiling (OOM)**: The model is 24 billion parameters (Devstral Small 2). Loaded as BF16 (2 bytes/param), it requires ~48 GB. The device has ~32 GB DRAM (8 banks × ~4.27 GB). The model weights saturate DRAM, leaving no room for activations. This is not a compiler or allocator bug — the model simply exceeds single-device capacity.

## Fix
1. **Loader fix** (tt-forge-models, `devstral_small_2_mlx_8bit/causal_lm/pytorch/loader.py`): Changed `model.model.layers` → `model.model.language_model.layers` in `load_shard_spec`.

2. **Test config** (tt-xla, `tests/runner/test_config/torch/test_config_inference_single_device.yaml`): Added `KNOWN_FAILURE_XFAIL` entry for `devstral_small_2_mlx_8bit/causal_lm/pytorch-2512_MLX_8bit-single_device-inference` with OOM reason.

## Verification
- pytest exit: xfailed
- Hardware:    blackhole-p150b
- Duration:    572.55s (0:09:32)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/devstral_small_2_mlx_8bit/causal_lm/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3b1918401a2e4a380fddb6bf1a9bd251a2a41f74 |
| tt-forge-models | 1a0956321d30d7fa7653e61858341061129e399e |
