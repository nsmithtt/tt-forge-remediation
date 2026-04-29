# Remediation Summary: devstral_small_mlx_4bit-causal_lm-pytorch-2507_MLX_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[devstral_small_mlx_4bit/causal_lm/pytorch-2507_MLX_4bit-single_device-inference]

## Result
XFAIL — Devstral-Small-2507 is a 22B-parameter Mistral model; dequantized to bfloat16 it requires ~44 GB DRAM, exceeding n150 single-device capacity of 12 GB

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
mlx-quant-config-no-quant-method

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized
```

The `quantization_config` in `config.json` is an MLX-native dict (`{"group_size": 64, "bits": 4, ...}`) with no `quant_method` key. transformers 5.x `AutoHfQuantizer.supports_quant_method()` raises `ValueError` instead of returning `False` when `quant_method` is absent.

## Root cause
Two issues are present:

**Loader bug (fixed):** `lmstudio-community/Devstral-Small-2507-MLX-4bit` ships its `config.json` with a `quantization_config` block in MLX-native format. This format uses `{"group_size": 64, "bits": 4, ...}` and has no `quant_method` key. transformers 5.x added a strict check in `AutoHfQuantizer.supports_quant_method()` that raises `ValueError` (rather than returning `False`) when `quant_method` is absent, preventing any model load attempt.

The weights are stored as MLX-packed uint32 (8×int4 per element) with companion `.scales` and `.biases` tensors that standard `from_pretrained` cannot load due to shape mismatches.

**Hardware capacity (hardware-class):** Devstral-Small-2507 is a 22B-parameter model (40 layers, hidden_size=5120, intermediate_size=32768, vocab_size=131072). Dequantized to bfloat16 it requires approximately 44 GB of DRAM, which far exceeds the n150 single-device capacity of 12 GB. On-silicon execution fails with:
```
TT_FATAL: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196977728 B, free: 76412288 B, largest free block: 37030336 B)
```

## Fix
**Loader fix** (`tt_forge_models`, `devstral_small_mlx_4bit/causal_lm/pytorch/loader.py`):
- Load `AutoConfig` and delete `quantization_config` before constructing the model to bypass the `ValueError`.
- Initialize `MistralForCausalLM` directly from the stripped config on CPU.
- Add `_mlx_dequantize()` helper that unpacks MLX-format packed-uint32 (8 int4 values per element LSB-first) using companion `.scales` and `.biases` tensors.
- Add `_load_mlx_state_dict()` that discovers all safetensors shards, identifies quantized bases, and dequantizes packed int4 weights to bfloat16.
- Remove the redundant `AutoModelForCausalLM` import path.

This follows the same pattern established for `mlx_community_gemma_3n_e2b_it_4bit` (commit `8f20989b38`) but simplified for a pure Mistral (no conv/NHWC reshaping, no key remapping).

**Test config** (`tt-xla`, `tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `KNOWN_FAILURE_XFAIL` entry for `devstral_small_mlx_4bit/causal_lm/pytorch-2507_MLX_4bit-single_device-inference` citing hardware capacity.

## Verification
- pytest exit: XFAIL (1 xfailed — OOM on n150 as expected)
- Hardware:    n150
- Duration:    597.62s (0:09:57) — includes ~11 GB safetensors download
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: `devstral_small_mlx_4bit/causal_lm/pytorch/loader.py` — MLX dequantize fix
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry
- `tt-xla`: `third_party/tt_forge_models` submodule pointer updated

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 02c53307e3eaa0f82b53fa0b82e48bc00d32dfbc |
| tt-forge-models | 8ee4478d81347d8744d848ee5638b3a97a5c7f82 |
