# Remediation Summary: magistral_small_mlx_4bit-causal_lm-pytorch-2506_MLX_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[magistral_small_mlx_4bit/causal_lm/pytorch-2506_MLX_4bit-single_device-inference]

## Result
XFAIL — Magistral Small 24B BF16 (~46 GB) exceeds p150b DRAM (~32 GB); loader fix applied separately

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
mlx-4bit-no-quant-method

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized

## Root cause
Two issues:

1. **Loader bug**: `lmstudio-community/Magistral-Small-2506-MLX-4bit` has a `quantization_config` in `config.json` with only `{"group_size": 64, "bits": 4}` — no `quant_method` field. `AutoModelForCausalLM.from_pretrained` raises `ValueError` in transformers 5.x because `AutoHfQuantizer.supports_quant_method` requires this field.

2. **Hardware-class OOM**: After the loader fix, the model successfully reaches the TT device but fails with `Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks` during the tilize stage. Magistral Small is 24B parameters; at BF16 (2 bytes/param) the weights alone require ~46 GB. The p150b has 8 GDDR banks × ~4 GB each ≈ 32 GB total DRAM, which is insufficient.

## Fix
**Loader fix** (`magistral_small_mlx_4bit/causal_lm/pytorch/loader.py` in tt-forge-models):
- Strip `quantization_config` from `AutoConfig` before model creation
- Create model skeleton via `AutoModelForCausalLM.from_config`
- Load safetensors shards and dequantize MLX 4-bit affine weights: each `uint32` holds 8 unsigned nibbles (little-endian); dequant as `w_float = nibble * scale + bias` per group-64 of bf16 scale+bias pairs
- Skip `.scales` and `.biases` keys; pass layernorm/norm bf16 weights through unchanged
- `model.load_state_dict(strict=False)` + `model.tie_weights()`

**Test config XFAIL** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in tt-xla):
- Added `magistral_small_mlx_4bit/causal_lm/pytorch-2506_MLX_4bit-single_device-inference` with `status: KNOWN_FAILURE_XFAIL`

## Verification
- pytest exit: FAIL (OOM after loader fix)
- Hardware:    blackhole-p150b
- Duration:    890.83s (0:14:50)
- Tier A attempts: N/A

## Files changed
- `magistral_small_mlx_4bit/causal_lm/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0a5c295c7bd0f9e2c985f56d37fab2d1f67111c3 |
| tt-forge-models | 79b85a0fd7d985cd0d73a1ce3219b4f88a0dfc0e |
