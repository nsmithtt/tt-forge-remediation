# Remediation Summary: lmstudio_qwen_2_5_coder_7b_instruct_mlx_8bit-causal_lm-pytorch-7B_Instruct_MLX_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_qwen_2_5_coder_7b_instruct_mlx_8bit/causal_lm/pytorch-7B_Instruct_MLX_8bit-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mlx-affine-8bit-quantization-config-no-quant-method

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ValueError: The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized

## Root cause
The loader bug is in `lmstudio_qwen_2_5_coder_7b_instruct_mlx_8bit/causal_lm/pytorch/loader.py`.

`lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-8bit` stores MLX 8-bit quantized weights (uint32-packed int8 with per-group bf16 scales and biases, group_size=64) and sets `quantization_config: {"group_size": 64, "bits": 8}` in config.json without a `quant_method` field. Transformers >=5.x raises `ValueError` when it encounters a `quantization_config` without `quant_method`. The model must be dequantized manually: the weights are uint32-packed int8 values that need to be unpacked, scaled and offset with per-group bf16 scales/biases, and converted to bfloat16 before loading into the standard Qwen2.5 architecture.

## Fix
Changed `lmstudio_qwen_2_5_coder_7b_instruct_mlx_8bit/causal_lm/pytorch/loader.py` in `tt-forge-models`:

1. Added `_dequantize_mlx_affine_8bit()` helper that unpacks uint32→uint8, reshapes to groups, applies per-group bf16 scales and biases (`x = u8 * scale + bias`), and casts to bfloat16.
2. In `load_model()`: load config, strip `quantization_config`, download sharded safetensors via `model.safetensors.index.json`, dequantize with the helper, then create model via `AutoModelForCausalLM.from_config()` + `load_state_dict(strict=False)`.
3. In `load_config()`: strip `quantization_config` before returning.
4. Removed erroneous `torch_dtype` kwarg from `AutoTokenizer.from_pretrained()`.

Remediation branch: `remediation/lmstudio_qwen_2_5_coder_7b_instruct_mlx_8bit-causal_lm-pytorch-7B_Instruct_MLX_8bit-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    364.70s (0:06:04)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/lmstudio_qwen_2_5_coder_7b_instruct_mlx_8bit/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 17ec2e75a (updated to include tt_forge_models fix) |
| tt-forge-models | 77c96f41fa7d40aa9679451abdb14e25566a235e |
