# Remediation Summary: llama_3_3_8b_instruct_heretic_mlx-causal_lm-pytorch-3.3_8B_Instruct_Heretic_MLX_8Bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_3_8b_instruct_heretic_mlx/causal_lm/pytorch-3.3_8B_Instruct_Heretic_MLX_8Bit-single_device-inference]

## Result
SILICON_PASS — loader dequantizes MLX affine-8bit weights; test passes on silicon in 385.45s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mlx-affine-8bit-no-quant-method

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

## Root cause
The `coderavi/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning-mlx-8Bit` checkpoint was quantized with Apple MLX affine-8bit. Its `config.json` contains `quantization_config: {group_size: 64, bits: 8, mode: "affine"}` but no `quant_method` field. Transformers >=5.x (specifically `quantizers/auto.py`) raises `ValueError` when it encounters a `quantization_config` without `quant_method`. The weights are stored as uint32-packed int8 with per-group bf16 scales and biases across two safetensors shards — they cannot be loaded directly as float weights without manual dequantization.

## Fix
Changed `llama_3_3_8b_instruct_heretic_mlx/causal_lm/pytorch/loader.py` in `tt-forge-models`:

1. Added `_dequantize_mlx_affine_8bit(raw_sd, group_size)` function that unpacks uint32→uint8, expands per-group scales/biases, and computes `x_bf16 = (w_u8 * scale + bias).bfloat16()`.
2. In `load_model`: load config with `AutoConfig.from_pretrained`, delete `quantization_config`, download both safetensors shards via the `model.safetensors.index.json` index, merge and dequantize, then create model via `AutoModelForCausalLM.from_config(config) + load_state_dict(strict=False)`. `strict=False` is needed because `lm_head.weight` is a tied weight not stored in the checkpoint.
3. In `load_config`: strip `quantization_config` to keep config clean.
4. Guarded `apply_chat_template` with `if self.tokenizer.chat_template is not None`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    385.45s (0:06:25)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: llama_3_3_8b_instruct_heretic_mlx/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ac521d067dd0890497ab35ea897e94f812b86a0e |
