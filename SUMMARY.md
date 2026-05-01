# Remediation Summary: mlx_community_qwen_1_5_0_5b_chat_4bit-causal_lm-pytorch-0_5B_Chat_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_qwen_1_5_0_5b_chat_4bit/causal_lm/pytorch-0_5B_Chat_4bit-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mlx-affine-4bit-weight-shape-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

From `transformers/utils/loading_report.py:278` during `AutoModelForCausalLM.from_pretrained()`.

The loading report showed all projection weight matrices (q/k/v/o_proj, gate/up/down_proj, embed_tokens.weight) as MISMATCH with checkpoint shapes 8× smaller than model shapes (e.g. `[1024, 128]` uint32 vs `[1024, 1024]` bfloat16). Extra keys `*.biases` and `*.scales` were UNEXPECTED.

## Root cause
`mlx-community/Qwen1.5-0.5B-Chat-4bit` stores weights in MLX's native affine 4-bit quantization format. Each weight matrix is packed as uint32, with 8 nibbles (4-bit values) per element (LSB-first). Per-group (group_size=64) float16 `.scales` and `.biases` companion tensors are stored alongside each quantized weight. The model config contains `"quantization": {"group_size": 64, "bits": 4}` (an MLX-specific key, not a transformers `quantization_config`).

`AutoModelForCausalLM.from_pretrained()` has no knowledge of this format: it initializes the architecture with full-precision shapes and then finds shape mismatches everywhere, raising RuntimeError. Simply using `ignore_mismatched_sizes=True` would load randomly-initialized weights, not the actual model.

## Fix
In `mlx_community_qwen_1_5_0_5b_chat_4bit/causal_lm/pytorch/loader.py`:

1. Added `_mlx_dequantize(weight, scales, biases)`: unpacks uint32-packed nibbles (8 per element, LSB-first) into float32, then applies per-group affine dequantization `w = packed_int * scale + bias`, returning bfloat16.

2. Added `_load_mlx_state_dict(pretrained_model_name, dtype)`: downloads the safetensors, identifies quantized weight bases (keys with companion `.scales`), dequantizes them via `_mlx_dequantize`, passes float tensors through unchanged.

3. In `load_model()`: loads `AutoConfig`, strips `config.quantization` (MLX-specific, not a transformers quantizer), creates `Qwen2ForCausalLM` directly from config (bypassing `from_pretrained`), loads the dequantized state dict with `strict=False` (lm_head.weight is tied to embed_tokens.weight and not stored separately).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    152.23s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mlx_community_qwen_1_5_0_5b_chat_4bit/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 68b669335ab526f11c9273ec00c9047d459bea11 |
