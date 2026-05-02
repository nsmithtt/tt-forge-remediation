# Remediation Summary: huihui_qwen_3_5_9b_abliterated_mlx_4bit-causal_lm-pytorch-9B_Abliterated_MLX_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen_3_5_9b_abliterated_mlx_4bit/causal_lm/pytorch-9B_Abliterated_MLX_4bit-single_device-inference]

## Result
FAIL — PCC=0.7503 (required ≥0.99); GatedDeltaNet SSM recurrence CPU-hoisted via dynamic_update_slice (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
dynamic-update-slice-cpu-fallback-pcc-loss

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7503645774825679. Required: pcc=0.99.
```

## Root cause

**Loader bugs (fixed):** The checkpoint is a VLM (`Qwen3_5ForConditionalGeneration`) with
weights stored under the `language_model.*` prefix and MLX affine int4 quantization (uint32-packed
nibbles, separate BF16 scales/biases per group of 64). Four loader bugs were fixed:
1. `AutoModelForCausalLM.from_pretrained` produced all-random weights — every key was unexpected
   because the model expects `model.*` but the checkpoint has `language_model.*`. Fix: extract
   `text_config`, construct `Qwen3_5ForCausalLM` directly, and use a custom `_load_mlx4bit_state_dict`
   that strips the VLM prefix and dequantizes MLX affine int4 weights to BF16 via numpy.
2. BF16 tensors cannot be converted to numpy directly; fix: cast scales/biases to float32 first.
3. Conv1d kernel axes are transposed in MLX (`[out, K, in]`) vs PyTorch (`[out, in, K]`); fix:
   `tensor.permute(0, 2, 1).contiguous()` for all `conv1d.weight` tensors.
4. `use_cache=True` (default) caused `Qwen3_5DynamicCache` in model output, which the test harness
   `tree_map` cannot traverse; fix: `text_config.use_cache = False`.

**Terminal compiler bug (Tier B):** After the loader fixes the model compiles and runs on
p150b silicon, but PCC=0.750. The Qwen3.5 9B has 32 decoder layers: 24 use `linear_attention`
(GatedDeltaNet/GDA) and 8 use standard `full_attention`. Because `flash-linear-attention` is
not installed, GDA layers fall back to the pure-Python `torch_chunk_gated_delta_rule` loop,
which issues in-place variable-index tensor updates that lower to `dynamic_update_slice` ops
in StableHLO. The tt-mlir compiler CPU-hoists these ops (no TT silicon lowering exists for
arbitrary `dynamic_update_slice`), so all 24 GDA recurrence loops execute on CPU with
device↔CPU data transfers between layers. The accumulated precision loss across 24 layers
drives PCC from the BF16 floor (~0.88) down to 0.750.

## Fix

Loader fixes in `tt-xla/third_party/tt_forge_models/huihui_qwen_3_5_9b_abliterated_mlx_4bit/causal_lm/pytorch/loader.py`
on remediation branch `remediation/huihui_qwen_3_5_9b_abliterated_mlx_4bit-causal_lm-pytorch-9B_Abliterated_MLX_4bit-single_device-inference`
in the tt-forge-models repo. Four commits:
1. `32d19e1a7a` — Fix MLX 4-bit loader: dequantize affine int4 weights and remap VLM keys
2. `a4988dde83` — Use numpy for MLX affine int4 dequantization to bypass TT XLA CPU tensor interception
3. `c805fb5f33` — Disable use_cache to prevent Qwen3_5DynamicCache in model output
4. `8ae5a9df15` — Cast BF16 scales/biases to float32 before numpy conversion

The terminal bug lives in tt-mlir: `dynamic_update_slice` ops from `torch_chunk_gated_delta_rule`
are CPU-hoisted instead of being lowered to a TT kernel.

## Tier B justification

**new-infrastructure**: Implementing a native TT lowering for `dynamic_update_slice` (or an
equivalent GDA/GLA recurrence kernel) requires new infrastructure — either a custom tt-metal
kernel for the GatedDeltaNet scan or a tt-mlir lowering pass for scatter/update ops. This is
not a single-function bounded fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    3801.88s (1:03:21)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huihui_qwen_3_5_9b_abliterated_mlx_4bit/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ff8eb7838ea1d4f4ccb3aa2114e28ac8e8beeeb2 |
| tt-forge-models | 8ae5a9df155ee14bd1fb0fddd2a7a02e677e0903 |
