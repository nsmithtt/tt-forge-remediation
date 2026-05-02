# Remediation Summary: mlx_community_qwen3_5_0_8b_mlx_bf16-causal_lm-pytorch-0_8B_MLX_bf16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_qwen3_5_0_8b_mlx_bf16/causal_lm/pytorch-0_8B_MLX_bf16-single_device-inference]

## Result
FAIL — PCC=0.433 on TT silicon from GatedDeltaNet (linear attention) layers; Tier B compiler bug

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-qwen35-gated-delta-net-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes):
```
AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'self_attn'
```
(session contamination from minicpm loaders + unconditional self_attn access in load_shard_spec)

After loader fixes:
```
TypeError: equal(): argument 'input' (position 1) must be Tensor, not Qwen3_5DynamicCache
```

After use_cache=False fix:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.4330129355194529. Required: pcc=0.99.
```

## Root cause
The `mlx-community/Qwen3.5-0.8B-MLX-bf16` checkpoint is a VLM (`Qwen3_5ForConditionalGeneration`) whose safetensors weights use a `language_model.` key prefix. Four loader bugs needed fixing before reaching silicon:

1. **Weight loading mismatch**: The original loader used `AutoModelForCausalLM.from_pretrained`, which created `Qwen3_5ForCausalLM` with `model.` prefix expectations while the checkpoint has `language_model.` prefixed keys — all weights MISSING. Fixed by using `AutoModelForCausalLM.from_config(text_config)` plus manual safetensors load with prefix stripping.

2. **Conv1d weight layout**: MLX stores `conv1d.weight` as `(out, kernel, in)` while PyTorch expects `(out, in, kernel)`. Fixed by `v.permute(0, 2, 1)` during weight remapping.

3. **load_shard_spec unconditional self_attn access**: `Qwen3_5DecoderLayer` has either `self_attn` (full_attention layers at indices 3,7,11,15,19,23) or `linear_attn` (GatedDeltaNet layers at all others). The original `load_shard_spec` accessed `layer.self_attn` unconditionally, crashing on GDA layers. Fixed with `hasattr(layer, "self_attn")` guard (prior commit on the remediation branch).

4. **Qwen3_5DynamicCache TypeError**: `Qwen3_5DynamicCache` is not a subclass of `transformers.cache_utils.Cache`, so the evaluator's `tree_map` comparison calls `torch.equal(Qwen3_5DynamicCache, ...)` and raises TypeError. Fixed by `inputs["use_cache"] = False`.

After all four loader fixes, the model compiles and runs on TT silicon. However, PCC=0.433 is far below the required 0.99. The model has 24 layers (18 GatedDeltaNet linear-attention, 6 standard attention at every 4th layer). The GatedDeltaNet recurrent state-space scan produces wrong results on WH silicon — the same bug fingerprint (`ttmlir-qwen35-gated-delta-net-pcc`) as identified for arthurcollet-omnicoder-9b-mlx-mxfp8. The lower PCC here (0.433 vs 0.9807) is consistent with a higher fraction of GDA layers (18/24 = 75% vs fewer in the 9B model).

## Fix
Loader fixes are in `remediation/mlx_community_qwen3_5_0_8b_mlx_bf16-causal_lm-pytorch-0_8B_MLX_bf16-single_device-inference` branch of tt-forge-models:
- `mlx_community_qwen3_5_0_8b_mlx_bf16/causal_lm/pytorch/loader.py` — weight loading with `language_model.` prefix stripping + conv1d permutation + self_attn hasattr guard + use_cache=False

Proposed compiler fix: Correctly implement the GatedDeltaNet recurrent scan for TT silicon in tt-mlir. The GDA forward pass applies a causal state-space update across the sequence dimension using delta-rule recurrence. The compiler either miscomputes or drops this recurrence, causing divergence from CPU reference.

## Tier B justification
**Indicator**: cross-cutting

The GatedDeltaNet computation involves a causal recurrent scan over the full sequence across 18 of 24 model layers. Fixing it requires implementing correct causal state-space recurrence in the compiler, which touches multiple lowering passes and likely requires new runtime infrastructure for the sequential state update. This is not a single-function scoped fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    2098.86s (0:34:58) [after loader fixes]
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mlx_community_qwen3_5_0_8b_mlx_bf16/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 05559bc6dbac682b55290582cc6bd7f36dbf93b7 |
| tt-forge-models | 9885b199dc1926ec58829f7eda441c473c48628b |
