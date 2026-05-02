# Remediation Summary: fujin_9b_gguf-causal_lm-pytorch-FUJIN_9B_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fujin_9b_gguf/causal_lm/pytorch-FUJIN_9B_I1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — PCC=0.30 (required 0.99) caused by ttmlir-f32-precision-not-preserved in GatedDeltaNet SSM layers

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.30373505985147253. Required: pcc=0.99.
```

## Root cause

Fujin 9B is a fine-tune of Qwen3.5-9B-SSM-Hybrid, which has 32 decoder layers: 8 full-attention layers (at indices 3, 7, 11, 15, 19, 23, 27, 31) and 24 GatedDeltaNet linear-attention (SSM) layers. Because the `flash-linear-attention` (`fla`) library is not installed, inference falls back to the pure-PyTorch `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` implementations in transformers.

These functions explicitly upcast all inputs to `torch.float32` for numerical stability:
```python
query, key, value, beta, g = [
    x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
]
```

The GatedDeltaNet computation involves cumulative products of decay values (`g.cumsum` + `exp()`) that have large dynamic range and are numerically sensitive to precision. TT hardware does not preserve float32 — operations that are lowered as f32 in StableHLO are executed in bfloat16, losing the 7 vs 23 mantissa bits required for stable delta-rule accumulation.

In addition, `torch_chunk_gated_delta_rule` contains a Python loop with in-place triangular updates:
```python
for i in range(1, chunk_size):  # chunk_size=64, runs 63 iterations
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
```
This loop unrolls into 63 sequential scatter operations per chunk per layer in the StableHLO graph. With 24 SSM layers × 2 chunks × 63 iterations = 3024 scatter ops, compilation takes ~32 minutes. The accumulated bfloat16 round-off errors across these scatter operations reduce PCC to 0.30.

## Fix

**Loader bugs fixed** (5 bugs across 2 repos before hitting the compiler bug):

1. **`model_to_load` TypeError** (tt_forge_models): Other loaders imported during pytest collection globally patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with functions that lack the `model_to_load` kwarg added in transformers 5.x. Added `_find_real()` gc scan to restore the original function inside a context manager.

2. **Wrong model class** (tt_forge_models): `qwen35` was absent from `GGUF_CONFIG_MAPPING`, so `model_type="qwen35"` was not remapped to `"qwen3_5_text"`, causing AutoModelForCausalLM to load `Qwen3ForCausalLM` instead of `Qwen3_5ForCausalLM`. Added `qwen35` to the mapping and wrapped `load_gguf_checkpoint` to convert the model_type.

3. **`get_gguf_hf_weights_map` NotImplementedError** (tt_forge_models): After remapping to `qwen3_5_text`, the weights map lookup failed because gguf-py's `MODEL_ARCH_NAMES` knows `"qwen35"` but not `"qwen3_5_text"`. Wrapped `get_gguf_hf_weights_map` to translate back to `"qwen35"` before the lookup.

4. **`conv1d.weight` shape mismatch and `dt_bias` missing** (tt_forge_models): GGUF dequantizes `ssm_conv1d.weight` to shape `(H, K)` but `nn.Conv1d` expects `(H, 1, K)`; the gguf-py tensor-name map does not include `ssm_dt.bias`. Added `_Qwen35TensorProcessor` with `process()` to unsqueeze axis 1, and `perform_fallback_tensor_mapping()` to add `blk.N.ssm_dt.bias → model.layers.N.linear_attn.dt_bias`.

5. **`Qwen3_5DynamicCache` TypeError in comparison** (tt-xla): `Qwen3_5DynamicCache` does not inherit from `transformers.Cache`, so the `isinstance(tensor, Cache)` guard in `convert_and_match` did not fire. Added duck-type detection (`hasattr key_cache and value_cache`) and a new branch in `_cache_to_legacy` to handle the flat per-layer list pattern (attention layers use `key_cache`/`value_cache`; SSM layers use `conv_states`/`recurrent_states`).

**Terminal compiler bug** (Tier B, not fixed): After all loader fixes the test runs to completion but PCC=0.30. Root cause is `ttmlir-f32-precision-not-preserved` in the GatedDeltaNet SSM layers.

The proposed fix requires either:
- A TT-native SSM kernel that operates in f32 without lowering to bf16
- Or preserving float32 through all StableHLO lowering passes in tt-mlir (cross-cutting change across all lowering patterns)

## Tier B justification
cross-cutting — preserving f32 precision through every lowering pass requires coordinated changes across many ops in tt-mlir, not a single scoped fix.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 3951.09s (1:05:51)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/fujin_9b_gguf/causal_lm/pytorch/loader.py` (rewritten — 4 loader bugs)
- `tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py` (duck-type cache detection)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c3490b08f88a8b3231aefb7c1a683a5cdfe83ae4 |
| tt-forge-models | 5197d3d3cd44401ffa64b5fb13c1501930ee6bb2 |
