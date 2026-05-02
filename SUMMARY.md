# Remediation Summary: mlx_community_qwen3_5_35b_a3b_bf16-causal_lm-pytorch-35B_A3B_bf16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_qwen3_5_35b_a3b_bf16/causal_lm/pytorch-35B_A3B_bf16-single_device-inference]

## Result
XFAIL — Model exceeds single-device DRAM: ~35B params × 2 bytes BF16 ≈ 70 GB > p150b 32 GB; confirmed OOM (INTERNAL: Error code: 13)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-35b-moe-exceeds-p150b-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

The test failed in three stages as loader bugs were fixed:

**Stage 1 (original):** `load_shard_spec` crashed because it assumed all decoder layers have `self_attn`, but Qwen3.5 35B A3B is a hybrid model where `linear_attention` layers use `linear_attn` (GatedDeltaNet) instead.

**Stage 2 (after hasattr guard):** `partition_fx_graph_for_cpu_fallback` segfaulted because `Qwen3_5MoeExperts.forward` uses `nonzero()` producing dynamic shapes that crash the XLA Dynamo FX partitioner.

**Stage 3 (after batched_mm fix):** `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` — OOM when dispatching the 70 GB model to the 32 GB p150b device.

## Root cause
The model has 35B total parameters (256 experts × 40 layers × gate_up_proj[256,1024,2048] + down_proj[256,512,2048]) = ~35B params × 2 bytes BF16 = ~70 GB. This exceeds the p150b single-device DRAM (32 GB). Hardware-class ceiling.

## Fix
Four loader bugs were fixed in `tt_forge_models/mlx_community_qwen3_5_35b_a3b_bf16/causal_lm/pytorch/loader.py`:

1. **`load_shard_spec` hasattr guard** (commit `fe6cad2998` by prior developer): wrapped `layer.self_attn` accesses with `if hasattr(layer, "self_attn")` since hybrid GLA layers have `linear_attn` instead.

2. **`linear_attn` shard specs** (commit `690e9c5b85`): added shard specs for GLA layers: `in_proj_qkv.weight → ("model","batch")`, `in_proj_z.weight → ("model","batch")`, `out_proj.weight → ("batch","model")`.

3. **`use_cache=False`** (commit `690e9c5b85`): `Qwen3_5DynamicCache` is not a `transformers.Cache` subclass so the evaluator's `convert_and_match` raises TypeError; adding `inputs["use_cache"] = False` prevents returning `past_key_values`.

4. **`_experts_implementation = "batched_mm"`** (commit `223f932729`): `Qwen3_5MoeExperts.forward` default uses `nonzero()` which causes dynamic shapes crashing `partition_fx_graph_for_cpu_fallback`; setting `batched_mm` uses static-shape gather+matmul instead.

The test config was updated to `KNOWN_FAILURE_XFAIL` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (RuntimeError: INTERNAL: Error code: 13)
- Hardware:    blackhole-p150b
- Duration:    6925.20s (1:55:25)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mlx_community_qwen3_5_35b_a3b_bf16/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5c7b4389f76fff08043f3b7a2cd18a50f9c7a9cf |
| tt-forge-models | 223f932729b290ccaed3123a33a84fcf0f68baf1 |
