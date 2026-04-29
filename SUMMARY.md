# Remediation Summary: egan_ai_qwen3_5_9b_terminal_merge-causal_lm-pytorch-9B_Terminal_Merge-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[egan_ai_qwen3_5_9b_terminal_merge/causal_lm/pytorch-9B_Terminal_Merge-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
qwen3-5-decoder-layer-hybrid-attn-shard-spec, qwen3-5-dynamic-cache-not-cache-subclass

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
third_party/tt_forge_models/egan_ai_qwen3_5_9b_terminal_merge/causal_lm/pytorch/loader.py:143: in load_shard_spec
    shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'self_attn'
```

After fixing the above, a second bug surfaced:
```
tests/infra/evaluators/torch_comparison_evaluator.py:93: in _equal_leaf
    return torch.equal(x, y)
TypeError: equal(): argument 'input' (position 1) must be Tensor, not Qwen3_5DynamicCache
```

## Root cause
Two loader-layer bugs:

**Bug 1 (tt_forge_models):** `load_shard_spec` in the EganAI Qwen3.5 9B loader assumed all
`Qwen3_5DecoderLayer` instances have a `self_attn` attribute. In transformers 5.x, this model
is a hybrid SSM+attention architecture where each layer has a `layer_type` of either
`"full_attention"` (with `self_attn`) or `"linear_attention"` (with `linear_attn` /
`Qwen3_5GatedDeltaNet`). Accessing `layer.self_attn` on a `linear_attention` layer raises
`AttributeError`.

**Bug 2 (tt-xla):** `Qwen3_5DynamicCache` (the custom cache class returned by
`Qwen3_5ForCausalLM`) does not inherit from `transformers.Cache`. The `isinstance(tensor,
Cache)` guard in `convert_and_match` therefore never triggers, leaving the raw cache object in
the pytree. `_compare_equal` then calls `torch.equal(Qwen3_5DynamicCache, ...)` which raises
`TypeError`. Unlike standard transformers cache classes, `Qwen3_5DynamicCache` stores
per-layer state in flat lists (`key_cache`, `value_cache`, `conv_states`, `recurrent_states`).

## Fix
**Fix 1 — `tt_forge_models` loader** (`egan_ai_qwen3_5_9b_terminal_merge/causal_lm/pytorch/loader.py`):

Guard the `self_attn` shard-spec entries with `if layer.layer_type == "full_attention":` and
add a corresponding `elif layer.layer_type == "linear_attention":` branch that shards the
`linear_attn` projections (`in_proj_qkv`, `in_proj_z`, `out_proj`).

**Fix 2 — `tt-xla` comparison evaluator** (`tests/infra/evaluators/torch_comparison_evaluator.py`):

1. Extended `convert_and_match` to also trigger on duck-type detection:
   `hasattr(tensor, "key_cache") and hasattr(tensor, "value_cache")`.
2. Extended `_cache_to_legacy` to handle the flat-list pattern: iterates per-layer indices,
   appending `(key_cache[i], value_cache[i])` for full-attention layers and
   `(conv_states[i], recurrent_states[i])` for linear-attention layers.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    3255.55s (0:54:15)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/egan_ai_qwen3_5_9b_terminal_merge/causal_lm/pytorch/loader.py` (tt_forge_models)
- `tests/infra/evaluators/torch_comparison_evaluator.py` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 88cfe6da6403108e6d72da9413731520bb5e4581 |
| tt-forge-models | b3b4df2c5f |
