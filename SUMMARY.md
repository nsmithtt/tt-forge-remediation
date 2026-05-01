# Remediation Summary: glm-causal_lm-pytorch-Z1_9B_0414-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm/causal_lm/pytorch-Z1_9B_0414-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
glm4-load-shard-spec-gate-up-proj-fused-weight

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'Glm4MLP' object has no attribute 'up_proj'

(The hf-bringup results recorded `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
as the failure_detail because pytest prints it as the very last line; the actual error is the AttributeError above.)

## Root cause
The `load_shard_spec` method in `glm/causal_lm/pytorch/loader.py` unconditionally accessed
`layer.mlp.up_proj` and `layer.mlp.gate_proj`. `GLM-Z1-9B-0414` uses `Glm4ForCausalLM`
(model_type `glm4`) whose MLP (`Glm4MLP`) has a single fused `gate_up_proj` linear
(hidden_size × 2*intermediate_size). The shard spec was written only for the MoE variant
`Glm4MoeMLP` (model_type `glm4_moe`) which has separate `gate_proj` + `up_proj`.

The loader also unconditionally added `q/k/v_proj.bias` entries to the shard spec, but
`GLM-Z1-9B-0414` has `attention_bias=False`, so those bias tensors are `None`.

## Fix
In `tt_forge_models` repo, file `glm/causal_lm/pytorch/loader.py`, `load_shard_spec`:

1. Added `hasattr(layer.mlp, "gate_up_proj")` guard: if fused, use `gate_up_proj.weight`;
   otherwise use the separate `gate_proj.weight` + `up_proj.weight`.
2. Added `is not None` guards around `q_proj.bias`, `k_proj.bias`, `v_proj.bias` entries
   to skip them when the model has `attention_bias=False`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    279.35s (0:04:39)
- Tier A attempts: N/A

## Files changed
- `glm/causal_lm/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9a00172c79919acaf553c2570e2b1c1dc9d14011 |
| tt-forge-models | 356aefa348f4a89e4b3be60e2df9d56cddaba40a |
