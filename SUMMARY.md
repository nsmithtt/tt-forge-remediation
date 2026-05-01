# Remediation Summary: glm-causal_lm-pytorch-Z1_32B_0414-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm/causal_lm/pytorch-Z1_32B_0414-single_device-inference]

## Result
XFAIL — 32B BF16 model (~62 GB) exceeds single p150b DRAM (32 GB); DRAM OOM at inference execution

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-z1-32b-dram-oom

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'Glm4MLP' object has no attribute 'up_proj'

(Proximate loader bug in load_shard_spec; after fix the terminal failure is DRAM OOM)

## Root cause
Two issues found:

**1. Loader bug (fixed):** `load_shard_spec` in `glm/causal_lm/pytorch/loader.py` unconditionally
accesses `layer.mlp.up_proj` and `layer.mlp.gate_proj`. The GLM-Z1-32B-0414 uses
`Glm4ForCausalLM` (model_type `glm4`) whose MLP (`Glm4MLP`) has a single fused
`gate_up_proj` linear (hidden_size × 2*intermediate_size). The other GLM variants use
`Glm4MoeForCausalLM` (model_type `glm4_moe`) whose `Glm4MoeMLP` has separate
`gate_proj` + `up_proj`. The shard spec was written for the MoE variant only.

The loader also unconditionally added bias tensors (`q_proj.bias`, `k_proj.bias`,
`v_proj.bias`) to the shard spec, but `GLM-Z1-32B-0414` has `attention_bias=False`,
making those tensors `None` — valid Python dict key but incorrect semantics.

**2. Hardware capacity ceiling:** After the loader fix, the model loads and compiles
successfully, then hits DRAM OOM during inference execution:
```
Out of Memory: Not enough space to allocate 566231040 B DRAM buffer across 8 banks
(allocated: 4088626368 B, free: 184763648 B, largest free block: 35389440 B)
```
The 32B BF16 model requires approximately:
- 61 layers × (gate_up_proj 6144×46080 + down_proj 23040×6144 + attention) × 2 bytes ≈ 62 GB
The p150b has 32 GB DRAM, so 32B BF16 cannot fit on a single device.

## Fix
**Loader fix** in `tt_forge_models` repo, file `glm/causal_lm/pytorch/loader.py`:
- `load_shard_spec`: added `hasattr(layer.mlp, "gate_up_proj")` guard to handle
  both fused (`Glm4MLP`) and unfused (`Glm4MoeMLP`) MLP styles
- Also guarded `q/k/v_proj.bias` entries with `is not None` checks to skip
  `attention_bias=False` variants

**Test config** in `tt_xla` repo, file `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
- Added `status: KNOWN_FAILURE_XFAIL` entry for `glm/causal_lm/pytorch-Z1_32B_0414-single_device-inference`

## Verification
- pytest exit: FAIL (DRAM OOM after loader fix)
- Hardware:    blackhole-p150b
- Duration:    237.56s (0:03:57) — second run after loader fix
- Tier A attempts: N/A

## Files changed
- `glm/causal_lm/pytorch/loader.py` (tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bd2af516aa1845d69bcda95f0834d1063a20e61b |
| tt-forge-models | 01cbd8298e3f6fc4f2e155eb045e6d4f529f8069 |
