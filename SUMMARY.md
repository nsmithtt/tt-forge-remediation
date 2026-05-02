# Remediation Summary: qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored-causal_lm-pytorch-9B_CLAUDE_4_6_HIGHIQ_THINKING_HERETIC_UNCENSORED-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored/causal_lm/pytorch-9B_CLAUDE_4_6_HIGHIQ_THINKING_HERETIC_UNCENSORED-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
qwen35-hybrid-load-shard-spec-missing-self-attn-guard

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'self_attn'

Raised in load_shard_spec at:
  third_party/tt_forge_models/qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored/causal_lm/pytorch/loader.py:151
    shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")

After fixing that, a second loader bug surfaced during comparison:
  TypeError: equal(): argument 'input' (position 1) must be Tensor, not Qwen3_5DynamicCache

## Root cause
Two loader bugs, both in the same file:

1. **Hybrid architecture not handled in load_shard_spec**: Qwen3.5 uses a
   hybrid GDA+attention architecture with 32 layers, only 8 of which
   (indices 3, 7, 11, 15, 19, 23, 27, 31) are `full_attention` with a
   `self_attn` sub-module. The remaining 24 are `linear_attention` layers
   backed by `Qwen3_5GatedDeltaNet` (accessible as `layer.linear_attn`).
   `load_shard_spec` iterated all 32 layers and unconditionally accessed
   `layer.self_attn`, crashing immediately on the first GDA layer.

2. **use_cache=True returns Qwen3_5DynamicCache in output**: With the
   default `use_cache=True`, the model returns a `Qwen3_5DynamicCache`
   object alongside the logits tensor. The comparison evaluator calls
   `torch.equal()` on every output element; `torch.equal` requires Tensors
   and raises `TypeError` on the Cache object.

## Fix
Both fixes are in `tt_forge_models` on branch
`remediation/qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored-causal_lm-pytorch-9B_CLAUDE_4_6_HIGHIQ_THINKING_HERETIC_UNCENSORED-single_device-inference`.

**Commit 1** — `1eaf833184` — Fix load_shard_spec for Qwen3.5 hybrid GLA+attention architecture:
- `qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored/causal_lm/pytorch/loader.py`
- Added `if hasattr(layer, "self_attn"):` guard around the four attention
  shard-spec lines; added a corresponding `elif hasattr(layer, "linear_attn"):` branch
  that shards `in_proj_qkv`, `in_proj_z`, and `out_proj` for GDA layers.

**Commit 2** — `7dcf51d515` — Disable use_cache to prevent Qwen3_5DynamicCache in model output:
- `qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored/causal_lm/pytorch/loader.py`
- Added `model.config.use_cache = False` after `from_pretrained`, preventing
  the model from returning a Cache object alongside logits.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    3175.55s (0:52:55)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: `qwen_3_5_9b_claude_4_6_highiq_thinking_heretic_uncensored/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7dcf51d5159ea994f752ac32ec26dc8b562e562e |
