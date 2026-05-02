# Remediation Summary: qwen_3_5_9b_omnicoder_claude_polaris-causal_lm-pytorch-9B_OmniCoder_Claude_Polaris-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_5_9b_omnicoder_claude_polaris/causal_lm/pytorch-9B_OmniCoder_Claude_Polaris-single_device-inference]

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
  third_party/tt_forge_models/qwen_3_5_9b_omnicoder_claude_polaris/causal_lm/pytorch/loader.py:146
    shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")

After fixing that, a second loader bug surfaced during comparison:
  TypeError: equal(): argument 'input' (position 1) must be Tensor, not Qwen3_5DynamicCache

## Root cause
Two loader bugs, both in the same file:

1. **Hybrid architecture not handled in load_shard_spec**: nightmedia/Qwen3.5-9B-OmniCoder-Claude-Polaris
   uses a hybrid GDA+attention architecture with 36 layers. The text_config has
   layer_types alternating between "linear_attention" (GatedDeltaNet, 27 layers)
   and "full_attention" (standard attention, 9 layers) with full_attention_interval=4.
   Each Qwen3_5DecoderLayer has either:
   - layer.linear_attn (Qwen3_5GatedDeltaNet) for linear_attention layers, OR
   - layer.self_attn (Qwen3_5Attention) for full_attention layers.
   load_shard_spec iterated all 36 layers and unconditionally accessed layer.self_attn,
   crashing immediately on the first GDA layer. Session contamination from MiniCPM
   Module.__getattr__ patches intercepted the missing attribute and re-raised the
   AttributeError from torch.nn.modules.module:1964.

2. **use_cache=True returns Qwen3_5DynamicCache in output**: With the default
   use_cache=True, the model returns a Qwen3_5DynamicCache object alongside the
   logits tensor. The comparison evaluator calls torch.equal() on every output
   element; torch.equal raises TypeError on the Cache object.

## Fix
Both fixes are in tt_forge_models on branch
remediation/qwen_3_5_9b_omnicoder_claude_polaris-causal_lm-pytorch-9B_OmniCoder_Claude_Polaris-single_device-inference.

**Commit** 4968fd51b8 — Fix load_shard_spec + add use_cache=False:
- qwen_3_5_9b_omnicoder_claude_polaris/causal_lm/pytorch/loader.py
- Added `if hasattr(layer, "self_attn"):` guard around the four attention
  shard-spec lines; added a corresponding `elif hasattr(layer, "linear_attn"):`
  branch that shards in_proj_qkv, in_proj_z, and out_proj for GDA layers.
- Added `inputs["use_cache"] = False` in load_inputs to prevent
  Qwen3_5DynamicCache from appearing in the model output.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    3208.81s (0:53:28)
- e2e perf avg: 46.95s per inference

## Files changed
- tt_forge_models: qwen_3_5_9b_omnicoder_claude_polaris/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 65c51cfc685307b67a4dfb707a9a8f0334a29a1c |
| tt-forge-models | 4968fd51b8d111a90054e142dee2595c4b3916a2 |
