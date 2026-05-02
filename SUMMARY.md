# Remediation Summary: qwen_3_5_27b_claude_4_6_opus_reasoning_distilled_gguf-causal_lm-pytorch-27B_Claude_4_6_Opus_Reasoning_Distilled_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_5_27b_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch-27B_Claude_4_6_Opus_Reasoning_Distilled_GGUF-single_device-inference]

## Result
FAIL — qwen35 hybrid SSM+GLA+FullAttention architecture has no transformers model class; qwen35→qwen3 mapping fails with size mismatches on full-attention layers

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-hybrid-gla-ssm-no-transformers-class

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

Size mismatches in full-attention layers (every 4th layer, indices 3, 7, 11...63):
```
model.layers.{3...63}.self_attn.q_proj.weight  | MISMATCH | ckpt: torch.Size([12288, 5120]) vs model: torch.Size([3072, 5120])
model.layers.{3...63}.self_attn.k_proj.weight  | MISMATCH | ckpt: torch.Size([1024, 5120])  vs model: torch.Size([512, 5120])
model.layers.{3...63}.self_attn.v_proj.weight  | MISMATCH | ckpt: torch.Size([1024, 5120])  vs model: torch.Size([512, 5120])
model.layers.{3...63}.self_attn.o_proj.weight  | MISMATCH | ckpt: torch.Size([5120, 6144])  vs model: torch.Size([5120, 3072])
model.layers.{3...63}.self_attn.q_norm.weight  | MISMATCH | ckpt: torch.Size([256])         vs model: torch.Size([128])
model.layers.{3...63}.self_attn.k_norm.weight  | MISMATCH | ckpt: torch.Size([256])         vs model: torch.Size([512, 5120])
```

Original test-suite failure (before loader fixes): `TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology.`

## Root cause
`mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` uses the `qwen35` GGUF architecture, which is a **hybrid SSM+GLA+FullAttention** model (`full_attention_interval: 4`):

- Layers 0, 1, 2, 4, 5, 6, ... (non-multiples-of-4) are GLA+SSM layers with combined `attn_qkv` and `ssm_*` tensors.
- Layers 3, 7, 11, ... (every 4th) are full-attention layers with separate `attn_q/k/v` tensors.

The loader's `_patch_qwen35_support()` maps `qwen35 → qwen3`, which works for simple Qwen3.5 models but fails here because the Qwen3.5 full-attention layers use a different Q head dimension than Qwen3: the GGUF records `attention.key_length: 256` (head_dim for K/V) but the actual Q projection tensors have output dim 12288 = 48 Q-heads × 256 (the Q head_dim is larger than what the Qwen3 config will infer). Additionally, the GLA layers (`linear_attn`) have no Qwen3 equivalent at all. Transformers 5.x has no model class that implements this hybrid architecture.

The original MGD topology error occurred in a prior run where session contamination from another narrow-sig patch allowed the model to load with silently reinitialized (wrong-shape) weights, which then triggered a runtime fault.

## Fix
No fix attempted. The correct fix requires implementing a new transformers model class for the Qwen3.5 hybrid SSM+GLA+FullAttention architecture. The existing `Qwen3ForCausalLM` class cannot represent this architecture even with config changes.

Two loader fixes were applied before the architecture mismatch was encountered:
1. **26 narrow-sig patches fixed** (`tt_forge_models`): Cherry-picked `0729432440` from an existing remediation branch to add `**kwargs` to all 26 loaders that defined `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without a `**kwargs` acceptor, making them reject transformers 5.2's new `model_to_load` kwarg.
2. **qwen35 patch added to target loader** (`tt_forge_models`): Added `_patch_qwen35_support()` and wide-sig `_patched_load_gguf_checkpoint` to the target loader, which previously had no GGUF patch and relied on session contamination.

These fixes are correct and should be kept; they expose the deeper architecture incompatibility.

## Tier B justification
Tier B indicator: **new-infrastructure**

The qwen35 hybrid architecture interleaves GLA (GatedDeltaNet linear attention) layers with full-attention layers. The GLA layers require an entirely new model class with support for the `ssm_*` tensors, the `attn_qkv` combined projection, and the `full_attention_interval` scheduling logic. No such class exists in transformers 5.x for `qwen35`. Mapping to `qwen3` causes irreconcilable size mismatches on the full-attention Q projections (12288 vs 6144 output dim) because the Qwen3.5 Q head_dim (512) differs from what the config implies (256).

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 707.18s (0:11:47) — model dequantization completed but loading failed with size mismatch
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models` (submodule pointer updated):
  - `qwen_3_5_27b_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch/loader.py` — added `_patch_qwen35_support()` + wide-sig `_patched_load_gguf_checkpoint` with full module-level monkey-patching
  - 26 other GGUF loaders — added `**kwargs` to `_patched_load_gguf_checkpoint` signature and forwarded to `_orig_load_gguf_checkpoint`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a0233da0b6f90bc94af9ecb0587f6bcba907db4a |
| tt-forge-models | 4ce2a565f206339f060e41f6eeeba16b40d98f94 |
