# Remediation Summary: abhiray_qwen3_5_9b_abliterated_claude_4_6_opus_reasoning_distilled_gguf-causal_lm-pytorch-9B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[abhiray_qwen3_5_9b_abliterated_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch-9B_GGUF-single_device-inference]

## Result
FAIL — qwen35 is a hybrid SSM+attention architecture not supported in transformers; mapping to qwen3 produces size mismatches in every 4th (full-attention) layer

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-ssm-unsupported-arch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
On tt-forge-models branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-21` (which already has the `*args, **kwargs` signature fix for `_patched_load_gguf_checkpoint`):

```
E   RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

With mismatches:
```
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH | ckpt: torch.Size([8192, 4096]) vs model:torch.Size([2048, 4096])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH | ckpt: torch.Size([1024, 4096]) vs model:torch.Size([512, 4096])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.v_proj.weight | MISMATCH | ckpt: torch.Size([1024, 4096]) vs model:torch.Size([512, 4096])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.o_proj.weight | MISMATCH | ckpt: torch.Size([4096, 4096]) vs model:torch.Size([4096, 2048])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_norm.weight | MISMATCH | ckpt: torch.Size([256]) vs model:torch.Size([128])
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_norm.weight | MISMATCH | ckpt: torch.Size([256]) vs model:torch.Size([128])
```

On older branches without the `*args, **kwargs` fix, the error precedes this:
```
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause

The GGUF file declares architecture `qwen35`, which is a hybrid SSM+attention model (similar to Jamba):

- **Non-full-attention layers (0,1,2,4,5,6,...)**: each layer has both SSM components (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`) and a fused local-attention projection (`attn_qkv`, `attn_gate`).
- **Full-attention layers (every 4th: 3,7,11,...,31)**: standard multi-head attention only, but with 64 Q-heads and 8 KV-heads (vs 16 Q-heads and 4 KV-heads in non-full layers).

Key GGUF metadata: `qwen35.full_attention_interval: [4]`, `qwen35.ssm.state_size: [128]`, `qwen35.ssm.conv_kernel: [4]`, `qwen35.ssm.inner_size: [4096]`.

The existing loaders map `qwen35→qwen3` and correct `model_type`, but `Qwen3ForCausalLM` is a pure attention transformer with uniform `num_attention_heads=16` across all layers. It has no SSM module slots and cannot represent the different head counts for full-attention layers. The result is:
- Weight size mismatches for every 4th layer's Q/K/V/O projections and norms
- SSM weights in non-full layers are silently dropped (no matching module)

The fix requires a new `Qwen35ForCausalLM` model class in transformers that implements the hybrid SSM+attention architecture with layer-type dispatch.

## Fix
**Proposed fix**: Implement a `Qwen35ForCausalLM` class in `transformers` (or a standalone module) that:
1. Dispatches each layer by index based on `full_attention_interval`
2. Full-attention layers use standard `Qwen3Attention` with `num_attention_heads=64, num_key_value_heads=8`
3. Other layers use an SSM block with local attention (Mamba2-like)
4. Registers `qwen35` in `GGUF_CONFIG_MAPPING` with a proper `Qwen35Config`

This would live in `transformers/models/qwen35/` or as a remote-code model via HuggingFace.

The `_patched_load_gguf_checkpoint` signature fix (changing `(gguf_path, return_tensors=False)` to `(*args, **kwargs)`) was a prerequisite and is already committed in the remediation branch `deba21f33e` in `tt-forge-models`.

## Tier B justification
new-infrastructure — the qwen35 model class with hybrid SSM+attention does not exist in transformers; implementing it requires a new model architecture module spanning multiple files (config, modeling, GGUF mapping).

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 423.50s (second run on remediation branch); 0:07:03 on hf-bringup-21 branch
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: abhiray_qwen3_5_9b_abliterated_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch/loader.py` — added `_patch_qwen35_support()` and `_patched_load_gguf_checkpoint(*args, **kwargs)` (prerequisite signature fix; does not make test pass)
- `tt-forge-models: tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py` and 25 other GGUF loaders — widened `_patched_load_gguf_checkpoint` signature from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9e09fdaf6c95905dc90605c76d2628d4949392af (hf-bringup-21); deba21f33e62690368a7d8878c00506cebd8e62c (remediation branch with loader patch) |
