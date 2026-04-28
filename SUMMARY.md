# Remediation Summary: abhiray_qwen3_5_4b_abliterated_claude_4_6_opus_reasoning_distilled_gguf-causal_lm-pytorch-4B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[abhiray_qwen3_5_4b_abliterated_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch-4B_GGUF-single_device-inference]

## Result
FAIL — GGUF checkpoint uses a hybrid SSM+attention architecture unknown to transformers; no existing model class supports it

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-unsupported-hybrid-ssm-mha-architecture

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
First error (original branch, loader collision): TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

After fixing the GGUF wrapper signatures:
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
  model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH | ckpt: torch.Size([8192, 2560]) vs model: torch.Size([2048, 2560])
  model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH | ckpt: torch.Size([1024, 2560]) vs model: torch.Size([512, 2560])
  (and similar mismatches for v_proj, o_proj, q_norm, k_norm in those layers)

The CI failure "Test exceeded configured timeout and was killed" was likely due to a slow GGUF download before the TypeError was raised.

## Root cause
Two distinct issues were found:

**Issue 1 (fixed):** Multiple GGUF loader modules define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and monkey-patch it onto `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. Pytest's test discovery imports all model loaders, so the broken narrow signature patches the global function. When the abhiray test later calls `AutoModelForCausalLM.from_pretrained`, transformers 5.x passes `model_to_load=dummy_model` to `load_gguf_checkpoint`, which fails because the narrow-signature wrapper doesn't accept it.

**Issue 2 (Tier B):** The GGUF file `Qwen3.5-4B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf` contains a **hybrid SSM+attention architecture** — NOT a standard Qwen3.5 transformer:
- 24 SSM layers (layers 0,1,2, 4,5,6, ..., 28,29,30) with fields `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_out`, `attn_gate`, `attn_qkv`
- 8 full MHA layers (layers 3,7,11,15,19,23,27,31) with per-layer heads: 64q/8kv heads at head_dim=128 vs the config-default 16q/4kv
- GGUF metadata field `qwen35.full_attention_interval: [4]` and SSM fields `ssm.conv_kernel`, `ssm.state_size`, `ssm.inner_size`, etc.

The GGUF architecture string "qwen35" is mapped to standard `Qwen3ForCausalLM` (a pure transformer), which cannot accept these hybrid weights. There is no transformers model class that implements this architecture.

## Fix

**Issue 1 fix (committed):** Updated all 26 GGUF loader wrappers from `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to `_patched_load_gguf_checkpoint(*args, **kwargs)` and forwarding args to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Also added the full qwen35 patching to the abhiray loader itself (it was previously relying on import-order side-effects from other loaders).

**Issue 2 proposed fix:** The GGUF architecture "qwen35" in this checkpoint refers to a novel hybrid SSM+MHA model. A correct fix requires either:
(a) A new transformers model class for this architecture (Mamba-2/hybrid-style with periodic full attention), or
(b) Mapping "qwen35" to an existing transformers class that happens to support the same hybrid structure (none currently exists).
The fix lives in the loader layer — specifically in `_patch_qwen35_support()` and the GGUF architecture-to-class mapping — but only once the underlying transformers model class is available.

## Tier B justification
new-infrastructure: the hybrid SSM+attention architecture in this GGUF checkpoint has no corresponding transformers model class. Implementing the architecture and its weight-loading logic from scratch requires new infrastructure, not a scoped single-file change.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~351s (to reproduce the architecture mismatch after the first fix)
- Tier A attempts: N/A

## Files changed
tt-forge-models:
- abhiray_qwen3_5_4b_abliterated_claude_4_6_opus_reasoning_distilled_gguf/causal_lm/pytorch/loader.py — added qwen35 patching with (*args, **kwargs) wrapper
- 26 other GGUF loader files — updated `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `(*args, **kwargs)`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4733b172ce71ffa8f40328bbd386f0d62c09b1b6 |
