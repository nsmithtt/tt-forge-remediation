# Remediation Summary: bartowski_glm_4_9b_chat_gguf-causal_lm-pytorch-9B_CHAT_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_glm_4_9b_chat_gguf/causal_lm/pytorch-9B_CHAT_Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
chatglm-gguf-tokenizer-glm4-converter-missing-merge-fix

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: argument 'merges': failed to extract enum PyMerges ('Merges | Filename')

## Root cause
Three interacting bugs in the loader layer:

**Bug 1 (primary): `chatglm-gguf-tokenizer-glm4-converter-missing-merge-fix`**

The GLM-4-9B-Chat GGUF tokenizer vocabulary has tokens with embedded spaces. When loading ChatGLM merges from GGUF, space-splitting produces 3-tuples instead of 2-tuples for some entries (e.g. `"ł éĻ ¤"` → `("ł", "éĻ", "¤")`). `GGUFChatGLMConverter` (defined in the bartowski_glm loader) correctly handles this, but was registered only for architecture key "chatglm". In the full pytest session, cross-loader patching causes the tokenizer to see architecture "glm4" (after `patched_load_gguf_checkpoint` remaps chatglm→glm4). `GGUF_TO_FAST_CONVERTERS["glm4"]` was a different loader's `GGUFQwen2Converter`, which passes raw 3-tuple merges directly to `BPE()`, raising the TypeError.

**Bug 2 (secondary): `gguf-load-checkpoint-model-to-load-kwarg`**

26 loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without `**kwargs`. Transformers 5.x `modeling_utils.py` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which fails with `TypeError: unexpected keyword argument 'model_to_load'`.

**Bug 3: GLM-4-9B model weight mismatch after GGUF load**

Two additional model weight issues prevented correct inference:
- The GGUF file stores attention weights as fused `attn_qkv` tensors, but `Glm4ForCausalLM` expects separate `q_proj`, `k_proj`, `v_proj`. The gguf-py name mapping routes these to a non-existent `qkv_proj`, leaving all three projections randomly initialized.
- `Glm4ForCausalLM` has `post_self_attn_layernorm` and `post_mlp_layernorm` (extra RMSNorm layers not present in the chatglm GGUF). They default to weight=1 which computes `x/rms(x)` instead of identity, corrupting activations.

## Fix
All fixes are in `tt-xla/third_party/tt_forge_models` on branch `remediation/bartowski_glm_4_9b_chat_gguf-causal_lm-pytorch-9B_CHAT_Q4_K_M-single_device-inference`.

**Fix 1** (`eb234a1823`, then updated in `d575d77d91`): In `bartowski_glm_4_9b_chat_gguf/causal_lm/pytorch/loader.py` and `glm_4_9b_chat_abliterated_gguf/causal_lm/pytorch/loader.py`, register `GGUFChatGLMConverter` for all architecture keys that may appear: "chatglm", "glm", and "glm4". Unconditional registration (no guard) so it overwrites whatever converter was previously registered under these keys.

**Fix 2** (`400ff45bb6`): Added `**kwargs` forwarding to 26 loaders with fixed-signature `_patched_load_gguf_checkpoint`. Changed signature from `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, **kwargs)` and forwarded through to `_orig_load_gguf_checkpoint`.

**Fix 3** (`eebf0efb56`): Added `_patch_fused_qkv()` to `bartowski_glm_4_9b_chat_gguf/causal_lm/pytorch/loader.py`. Reads the raw GGUF `blk.N.attn_qkv.weight` and `blk.N.attn_qkv.bias` tensors directly via `GGUFReader`, dequantizes them, and splits into `q_proj`/`k_proj`/`v_proj` weights on the loaded model.

**Fix 4** (`ef5e2764d5`): Added `_patch_extra_norms()` to replace `post_self_attn_layernorm` and `post_mlp_layernorm` in each `Glm4ForCausalLM` layer with `nn.Identity()`, restoring semantically correct GlmForCausalLM behavior.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    598.04s (0:09:58)
- Tier A attempts: N/A

## Files changed
- `bartowski_glm_4_9b_chat_gguf/causal_lm/pytorch/loader.py`
- `glm_4_9b_chat_abliterated_gguf/causal_lm/pytorch/loader.py`
- 26 other loaders with `_patched_load_gguf_checkpoint` signature fix (commit 400ff45bb6)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ef5e2764d5c7dd8d2d5060c28a4f87b23d5ba7b3 |
