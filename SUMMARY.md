# Remediation Summary: libretranslate_gemma3-causal_lm-pytorch-4B_IT_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[libretranslate_gemma3/causal_lm/pytorch-4B_IT_Q4_0-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; test passes on TT silicon in 383s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` in the issue title is an unrelated SWIG import warning, not the actual failure.)

## Root cause
Two sequential loader bugs:

**Bug 1 — session contamination from narrow-sig GGUF patches (loader):**
During pytest collection every loader module is imported, including ~26 Qwen3.5/gpt-oss GGUF loaders that install a module-level monkey-patch of `transformers.integrations.gguf.load_gguf_checkpoint`. These patchers used a narrow signature `(gguf_path, return_tensors=False)`. In transformers 5.2.0 `from_pretrained` began passing `model_to_load=dummy_model` to `load_gguf_checkpoint`. Because the narrow-sig patch was the last one installed when libretranslate_gemma3's test ran, it raised `TypeError: unexpected keyword argument 'model_to_load'`.

**Bug 2 — XLA lazy slice OOB for Gemma3 sliding-window cache (tt-xla):**
After fixing Bug 1, `from_pretrained` succeeded and the model began running. Gemma3 uses a sliding-window KV cache: `full_value_states[:, :, -sliding_window+1:, :]` = `[:, :, -1023:, :]` on a tensor whose seq_len dim has only 23 elements at inference time. PyTorch eager silently clamps -1023 to 0; the XLA lazy backend raises `ValueError: Value out of range (expected to be in range of [-23, 22], but got -1023)`.

## Fix
**Fix 1 — widen 26 narrow-sig patches in tt_forge_models:**
Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(*args, **kwargs):` and forwarded `*args, **kwargs` to `_orig_load_gguf_checkpoint` in all 26 affected loaders. Committed on `remediation/libretranslate_gemma3-causal_lm-pytorch-1B_IT_Q4_0-single_device-inference` in tt_forge_models (shared fix with the 1B variant).

**Fix 2 — pre-clamp slice start/end in TorchFunctionOverride (tt-xla):**
Added a guard in `TorchFunctionOverride.__torch_function__` (`python_package/tt_torch/torch_overrides.py`): when `func is torch.ops.aten.slice.Tensor` and the start or end index is a negative integer below `-size`, clamp it to `-size` before forwarding to XLA. This matches PyTorch eager semantics.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    383.54s (0:06:23)
- Tier A attempts: N/A

## Files changed
tt_forge_models (26 files, all in `*/causal_lm/pytorch/loader.py`):
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf
- unified_reward_flex_qwen35_27b_gguf
- gpt_oss_swallow_120b_rl_v0_1_gguf
- gpt_oss_swallow_20b_rl_v0_1_gguf
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf
- mradermacher_vilm_0_8b_sft_gguf
- mradermacher_qwen3_5_4b_gabliterated_gguf
- mradermacher_qwen3_5_4b_unredacted_max_gguf
- mradermacher_qwen3_5_4b_unfiltered_gguf
- mradermacher_qwen3_5_9b_abliterated_i1_gguf
- mradermacher_qwen3_5_4b_abliterated_i1_gguf
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf
- mradermacher_qwen3_5_27b_gguf
- mradermacher_qwen3_5_27b_homebrew_gguf
- mradermacher_qwen3_5_27b_tainted_heresy_gguf
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf
- mradermacher_bartleby_qwen3_5_4b_gguf
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf
- qwen_3_5_imatrix_gguf
- dmind_3_mini_i1_gguf
- daniloreddy_qwen3_5_0_8b_gguf
- bartowski_coniccat_qwen3_5_27b_writer_gguf

tt-xla:
- `python_package/tt_torch/torch_overrides.py` — pre-clamp negative slice start/end

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 97d5c80d2fa7d404df466bd0a43842d2d2d578e7 |
| tt-forge-models | 50255ca5e15febae11c281f79fd53dfd0df97e9e |
