# Remediation Summary: darkidol_ballad_gguf-causal_lm-pytorch-4B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[darkidol_ballad_gguf/causal_lm/pytorch-4B_i1_GGUF-single_device-inference]

## Result
FAIL — qwen35 hybrid attention layer size mismatch: full-attention layers (3,7,11,15,19,23,27,31) store weights sized for 64 q-heads but qwen3 config expects 16 q-heads; new-infrastructure required for proper qwen3.5 hybrid support

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-layer-size-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
First error (before loader fix):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Second error (after loader fix, underlying bug):
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

Mismatched weights on layers {3, 7, 11, 15, 19, 23, 27, 31}:
```
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight  MISMATCH  ckpt: [8192, 2560] vs model: [2048, 2560]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight  MISMATCH  ckpt: [1024, 2560] vs model: [512, 2560]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.v_proj.weight  MISMATCH  ckpt: [1024, 2560] vs model: [512, 2560]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.o_proj.weight  MISMATCH  ckpt: [2560, 4096] vs model: [2560, 2048]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_norm.weight  MISMATCH  ckpt: [256] vs model: [128]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_norm.weight  MISMATCH  ckpt: [256] vs model: [128]
```

## Root cause
Two layered bugs:

**Bug 1 (fixed):** During pytest test collection, all loader modules are imported. The qwen3.5-based GGUF loaders (bartowski, daniloreddy, dmind, etc.) each install a global monkey-patch on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. These patches had the old signature `(gguf_path, return_tensors=False)`, missing `model_to_load`. When transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, the stale patch raises TypeError. The darkidol loader itself does not install any patch, but it is victimized by an earlier loader's patch being in place when its test runs.

**Bug 2 (Tier B):** Darkidol Ballad 4B is a Qwen3.5 fine-tune. Its GGUF has `model_type=qwen35`. The qwen35→qwen3 patch remaps this to `qwen3`. Qwen3.5 uses a hybrid architecture where every 4th layer (`full_attention_interval=4`) is a full-attention layer with 4× more q-heads (64 vs 16) and 2× more KV-heads (8 vs 4) than the base GLA layers. The qwen3 model class has no concept of per-layer variable attention head counts, so it initializes all layers with the base GLA head count. The full-attention layers' checkpoint weights (sized for the larger head counts) cannot fit into the model's smaller tensors, causing the `ignore_mismatched_sizes` RuntimeError. Proper loading requires either a transformers `qwen35` model class that understands the hybrid layout, or a GGUF tensor name remapping pass that correctly expands the per-layer attention configs.

## Fix
**Bug 1 fixed:** Added `model_to_load=None` parameter to `_patched_load_gguf_checkpoint` in 26 loader files in `tt-forge-models`, and passed it through to `_orig_load_gguf_checkpoint`. The two files already using `*args, **kwargs` (`gpt_oss_swallow_20b_sft_v0_1_gguf` and `mradermacher_qwen_3_5_27b_derestricted_gguf`) were left unchanged.

**Bug 2 proposed fix (Tier B):** The darkidol loader (and all qwen3.5-based loaders) would need to load using a proper qwen35 model class, or the GGUF loading infrastructure would need to read `full_attention_interval` from the GGUF metadata and configure per-layer attention head counts when constructing the qwen3 model config. This is new infrastructure in transformers / the GGUF loading stack.

## Tier B justification
new-infrastructure: Loading qwen3.5 models with `full_attention_interval > 0` requires either a new `Qwen3_5Model` class in transformers with per-layer variable attention head counts, or a GGUF loading pass that populates per-layer overrides — neither exists today.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    379.04s (0:06:19) for second run (after loader fix)
- Tier A attempts: N/A

## Files changed
In `tt-forge-models` (remediation/darkidol_ballad_gguf-4B_i1_GGUF-single_device-inference):
- bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5a1e473f6328d956803a6c214d373b8d0302e6ee |
| tt-forge-models | 8c9458125277713d41a52f0c0adf1e73b737d1d2 |
