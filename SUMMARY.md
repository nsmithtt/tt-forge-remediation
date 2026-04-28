# Remediation Summary: l3_1_moe_6x8b_dantes_peak_horror_r1_gguf-causal_lm-pytorch-DANTES_PEAK_HORROR_R1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[l3_1_moe_6x8b_dantes_peak_horror_r1_gguf/causal_lm/pytorch-DANTES_PEAK_HORROR_R1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — GGUF Llama MoE weight mapping missing: ffn_gate_exps/ffn_up_exps/ffn_down_exps tensors not loaded into model, all 32 MLP layers randomly initialized

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-llama-moe-weight-mapping-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8143305622014877. Required: pcc=0.95.

The test first failed during reproduction with a TypeError (different from the reported PCC failure), caused by a
separate bug in 26 GGUF loaders that globally patch load_gguf_checkpoint without the `model_to_load` kwarg added
in transformers 5.x. That bug was fixed. After the fix, the model loads but produces PCC 0.8143 vs the required
threshold.

## Root cause
Two bugs are present. Bug 1 was fixed; Bug 2 is the residual unfixed root cause of the PCC failure.

**Bug 1 (FIXED):** 26 GGUF loaders in tt_forge_models define `_patched_load_gguf_checkpoint(gguf_path,
return_tensors=False)` and globally replace `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with
this patched version. In transformers 5.x, `load_gguf_checkpoint` is now called by `modeling_utils.py` with
`model_to_load=dummy_model` as a kwarg. Because the patched functions do not accept `model_to_load`, a TypeError
is raised whenever any of these loaders has been imported (which happens at test collection time for all tests in
the session). Fixed by adding `model_to_load=None` to the signature of all 26 affected patchers and forwarding it
to the original function.

**Bug 2 (UNFIXED):** The GGUF file declares `general.architecture = "llama"` with `llama.expert_count = 6` and
`llama.expert_used_count = 6` (all-expert dense routing). Its FFN tensors use MoE naming:
`blk.N.ffn_gate_exps.weight`, `blk.N.ffn_up_exps.weight`, `blk.N.ffn_down_exps.weight`, and the routing gate
`blk.N.ffn_gate_inp.weight`. However, `LlamaTensorProcessor` (the GGUF tensor processor for the "llama"
architecture) has no MoE weight-handling logic — only `Qwen2MoeTensorProcessor` implements that. There is also no
`LlamaMoEForCausalLM` model class in transformers.

Consequently, `AutoModelForCausalLM.from_pretrained` creates a standard `LlamaForCausalLM` instance (no expert
layers), and `get_gguf_hf_weights_map` maps `model.layers.N.mlp.gate_proj` → `blk.N.ffn_gate` (not
`ffn_gate_exps`). The expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) are therefore not found in
the weight key map and are silently skipped. All 32 layers report:

```
model.layers.{0...31}.mlp.gate_proj.weight | MISSING |
model.layers.{0...31}.mlp.down_proj.weight | MISSING |
model.layers.{0...31}.mlp.up_proj.weight   | MISSING |
```

These parameters are randomly initialized. With 32 layers of random MLP weights, activations grow large and
unstable. A CPU f32 vs CPU bf16 comparison of the broken model gives PCC = -0.026, confirming the random weights
dominate. The TT-vs-CPU PCC of 0.8143 arises because numerical differences in bf16 computation (different
reduction order, overflow handling) are amplified through 32 layers of random large-magnitude weights.

## Fix
**Bug 1 fix (committed):** In the remediation branch of tt_forge_models
(`remediation/l3_1_moe_6x8b_dantes_peak_horror_r1_gguf-causal_lm-pytorch-DANTES_PEAK_HORROR_R1_Q4_K_M_GGUF-single_device-inference`),
added `model_to_load=None` to `_patched_load_gguf_checkpoint` and forwarded it to `_orig_load_gguf_checkpoint` in
26 loaders. Two loaders that already used `*args, **kwargs` were left unchanged.

Files changed:
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

**Bug 2 proposed fix (not implemented):**

1. Add `expert_count` → `num_local_experts` and `expert_used_count` → `num_experts_per_tok` to
   `GGUF_CONFIG_MAPPING["llama"]` in `transformers.modeling_gguf_pytorch_utils`.
2. Implement a `LlamaMoEForCausalLM` model class (or integrate with the existing `MixtralForCausalLM` by
   overriding `updated_architecture = "mixtral"` when `expert_count > 0` and the model is a standard Llama
   attention variant without sliding window).
3. Add MoE tensor handling to `LlamaTensorProcessor` mirroring `Qwen2MoeTensorProcessor.perform_fallback_tensor_mapping`
   to map `blk.N.ffn_gate_exps`, `blk.N.ffn_up_exps`, `blk.N.ffn_down_exps` to the appropriate expert weight
   keys.

This fix spans 3+ files and requires implementing a non-trivial new model class or a careful Mixtral config
override, making it new infrastructure rather than a targeted bug fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    933.91s (0:15:33) — run with Bug 1 fixed, Bug 2 still present
- Tier A attempts: N/A

## Files changed
### tt_forge_models (26 loader files — Bug 1 fix)
See Fix section for full list.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a5d7520f387b9837d45e3a4815e95486a1f951fa |
| tt-forge-models | 35aa9bc86a355447aaecbdc218f0befafb281677 |
