# Remediation Summary: avalon2244_qwen3_5_4b_claude_opus_4_6_distilled_gguf-causal_lm-pytorch-4B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[avalon2244_qwen3_5_4b_claude_opus_4_6_distilled_gguf/causal_lm/pytorch-4B_GGUF-single_device-inference]

## Result
FAIL — qwen35 GGUF hybrid SSM+Attention architecture has no transformers GGUF loading support; weight mapping is unimplemented for this architecture

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-ssm-architecture-not-supported

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

State dict report showed:
- `model.layers.{0...30}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight` — MISSING (no mapping from GGUF tensor names to HF parameter names)
- `model.layers.{3,7,11,15,19,23,27,31}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight` — MISMATCH (checkpoint has full-attention head sizes, model config has sliding-window head sizes)

The original test failure (`TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`) was fixed first — see Fix section.

## Root cause

**First bug (fixed):** Other Qwen3.5 GGUF loaders imported during test collection monkey-patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow `(gguf_path, return_tensors=False)` signature. When transformers 5.2.0 upgraded `from_pretrained` to pass `model_to_load=dummy_model`, the patched function raised `TypeError`. 26 affected loaders were fixed.

**Second bug (unfixed, Tier B):** Inspecting the GGUF file reveals it declares `general.architecture = "qwen35"` with SSM metadata fields (`qwen35.ssm.conv_kernel`, `qwen35.ssm.state_size`, `qwen35.ssm.inner_size`, `qwen35.full_attention_interval = 4`) and tensor patterns:

- SSM/linear-attention layers (most blocks): `blk.N.attn_qkv.weight`, `blk.N.attn_gate.weight`, `blk.N.ssm_a`, `blk.N.ssm_alpha.weight`, `blk.N.ssm_beta.weight`, `blk.N.ssm_conv1d.weight`, `blk.N.ssm_dt.bias`, `blk.N.ssm_norm.weight`, `blk.N.ssm_out.weight`
- Full-attention layers (every 4th block: 3, 7, 11, 15, 19, 23, 27, 31): separate `blk.N.attn_q.weight` [8192, 2560], `blk.N.attn_k.weight` [1024, 2560], `blk.N.attn_v.weight` [1024, 2560], `blk.N.attn_output.weight` [4096, 2560]

This is a hybrid Mamba (SSM) + Attention architecture, similar to Jamba/Falcon-Mamba, NOT the standard Qwen3 transformer. Transformers 5.2.0 lists `"qwen35"` in neither `GGUF_SUPPORTED_ARCHITECTURES` nor `GGUF_TO_TRANSFORMERS_MAPPING`. The existing monkey-patch approach (mapping `model_type: "qwen35" → "qwen3"`) creates a standard Qwen3 config which has no SSM parameters, causing all SSM-layer weights to appear MISSING and the full-attention layers (with 64 Q heads vs the config's 16) to appear MISMATCHED.

The closest matching transformers model class is `Qwen3NextForCausalLM`, which has `linear_attn` submodules and a `decoder_sparse_step` (= `full_attention_interval`). However, the GGUF-to-HF weight name mapping for `blk.N.ssm_*` → `model.layers.N.linear_attn.*` is unimplemented and non-trivial: `in_proj_qkvz.weight` must be assembled from `attn_qkv.weight` + `attn_gate.weight`, and `in_proj_ba.weight` must be mapped from `ssm_alpha.weight` + `ssm_beta.weight`. Additionally, the Qwen3Next config fields for SSM (`linear_conv_kernel_dim`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`, `linear_num_value_heads`) have no documented mapping from the GGUF metadata fields.

## Fix
**Applied:** Fixed the narrow `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature in 26 Qwen3.5/gpt-oss-swallow GGUF loaders to `_patched_load_gguf_checkpoint(*args, **kwargs)`, forwarding all arguments to `_orig_load_gguf_checkpoint`. This allows transformers 5.2.0's `from_pretrained` to pass `model_to_load=dummy_model` correctly.

Changed files in `tt_forge_models` (remediation branch `remediation/avalon2244_qwen3_5_4b_claude_opus_4_6_distilled_gguf-causal_lm-pytorch-4B_GGUF-single_device-inference`):
- 26 `*/causal_lm/pytorch/loader.py` files under `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `gpt_oss_swallow_{20b_sft_v0_1_mxfp4_moe,20b_rl_v0_1,120b_rl_v0_1}_gguf`, `mradermacher_{bartleby_qwen3_5_4b,crow_4b_opus_4_6_distill_heretic_qwen3_5,gpt_oss_swallow_120b_rl_v0_1,huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1,luna_qwen3_5_27b_v5_i1,qwen3_5_27b,qwen3_5_27b_homebrew,qwen3_5_27b_tainted_heresy,qwen3_5_4b_abliterated_i1,qwen3_5_4b_ara_heresy_v2,qwen3_5_4b_gabliterated,qwen3_5_4b_sompoa_heresy_v2,qwen3_5_4b_unfiltered,qwen3_5_4b_unredacted_max,qwen3_5_9b_abliterated_i1,vilm_0_8b_sft}_gguf`, `qwen_3_5_imatrix_gguf`, `tvall43_qwen3_5_{4b_heretic_v2_i1,2b_heretic_v3b_i1}_gguf`, `unified_reward_flex_qwen35_27b_gguf`

**Proposed (unfixed):** Add `"qwen35"` GGUF architecture support targeting `Qwen3NextForCausalLM`:
1. Register `"qwen35"` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_FAST_CONVERTERS`
2. Add config field mappings: `block_count→num_hidden_layers`, `embedding_length→hidden_size`, `feed_forward_length→intermediate_size`, `attention.head_count→num_attention_heads`, `attention.head_count_kv→num_key_value_heads`, `ssm.conv_kernel→linear_conv_kernel_dim`, `ssm.inner_size` (maps to linear attention inner size), `full_attention_interval→decoder_sparse_step`
3. Implement per-block-type tensor name mapping (linear-attention blocks vs full-attention blocks)
4. Assemble `in_proj_qkvz` from `attn_qkv` + `attn_gate` and `in_proj_ba` from `ssm_alpha` + `ssm_beta`

## Tier B justification
new-infrastructure: Implementing GGUF loading support for the `qwen35` hybrid SSM+Attention architecture requires registering a new architecture, adding ~30 config field mappings, implementing per-block-type tensor name mapping (different tensor sets for SSM vs full-attention blocks), and constructing concatenated weight matrices from multiple GGUF tensors — none of which exist in the current codebase. The exact mapping from GGUF SSM tensor names to Qwen3Next HF parameter names is undocumented and requires empirical verification.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    99.77s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9716f997056163e40eeb391f6034512b34fe062d |
| tt-forge-models | 634452dc678db645d1c88e4034cb9690593794e6 |
