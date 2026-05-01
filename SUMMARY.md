# Remediation Summary: luna_qwen3_5_9b_v5_i1_gguf-causal_lm-pytorch-9B_v5_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[luna_qwen3_5_9b_v5_i1_gguf/causal_lm/pytorch-9B_v5_i1_GGUF-single_device-inference]

## Result
FAIL — qwen35 hybrid SSM+attention arch has no GGUF support in transformers; size mismatches on all 32 layers

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-hybrid-gguf-no-transformers-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before fix):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Failure after loader fix:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

Loading report (representative mismatches):
```
model.layers.{0...31}.post_attention_layernorm.weight               | MISSING
model.layers.{0...30}.self_attn.v_proj.weight                       | MISSING
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH | ckpt: [1024,4096] vs model: [512,4096]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH | ckpt: [8192,4096] vs model: [2048,4096]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.v_proj.weight | MISMATCH | ckpt: [1024,4096] vs model: [512,4096]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.o_proj.weight | MISMATCH | ckpt: [4096,4096] vs model: [4096,2048]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_norm.weight | MISMATCH | ckpt: [256] vs model: [128]
model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_norm.weight | MISMATCH | ckpt: [256] vs model: [128]
```

## Root cause
Two bugs, one loader-layer and one Tier B:

**Bug 1 (fixed) — cross-loader narrow-sig TypeError:** 26 GGUF loaders in tt-forge-models patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time using a narrow signature `(gguf_path, return_tensors=False)`. transformers 5.2.0 added `model_to_load=` kwarg and calls it as `load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)`. During pytest collection all loaders are imported; whichever narrow-sig wrapper was installed last raised `TypeError` when the luna loader called `AutoModelForCausalLM.from_pretrained`.

**Bug 2 (Tier B) — qwen35 hybrid GGUF architecture:** Luna-Qwen3.5-9B-v5 uses the `qwen35` GGUF architecture, which is Qwen3.5's hybrid SSM+attention model (GatedDeltaNet SSM layers + full-attention every 4th layer at indices 3,7,11,...,31). The cross-loader patches remap `model_type: qwen35 → qwen3`, loading into `Qwen3ForCausalLM` which has uniform 16 Q-heads / 4 KV-heads attention throughout. The full-attention layers in the checkpoint have 64 Q-heads / 8 KV-heads, causing `[8192,4096]` vs `[2048,4096]` shape mismatches on q_proj. The non-full-attention (SSM/GDA) layers are missing `post_attention_layernorm` and `self_attn.v_proj` entirely. transformers has no `Qwen3_5ForCausalLM` class with GGUF support.

## Fix
**Bug 1 fix (committed):** Updated all 26 `_patched_load_gguf_checkpoint` wrappers in tt-forge-models from narrow signature `(gguf_path, return_tensors=False)` to `(*args, **kwargs)`, and updated their inner `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` calls to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Files in:
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_*` (many variants)
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_*/`
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_*/`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/`

**Bug 2 proposed fix (Tier B):** The fix requires adding GGUF infrastructure for the `qwen35` architecture to transformers:
1. Add `qwen35` to `GGUF_CONFIG_MAPPING` pointing to `Qwen3_5Config`
2. Add GGUF tensor-name mappings for all GatedDeltaNet/SSM layer tensor types (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`, `attn_qkv`, `attn_gate` for non-full-attention layers; separate `attn_q`/`attn_k`/`attn_v`/`attn_output` with correct head counts for full-attention layers)
3. Wire up `GGUF_TO_FAST_CONVERTERS["qwen35"]` for tokenizer loading
This is upstream transformers work — no single-PR Tier A fix is possible.

## Tier B justification
Indicator: **new-infrastructure**. Correctly loading a `qwen35` GGUF model requires adding a complete new tensor-name mapping table for the hybrid SSM+GDA+attention layer types, plus wiring up `Qwen3_5Config` as the GGUF config class. This is 3+ files and must be done in upstream transformers (or by vendoring the model class), which is beyond a scoped one-repo Tier A fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    449.10s (Bug 2 run)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aac9013b3813b345764ce190059428e6d0d4e278 |
| tt-forge-models | ef6227fd1a9b7ed651a35d0c023070c427afe1a5 |
