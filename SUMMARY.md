# Remediation Summary: luna_qwen3_5_4b_v5_i1_gguf-causal_lm-pytorch-4B_v5_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[luna_qwen3_5_4b_v5_i1_gguf/causal_lm/pytorch-4B_v5_i1_GGUF-single_device-inference]

## Result
FAIL — qwen35 GGUF hybrid tensor mapping missing for GLA/SSM layers in transformers

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-no-transformers-tensor-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
First failure (fixed):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Second failure (Tier B, unfixed):
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```
With mismatches:
```
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.q_proj.weight | MISMATCH | ckpt: torch.Size([8192, 2560]) vs model:torch.Size([2048, 2560])
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.k_proj.weight | MISMATCH | ckpt: torch.Size([1024, 2560]) vs model:torch.Size([512, 2560])
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.v_proj.weight | MISMATCH | ckpt: torch.Size([1024, 2560]) vs model:torch.Size([512, 2560])
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.o_proj.weight | MISMATCH | ckpt: torch.Size([2560, 4096]) vs model:torch.Size([2560, 2048])
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.q_norm.weight | MISMATCH | ckpt: torch.Size([256]) vs model:torch.Size([128])
model.layers.{3, 7, 11, 15, 19, 23, 27, 31}.self_attn.k_norm.weight | MISMATCH | ckpt: torch.Size([256]) vs model:torch.Size([128])
```

## Root cause
**Bug 1 (fixed — loader):** 26 GGUF loaders on branch `hf-bringup-16` had a narrow signature:
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):`
transformers 5.2.0 calls `load_gguf_checkpoint` with `model_to_load=dummy_model` (positional kwarg). During pytest collection all loaders are imported, so whichever loader ran first would install this narrow-sig patch globally; the luna loader (which itself has no patch) then failed when transformers called the stale patch. Fixed by widening all 26 loaders to `(*args, **kwargs)`.

**Bug 2 (unfixed — Tier B):** The `luna_qwen3_5_4b_v5_i1_gguf` model uses the `qwen35` GGUF architecture, which is a hybrid GatedDeltaNet (linear attention / SSM) + full self-attention model with `full_attention_interval=4`:
- Layers 0,1,2 (and 4,5,6, etc.): GLA / linear-attention — GGUF tensor names `blk.N.attn_qkv`, `blk.N.attn_gate`, `blk.N.ssm_a`, `blk.N.ssm_alpha`, `blk.N.ssm_beta`, `blk.N.ssm_conv1d`, `blk.N.ssm_dt`, `blk.N.ssm_norm`, `blk.N.ssm_out`
- Layers 3,7,11,15,19,23,27,31: Full self-attention — GGUF tensor names `blk.N.attn_q`, `blk.N.attn_k`, `blk.N.attn_v`, `blk.N.attn_output`, `blk.N.attn_q_norm`, `blk.N.attn_k_norm`

The existing `_patch_qwen35_support()` converts `qwen35` → `qwen3` in the GGUF config mapping, which creates a uniform `Qwen3ForCausalLM` with global `num_attention_heads=16, head_dim=128`. The full-attention layers in the GGUF have `32 Q-heads × 256 head_dim` (4× larger), causing shape mismatches.

The correct target class is `Qwen3_5ForCausalLM` / `Qwen3_5TextModel` (model_type `qwen3_5_text`), which IS present in transformers 5.x and properly models the hybrid architecture with per-layer `layer_types`. However, `qwen35` is absent from `GGUF_CONFIG_MAPPING`, and there are no GGUF tensor name → `Qwen3_5TextModel` weight name mappings for GLA layers (`linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, SSM weights, etc.) in `get_gguf_hf_weights_map`.

## Fix
Bug 1 was fixed in tt-forge-models branch `remediation/luna_qwen3_5_4b_v5_i1_gguf-causal_lm-pytorch-4B_v5_i1_GGUF-single_device-inference` (commit `d595410479`):
- 26 files in `tt-xla/third_party/tt_forge_models/` changed:
  `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(*args, **kwargs):`
  `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)`

Bug 2 fix (proposed, Tier B): In `transformers/modeling_gguf_pytorch_utils.py`:
1. Add `'qwen35': {<key mappings>}` to `GGUF_CONFIG_MAPPING`, mapping to `model_type: qwen3_5_text`. Key mappings need `full_attention_interval`, `ssm.*` fields, and per-layer head counts.
2. Add GLA tensor name mappings in `get_gguf_hf_weights_map` for `MODEL_ARCH.QWEN35`: `blk.N.attn_qkv` → `model.layers.N.linear_attn.in_proj_qkv`, `blk.N.attn_gate` → `model.layers.N.linear_attn.in_proj_z`, `blk.N.ssm_*` → `model.layers.N.linear_attn.*`.
3. Ensure full-attention layers map to `model.layers.N.self_attn.*` using separate q/k/v tensors.

## Tier B justification
**Indicator:** new-infrastructure

The `qwen35` GGUF architecture requires adding an entirely new tensor name mapping path for GLA/SSM layers to transformers' `get_gguf_hf_weights_map`. The `Qwen3_5TextModel` uses a different parameter namespace (`linear_attn.*`, SSM-specific tensors) with no existing bridge from gguf-py's QWEN35 tensor names. This requires changes to transformers' GGUF loading infrastructure and potentially gguf-py's MODEL_ARCH enum — cross-repo new-infrastructure work.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 346.51s (second run after narrow-sig fix)
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
| tt-xla          | 227a052cbafd0cbd72b469415a18ea236657cdea |
| tt-forge-models | d595410479de7df9ca9b27e3ab97bdb809f83f7f |
