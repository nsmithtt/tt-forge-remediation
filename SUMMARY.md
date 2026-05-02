# Remediation Summary: mradermacher_qwen3_5_27b_engineer_deckard_gemini_i1_gguf-causal_lm-pytorch-27B_Engineer_Deckard_Gemini_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_5_27b_engineer_deckard_gemini_i1_gguf/causal_lm/pytorch-27B_Engineer_Deckard_Gemini_i1_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5 hybrid GLA+attention architecture not supported: `full_attention_interval=4` produces GLA layers (missing self_attn weights) and non-standard Q projection dims that exceed what Qwen3ForCausalLM can represent

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
Original failure (before this session): `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

After installing gguf>=0.10.0 (already present in venv), reproduced as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing narrow-sig patches, secondary failure:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.

model.layers.{0...62}.self_attn.k_norm.weight         | MISSING
model.layers.{0...63}.post_attention_layernorm.weight | MISSING
model.layers.{3...63}.self_attn.v_proj.weight         | MISMATCH | ckpt: torch.Size([1024, 5120]) vs model: torch.Size([512, 5120])
model.layers.{3...63}.self_attn.q_proj.weight         | MISMATCH | ckpt: torch.Size([12288, 5120]) vs model: torch.Size([3072, 5120])
model.layers.{3...63}.self_attn.q_norm.weight         | MISMATCH | ckpt: torch.Size([256]) vs model: torch.Size([128])
model.layers.{3...63}.self_attn.k_proj.weight         | MISMATCH | ckpt: torch.Size([1024, 5120]) vs model: torch.Size([512, 5120])
model.layers.{3...63}.self_attn.o_proj.weight         | MISMATCH | ckpt: torch.Size([5120, 6144]) vs model: torch.Size([5120, 3072])
model.layers.{3...63}.self_attn.k_norm.weight         | MISMATCH | ckpt: torch.Size([256]) vs model: torch.Size([128])
```

## Root cause

Two bugs, one fixed and one Tier B:

**Bug 1 (fixed): narrow-sig `_patched_load_gguf_checkpoint` contamination.**
26 other GGUF loaders patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow-signature function `(gguf_path, return_tensors=False)`. When pytest collects all tests, these loaders are imported and their patches persist globally. transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` which fails with TypeError. Fix: widen all 26 patches to `(*args, **kwargs)`.

**Bug 2 (Tier B): Qwen3.5 hybrid GLA architecture incompatible with Qwen3ForCausalLM.**
GGUF metadata shows `qwen35.full_attention_interval=4` (every 4th layer is full attention, others are GLA/Gated Linear Attention), 64 layers, `key_length=256` (RoPE portion of head), `head_count=24`. The `_patch_qwen35_support()` maps `qwen35` → `qwen3`, setting `head_dim=256` from `key_length`. But the actual q_proj tensor is `[12288, 5120]` = 24 heads × 512 = 2× the RoPE head_dim (full q-head dim is 512, rope portion is 256). The GLA layers (those without `full_attention_interval` membership) produce MISSING self_attn weights. `Qwen3ForCausalLM` has no support for: (1) hybrid GLA+attention layer types, (2) asymmetric Q vs K/V head dimensions. Loading the model produces systematic MISSING and MISMATCH errors for 64 layers.

## Fix

**Bug 1 (applied):** In `tt_forge_models`, create remediation branch and change:
```python
# BEFORE (26 loaders)
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# AFTER
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
Also added `requirements.txt` with `gguf>=0.10.0` to `mradermacher_qwen3_5_27b_engineer_deckard_gemini_i1_gguf/causal_lm/pytorch/`.

**Bug 2 (proposed):** The loader needs to either:
- Use the HuggingFace `Qwen3_5ForCausalLM` class if/when one is added to transformers that supports `full_attention_interval`, GLA layers, and asymmetric Q head dims, OR
- Load the model using custom GLA-aware code (similar to how `qwen35` pytorch models are handled) that can represent hybrid architectures

This requires new infrastructure in transformers or a custom model class in tt_forge_models.

## Tier B justification
new-infrastructure: Loading Qwen3.5 hybrid GLA+attention architectures as `Qwen3ForCausalLM` produces systematic tensor shape mismatches across all 64 layers. A correct loader requires either a new `Qwen3_5ForCausalLM` class in transformers (supporting `full_attention_interval` and asymmetric Q/K head dims) or a custom implementation. This is not a targeted fix to one function but requires a new model class supporting GLA+self_attn hybrid layers.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 794.96s (0:13:14)
- Tier A attempts: N/A

## Files changed
In `tt_forge_models` remediation branch (`remediation/mradermacher_qwen3_5_27b_engineer_deckard_gemini_i1_gguf-causal_lm-pytorch-27B_Engineer_Deckard_Gemini_i1_GGUF-single_device-inference`):
- `mradermacher_qwen3_5_27b_engineer_deckard_gemini_i1_gguf/causal_lm/pytorch/requirements.txt` (added, `gguf>=0.10.0`)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py` (narrow-sig fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9a23c0103da6f6854d8f3d575c3b6ae28d9c6589 (branch: remediation/mradermacher_qwen3_5_27b_engineer_deckard_gemini_i1_gguf-causal_lm-pytorch-27B_Engineer_Deckard_Gemini_i1_GGUF-single_device-inference) |
