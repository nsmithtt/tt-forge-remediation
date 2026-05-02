# Remediation Summary: mradermacher_deepseek_nsfw_qwen3_i1_gguf-pytorch-i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_deepseek_nsfw_qwen3_i1_gguf/causal_lm/pytorch-i1_GGUF-single_device-inference]

## Result
FAIL — loader TypeError fixed (26 narrow-sig patchers updated); residual PCC=0.9846 < 0.99 is ttmlir-bf16-matmul-precision-floor on BH p150b for Qwen3-4B (36-layer) architecture

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Two-layer failure:

**Layer 1 (loader):** At pytest collection time, `TorchDynamicLoader.setup_test_discovery` calls `import_model_loader` for every model loader to discover variants. 26 Qwen3.5/gpt-oss loaders install a module-level GGUF patch via `_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint`. These patches used a narrow signature `(gguf_path, return_tensors=False)`. When transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, the installed narrow-sig wrapper raises TypeError. The mradermacher loader has no such patch of its own — it is a victim of session contamination.

**Layer 2 (tt-mlir):** After fixing the TypeError, the model (Qwen3-4B, 36 layers, embedding_length=4096, Q4_K_M GGUF) runs to completion on BH p150b but achieves PCC=0.9846, below the required 0.99. This matches the known WH/BH BF16 matmul precision floor pattern for Qwen3 architectures. The gap is consistent with multi-layer BF16 accumulation error; no incorrect op path is apparent.

## Fix
**Loader fix (applied):** In `tt-forge-models` on branch `remediation/mradermacher_deepseek_nsfw_qwen3_i1_gguf-single_device-inference`, changed all 26 `_patched_load_gguf_checkpoint` function signatures from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` and updated the inner call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)`.

Affected files:
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

**PCC fix (not attempted):** Would require preserving FP32 accumulation through tt-mlir BF16 matmul lowering — cross-cutting Tier B change.

## Tier B justification
cross-cutting — fixing the BF16 matmul precision floor requires changing accumulation precision through every matmul lowering in tt-mlir; this is not a scoped single-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    408.11s (0:06:48)
- Tier A attempts: N/A

## Files changed
In `tt-forge-models` (branch `remediation/mradermacher_deepseek_nsfw_qwen3_i1_gguf-single_device-inference`, commit `3bb94c642900ecf8ae9ed53dd2be17f2fe9cda5d`):
- 26 `*/causal_lm/pytorch/loader.py` files: narrow-sig `_patched_load_gguf_checkpoint` → `*args, **kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3bb94c642900ecf8ae9ed53dd2be17f2fe9cda5d |
