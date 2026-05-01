# Remediation Summary: glm_4_7_trashflash_think_sorete_1b_i1_gguf-causal_lm-pytorch-4_7_TrashFlash_Think_Sorete_1B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_trashflash_think_sorete_1b_i1_gguf/causal_lm/pytorch-4_7_TrashFlash_Think_Sorete_1B_i1_GGUF-single_device-inference]

## Result
FAIL — PCC=0.957 below required 0.99; residual ttmlir-bf16-matmul-precision-floor after loader and slice fixes

## Stack layer
loader, tt-xla, tt-mlir

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
Original error:
```
RuntimeError: Value out of range (expected to be in range of [-12, 11], but got -511)
While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})
Original traceback: cache_utils.py:214: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

Secondary errors encountered during debugging:
1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` — 26 GGUF loaders with narrow-sig patch contaminating the session
2. `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set` — GGUF i1 tokenizer has no chat template

Residual error after fixes:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9570118207998776. Required: pcc=0.99.
```

## Root cause
Three bugs chained:

**Bug 1 (loader, session contamination):** 26 GGUF loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` with a narrow signature and install it as a module-level monkey-patch. During pytest collection, all loaders are imported, so these narrow-sig patches replace `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. When transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` inside `from_pretrained`, the narrow-sig function rejects the new kwarg.

**Bug 2 (loader):** The GLM-4.7-TrashFlash-Think.Sorete-1B i1 GGUF tokenizer has no `chat_template` set. The loader unconditionally calls `tokenizer.apply_chat_template(...)` which raises `ValueError`.

**Bug 3 (tt-xla, Tier A fixed):** The Gemma-3 sliding window attention cache (`cache_utils.py:214`) computes `full_value_states[:, :, -sliding_window + 1:, :]`. With `sliding_window=512` and `seq_len=12` test tokens, `start = -511 < -12 = -seq_len`. PyTorch eager silently clamps this; the XLA lazy backend raises "Value out of range". The `clamp_out_of_range_slice_starts` FX pass was added to `tt-xla/python_package/tt_torch/backend/passes.py` to pre-clamp out-of-range static negative slice starts.

**Residual bug (tt-mlir, Tier B):** After all three fixes, the model compiles and runs but produces PCC=0.957. CPU BF16 vs CPU FP32 gives PCC=0.9896 — the TT silicon loses an additional 0.033 PCC below the BF16 floor. This is the known `ttmlir-bf16-matmul-precision-floor` issue on Blackhole p150b hardware with BF16 matmul accumulation.

## Fix
**Bug 1:** Updated 26 GGUF loaders in `tt-forge-models` to use `(*args, **kwargs)` signature for `_patched_load_gguf_checkpoint` and pass them through to the original function. Affects all `mradermacher_qwen3_5_*`, `qwen_3_5_imatrix_gguf`, `gpt_oss_swallow_*`, `tvall43_qwen3_5_*`, `bartowski_*`, `daniloreddy_*`, `dmind_*`, `unified_reward_*` loaders.

**Bug 2:** Added `chat_template is not None` guard to `load_inputs` in `glm_4_7_trashflash_think_sorete_1b_i1_gguf/causal_lm/pytorch/loader.py`; falls back to plain `sample_text` when no template is set.

**Bug 3 (Tier A):** Added `clamp_out_of_range_slice_starts` FX pass to `tt-xla/python_package/tt_torch/backend/passes.py`, called from `torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`. The pass iterates `aten.slice.Tensor` nodes and clamps any static `start < -dim_size` to `-dim_size`.

**Residual (proposed):** The BF16 matmul accumulation precision in tt-mlir would need to be improved (F32 intermediate accumulation, or math fidelity override) — a cross-cutting change across the lowering pipeline.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting: Improving BF16 matmul accumulation precision to close the 0.033 PCC gap (0.9896 CPU BF16 → 0.957 TT BF16) would require changes to matmul lowering across multiple passes in tt-mlir and/or the tt-metal kernel. This is the same class of issue seen in Gemma 7B (n150), GaMS3-12B (p150b), and BlackSheep-RP 12B (p150b).

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    333.30s
- Tier A attempts: 1

## Files changed
**tt-forge-models (remediation branch):**
- `glm_4_7_trashflash_think_sorete_1b_i1_gguf/causal_lm/pytorch/loader.py` — chat_template guard
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

**tt-xla (remediation branch):**
- `python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts`
- `python_package/tt_torch/backend/backend.py` — import and call new pass

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 34f4193ca04e0dd1e580642da24abf402a3dddf9 |
| tt-forge-models | 789938654a7bf1f828c5473b8a4bc3bd7df264dc |
