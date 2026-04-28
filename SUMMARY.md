# Remediation Summary: gpt_5_distill_qwen3_4b_instruct_gguf-causal_lm-pytorch-4B_Instruct_Q4_K_S_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_5_distill_qwen3_4b_instruct_gguf/causal_lm/pytorch-4B_Instruct_Q4_K_S_GGUF-single_device-inference]

## Result
FAIL — WH BF16 matmul fp32-dest-acc precision floor: TT silicon PCC=0.878 vs CPU BF16 PCC=0.9998; Tier B cross-cutting fix required

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
tt-metal-bf16-matmul-fp32-dest-acc-precision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8779489109489146. Required: pcc=0.95.

## Root cause

Two issues were found:

**Issue 1 (Loader — fixed):** The test failed locally with `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Transformers 5.2.0 added a `model_to_load=dummy_model` kwarg to its `load_gguf_checkpoint` call in `modeling_utils.py:4016`. Twenty-six GGUF loaders (alphabetically before `gpt_5_distill_qwen3_4b_instruct_gguf`) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a narrow signature `(gguf_path, return_tensors=False)` that does not accept `model_to_load`. When pytest collects all tests, these loaders are imported before the target test and leave the narrow-signature patch in place.

**Issue 2 (tt-metal — Tier B):** After the loader fix, the test runs through to completion but gives PCC=0.878 on TT silicon vs. the CPU BF16 golden. The model is `Qwen3ForCausalLM` with 36 layers loaded in BF16. CPU BF16 vs CPU FP32 PCC is 0.9998 (essentially lossless), so the entire precision gap (0.9998 → 0.878) is attributable to the WH BF16 matmul fp32-dest-acc precision bug in tt-metal: WH BF16 matrix-multiply units do not accumulate to FP32, causing compounding rounding error across 36 transformer layers. This is the same bug as seen in Qwen3 4B (AQ-MedAI/Diver-Retriever-4B-1020, PCC=0.864) and Gemma 7B (tt-metal #39518 / tt-xla #2861).

## Fix

**Issue 1 (applied):** Updated all 26 narrow-signature `_patched_load_gguf_checkpoint` functions in tt-forge-models to use `(*args, **kwargs)` and pass them through to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Files changed in `tt-forge-models` on branch `remediation/gpt-5-distill-qwen3-4b-instruct-gguf-single-device-inference`:

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

**Issue 2 (not attempted):** Tier B — the WH BF16 matmul fp32-dest-acc precision bug requires cross-cutting changes to tt-metal's WH matmul kernel to use FP32 accumulators for BF16 inputs. The FP32 workaround used for small models is infeasible here: 4B params × 4 bytes = ~16 GB VRAM, exceeding n150's 12 GB capacity.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

Fixing the WH BF16 matmul precision floor requires changing FP32 accumulation behavior across all BF16 matmul kernels in tt-metal, touching many files across tt-metal and potentially tt-mlir. This is tracked as tt-metal #39518 / tt-xla #2861.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    354.23s (0:05:54)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 GGUF loader files — `_patched_load_gguf_checkpoint` signature fixed
- `tt-xla`: `third_party/tt_forge_models` submodule pointer bumped

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 29fd64afcd7fcc6c880304d0203ce5c1674057ed |
| tt-forge-models | 991af7b232ec4ded32d5a20d5f5cd01e727b3dff |
