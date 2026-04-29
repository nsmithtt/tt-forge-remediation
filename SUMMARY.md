# Remediation Summary: bella_bartender_8b_llama3_1_i1_gguf-causal_lm-pytorch-bella_bartender_v2_8b_i1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bella_bartender_8b_llama3_1_i1_gguf/causal_lm/pytorch-bella_bartender_v2_8b_i1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — PCC=0.978 below required 0.99 after loader fix; compiler precision bug remains

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original (reproduced): TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

After loader fix: AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9781789913733782. Required: pcc=0.99.

## Root cause
Two independent bugs:

**Bug 1 (loader — fixed):** 26 qwen35-related loaders in `tt_forge_models` define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it as a global replacement for `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. Transformers 5.x changed `load_gguf_checkpoint` to accept a `model_to_load` keyword argument. During pytest collection `TorchDynamicLoader.setup_test_discovery` imports all loaders, installing the broken patch. When bella_bartender's test subsequently calls `AutoModelForCausalLM.from_pretrained`, transformers calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)` which hits the patched version and raises `TypeError`. The loader alphabetically preceding bella_bartender that installs this patch is `bartowski_coniccat_qwen3_5_27b_writer_gguf`.

**Bug 2 (tt-mlir — unfixed):** After fixing the loader, the model compiles and runs on silicon but produces PCC=0.978 against the CPU reference. The bella_bartender_v2_8b model is a Q4_K_M GGUF loaded with `torch_dtype=bfloat16`. Both reference and device runs use bfloat16 weights, yet the TT compiler introduces sufficient additional numerical error to drop PCC below the 0.99 threshold. This is consistent with the known `ttmlir-f32-precision-not-preserved` pattern where compiler lowerings accumulate bfloat16 rounding error beyond what CPU torch accumulates.

## Fix
**Bug 1 (applied):** Updated all 26 broken loaders in tt-forge-models to accept `model_to_load=None, **kwargs` in `_patched_load_gguf_checkpoint` and forward them to `_orig_load_gguf_checkpoint`. Branch: `remediation/bella_bartender_8b_llama3_1_i1_gguf-causal_lm-pytorch-bella_bartender_v2_8b_i1_Q4_K_M_GGUF-single_device-inference` in tt-forge-models.

Files changed:
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
- mradermacher_qwen_3_5_27b_derestricted_gguf (already had *args/**kwargs — not changed)
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

**Bug 2 (not fixed):** Proposed fix would live in tt-mlir lowering passes to preserve bfloat16 numerical fidelity (cross-cutting change).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The PCC gap (0.978 vs 0.99) reflects accumulated bfloat16 rounding error introduced across multiple lowering passes in tt-mlir. Fixing it would require improving numerical precision through every affected op lowering (matmul, attention, etc.), which is a cross-cutting change across multiple files.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    446.69s (0:07:26) — model ran on silicon, failed PCC check
- Tier A attempts: N/A

## Files changed
- tt-forge-models: 26 loader.py files (see Fix section above)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 12a70ac91dad670992213c308c224b547b1c867b |
