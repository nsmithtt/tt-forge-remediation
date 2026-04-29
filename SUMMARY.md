# Remediation Summary: bartowski_ready_art_brisk_evolution_12b_v0_1_gguf-causal_lm-pytorch-ReadyArt_Brisk-Evolution-12B-v0.1-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_ready_art_brisk_evolution_12b_v0_1_gguf/causal_lm/pytorch-ReadyArt_Brisk-Evolution-12B-v0.1-GGUF-single_device-inference]

## Result
FAIL — loader TypeError fixed; secondary PCC=0.9833 < 0.99 is WH BF16 matmul precision (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
wh-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(Reported as `AttributeError: 'NoneType' object has no attribute 'config'` in the CI log; the
test framework wraps the original exception. The actual error is the TypeError above, which
surfaces when running pytest directly with -svv.)

## Root cause

**Loader layer (fixed):** 26 GGUF loaders in `tt_forge_models` monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time to register the
`qwen35` architecture alias. Their patched function used the narrow signature
`(gguf_path, return_tensors=False)`. Transformers 5.2.0 added a `model_to_load=dummy_model`
keyword argument to this call site (`modeling_utils.py:4016`). When pytest collects tests, any
of these 26 loaders can be imported before this test runs, leaving the narrow-signature patcher
installed. The subsequent call from `AutoModelForCausalLM.from_pretrained` then raises
`TypeError`.

The ReadyArt_Brisk-Evolution-12B model uses the `llama` GGUF architecture (already registered
in transformers) and does not itself need architecture patching.

**Compiler layer (unfixed, Tier B):** After fixing the TypeError the model compiles and runs,
but output PCC is 0.9833 vs the required 0.99 threshold. The model has 50 Llama transformer
layers with hidden_size=4096 and intermediate_size=14336. PCC shortfall of 0.007 is consistent
with the known WH BF16 matmul accumulation difference from x86 CPU that was previously
classified as Tier B for Gemma 7B (PCC≈0.915, tt-xla #2861) and Qwen3 4B (PCC≈0.864).

## Fix

**Loader fix (committed to remediation branch):**
Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `(*args, **kwargs)`
in 26 GGUF loader files, with the inner call updated from
`_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `(*args, **kwargs)`.

Committed as `6b71b221e6` on
`remediation/bartowski_ready_art_brisk_evolution_12b_v0_1_gguf-causal_lm-pytorch-ReadyArt_Brisk-Evolution-12B-v0.1-GGUF-single_device-inference`
in `tt_forge_models`.

**Proposed compiler fix (not attempted):**
Preserve FP32 accumulation for BF16 matmuls through the StableHLO→TTIR lowering pipeline on
WH hardware. Would require cross-cutting changes across the matmul lowering passes in tt-mlir.

## Tier B justification
cross-cutting — Fixing the WH BF16 matmul precision floor requires preserving higher-precision
accumulation through every matmul lowering pass; not a single scoped change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    565.57s (9:25) for the run that revealed the PCC failure
- Tier A attempts: N/A

## Files changed
tt_forge_models (remediation branch `6b71b221e6`):
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen_3_5_27b_derestricted_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- (and 6 more in bartowski_skywork_skywork_swe_32b_gguf, daniloreddy, dmind, gpt_oss_swallow_20b, abhiray)
- `bartowski_skywork_skywork_swe_32b_gguf/causal_lm/pytorch/requirements.txt` (added gguf>=0.10.0)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 6b71b221e6d550686e6adad9a41a9b407c7ea504 |
