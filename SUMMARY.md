# Remediation Summary: lewdiculous_captainerisnebula_12b_chimera_v1_1_gguf_iq_imatrix-causal_lm-pytorch-12B_Chimera_v1.1_GGUF_IQ_Imatrix-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lewdiculous_captainerisnebula_12b_chimera_v1_1_gguf_iq_imatrix/causal_lm/pytorch-12B_Chimera_v1.1_GGUF_IQ_Imatrix-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The CI failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

On reproduction with gguf installed, two additional bugs surfaced:
1. `OSError: Lewdiculous/CaptainErisNebula-12B-Chimera-v1.1-GGUF-IQ-Imatrix does not appear to have a file named CaptainErisNebula-12B-Chimera-v1.1-IQ4_XS-imatrix.gguf`
2. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

## Root cause
Three loader bugs:

1. **Missing requirements.txt**: The loader had no `requirements.txt`, so `gguf>=0.10.0` was not installed in CI, causing the ImportError before even reaching the HuggingFace download.

2. **Wrong GGUF filename**: The loader had `GGUF_FILE = "CaptainErisNebula-12B-Chimera-v1.1-IQ4_XS-imatrix.gguf"` but the actual filename in the HuggingFace repo is `CaptainErisNebula-12B-Chimera-v1.1-IQ4_XS-imat.gguf` (shortened suffix).

3. **Narrow `_patched_load_gguf_checkpoint` signature in 26 other loaders**: Transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. Twenty-six loaders in tt_forge_models patched this function with the old narrow signature `(gguf_path, return_tensors=False)` that does not forward `**kwargs`. During pytest collection all loaders are imported, so whichever of these loaders was collected last overwrote `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with the narrow-signature stub. When the lewdiculous model's `AutoModelForCausalLM.from_pretrained()` locally imported `load_gguf_checkpoint` from that module, it got the stubbed version that rejected `model_to_load`.

## Fix
All fixes are in `tt_forge_models` (submodule of tt-xla):

1. **Added** `lewdiculous_captainerisnebula_12b_chimera_v1_1_gguf_iq_imatrix/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`
2. **Fixed** `lewdiculous_captainerisnebula_12b_chimera_v1_1_gguf_iq_imatrix/causal_lm/pytorch/loader.py`: `GGUF_FILE` changed from `IQ4_XS-imatrix.gguf` to `IQ4_XS-imat.gguf`
3. **Fixed 26 loaders** with narrow-signature `_patched_load_gguf_checkpoint`: added `**kwargs` to the function signature and forwarded to `_orig_load_gguf_checkpoint`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    567.02s (0:09:27)
- Tier A attempts: N/A

## Files changed
- `lewdiculous_captainerisnebula_12b_chimera_v1_1_gguf_iq_imatrix/causal_lm/pytorch/requirements.txt` (created)
- `lewdiculous_captainerisnebula_12b_chimera_v1_1_gguf_iq_imatrix/causal_lm/pytorch/loader.py`
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

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e63e6550237d34b572cb63231370c4c39256db6f |
| tt-forge-models | 1053234ca721fd5d56bd97885341fbbc7faade7d |
