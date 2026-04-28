# Remediation Summary: davidau_openai_gpt_oss_20b_coder_neo_code_di_matrix_gguf/causal_lm/pytorch-CODER_NEO_CODE_DIMAT_IQ4_NL-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[davidau_openai_gpt_oss_20b_coder_neo_code_di_matrix_gguf/causal_lm/pytorch-CODER_NEO_CODE_DIMAT_IQ4_NL-single_device-inference]

## Result
XFAIL — GPT-OSS 20B dequantizes from GGUF IQ4_NL to bfloat16 (~40 GB), far exceeding single-device DRAM (12 GB n150, 24 GB p150b); the loader bug (TypeError) was fixed as a prerequisite

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-20b-gpt-oss-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Fatal Python error: Segmentation fault

Current thread 0x000079d606d5e080 (most recent call first):
  File ".../torch/_ops.py", line 841 in __call__
  File ".../tt_torch/torch_overrides.py", line 34 in __torch_function__
  File ".../torch/_ops.py", line 841 in __call__
  File ".../torch/fx/interpreter.py", line 336 in call_function
  File ".../torch/fx/interpreter.py", line 256 in run_node
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 859 in extract_compiled_graph_helper
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 737 in extract_compiled_graph
  File ".../tt_torch/backend/backend.py", line 215 in _call_experimental_compile
```

Preceded by a loader-layer TypeError (fixed): `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

## Root cause

**Loader bug (fixed):** Transformers 5.2.0 added `model_to_load=None` to `load_gguf_checkpoint()`. Twenty-six GGUF model loaders had monkey-patched that function with a fixed signature `(gguf_path, return_tensors=False)`, which rejected the new kwarg. During pytest collection all 26 loaders are imported (via `TorchDynamicLoader.setup_test_discovery`), leaving `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` pointing at the last loader's broken patch (`unified_reward_flex_qwen35_27b_gguf`). When the DavidAU test ran, `AutoModelForCausalLM.from_pretrained` called the monkey-patched function with `model_to_load=dummy_model`, raising TypeError.

**Hardware-class failure (XFAIL):** After fixing the loader bug the model loads successfully but crashes with a segfault in `partition_fx_graph_for_cpu_fallback`. The DavidAU model is GPT-OSS 20B (Qwen3-MoE architecture). Transformers dequantizes GGUF IQ4_NL weights to bfloat16, producing ~40 GB of weights in host RAM. `_call_experimental_compile` moves all parameters to the XLA device via `arg.to(torch.device("xla"))`. No single TT device can accommodate 40 GB (n150: 12 GB, p150b: 24 GB), causing the XLA/TT C++ layer to crash without raising a Python exception. The non-GGUF equivalent (`gpt_oss/pytorch-20B-single_device-inference`) is already marked `EXCLUDE_MODEL # Too large for single chip, run as tensor_parallel instead` in the same test config.

## Fix

**Loader fix (tt_forge_models, committed):** Updated all 26 GGUF loaders that had a broken `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature to use `(*args, **kwargs)` and pass through to `_orig_load_gguf_checkpoint(*args, **kwargs)`. This allows the entire monkey-patch chain to forward the new `model_to_load` kwarg to the real transformers function.

Files changed (26 loaders in tt_forge_models):
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

**XFAIL config (tt-xla, committed):** Added `KNOWN_FAILURE_XFAIL` entry for this model to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (segfault — hardware-class, not attempted on device after XFAIL classification)
- Hardware:    n150
- Duration:    ~23 min (model load + segfault during compilation)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: 26 GGUF loader files (signature fix)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 38495772581466c7bbf4f1427737051ad0c59ae1 |
| tt-forge-models | b29da7426c414f70db0d56751ff3ccfff5557dfa |
