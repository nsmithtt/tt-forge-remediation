# Remediation Summary: gpt_oss_heretic_gguf-causal_lm-pytorch-20B_Heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_heretic_gguf/causal_lm/pytorch-20B_Heretic_GGUF-single_device-inference]

## Result
XFAIL — 20B GPT-OSS model in BF16 (~40 GB) exceeds n150 DRAM capacity (~32 GB); INTERNAL: Error code: 13 during device execution

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: `raise AttributeError(` — gpt-oss GGUF architecture not registered in GGUF_SUPPORTED_ARCHITECTURES and GGUF_TO_TRANSFORMERS_MAPPING, plus narrow `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` from another GGUF loader imported during collection causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

After loader fixes: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` from `torch_xla._XLAC._run_cached_graph` during device execution.

## Root cause
Multiple loader bugs stacked:
1. **Missing `gguf>=0.10.0`** in requirements.txt → ImportError from `modeling_gguf_pytorch_utils`.
2. **`_patched_load_gguf_checkpoint` narrow signature** (`gguf_path, return_tensors=False`) in several GGUF loaders (gpt_oss_swallow_120b, gutenocr_3b, qwen_3_5_imatrix, mradermacher_*) imported at collection time; transformers 5.2.0 passes `model_to_load=dummy_model` → `TypeError`. The gpt_oss_heretic_gguf loader itself does not patch, but is a victim of another loader's global patch.
3. **gpt-oss GGUF arch not registered** — GGUF file reports `general.architecture = gpt-oss`, which is not in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING` → `AttributeError` / `ValueError`. Fix: register as alias for qwen3_moe (same architecture) with custom tensor processors.
4. **`Qwen3MoeExperts` for-loop segfault** — XLA graph tracing segfaults on the Python for-loop over dynamically-sized expert tensors; fix: set `config._experts_implementation = "batched_mm"`.
5. **`load_shard_spec` wrong attribute names** — original loader referenced `layer.mlp.router.weight` (doesn't exist; correct name is `layer.mlp.gate.weight`) and non-existent bias attributes.

After all loader fixes, the model loads and compiles successfully. At execution, it fails with `INTERNAL: Error code: 13`. Root cause: the GPT-OSS 20B model has 24 layers × 32 experts, hidden_size=2880, moe_intermediate_size=2880 → ~20B parameters. Dequantized to BF16: ~40 GB. n150 DRAM capacity: ~32 GB. This is a hardware capacity ceiling.

## Fix
**Loader fixes (tt-forge-models, remediation branch `remediation/gpt_oss_heretic_gguf-causal_lm-pytorch-20B_Heretic_GGUF-single_device-inference`):**
- `gpt_oss_heretic_gguf/causal_lm/pytorch/requirements.txt` — add `gguf>=0.10.0`
- `gpt_oss_heretic_gguf/causal_lm/pytorch/loader.py` — add `_patch_gpt_oss_support()` to register gpt-oss as qwen3_moe alias; `_patched_load_gguf_checkpoint(*args, **kwargs)` to fix model_type; set `_experts_implementation=batched_mm`; fix `load_shard_spec` (router→gate)
- `gpt_oss_swallow_120b_rl_v0_1_gguf`, `gutenocr_3b_i1_gguf`, `mradermacher_qwen3_5_27b_homebrew_gguf`, `qwen_3_5_imatrix_gguf` — fix narrow `_patched_load_gguf_checkpoint` signature to `*args, **kwargs`

**Test config (tt-xla, remediation branch):**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add `KNOWN_FAILURE_XFAIL` entry for this test

## Verification
- pytest exit: FAIL (INTERNAL: Error code: 13 — hardware OOM)
- Hardware: n150
- Duration: 964.36s (0:16:04) — model loaded and compiled; failed at device execution
- Tier A attempts: N/A

## Files changed
**tt-forge-models:**
- `gpt_oss_heretic_gguf/causal_lm/pytorch/requirements.txt` (new)
- `gpt_oss_heretic_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gutenocr_3b_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py` (and 3 other mradermacher loaders)
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`

**tt-xla:**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 86b8d4fd11d14bb2a53845945f7dbe4685527169 |
| tt-forge-models | 48189a428e830f8d27ddd33a62420a0a6b4bed1b |
