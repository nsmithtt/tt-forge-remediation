# Remediation Summary: qwen3_30b_a3b_instruct_2507_malaysian_dora_gguf-causal_lm-pytorch-30B_A3B_Instruct_2507_Malaysian_DoRA_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen3_30b_a3b_instruct_2507_malaysian_dora_gguf/causal_lm/pytorch-30B_A3B_Instruct_2507_Malaysian_DoRA_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B Q4_K_M GGUF dequantizes to ~60 GB BF16, exceeding p150b single-device DRAM (~31.84 GB); INTERNAL: Error code: 13 (OOM) after 1497s

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen3-30b-a3b-dram-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Test ran for 1497.81s (0:24:57) before OOM.

## Root cause
The original segfault was caused by two loader bugs:
1. **Missing `gguf>=0.10.0` requirement**: The model needed the `gguf` package but had no `requirements.txt`.
2. **Qwen3MoeExperts for-loop segfault**: `Qwen3MoeExperts.forward` uses a Python for-loop over a dynamically-sized `expert_hit` tensor that XLA/torch.compile cannot statically trace, causing a segfault in `partition_fx_graph_for_cpu_fallback`. Fix: set `model.config._experts_implementation = "batched_mm"`.
3. **Cross-loader contamination of `load_gguf_checkpoint`**: 26+ GGUF loaders imported before this one replace `_gguf_utils.load_gguf_checkpoint` with narrow-sig wrappers `def _patched(gguf_path, return_tensors=False):` that reject the `model_to_load` kwarg added in transformers 5.2. The initial fix of capturing `_orig_load_gguf_checkpoint` at import time still captured a contaminated narrow-sig version. Fix: `_find_real_load_gguf_checkpoint()` traces the chain via `func.__globals__['_orig_load_gguf_checkpoint']` until reaching the function whose `__code__.co_filename` matches the real transformers source file; re-applied at `load_model()` call time to handle loaders imported after this one.

After all three loader bugs were fixed, the model loaded and compiled successfully but failed at device execution with `INTERNAL: Error code: 13` (OOM). The Qwen3-30B-A3B Q4_K_M GGUF dequantizes from ~17 GB to ~60 GB BF16 (30B params × 2 bytes), which exceeds the p150b single-device DRAM capacity of ~31.84 GB.

## Fix
**Loader fixes** (tt_forge_models, `qwen3_30b_a3b_instruct_2507_malaysian_dora_gguf/causal_lm/pytorch/`):
- `requirements.txt` — new file, adds `gguf>=0.10.0`
- `loader.py` — three fixes:
  - Set `model.config._experts_implementation = "batched_mm"` after `from_pretrained` to prevent Qwen3MoeExperts for-loop segfault in XLA
  - `_find_real_load_gguf_checkpoint()` function that traces narrow-sig wrapper chains to find real transformers `load_gguf_checkpoint`
  - Re-apply the patch in `load_model()` to prevent clobbering by later-imported loaders

**Test config** (tt-xla, `tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `KNOWN_FAILURE_XFAIL` entry for this model with OOM reason

## Verification
- pytest exit: FAIL (INTERNAL: Error code: 13 — OOM)
- Hardware: blackhole-p150b
- Duration: 1497.81s (0:24:57)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/qwen3_30b_a3b_instruct_2507_malaysian_dora_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt_forge_models/qwen3_30b_a3b_instruct_2507_malaysian_dora_gguf/causal_lm/pytorch/loader.py` (3 fixes)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e571f5f0ca4d68697a68aadadc54f6b49992895b |
| tt-forge-models | ffab433831ada603bb1633417df12be0aff302c1 |
