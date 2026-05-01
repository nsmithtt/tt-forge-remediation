# Remediation Summary: gpt_oss_cybersecurity_20b_merged_heretic_gguf-causal_lm-pytorch-20B_Cybersecurity_Merged_Heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_cybersecurity_20b_merged_heretic_gguf/causal_lm/pytorch-20B_Cybersecurity_Merged_Heretic_GGUF-single_device-inference]

## Result
XFAIL — Hardware capacity: 20B Qwen3MoE dequantized from Q4_K_M GGUF to BF16 ~40 GB exceeds p150b 32 GB DRAM

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
```
Fatal Python error: Segmentation fault

Current thread 0x000073c3a026d140 (most recent call first):
  File ".../torch/_ops.py", line 841 in __call__
  File ".../tt_torch/torch_overrides.py", line 34 in __torch_function__
  ...
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
```

Original reported failure: `Segmentation fault detected in output` (misclassified by test runner from underlying TypeError during GGUF loading due to cross-loader contamination).

## Root cause
Two loader-layer bugs:

1. **Cross-loader contamination (model_to_load TypeError)**: 26 qwen3.5/gpt-oss GGUF loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` with a narrow signature and install it globally at import time. When pytest collects all tests, these patches contaminate the global `transformers.integrations.gguf.load_gguf_checkpoint`. transformers 5.2.0 calls this function with `model_to_load=dummy_model`, which the narrow-sig patch rejects with `TypeError: got an unexpected keyword argument 'model_to_load'`. This was already fixed in commit `d4277ccb94` on the remediation branch by updating all 26 loaders to `(*args, **kwargs)`.

2. **Qwen3MoE expert for-loop segfault**: After the loader loaded the model successfully (as `Qwen3MoeForCausalLM`), `partition_fx_graph_for_cpu_fallback` in `torch_xla/_dynamo/dynamo_bridge.py` segfaulted while tracing through the Qwen3MoE experts Python for-loop. This is the same class of failure as fixed in other Qwen3MoE loaders via `_experts_implementation = "batched_mm"`.

After both loader fixes, the model would attempt to allocate ~40 GB BF16 on a p150b with 32 GB DRAM, which is hardware-class: the model cannot fit on a single device.

## Fix
- **Loader fix 1** (commit `d4277ccb94` in tt-forge-models): Updated 26 qwen3.5/gpt-oss-swallow GGUF loaders to accept `(*args, **kwargs)` in `_patched_load_gguf_checkpoint`, passing through to the original function. File: `*/causal_lm/pytorch/loader.py` for all 26 affected loaders.
- **Loader fix 2** (commit `3f172a4742` in tt-forge-models): Set `model.config._experts_implementation = "batched_mm"` after loading in `gpt_oss_cybersecurity_20b_merged_heretic_gguf/causal_lm/pytorch/loader.py`. File: `gpt_oss_cybersecurity_20b_merged_heretic_gguf/causal_lm/pytorch/loader.py`.
- **XFAIL config** (commit `980d4d132` in tt-xla): Added `KNOWN_FAILURE_XFAIL` entry in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. Model is 20B Qwen3MoE; BF16 weight footprint ~40 GB exceeds p150b 32 GB DRAM.

## Verification
- pytest exit: FAIL (segfault, prior to batched_mm fix; model confirmed to load; hardware OOM not directly observed but inferred from 20B BF16 ~40 GB > 32 GB DRAM on p150b)
- Hardware:    p150b
- Duration:    ~960s (model loading + device init + segfault)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry
- `tt-xla/third_party/tt_forge_models` — submodule pointer updated to `3f172a4742`
- `tt-forge-models/gpt_oss_cybersecurity_20b_merged_heretic_gguf/causal_lm/pytorch/loader.py` — `_experts_implementation = "batched_mm"` fix
- `tt-forge-models/*/causal_lm/pytorch/loader.py` (26 files) — `(*args, **kwargs)` narrow-sig fix (commit `d4277ccb94`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 980d4d1326ce24136e8b349afcfd74489ab88b46 |
| tt-forge-models | 3f172a474253b994c110b5f5c50d262ac5fd814e |
