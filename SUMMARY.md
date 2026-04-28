# Remediation Summary: gpt_oss_heretic_ara_gguf-causal_lm-pytorch-20B_Heretic_Ara_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_heretic_ara_gguf/causal_lm/pytorch-20B_Heretic_Ara_GGUF-single_device-inference]

## Result
XFAIL — 20B MoE model dequantizes to bfloat16 (~40 GB), exceeding single-device DRAM capacity (~34 GB on Blackhole p150b)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
oom-20b-moe-bfloat16-dram-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

(Original failure before loader fixes were applied.)

After loader fixes, actual hardware failure:
```
TT_FATAL: Out of Memory: Not enough space to allocate 1061683200 B DRAM buffer across 8 banks, where each bank needs to store 132710400 B, but bank size is 4273390016 B (allocated: 4221954688 B, free: 51435328 B, largest free block: 45589184 B)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Three bugs were fixed in the loader layer before the hardware capacity ceiling was exposed:

1. **`gguf-load-checkpoint-model-to-load-kwarg` (global patch leak)**: 26 GGUF loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it globally at import time. transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`; the patched function did not accept it, causing `TypeError` when any of those loaders was imported before `gpt_oss_heretic_ara_gguf` in a pytest session.

2. **`load_shard_spec` attribute errors**: The loader's `load_shard_spec` referenced `layer.mlp.router` (does not exist on `Qwen3MoeSparseMoeBlock`; correct name is `gate`) and `layer.mlp.experts.gate_up_proj_bias` / `down_proj_bias` (no bias parameters on `Qwen3MoeExperts`).

3. **`qwen3moe-experts-for-loop-segfault`**: `Qwen3MoeExperts.forward()` iterates experts via a Python for-loop over a dynamically-sized tensor; XLA cannot trace this statically and segfaults during `partition_fx_graph_for_cpu_fallback`. Fixed by setting `model.config._experts_implementation = "batched_mm"`.

After all loader fixes, the model loads and compiles successfully but fails at runtime with an OOM. The model has 20B parameters; when dequantized from Q4_K_M GGUF to bfloat16 (as required by the test infrastructure), the weight tensor size is ~40 GB (20B × 2 bytes), which exceeds the ~34 GB DRAM available on a single Blackhole p150b device.

## Fix
Three loader fixes in `tt_forge_models` on branch `remediation/gpt_oss_heretic_ara_gguf-causal_lm-pytorch-20B_Heretic_Ara_GGUF-single_device-inference`:

- **Commit 1** (`85ce1564`): Updated `_patched_load_gguf_checkpoint` in all 26 affected loaders to accept `**kwargs` and forward them to `_orig_load_gguf_checkpoint`. Files: 26 `loader.py` files across different GGUF model directories.
- **Commit 2** (`0bdf8d93`): Fixed `load_shard_spec` in `gpt_oss_heretic_ara_gguf/causal_lm/pytorch/loader.py`: `layer.mlp.router.weight` → `layer.mlp.gate.weight`, removed non-existent `gate_up_proj_bias` and `down_proj_bias` lines.
- **Commit 3** (`87e2ecdb`): Added `model.config._experts_implementation = "batched_mm"` after `.eval()` in `gpt_oss_heretic_ara_gguf/causal_lm/pytorch/loader.py`.

Hardware capacity disposition: added `KNOWN_FAILURE_XFAIL` entry in `tt-xla` at `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (commit `89ebf1e3`).

## Verification
- pytest exit: FAIL (OOM at device execution)
- Hardware:    blackhole-p150b
- Duration:    1291.04s (0:21:31) — model loading and compilation succeeded; OOM at inference
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gpt_oss_heretic_ara_gguf/causal_lm/pytorch/loader.py` (shard_spec + batched_mm fix)
- `tt_forge_models/<26 GGUF loaders>/causal_lm/pytorch/loader.py` (_patched_load_gguf_checkpoint kwarg fix)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 89ebf1e33399660e9320b97c040671bb8589d81d |
| tt-forge-models | 87e2ecdb0b1de7144850b4ef501a223419c49136 |
