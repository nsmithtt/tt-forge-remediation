# Remediation Summary: gpt_oss_20b_ilograph_instruct_i1_gguf-causal_lm-pytorch-20B_ilograph_instruct_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_ilograph_instruct_i1_gguf/causal_lm/pytorch-20B_ilograph_instruct_i1_GGUF-single_device-inference]

## Result
XFAIL — 20B Qwen3MoE model dequantized to BF16 (~31.4 GB) fills p150b 32 GB DRAM; activation buffers OOM during inference

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-dram-capacity-20b-moe-p150b

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

## Root cause
Four sequential loader bugs, then a hardware capacity ceiling:

1. **Missing `gguf>=0.10.0` in requirements.txt** — the loader had no requirements.txt, so
   the `gguf` package was never installed before `from_pretrained(..., gguf_file=...)` was
   called. transformers raises ImportError immediately.

2. **`gpt-oss` GGUF architecture not registered** — the GGUF file stores arch=`gpt-oss`
   which is not in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING`. After
   gguf install, transformers raises `KeyError: 'gpt-oss'`.

3. **`load_shard_spec` assumed dense MLP** — the loader's shard spec iterated
   `layer.mlp.up_proj.weight` etc., but the gpt-oss/qwen3_moe architecture uses
   `Qwen3MoeSparseMoeBlock` with fused `Qwen3MoeExperts` (3D `gate_up_proj`/`down_proj`
   tensors), so `AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'`.

4. **Qwen3MoeExperts for-loop XLA segfault** — the default `_experts_implementation` uses a
   Python for-loop over expert MLPs that segfaults in the XLA dynamo bridge. Setting
   `model.config._experts_implementation = "batched_mm"` routes the forward to batched
   matrix multiply, which XLA handles correctly.

5. **Hardware capacity** — after all loader fixes, the model's BF16 weights consume ~31.4 GB
   of p150b's 32 GB DRAM (8 × 4 GB banks). The first 1 GB activation buffer during inference
   hits OOM: `Not enough space to allocate 1061683200 B DRAM buffer ... free: 51434624 B`.

## Fix
Loader fixes in `tt-forge-models` on the remediation branch
(`remediation/gpt_oss_20b_ilograph_instruct_i1_gguf-causal_lm-pytorch-20B_ilograph_instruct_i1_GGUF-single_device-inference`):

- **Commit `6c5205153d`** — `gpt_oss_20b_ilograph_instruct_i1_gguf/causal_lm/pytorch/requirements.txt` added (`gguf>=0.10.0`); `loader.py` extended with `_patch_gpt_oss_support()` to register `gpt-oss` as an alias for `qwen3_moe` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, `GGUF_TO_FAST_CONVERTERS`, and `GGUF_CONFIG_DEFAULTS_MAPPING`; `_patched_load_gguf_checkpoint(*args, **kwargs)` installed on all modules caching `load_gguf_checkpoint` at import time (compatible with the `model_to_load` kwarg added in transformers 5.2.0).
- **Commit `811a428c19`** — `load_shard_spec` updated to handle `Qwen3MoeExperts` fused weights: `mlp.experts.gate_up_proj` / `mlp.experts.down_proj` with `(None, "model", "batch")` / `(None, "batch", "model")`, guarded by `hasattr`.
- **Commit `a42e771554`** — `model.config._experts_implementation = "batched_mm"` set after `from_pretrained` to prevent XLA segfault in the expert for-loop.

Test config in `tt-xla` on the remediation branch:

- **Commit `7b4ff141f`** — `gpt_oss_20b_ilograph_instruct_i1_gguf/causal_lm/pytorch-20B_ilograph_instruct_i1_GGUF-single_device-inference` marked `KNOWN_FAILURE_XFAIL` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (OOM after all loader fixes applied — hardware capacity)
- Hardware:    blackhole-p150b
- Duration:    1268.90s (0:21:08) before OOM
- Tier A attempts: N/A

## Files changed
- `gpt_oss_20b_ilograph_instruct_i1_gguf/causal_lm/pytorch/requirements.txt` (created)
- `gpt_oss_20b_ilograph_instruct_i1_gguf/causal_lm/pytorch/loader.py` (3 fixes)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7b4ff141f22ee8a002fc3533740e499b59d722a6 |
| tt-forge-models | a42e7715546f3afa485b1c776076e819a5bc79c5 |
