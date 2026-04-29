# Remediation Summary: bereaved_compound_v1_0_24b_i1_gguf/causal_lm/pytorch-BereavedCompound-v1.0-24b-i1-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bereaved_compound_v1_0_24b_i1_gguf/causal_lm/pytorch-BereavedCompound-v1.0-24b-i1-GGUF-single_device-inference]

## Result
XFAIL — 24B model in BF16 (~31 GB) exhausts single-device DRAM; no room for inference activations

## Stack layer
loader, hardware-class

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
First failure (masked, loader bug):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Second failure (hardware-class, after loader fix):
```
RuntimeError: TT_FATAL @ .../bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4196977728 B, free: 76412288 B, largest free block: 37030336 B)
```

## Root cause
Two sequential failures:

**Loader bug (fixed):** `setup_test_discovery` imports all loaders during pytest collection. 26 GGUF loaders register qwen35 architecture support by monkey-patching `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a narrow signature `(gguf_path, return_tensors=False)`. transformers 5.2.0 now calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)` — the extra kwarg raises `TypeError`. The bereaved_compound loader itself has no such patch but is victimized by the other loaders' patches during the same pytest session.

**Hardware capacity ceiling:** After the loader fix, the model loads and compiles. At inference time, the device (8 DRAM banks × 4.27 GB = ~34 GB total) has 31.3 GB occupied by model weights (the 24B model dequantized to BF16 ≈ 48 GB nominal, but consumed 31.3 GB due to the GGUF quantized loading path). Only 583 MB free remain, and the tilize activation buffer needs 320 MB contiguous but the largest free block per bank is only 35.3 MB (memory fragmentation). This is hardware-class: no single Tenstorrent device has enough DRAM for a 24B BF16 model plus inference activations.

## Fix
**Loader fix** (tt-forge-models, 26 files): Changed all `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signatures to `_patched_load_gguf_checkpoint(*args, **kwargs)` and forwarded `*args, **kwargs` to the original function. Branch: `remediation/bereaved-compound-v1-0-24b-i1-gguf-gguf-load-checkpoint-model-to-load-kwarg`.

**Test config** (tt-xla): Added `KNOWN_FAILURE_XFAIL` entry in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` for the bereaved_compound model.

## Verification
- pytest exit: FAIL (hardware-class OOM after loader fix)
- Hardware:    wormhole (p150b equivalent — 8 DRAM banks × 4.27 GB = 34 GB)
- Duration:    683.79s (0:11:23) for post-fix run
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/{bartowski_coniccat,daniloreddy,dmind,gpt_oss_swallow*,mradermacher_*,qwen_3_5_imatrix,tvall43_*,unified_reward_flex_qwen35_27b}_gguf/causal_lm/pytorch/loader.py` (26 files) — fix `_patched_load_gguf_checkpoint` signature
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add KNOWN_FAILURE_XFAIL

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d47e391c1446e7b40e65d539caaf1b993683371b |
| tt-forge-models | 65652991203db617586fc08cdcd7ac7c8b96970f |
