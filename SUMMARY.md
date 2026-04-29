# Remediation Summary: bartowski_magnum_v4_72b_gguf-causal_lm-pytorch-magnum_v4_72b_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_magnum_v4_72b_gguf/causal_lm/pytorch-magnum_v4_72b_GGUF-single_device-inference]

## Result
XFAIL — 72B param model dequantized to BF16 (~144 GB) exceeds single-device DRAM capacity

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
Original CI failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, device OOM:
```
RuntimeError: TT_FATAL @ bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 484442112 B DRAM buffer across 8 banks,
where each bank needs to store 60555264 B, but bank size is 4273390016 B
(allocated: 4092737984 B, free: 180652032 B, largest free block: 43778048 B)
```

## Root cause
Two bugs in the loader layer, plus a hardware capacity ceiling:

1. **gguf-load-checkpoint-model-to-load-kwarg (loader bug)**: transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. 26 other GGUF model loaders in tt_forge_models define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` at module level. During pytest collection, `TorchDynamicLoader.setup_test_discovery` imports every loader via `exec_module`, which executes these module-level patches. By the time the bartowski loader runs its `from_pretrained(gguf_file=...)`, the global `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` has been replaced by a patched version that does not accept `model_to_load`, causing TypeError. The original CI failure (`ImportError`) was the same patching chain but with gguf package not installed in that environment.

2. **Missing gguf>=0.10.0 requirement (loader bug)**: The bartowski_magnum_v4_72b_gguf loader had no `requirements.txt`, so environments without the `gguf` package would fail with ImportError before hitting the TypeError.

3. **Hardware capacity ceiling**: With both loader bugs fixed, the model loads (42 GB GGUF Q4_K_M, dequantized to BF16 ≈ 144 GB), but the 72B param model at BF16 exceeds the device DRAM (~34 GB on n150). The device OOM occurs during tensor tilization with ~30.5 GB already allocated.

## Fix
**tt_forge_models** (`remediation/bartowski_magnum_v4_72b_gguf-causal_lm-pytorch-magnum_v4_72b_GGUF-single_device-inference`):
- Cherry-picked commit `8d60d0229d` (ported from `remediation/drt-7b-i1-gguf-gguf-load-checkpoint-model-to-load-kwarg`): changed all 26 broken `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signatures to `(*args, **kwargs)` and updated internal `_orig_load_gguf_checkpoint` calls to forward `*args, **kwargs`.
- Added `bartowski_magnum_v4_72b_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.

**tt-xla** (`remediation/bartowski_magnum_v4_72b_gguf-causal_lm-pytorch-magnum_v4_72b_GGUF-single_device-inference`):
- Added `KNOWN_FAILURE_XFAIL` entry for this test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (OOM on device, hardware-class)
- Hardware:    n150
- Duration:    1450.57s (0:24:10)
- Tier A attempts: N/A

## Files changed
- `bartowski_magnum_v4_72b_gguf/causal_lm/pytorch/requirements.txt` (new)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `drt_7b_i1_gguf/causal_lm/pytorch/requirements.txt` (new, via cherry-pick)
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
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b  |
| tt-xla          | 5962a8422  |
| tt-forge-models | 460541d628 |
