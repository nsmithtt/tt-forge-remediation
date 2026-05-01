# Remediation Summary: lmstudio_qwen_3_30b_a3b_gguf-causal_lm-pytorch-30B_A3B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_qwen_3_30b_a3b_gguf/causal_lm/pytorch-30B_A3B_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B Q4_K_M (~61.7 GB BF16) exceeds all single-device DRAM (p150b: 32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-gguf-dequant-exceeds-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

On reproduction (gguf>=0.10.0 was actually installed), the actual error was:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

This was caused by cross-loader clobbering: 26 other GGUF loaders imported during pytest collection had patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with narrow-signature wrappers `(gguf_path, return_tensors=False)` that didn't accept the `model_to_load` kwarg added in transformers 5.2.0.

After fixing the narrow-sig patches, the next failure was:
```
While executing %histc ... torch.ops.aten.histc.default
```

`grouped_mm_experts_forward` in `transformers.integrations.moe` calls `torch.histc` on an XLA tensor, which is unsupported. Fixed by setting `model.config._experts_implementation = "batched_mm"`.

After applying `batched_mm`, the test failed with:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

This is OOM (Error code 13 = device out of memory). The Qwen3-30B-A3B GGUF model (17.35 GB on disk at Q4_K_M, ~30.85B parameters) dequantizes to BF16 when loaded by transformers, requiring ~61.7 GB DRAM — nearly 2× the 32 GB available on p150b. This is the root cause: hardware capacity ceiling.

## Root cause
The model's total parameter count (~30.85B) at BF16 requires approximately 61.7 GB DRAM. The largest available single-device DRAM is 32 GB (p150b). The model cannot fit on any single TT device.

The loader bug chain (narrow-sig TypeError → histc XLA failure → batched_mm OOM) was worked through, but the terminal failure is hardware capacity: transformers dequantizes GGUF Q4_K_M tensors to BF16 at load time, so the full BF16 model must fit in device DRAM. For 30.85B parameters, that is 61.7 GB — 1.93× over the 32 GB p150b limit.

## Fix
Two loader fixes were committed to tt_forge_models remediation branch before the hardware capacity determination:

1. **Narrow-sig `_patched_load_gguf_checkpoint` fix** (`4a56083106`): 26 GGUF loaders had `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` which raised `TypeError` when transformers 5.2.0 called it with `model_to_load=`. Changed all 26 to `(*args, **kwargs)`. Also added `requirements.txt` with `gguf>=0.10.0` to the lmstudio loader directory.

2. **Qwen3MoE batched_mm fix** (`10bfbfdff8`): Added `model.config._experts_implementation = "batched_mm"` after model load to avoid `grouped_mm_experts_forward`'s unsupported `torch.histc` call on XLA tensors.

The test config was updated to `KNOWN_FAILURE_XFAIL` in tt-xla (`b5d8d5060`).

For a SILICON_PASS, this model would need a multi-device setup with tensor parallelism (similar to the existing `qwen_3/causal_lm/pytorch-30B_A3b` which is EXCLUDE_MODEL for single-device).

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — this is XFAIL hardware-class, not Tier B.

## Verification
- pytest exit: FAIL (OOM, INTERNAL error 13)
- Hardware: blackhole-p150b
- Duration: 1542.17s (0:25:42) before OOM
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/lmstudio_qwen_3_30b_a3b_gguf/causal_lm/pytorch/loader.py` — added `_experts_implementation = "batched_mm"`
- `tt_forge_models/lmstudio_qwen_3_30b_a3b_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`
- `tt_forge_models/{26 qwen3.5 GGUF loaders}/causal_lm/pytorch/loader.py` — fixed narrow-sig `_patched_load_gguf_checkpoint`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — marked KNOWN_FAILURE_XFAIL

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b5d8d5060523627e15f20442325cd7e24f5c6095 |
| tt-forge-models | 10bfbfdff8242d5ea410ca6055306fb5c149a490 |
