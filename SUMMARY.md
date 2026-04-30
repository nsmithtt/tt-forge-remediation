# Remediation Summary: gemma3_27b_it_gguf-causal_lm-pytorch-27B_IT_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_27b_it_gguf/causal_lm/pytorch-27B_IT_Q4_K_M-single_device-inference]

## Result
XFAIL — 27B model dequantized to bfloat16 (~54 GB) exceeds single p150b device DRAM (~34 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
oom-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
where each bank needs to store 28901376 B, but bank size is 4273390016 B
(allocated: 4225128768 B, free: 48261248 B, largest free block: 13855040 B)
```

(The original reported failure `raise ImportError("Please install torch and gguf>=0.10.0 …")` was a
`TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` —
a loader bug caused by other GGUF loaders monkey-patching `load_gguf_checkpoint` globally at
pytest-collection time. That loader bug was fixed. An `aten.slice.Tensor` OOB-start error that
appeared next was also fixed. After both fixes, the test ran to completion and hit the hardware
capacity ceiling described above.)

## Root cause
The Gemma 3 27B IT GGUF (Q4_K_M) checkpoint is loaded via `AutoModelForCausalLM.from_pretrained`
with `torch_dtype=torch.bfloat16`, which dequantizes the GGUF weights to bfloat16 on load.
27 billion parameters × 2 bytes = ~54 GB. The single p150b Blackhole device has 8 GDDR banks of
~4.27 GB each = ~34.2 GB total on-device DRAM. At execution time the runtime attempts to tilize
(copy and reformat) model tensors into device DRAM, and runs out of space after allocating 33.8 GB
(99% of capacity). This is a genuine hardware capacity ceiling: the model size class (24B+) simply
exceeds single-device memory.

## Fix
No compiler fix. Two loader-layer bugs were fixed along the way (required to reach the OOM):

1. **`_patched_load_gguf_checkpoint` missing `model_to_load` kwarg** (tt_forge_models,
   `remediation/gemma3_27b_it_gguf-causal_lm-pytorch-27B_IT_Q4_K_M-single_device-inference`,
   commit `db85cb5642`): ~26 loader files monkey-patched `load_gguf_checkpoint` globally with a
   function that only accepted `(gguf_path, return_tensors=False)`, missing the `model_to_load`
   kwarg added in transformers 5.2.0. Fixed by widening all patched functions to `(*args, **kwargs)`
   with pass-through calls.

2. **`aten.slice.Tensor` OOB negative start for sliding-window attention** (tt-xla,
   `python_package/tt_torch/torch_overrides.py`, commit `b1be1513d`): XLA lazy backend strictly
   validates slice indices; Gemma3's sliding-window cache uses `[-sliding_window+1:]` which yields
   start=-1023 on a tensor shorter than the window. Fixed by pre-clamping start/end to `-size`
   in `TorchFunctionOverride.__torch_function__` (cherry-picked from gemma3_1b_gguf remediation).

The test config was updated to `KNOWN_FAILURE_XFAIL` in:
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling)
- Hardware:    blackhole-p150b
- Duration:    1102.19s (0:18:22)
- Tier A attempts: N/A

## Files changed
- `tt-xla`: `python_package/tt_torch/torch_overrides.py` (slice OOB fix)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)
- `tt-xla`: `third_party/tt_forge_models` submodule pointer (→ tt_forge_models remediation commit)
- `tt_forge_models`: ~26 GGUF loader files (widened `_patched_load_gguf_checkpoint` to `*args, **kwargs`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ce1d011adb1d46c083b7be98838da6a15a364c89 |
| tt-forge-models | db85cb56426acf96012407805ff9011dec52799c |
