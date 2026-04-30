# Remediation Summary: gemma_3_27b_it_ultra_heretic_gguf-causal_lm-pytorch-27B_IT_ULTRA_HERETIC_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_27b_it_ultra_heretic_gguf/causal_lm/pytorch-27B_IT_ULTRA_HERETIC_GGUF-single_device-inference]

## Result
XFAIL — 27B model dequantized to bfloat16 (~54 GB) exceeds single p150b device DRAM (~34 GB)

## Stack layer
loader, tt-xla, hardware-class

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
Original reported failure:
```
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
```

On reproduction, two prior bugs blocked reaching the slice error:

1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`
   — 26 GGUF loaders in tt_forge_models patched `load_gguf_checkpoint` at import time with
   a narrow `(gguf_path, return_tensors=False)` signature; transformers 5.2.0 added
   `model_to_load=dummy_model` to the call.

2. After fixing #1, the test loaded the model and reached the Gemma 3 SlidingWindowCache
   path where `slice(tensor, dim, kv_seq_len - window_size, ...)` produces a start index of
   `23 - 1024 = -1001` on a dim-23 KV cache. XLA lazy backend rejects indices outside
   `[-size, size-1]` (unlike PyTorch eager which silently clamps).

After fixing both #1 and #2, the test loads and runs the 27B model but OOMs during inference:
```
RuntimeError: TT_FATAL @ tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
where each bank needs to store 28901376 B, but bank size is 4273390016 B
(allocated: 4225214784 B, free: 48175232 B, largest free block: 13683008 B)
```

## Root cause
Three separate issues in sequence:

1. **Loader bug** — 26 tt_forge_models loaders that monkey-patch `load_gguf_checkpoint`
   used a fixed `(gguf_path, return_tensors=False)` signature. Because pytest imports all
   loaders during collection, one of these patched the global before our test ran. The
   patched version does not accept the `model_to_load` kwarg added in transformers 5.2.0,
   causing a TypeError when `AutoModelForCausalLM.from_pretrained` invokes
   `load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)`.

2. **Compiler frontend bug** — `SlidingWindowCache.update()` does
   `full_value_states[:, :, -sliding_window+1:, :]`. With `sliding_window=1024` and
   `seq_len=23`, the start index is `23 - 1024 = -1001`, which falls below `-23` (the
   minimum allowed by XLA's slice kernel for a dim-23 tensor). PyTorch CPU silently clamps
   such indices, but the XLA lazy backend raises `RuntimeError: Value out of range`. This
   fires during `partition_fx_graph_for_cpu_fallback` when the graph is replayed on XLA
   tensors.

3. **Hardware capacity ceiling** — The `gemma-3-27b-it-ultra-heretic-Q4_K_M.gguf`
   checkpoint is loaded via `AutoModelForCausalLM.from_pretrained` with
   `torch_dtype=torch.bfloat16`, which dequantizes the GGUF weights to bfloat16 at load
   time. 27 billion parameters × 2 bytes = ~54 GB. The single p150b Blackhole device has
   8 GDDR banks of ~4.27 GB each = ~34.2 GB total on-device DRAM. At execution time the
   runtime tilizes (copies and reformats) model weights into device DRAM and runs out of
   space after allocating ~32.3 GB (94% of capacity). This is a genuine hardware capacity
   ceiling: the model size class (27B BF16) simply exceeds single-device memory.

## Fix
No compiler fix. Two loader-layer bugs were fixed along the way (required to reach the OOM):

1. **`_patched_load_gguf_checkpoint` missing `model_to_load` kwarg** (tt_forge_models,
   `remediation/gemma_3_27b_it_ultra_heretic_gguf-causal_lm-pytorch-27B_IT_ULTRA_HERETIC_GGUF-single_device-inference`,
   commit `f46a7d0222`): 26 loader files monkey-patched `load_gguf_checkpoint` globally with
   a function that only accepted `(gguf_path, return_tensors=False)`. Updated all 26 to
   `(gguf_path, return_tensors=False, **kwargs)` passing `**kwargs` through.

2. **`aten.slice.Tensor` OOB start clamping** (tt-xla,
   `remediation/gemma_3_27b_it_ultra_heretic_gguf-causal_lm-pytorch-27B_IT_ULTRA_HERETIC_GGUF-single_device-inference`,
   commit `a184aea41`): Added guard in `TorchFunctionOverride.__torch_function__` that clamps
   `start` to `-size` when `func is torch.ops.aten.slice.Tensor` and `start < -size`.
   File: `python_package/tt_torch/torch_overrides.py`.

3. **XFAIL entry** added to
   `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (commit
   `542f4f1c0`): `status: KNOWN_FAILURE_XFAIL` with reason citing the OOM.

## Verification
- pytest exit: FAIL (OOM after fixes applied)
- Hardware:    blackhole-p150b
- Duration:    978.26s (0:16:18)
- Tier A attempts: N/A

## Files changed
- `tt-xla`: `python_package/tt_torch/torch_overrides.py` (slice OOB fix, commit `a184aea41`)
- `tt-xla`: `third_party/tt_forge_models` submodule pointer → `f46a7d0222` (commit `c8027e2ba`)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry, commit `542f4f1c0`)
- `tt_forge_models`: 26 GGUF loader files (widened `_patched_load_gguf_checkpoint` to `**kwargs`, commit `f46a7d0222`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 542f4f1c057e786c708d9b6a4d393289424d7ec4 |
| tt-forge-models | f46a7d022223b05401fba0d1b38ef32f390f7477 |
