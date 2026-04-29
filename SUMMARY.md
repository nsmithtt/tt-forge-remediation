# Remediation Summary: big_tiger_gemma_gguf-causal_lm-pytorch-27B_V1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[big_tiger_gemma_gguf/causal_lm/pytorch-27B_V1_GGUF-single_device-inference]

## Result
XFAIL — 27B bfloat16 model (~54 GB) exceeds single-device DRAM (~32 GB on the test machine)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-dram-capacity-27b-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_FATAL @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 339738624 B DRAM buffer across 8 banks,
where each bank needs to store 42467328 B, but bank size is 4273390016 B
(allocated: 4211366208 B, free: 62023808 B, largest free block: 31353152 B)
```

Original reported failure (before loader fixes):
```
2026-04-23 21:35:35.524 | critical | Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```

## Root cause

Three issues were uncovered in sequence:

1. **Loader bug — wrong GGUF filenames.** The V1 variant pointed to
   `Big-Tiger-Gemma-27B-v1-Q4_K_M.gguf` but the HuggingFace repo
   (`TheDrummer/Big-Tiger-Gemma-27B-v1-GGUF`) only contains
   `Big-Tiger-Gemma-27B-v1c-Q4_K_M.gguf` (the "c" corrected release).
   The V3 variant similarly had the wrong prefix for the bartowski mirror.

2. **Loader bug — GGUF `load_gguf_checkpoint` narrow signature (26 loaders).**
   `transformers` 5.2.0 added a `model_to_load` keyword argument to
   `load_gguf_checkpoint`. Twenty-six other loaders in tt-forge-models
   monkey-patch this function with the narrow signature
   `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`,
   which breaks any GGUF test collected in the same pytest session.

3. **tt-xla bug — XLA lazy slice out-of-bounds negative start.**
   Gemma 2's `SlidingWindowCache.update()` computes
   `full_value_states[:, :, -self.sliding_window + 1 :, :]`
   where `sliding_window = 4096`, giving a start index of `-4095`.
   At prefill with only 22 tokens, the XLA lazy backend raises
   "Value out of range (expected to be in range of [-22, 21], but got -4095)".
   PyTorch eager silently clamps; the fix pre-clamps in
   `TorchFunctionOverride.__torch_function__`.

4. **Hardware capacity (XFAIL root cause).** After all loader and
   tt-xla fixes, the model loads as bfloat16 (~54 GB) and is sent to
   the TT device at inference time. The device has 8 DRAM banks × ~4 GB
   each ≈ 32 GB total DRAM. By the time a late-layer MLP `gate_proj`
   weight (340 MB = 4608 × 36864 × 2 bytes) is to be tilized
   onto device, only 59 MB free remains. This is not a software bug —
   the 27B bfloat16 model simply exceeds single-device DRAM.

The originally-reported `TT_FATAL: Chip 0 logical eth core … connects
to a remote mmio device` messages are non-fatal initialization warnings
on multi-card systems; tt-metal logs them as CRITICAL but continues
with "Skipping ethernet core". They do not cause the test to fail.

## Fix

**Loader fixes (tt-forge-models, remediation branch):**

- `big_tiger_gemma_gguf/causal_lm/pytorch/loader.py`: corrected GGUF
  filenames for both variants:
  - V1: `Big-Tiger-Gemma-27B-v1-Q4_K_M.gguf` →
    `Big-Tiger-Gemma-27B-v1c-Q4_K_M.gguf`
  - V3: `Big-Tiger-Gemma-27B-v3-Q4_K_M.gguf` →
    `TheDrummer_Big-Tiger-Gemma-27B-v3-Q4_K_M.gguf`
- 26 GGUF loaders: `_patched_load_gguf_checkpoint(gguf_path,
  return_tensors=False)` → `_patched_load_gguf_checkpoint(*args,
  **kwargs)` and corresponding call-site updated.

**tt-xla fixes (remediation branch):**

- `python_package/tt_torch/torch_overrides.py`: added slice-index
  clamping in `TorchFunctionOverride.__torch_function__`. When
  `func is torch.ops.aten.slice.Tensor` and not compiling, clamp
  `start` and `end` to `max(index, -size)` before dispatching.
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  added `KNOWN_FAILURE_XFAIL` entry for
  `big_tiger_gemma_gguf/causal_lm/pytorch-27B_V1_GGUF-single_device-inference`.

## Verification
- pytest exit: FAIL (device OOM — hardware capacity ceiling confirmed)
- Hardware:    p150 (8 DRAM banks × ~4 GB = 32 GB total)
- Duration:    779.16s (0:12:59) — final run
- Tier A attempts: 1 (slice clamp fix; resolved that bug, revealed OOM)

## Files changed
- `big_tiger_gemma_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)
- 26 GGUF loader files (tt-forge-models, `_patched_load_gguf_checkpoint` signature)
- `python_package/tt_torch/torch_overrides.py` (tt-xla)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eb991f788a27dea532b4fb824a007738f52db684 |
| tt-forge-models | 8a2eedac78f64ee88811d95950a5a4b0cdd6a253 |
