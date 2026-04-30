# Remediation Summary: gemma3_27b_qat_gguf-causal_lm-pytorch-27B_IT_QAT_Q4_0_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_27b_qat_gguf/causal_lm/pytorch-27B_IT_QAT_Q4_0_GGUF-single_device-inference]

## Result
XFAIL — 27B model GGUF Q4_0 dequantizes to bfloat16 on load (~54 GB), exceeding single p150b device DRAM (~34 GB)

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
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
```

## Root cause
Two issues were uncovered and addressed:

1. **`aten.slice.Tensor` OOB negative start** (tt-xla, `python_package/tt_torch/backend/passes.py`):
   Gemma 3's `SlidingWindowCache.update()` does `full_value_states[:, :, -sliding_window+1:, :]`
   where `sliding_window = 1024`. When `seq_len = 23` (< 1024), the start index is
   `-1023`, which is outside the valid range `[-23, 22]` for that dimension. PyTorch
   allows such out-of-range starts (semantically clamped to 0), but the XLA/TT backend
   raises `RuntimeError`. Fixed by `clamp_out_of_range_slice_starts()` FX pass.

2. **Hardware capacity ceiling**: After the slice fix, the model loads the 27B GGUF Q4_0
   checkpoint via `AutoModelForCausalLM.from_pretrained` with `torch_dtype=bfloat16`,
   which dequantizes all weights to bfloat16 on load. 27B parameters × 2 bytes ≈ 54 GB.
   The single p150b Blackhole device has 8 GDDR banks × ~4.27 GB = ~34.2 GB total DRAM.
   Execution fails at runtime with OOM after allocating 99% of device DRAM:
   ```
   TT_FATAL: Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8
   banks, where each bank needs to store 28901376 B, but bank size is 4273390016 B
   (allocated: 4225214784 B, free: 48175232 B, largest free block: 13683008 B)
   ```

## Fix
One loader-layer fix and one XFAIL config update:

1. **`clamp_out_of_range_slice_starts` FX pass** (tt-xla,
   `remediation/gemma3_27b_qat_gguf-causal_lm-pytorch-27B_IT_QAT_Q4_0_GGUF-single_device-inference`,
   commit `9fd078b0f`): Added `clamp_out_of_range_slice_starts()` to
   `python_package/tt_torch/backend/passes.py` and wired it into `torch_pass_pipeline()`
   in `python_package/tt_torch/backend/backend.py`. The pass iterates FX graph nodes for
   `aten.slice.Tensor` calls, checks the static `start` argument against the input
   tensor's dimension size via `node.meta["val"].shape`, and replaces any `start <
   -dim_size` with `-dim_size` (equivalent to clamping as PyTorch would).

2. **Test config XFAIL** (tt-xla, commit `b0ab5458d`): Added
   `gemma3_27b_qat_gguf/causal_lm/pytorch-27B_IT_QAT_Q4_0_GGUF-single_device-inference`
   with `KNOWN_FAILURE_XFAIL` to
   `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling)
- Hardware:    blackhole-p150b
- Duration:    788.82s (0:13:08)
- Tier A attempts: N/A

## Files changed
- `tt-xla`: `python_package/tt_torch/backend/passes.py` (slice OOB pass)
- `tt-xla`: `python_package/tt_torch/backend/backend.py` (import + call new pass)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b0ab5458d36e55054aead8936772b55af84f37b9 |
| tt-forge-models | db63951491b926079da15b7e6efca99cff5efa16 |
