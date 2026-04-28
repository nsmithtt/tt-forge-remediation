# Remediation Summary: anakin87_phi_3_5_mini_ita-causal_lm-pytorch-Phi_3_5_mini_ITA-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[anakin87_phi_3_5_mini_ita/causal_lm/pytorch-Phi_3_5_mini_ITA-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; test passes on silicon at PCC>=0.99

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
2026-04-23 19:45:41.095 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device (assert.hpp:104)
```

On the current branch the failure manifested as:
```
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -262143)

While executing %slice_11 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_5, 2, -262143, 9223372036854775807), kwargs = {})
Original traceback:
  File "transformers/models/phi3/modeling_phi3.py", line 254, in forward
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
  File "transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

After fixing the slice OOB, the next failure was:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9855162131403349. Required: pcc=0.99.
```

## Root cause

Two bugs:

**Bug 1 — tt-xla compiler frontend (Tier A):** The loader had `use_cache=True` (default), which causes Phi 3.5's `SlidingWindowCache` to execute. Inside `SlidingWindowCache.update()`, the cache slices with `full_value_states[:, :, -self.sliding_window + 1:, :]`. With `sliding_window=262144` and a short sequence (128 tokens), the start index is -262143, far outside the [-128, 127] valid range. PyTorch eager silently clamps this to 0 (returning all tokens), but the XLA lazy backend raises `Value out of range`. The fix is a graph-rewriting FX pass (`clamp_out_of_range_slice_starts`) added to the torch_pass_pipeline.

**Bug 2 — loader:** The loader had two issues compared to the equivalent `phi3/phi_3_5/pytorch-Mini_Instruct` loader (which passes): (a) missing `use_cache=False` — forcing the KV-cache path unnecessarily — and (b) `padding="max_length"` instead of `padding=True`. With `padding="max_length"` and max_length=128, the model processes ~100 padding tokens alongside ~25 real tokens, compounding WH BF16 numerical error enough to drop PCC below 0.99.

## Fix

**tt-xla** (`remediation/anakin87_phi_3_5_mini_ita-causal_lm-pytorch-Phi_3_5_mini_ITA-single_device-inference`):
- `python_package/tt_torch/backend/passes.py`: Added `clamp_out_of_range_slice_starts` FX pass that rewrites `aten.slice.Tensor` nodes whose start argument is more negative than `-dim_size`, clamping it to `-dim_size` (matching PyTorch eager semantics).
- `python_package/tt_torch/backend/backend.py`: Added `clamp_out_of_range_slice_starts` to `torch_pass_pipeline`.
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `anakin87_phi_3_5_mini_ita/causal_lm/pytorch-Phi_3_5_mini_ITA-single_device-inference` as `EXPECTED_PASSING`.

**tt-forge-models** (`remediation/anakin87_phi_3_5_mini_ita-causal_lm-pytorch-Phi_3_5_mini_ITA-single_device-inference`):
- `anakin87_phi_3_5_mini_ita/causal_lm/pytorch/loader.py`: Added `use_cache=False` to model_kwargs and changed `padding="max_length"` to `padding=True`, matching the base `phi3/phi_3_5` loader.

Note: the `clamp_out_of_range_slice_starts` pass was cherry-picked from `remediation/mpt_1b_redpajama_200b_dolly-aten-slice-oob` (commit dd2905c41).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    88.84s (0:01:28)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-forge-models/anakin87_phi_3_5_mini_ita/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b027e24e030b2132c3e4cb27f1204759cc2e4d4d |
| tt-forge-models | 04b3fca1425fc1880add283b2e2dc8c48acb0352 |
