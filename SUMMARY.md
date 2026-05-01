# Remediation Summary: infomaniak_ai_vllm_translategemma_27b_it-text_translation-pytorch-vllm-translategemma-27b-it-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[infomaniak_ai_vllm_translategemma_27b_it/text_translation/pytorch-vllm-translategemma-27b-it-single_device-inference]

## Result
XFAIL — 27B model exceeds single-device DRAM capacity on p150b; OOM during inference execution

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-27b-dram-oom

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
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

After tt_forge_models branch HEAD (which had already applied a `return_dict=False` fix), the
reproduced failure was:
```
RuntimeError: Value out of range (expected to be in range of [-91, 90], but got -1023)
While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_65, 2, -1023, 9223372036854775807), kwargs = {})
```
From `transformers/cache_utils.py` SlidingWindowCache: `full_value_states[:, :, -self.sliding_window + 1 :, :]`
with `sliding_window=1024`, producing start=-1023 on a 91-token sequence.

After the slice fix, the failure was:
```
TT_FATAL: Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
where each bank needs to store 28901376 B, but bank size is 4273390016 B
(allocated: 4225215296 B, free: 48174720 B, largest free block: 13682752 B)
```

## Root cause
Two independent issues in sequence:

**Issue 1 (Tier A, fixed):** XLA validates `aten.slice.Tensor` start indices strictly. The
SlidingWindowCache in Gemma3 computes `start = -sliding_window + 1 = -1023` when the sequence
length (91 tokens) is shorter than the window size (1024). XLA raises `Value out of range
(expected [-91, 90], got -1023)`. Standard PyTorch silently clamps such OOB starts.

**Issue 2 (hardware-class):** After the slice fix, the 27B parameter model in BF16 fills ~31.4 GB
of the 34.2 GB allocatable DRAM pool (8 banks × 4.27 GB). The remaining 367 MB is insufficient
for activation tensors (needs 220 MB but largest contiguous block is only 104 MB due to
fragmentation). Device: single p150b (Blackhole). Model `Infomaniak-AI/vllm-translategemma-27b-it`
is a 27B fine-tune; at BF16 it saturates single-device DRAM similarly to `gemma/pytorch-2_27B_IT`
which is already marked `EXCLUDE_MODEL`.

## Fix
**Issue 1:** Added `clamp_out_of_range_slice_starts` FX pass to
`python_package/tt_torch/backend/passes.py` (tt-xla). The pass iterates over all
`aten.slice.Tensor` nodes after export/decomposition, and for any with a negative start index
whose magnitude exceeds the known tensor size in that dimension, clamps it to `-size`. Wired into
`torch_pass_pipeline` in `python_package/tt_torch/backend/backend.py` after
`bypass_assert_tensor_metadata`.

**Issue 2:** Added `KNOWN_FAILURE_XFAIL` entry in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla) with OOM
reason. The hardware capacity ceiling means no compiler fix is possible on a single device.

## Verification
- pytest exit: 0 (xfailed)
- Hardware:    blackhole-p150b
- Duration:    354.38s (0:05:54)
- Tier A attempts: 1

## Files changed
- `python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts`
- `python_package/tt_torch/backend/backend.py` — imported and called the new pass
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d9697eee66f09eb8ccba8fe7604e729c7d8f82d9 |
| tt-forge-models | 9115d30c19a33c9e08879c15ec1a5e606911f28a |
