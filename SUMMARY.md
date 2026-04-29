# Remediation Summary: gemma3-causal_lm-pytorch-4B_Instruct_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3/causal_lm/pytorch-4B_Instruct_bnb_4bit-single_device-inference]

## Result
SILICON_PASS — slice OOB clamped in FX pass, test passes on n150

## Stack layer
tt-xla

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
RuntimeError: Value out of range (expected to be in range of [-256, 255], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807), kwargs = {})

The slice comes from transformers/cache_utils.py:214 in SlidingWindowCache.update():
  self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

where sliding_window=1024 and seq_len=256, so start=-1023 is out of range [-256, 255].

## Root cause
The XLA lazy backend (torch/csrc/lazy/core/helpers.cpp) raises "Value out of range" for
aten.slice.Tensor when start < -dim_size. PyTorch eager silently clamps such values to 0.
Gemma3's SlidingWindowCache computes a window slice start as -sliding_window+1 = -1023,
which is out of range for the 256-element sequence dimension used during inference.

Three loader bugs were also present on the CI bringup branch (already fixed there):
1. Missing bitsandbytes>=0.46.1 requirement
2. load_inputs returned a positional list causing attn_mask to be passed as pixel_values
   to Gemma3ForConditionalGeneration (which has pixel_values as the 2nd positional arg)
3. BNB 4-bit quantization requires CUDA; load_model must fall back to from_config when
   CUDA is unavailable on TT hardware

## Fix
Added `clamp_out_of_range_slice_starts` FX pass in tt-xla that walks the compiled
FX graph and clamps any aten.slice.Tensor start index that is below -dim_size to
-dim_size, matching PyTorch eager semantics.

Files changed in tt-xla on branch
`remediation/gemma3-causal_lm-pytorch-4B_Instruct_bnb_4bit-single_device-inference`:
- python_package/tt_torch/backend/passes.py — added clamp_out_of_range_slice_starts()
- python_package/tt_torch/backend/backend.py — import and call the new pass

The CI bringup branch (ip-172-31-23-5-tt-xla-dev/ubuntu/2026-04-22_16-54/hf-bringup-37)
in tt_forge_models already contained the three loader fixes listed above. This report's
submodule pointer for tt_forge_models points to the tip of that bringup branch
(07596e7c32).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    349.52s (0:05:49)
- Tier A attempts: 1

## Files changed
- tt-xla: python_package/tt_torch/backend/passes.py
- tt-xla: python_package/tt_torch/backend/backend.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ef46a1d363f027f92d70a6aa6a6fe0b2b393506c |
| tt-forge-models | 07596e7c3246ff006e967459b698782255fec814 |
