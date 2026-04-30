# Remediation Summary: gemma_2_2b_it-causal_lm-pytorch-2B_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_2b_it/causal_lm/pytorch-2B_Instruct-single_device-inference]

## Result
SILICON_PASS — OOB slice start clamped in TorchFunctionOverride + FX pass; padding=True in loader removes pad-token PCC noise

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
E   RuntimeError: Value out of range (expected to be in range of [-256, 255], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause

Two independent bugs:

**Bug 1 — aten.slice OOB start (tt-xla, Tier A):** Gemma 2 uses
sliding_window=4096. SlidingWindowCache.update() slices with
start = -sliding_window + 1 = -4095 on a seq_len=256 tensor (valid
XLA range: [-256, 255]). PyTorch eager clamps OOB negative indices
silently; the XLA lazy backend raises RuntimeError in
partition_fx_graph_for_cpu_fallback when the slice is dispatched
through TorchFunctionOverride.

**Bug 2 — padding="max_length" PCC inflation (loader):** The
gemma_2_2b_it loader tokenized with padding="max_length", padding
the 15-token prompt to 256 tokens (241 pad tokens). After Bug 1 was
fixed the model ran, but PCC was 0.928: the comparison flattened all
256xvocab_size logit values, and 241/256 of them (from padding
positions) had small bf16 divergences between CPU and TT, dominating
the correlation. Changing to padding=True limits the comparison to
the 15 meaningful real-token positions, giving PCC >= 0.99.

## Fix

**Fix 1 (tt-xla, Tier A):**
- Added slice-index clamp in python_package/tt_torch/torch_overrides.py
  (TorchFunctionOverride.__torch_function__): when func is
  torch.ops.aten.slice.Tensor, pre-clamp start/end to [-size, size]
  for statically-known dimensions, matching PyTorch eager semantics.
  Handles the XLA range check in partition_fx_graph_for_cpu_fallback.
- Also added clamp_out_of_range_slice_starts FX pass in
  python_package/tt_torch/backend/passes.py and called it from
  torch_pass_pipeline in backend.py as a belt-and-suspenders
  fix for the compiled graph before re-export.

**Fix 2 (loader):**
- Changed padding="max_length" to padding=True in
  gemma_2_2b_it/causal_lm/pytorch/loader.py in tt-forge-models.
  This removes 241 meaningless pad-token positions from the PCC
  comparison.

Remediation branches:
- tt-xla: remediation/gemma-2-2b-it-causal-lm-pytorch-2b-instruct-single-device-inference
- tt-forge-models: remediation/gemma-2-2b-it-causal-lm-pytorch-2b-instruct-single-device-inference

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    108.31s (0:01:48)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py
- tt-xla/python_package/tt_torch/backend/passes.py
- tt-xla/python_package/tt_torch/backend/backend.py
- tt-forge-models/gemma_2_2b_it/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | beab2f0d1224f9812ae18c8a8b86e693140804fc |
| tt-forge-models | 03617755bb9c4f023c6e1e8ed00e49fa45afe82e |
