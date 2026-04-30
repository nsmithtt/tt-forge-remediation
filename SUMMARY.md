# Remediation Summary: gemma3_emotional_1b_i1_gguf-causal_lm-pytorch-Emotional_1B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_emotional_1b_i1_gguf/causal_lm/pytorch-Emotional_1B_i1_GGUF-single_device-inference]

## Result
FAIL — after loader fixes and Tier A slice-OOB fix, PCC=0.979 remains below 0.99 threshold due to WH BF16 matmul precision floor (same root cause as Gemma3 1B Instruct, tt-xla #3860)

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix 1, second loader failure:
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!
```

After loader fix 2, the originally-reported failure:
```
RuntimeError: Value out of range (expected to be in range of [-12, 11], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})
```
Originating from:
```
transformers/cache_utils.py:214: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

After Tier A fix, final failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9789648865914291. Required: pcc=0.99.
```

## Root cause

**Bug 1 (loader):** 26 GGUF loaders in tt_forge_models monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow
signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 now calls it
with `model_to_load=dummy_model`, raising `TypeError`. The patch persists in the
process after collection imports, breaking the Gemma3 Emotional loader which
does not itself apply the patch.

**Bug 2 (loader):** The Gemma3 Emotional GGUF tokenizer (mradermacher
Q4_K_M) embeds no `chat_template` in the GGUF metadata. The loader called
`tokenizer.apply_chat_template` unconditionally, raising `ValueError`.

**Bug 3 (tt-xla, Tier A):** The XLA lazy backend validates `aten.slice.Tensor`
start/end indices strictly: values < -size raise "Value out of range". PyTorch
eager silently clamps such indices. Gemma3's SlidingWindowCache slices with
`full_value_states[:, :, -self.sliding_window + 1:, :]` where
`sliding_window=512` but the sequence length is 12 (12-token input), so
`start = -511 < -12`. The fix intercepts `aten.slice.Tensor` in
`TorchFunctionOverride.__torch_function__` and clamps start/end to `max(val, -size)`.

**Bug 4 (Tier B, BF16 precision floor):** After all fixes, the model executes
correctly but produces PCC=0.979 vs the required 0.99. Gemma3 1B Instruct (the
non-GGUF BF16 variant) gets PCC=0.956 on the same hardware, documented in tt-xla
#3860 as a WH BF16 matmul accumulation precision floor. The GGUF Q4_K_M model
gets slightly higher PCC (0.979) than the full-BF16 variant (0.956) because the
4-bit weight quantization already introduces sufficient noise that additional
BF16 rounding errors contribute proportionally less — but the floor is still
below 0.99. Fixing this requires preserving FP32 precision through every matmul
lowering pass (cross-cutting change across tt-mlir and tt-metal).

## Fix

**Loader fixes (tt_forge_models, remediation/gemma3-emotional-1b-i1-gguf-slice-oob):**
- `*/causal_lm/pytorch/loader.py` (26 files): Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(*args, **kwargs)` and `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)`
- `gemma3_emotional_1b_i1_gguf/causal_lm/pytorch/loader.py`: Guarded `apply_chat_template` call with `if self.tokenizer.chat_template is not None` fallback to `sample_text`

**Tier A compiler fix (tt-xla, remediation/gemma3-emotional-1b-i1-gguf-slice-oob):**
- `python_package/tt_torch/torch_overrides.py`: In `TorchFunctionOverride.__torch_function__`, intercept `func is torch.ops.aten.slice.Tensor` and clamp `start`/`end` to `max(val, -size)` when the value is a known int below `-size`.

**Proposed fix for Tier B PCC floor:** Enable FP32 accumulation for BF16 matmuls on WH silicon throughout the tt-mlir lowering pipeline. This is tracked in tt-xla #3860 for all Gemma3 1B models and requires coordinated changes across tt-mlir and tt-metal. No attempt made per Tier B rules.

## Tier B justification
cross-cutting — fixing the BF16 matmul precision floor requires preserving FP32
accumulation across all matmul lowering passes in tt-mlir and tt-metal kernels.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    331.31s (5:31) for final run
- Tier A attempts: 1

## Files changed
**tt_forge_models (remediation/gemma3-emotional-1b-i1-gguf-slice-oob):**
- 26 GGUF loader files: `_patched_load_gguf_checkpoint` signature fix
- `gemma3_emotional_1b_i1_gguf/causal_lm/pytorch/loader.py`: chat_template guard

**tt-xla (remediation/gemma3-emotional-1b-i1-gguf-slice-oob):**
- `python_package/tt_torch/torch_overrides.py`: aten.slice.Tensor index clamping

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 74848586376a458b5bcbdb61d8a994844b375c6f |
| tt-forge-models | 6a03e8d07595561b5f9c0794a86209151f959f33 |
