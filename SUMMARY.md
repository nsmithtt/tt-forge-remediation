# Remediation Summary: bartowski_mistralai_ministral_3_14b_reasoning_2512_gguf-causal_lm-pytorch-mistralai_Ministral_3_14B_Reasoning_2512_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_mistralai_ministral_3_14b_reasoning_2512_gguf/causal_lm/pytorch-mistralai_Ministral_3_14B_Reasoning_2512_GGUF-single_device-inference]

## Result
FAIL — PCC 0.979 < 0.99 after all loader and Tier A compiler fixes applied; gap is from TT hardware bfloat16 matmul precision vs CPU float32, not attributable to a single fixable op

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
Original failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After loader fix, second failure:
```
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)
While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
```

After slice fix, remaining failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.979075465284767. Required: pcc=0.99.
```

## Root cause

**Loader bug (fixed):** The GGUF file declares architecture `mistral3`, which transformers 5.x does not recognise. The loader was missing registration of `mistral3` as an alias for `mistral` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, and `GGUF_TO_FAST_CONVERTERS`. Additionally, the `_patched_load_gguf_checkpoint` wrapper used a fixed signature `(gguf_path, return_tensors=False)` that did not forward the new `model_to_load` kwarg added in transformers 5.2.0, causing a TypeError through the patching chain.

**Compiler bug — tt-xla (Tier A, fixed):** Mistral's sliding-window KV cache update computes `full_key_states[:, :, -sliding_window + 1 :, :]` where `sliding_window=4096`. With seq_len=128, start=-4095. The XLA lowering for `aten.slice.Tensor` rejects any start index outside `[-size, size-1]` (here `[-128, 127]`) with `RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)`. PyTorch CPU silently clamps these to 0; XLA raises an error via the lazy tensor validation in `torch/csrc/lazy/core/helpers.cpp`. The fix adds a `canonicalize_slice_indices` FX graph pass in `tt-xla/python_package/tt_torch/backend/passes.py` that clamps concrete negative start indices to `-size` before the graph reaches `bridge.extract_compiled_graph`.

**PCC gap (unfixed, Tier B):** After both fixes the model runs to completion but achieves PCC=0.979 against the CPU float32 reference. Measured PCC between CPU float32 and CPU bfloat16 is effectively 1.0, ruling out simple bfloat16 rounding. The remaining 2% gap is from TT hardware's bfloat16 matmul accumulation path vs CPU float32 for a Q4_K_M GGUF-quantized 14B model. Fixing this would require changing the hardware-level accumulation precision, which is Tier B (cross-cutting).

## Fix

**Loader (tt_forge_models):**
- `bartowski_mistralai_ministral_3_14b_reasoning_2512_gguf/causal_lm/pytorch/loader.py`: Added `_patch_mistral3_support()` to register `mistral3` in GGUF architecture tables and `GGUF_TO_FAST_CONVERTERS`. Changed `_patched_load_gguf_checkpoint` to accept `**kwargs` and forward them to `_orig_load_gguf_checkpoint`. Added `_patched_get_gguf_hf_weights_map` to remap `mistral → mistral3` for gguf-py weight-map lookup.
- 27 other GGUF loaders (cherry-pick of commit `0d7443d0e7`): updated old-style `(gguf_path, return_tensors=False)` signatures to `(gguf_path, return_tensors=False, **kwargs)` so the patching chain correctly propagates `model_to_load`.

**Compiler frontend (tt-xla):**
- `python_package/tt_torch/backend/passes.py`: Added `_get_node_shape()` helper (traverses passthrough ops like `tt.mark_argument_attributes` to find shape metadata) and `canonicalize_slice_indices()` FX pass that clamps out-of-bounds negative start indices to `-size`.
- `python_package/tt_torch/backend/backend.py`: Import and call `canonicalize_slice_indices` in `torch_pass_pipeline` after `bypass_dtype_promotion_and_redundant_cast`.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

N/A (Tier A fix was applied for the slice bug; the PCC issue is Tier B but is a separate second bug)

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 176.37s (0:02:56)
- Tier A attempts: 1

## Files changed
**tt_forge_models (remediation branch):**
- `bartowski_mistralai_ministral_3_14b_reasoning_2512_gguf/causal_lm/pytorch/loader.py`
- 27 additional loader files (batch fix for `model_to_load` kwarg forwarding)

**tt-xla (remediation branch):**
- `python_package/tt_torch/backend/passes.py`
- `python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8d39e329df35e800734fdd36b63773cd4e228cfe |
| tt-forge-models | 9f0e76f0229b90d9fffe5e3e6bebbcd00d793d6c |
