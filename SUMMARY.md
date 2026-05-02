# Remediation Summary: ministral_3_3b_instruct_2512_gguf-causal_lm-pytorch-3_3B_Instruct_2512_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ministral_3_3b_instruct_2512_gguf/causal_lm/pytorch-3_3B_Instruct_2512_GGUF-single_device-inference]

## Result
FAIL — PCC 0.9813 < required 0.99; residual TT precision loss after loader and Tier A slice fixes; CPU BF16/FP32=0.9999 confirms this is not a BF16 floor; consteval-on-host precision issue tt-xla #1242

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original error:
```
NotImplementedError: Unknown gguf model_type: mistral in gguf-py.
```

After loader fix, second error:
```
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)
While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
```

After both fixes, residual:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.981285906204096. Required: pcc=0.99.
```

## Root cause
Three layered issues:

**Issue 1 — Loader (fixed):** The ministral loader patches `load_gguf_checkpoint` to remap `model_type` "mistral3"→"mistral" so `AutoModelForCausalLM` resolves `MistralForCausalLM`. But `get_gguf_hf_weights_map` then receives `model_type="mistral"` from `hf_model.config.model_type` and fails because gguf-py 0.18 only has `MODEL_ARCH.MISTRAL3 → "mistral3"` (no `MODEL_ARCH.MISTRAL`). The loader needed a complementary patch on `get_gguf_hf_weights_map` to remap "mistral"→"mistral3" for the gguf-py arch lookup.

**Issue 2 — tt-xla Tier A (fixed):** `MistralForCausalLM` with `sliding_window=4096` produces `aten.slice.Tensor(kv_cache, 2, -4095, MAX_INT)` during the sliding-window KV update (`full_value_states[:, :, -sliding_window+1:, :]`). XLA's `partition_fx_graph_for_cpu_fallback` validates slice indices strictly and rejects -4095 (out of int8 range used internally). PyTorch CPU silently clamps. Fixed by clamping both start and end slice args in `TorchFunctionOverride.__torch_function__` before the op is dispatched.

**Issue 3 — tt-mlir Tier B (residual):** After both fixes the model runs on silicon but achieves PCC=0.9813 vs required=0.99. Measured CPU BF16/FP32=0.9999, confirming the gap is NOT BF16 accumulation — it is a real TT precision deviation. This is the known consteval-on-host precision issue (tt-xla #1242) where operations moved from host FP32 to device BF16 produce systematically lower PCC.

## Fix
**Fix 1 — loader** (`tt_forge_models/ministral_3_3b_instruct_2512_gguf/causal_lm/pytorch/loader.py`):
Added `patched_get_gguf_hf_weights_map` inside `_patch_transformers_mistral3_gguf()` that:
- Reads `hf_model.config.model_type` when `model_type` arg is None
- Remaps "mistral" → "mistral3" before delegating to the real function
- Patches `gguf_utils.get_gguf_hf_weights_map` at module-level

**Fix 2 — tt-xla** (`tt-xla/python_package/tt_torch/torch_overrides.py`):
Added slice index clamping in `TorchFunctionOverride.__torch_function__`:
- Detects `func is torch.ops.aten.slice.Tensor` before the graph is built
- Clamps both `args[2]` (start) and `args[3]` (end) to `-dim_size` when below the valid floor
- Fires inside `TorchFunctionMode`, before FX tracing, so the bad indices never reach XLA

**Residual (not fixed):** PCC=0.9813 < required 0.99 due to tt-xla #1242.

## Tier B justification
cross-cutting — fixing the consteval-on-host precision loss requires ensuring FP32-precision constants currently computed on TT device in BF16 are evaluated with higher precision; touches multiple lowering passes across tt-mlir/tt-xla.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    348.64s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/ministral_3_3b_instruct_2512_gguf/causal_lm/pytorch/loader.py` — add `patched_get_gguf_hf_weights_map` to remap "mistral"→"mistral3"
- `tt-xla/python_package/tt_torch/torch_overrides.py` — clamp OOB slice start/end in `TorchFunctionOverride`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b90ee6b60f43994fa47d920b27911bad886af1f7 |
| tt-forge-models | 086c163db1cdc4d997e2bcfe6c8d8271d19b2b45 |
