# Remediation Summary: baichuan-causal_lm-pytorch-tiny-random-baichuan2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[baichuan/causal_lm/pytorch-tiny-random-baichuan2-single_device-inference]

## Result
FAIL — loader meta tensor bug fixed; PCC 0.496 vs required 0.99 is a Tier B compiler-stack bug

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
f32-sdpa-no-composite-pcc-failure

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

Actual error (before loader fix):
NotImplementedError: Cannot copy out of meta tensor; no data!

  File "modeling_baichuan.py", line 130, in forward
    self.cos_cached = self.cos_cached.to(x.device)
  File "tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))

After loader fix (PCC failure on silicon):
AssertionError: Evaluation result 0 failed: PCC comparison failed.
  Calculated: pcc=0.49619159044736. Required: pcc=0.99.

## Root cause
**Loader bug (fixed):** Transformers 5.x changed `from_pretrained` to use lazy
meta-device initialization (`low_cpu_mem_usage=True` by default). The baichuan
`RotaryEmbedding` module stores `inv_freq`, `cos_cached`, and `sin_cached` as
plain Python attributes — not via `register_buffer` — so they are never
materialized from the checkpoint state_dict. After loading they remain as meta
tensors with shape but no data. When the forward pass calls
`self.cos_cached.to(x.device)`, the meta-tensor copy raises
`NotImplementedError: Cannot copy out of meta tensor; no data!`.

**Compiler-stack bug (unfixed):** After materializing the rotary buffers the model
compiles and runs to completion on TT silicon, but the output PCC is 0.496 vs the
required 0.99. The model is loaded in float32 (no dtype_override). The
`tt_torch` SDPA composite (`tenstorrent.scaled_dot_product_attention`) applies only
when all SDPA inputs are bfloat16; with float32 inputs the constraint check in
`_check_sdpa_constraints` returns False, and the model falls back to the StableHLO
SDPA decomposition path. The StableHLO SDPA path with the explicit additive
causal float mask (`torch.finfo(float32).min` for masked positions) and float32
inputs produces wrong numerical results on TT silicon. The exact compiler/kernel
bug is not isolatable without targeted hardware-level debugging.

CPU bfloat16 vs CPU float32 PCC = 0.9999, confirming the failure is in the TT
hardware execution path, not the dtype conversion alone.

## Fix
**Loader fix (applied):**
`baichuan/causal_lm/pytorch/loader.py` in `tt-forge-models`, committed on branch
`remediation/baichuan-causal_lm-pytorch-tiny-random-baichuan2-single_device-inference`.

After `AutoModelForCausalLM.from_pretrained(...)`, iterate all modules. When a
module has `cos_cached`, `sin_cached`, and `inv_freq` that are meta tensors,
recompute `inv_freq` from the shape already known (shape[0]*2 gives dim, base
hardcoded to 10000 matching the model's `__init__`), then recompute `cos_cached`
and `sin_cached` from `max_seq_len_cached` and the recomputed `inv_freq`.

**Compiler-stack bug (FAIL — proposed fix):**
The `_check_sdpa_constraints` function in
`tt-xla/python_package/tt_torch/composite_ops.py` rejects float32 SDPA inputs
and falls back to StableHLO decomposition. Either:
(a) The StableHLO SDPA lowering for float32 inputs with explicit additive causal
    masks needs to be corrected in tt-mlir's StableHLO → TTIR lowering, or
(b) The TT SDPA composite should be made to accept float32 inputs (with
    appropriate hardware conversion to bfloat16 inside the composite).

## Tier B justification
- **cross-cutting**: The fix requires either changing how float32 SDPA is lowered
  through the entire StableHLO → TTIR → TTNN pipeline, or changing the SDPA
  composite constraint check and ensuring the hardware handles float32→bfloat16
  conversion correctly inside the composite. Both options touch multiple files
  across tt-xla and tt-mlir.
- **internal-error-unknown-mechanism**: The exact location in the SDPA/lowering
  stack that produces PCC 0.496 for a float32 model with an explicit causal mask
  cannot be determined without targeted hardware-level debugging of intermediate
  computations.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    39.13s (after loader fix)
- Tier A attempts: N/A

## Files changed
- `baichuan/causal_lm/pytorch/loader.py` (tt-forge-models — loader fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6b085d2244abe1d1c4df5392658cf0ed3131716d |
| tt-forge-models | 7d89176fcd9298f86eb7e850ef1c0a7a8cddcda5 |
