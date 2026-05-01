# Remediation Summary: ltxv_0_9_6_gguf-pytorch-0.9.6_distilled_Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ltxv_0_9_6_gguf/pytorch-0.9.6_distilled_Q4_0-single_device-inference]

## Result
FAIL — Tier B compiler bug: GGUF Q4_0 dequantization produces UINT8 tensors that cannot be untilized on device

## Stack layer
loader, tt-xla, tt-mlir

## Tier
A

## Bug fingerprint
gguf-uint8-to-layout-device-untilize

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded
E
E   from user code:
E      File ".../diffusers/quantizers/gguf/utils.py", line 520, in dequantize_gguf_tensor
E       tensor = tensor.as_tensor()
E     File ".../diffusers/quantizers/gguf/utils.py", line 545, in as_tensor
E       return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)
E     File ".../diffusers/quantizers/gguf/utils.py", line 564, in __torch_function__
E       result = super().__torch_function__(func, types, args, kwargs)
E     [Previous line repeated 158 more times]
```

## Root cause

Three bugs were found in sequence:

**Bug 1 (loader):** `diffusers.quantizers.gguf.utils.GGUFParameter.as_tensor()` calls
`torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)` without
`DisableTorchFunctionSubclass`. Under torch.compile/dynamo, this triggers
`GGUFParameter.__torch_function__` which calls `super().__torch_function__()`.
Because `GGUFParameter` is still in `types`, the dispatch re-enters
`GGUFParameter.__torch_function__` again — infinite recursion.

**Bug 2 (tt-xla):** After fixing the as_tensor recursion, dynamo's symbolic-shape
guard nodes (`_guards_fn` call_module) were injected during `run_decompositions`
re-tracing. Their forward() closes over `L` (the dynamo locals dict) which is
absent when `PropagateUnbackedSymInts` executes the graph metadata pass, producing
`NameError: name 'L' is not defined`. These nodes are dead code (num_users=0);
skipping them during the metadata pass and erasing them from the compiled graph
resolves this.

**Bug 3 (tt-mlir, Tier A):** The LTXVideo transformer generates `tosa.pow` in two
places:
1. In the TTNN compilation path: no `tosa.pow → ttir.pow` lowering existed in
   `TosaToTTIRPatterns.cpp`, so the op reached `TTNNCommonToRuntime` pipeline
   unlowered and failed legalization.
2. In the CPU fallback path: `ttir.pow` was lowered to `tosa.pow` via
   `ElementwiseBinaryOpToTosaPattern` in `EltwiseBinary.cpp`, but
   `tosa::addTosaToLinalgPasses` does not legalize `tosa.pow` inside the nested
   `CPUModuleOp` context.

**Terminal bug (Tier B, tt-metal):** After fixing all three issues above, GGUF Q4_0
dequantization (`GGUFLinear.forward_native`) is traced by torch.compile into the
device graph. This emits UINT8 tensor operations that reach `to_layout` in the
runtime. `canUntilizeDataTypeOnDevice` does not support UINT8:
```
FATAL | Currently to_layout does not support device to device untilize
      | for output data type or memory layout: UINT8
```
Q4_0 weights cannot be fully dequantized before loading (the 2B model dequantized
to BF16 would be ~4 GB, but more importantly the GGUF linear module's forward
keeps weights quantized and dequantizes per-call). This requires either a new
tt-metal kernel for UINT8 layout conversion or a tt-xla/diffusers integration
that eagerly dequantizes GGUF weights before compilation.

## Fix

**Loader (tt_forge_models, commit 4e4a65887c):**
`ltxv_0_9_6_gguf/pytorch/loader.py` — monkey-patch
`diffusers.quantizers.gguf.utils.GGUFParameter.as_tensor` with a version that
wraps `_make_subclass` in `torch._C.DisableTorchFunctionSubclass()` to break the
infinite `__torch_function__` dispatch loop.

**tt-xla (commit 096e3e8e1):**
`python_package/tt_torch/backend/backend.py` — in `torch_pass_pipeline`:
1. Temporarily patch `PropagateUnbackedSymInts.run_node` to return None for
   dead `_guards_fn` call_module nodes so `run_decompositions` can complete.
2. After `program.module()`, erase any surviving `_guards_fn` nodes from the
   compiled graph before passes run.

**tt-mlir (Tier A, commits 8f45fd550 + dc05bc2cb on remediation branch):**
- `lib/Conversion/TosaToTTIR/TosaToTTIRPatterns.cpp`: Add
  `TosaToTTIRDefaultOpConversionPattern<tosa::PowOp, mlir::tt::ttir::PowOp>`
  in `addElementwiseBinaryOpsConversionPatterns` so the TTNN path handles `tosa.pow`.
- `lib/Conversion/TTIRToLinalg/EltwiseBinary.cpp`:
  - Remove `ElementwiseBinaryOpToTosaPattern<ttir::PowOp, tosa::PowOp>` from
    `populateTTIRToTosaEltwiseBinaryPatterns`.
  - Add `ElementwiseBinaryOpToMathPattern<ttir::PowOp, math::PowFOp>` to
    `populateTTIRToLinalgEltwiseBinaryPatterns` to use `linalg.generic(math.powf)`
    directly instead of the TOSA intermediate that fails in the `CPUModuleOp` context.

**Proposed fix for terminal Tier B bug:**
Add UINT8 → BF16 layout conversion support in tt-metal's `canUntilizeDataTypeOnDevice`
(in `layout_converter.cpp`), OR pre-dequantize all GGUF weights to BF16 before
`torch.compile` tracing so UINT8 never reaches the device. The pre-dequantization
approach requires looping over all `GGUFLinear` modules and replacing them with
`nn.Linear` with BF16 weights before the model is compiled.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

Adding UINT8 layout conversion requires new tt-metal kernel infrastructure
(`canUntilizeDataTypeOnDevice` must support UINT8, plus the associated
tilize/untilize kernels). The alternative (pre-dequantizing GGUF weights)
requires cross-cutting changes to diffusers integration. Both paths require
more than 3 files across multiple repos.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~83s (per run after loader+tt-xla+tt-mlir fixes applied)
- Tier A attempts: 1

## Files changed

**tt_forge_models (remediation/ltxv_0_9_6_gguf-pytorch-0.9.6_distilled_Q4_0-single_device-inference):**
- `ltxv_0_9_6_gguf/pytorch/loader.py`

**tt-xla (remediation/ltxv_0_9_6_gguf-pytorch-0.9.6_distilled_Q4_0-single_device-inference):**
- `python_package/tt_torch/backend/backend.py`

**tt-mlir (remediation/ltxv_0_9_6_gguf-pytorch-0.9.6_distilled_Q4_0-single_device-inference):**
- `lib/Conversion/TosaToTTIR/TosaToTTIRPatterns.cpp`
- `lib/Conversion/TTIRToLinalg/EltwiseBinary.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | dc05bc2cbac526d41d61750e58eca912aec2666e |
| tt-xla          | 3ae3526bd7ef32ac56b872a5835481e203a2347d |
| tt-forge-models | 4e4a65887ca652c3614a89546d65f9e5be377acd |
