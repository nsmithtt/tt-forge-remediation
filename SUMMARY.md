# Remediation Summary: ltx_video-pytorch-tiny_random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ltx_video/pytorch-tiny_random-single_device-inference]

## Result
FAIL — PCC=0.798 from ttnn SDPA with non-tile-aligned seq_len=8 (Tier B compiler bug)

## Stack layer
tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ttnn-sdpa-nonaligned-kv-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
(The summary message is the last line of output from the forked process.  The
actual primary failure was:)
```
torch._dynamo.exc.BackendCompilerFailed: backend='tt' raised:
NameError: name 'L' is not defined

While executing %_guards_fn : [num_users=0] = call_module[target=_guards_fn](args = (%args_0, %args_1, %args_2, %args_3), kwargs = {})
```

After fixing the guards_fn bug, the next failure was:
```
loc("custom-call.100"): error: 'ttir.scaled_dot_product_attention' op Attention mask at dim 2 must match query sequence length
ValueError: Error code: 13
```

After fixing the SDPA verifier bug, the final failure is:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7981389300419354. Required: pcc=0.99.
```

## Root cause

Three bugs in sequence:

1. **tt-xla (Tier A, fixed):** `_guards_fn` NameError.  `ep.module()` with
   `check_guards=True` (default) inserts a `_guards_fn` submodule whose
   generated code references `L` (Dynamo's locals dict).  When guard
   expressions reference keys not present in the model inputs, `L`
   substitution is incomplete, causing `NameError: name 'L' is not defined`
   during subsequent AOT re-export inside `run_decompositions`.  Fix: pass
   `check_guards=False` to both `ep.module()` call sites in
   `torch_pass_pipeline`.

2. **tt-mlir (Tier A, fixed):** SDPA attention mask dim-2 verifier rejects
   broadcast shape.  LTX-Video's cross-attention receives `encoder_attention_mask`
   of shape `[1, 8]`.  After the transformer block converts it to `[1, 1, 8]`
   and the attention processor runs `prepare_attention_mask` + `view`, the
   mask becomes `[1, H, 1, K_seq]` (dim-2 = 1, relying on PyTorch broadcast
   semantics).  `ttir.scaled_dot_product_attention` verifier required
   `mask.dim(2) == Q_seq` exactly, rejecting broadcast dim-2 = 1.  Fix:
   relax verifier to allow dim-2 = 1 in `TTIROps.cpp` and `TTNNOps.cpp`;
   add `expandMaskQueryDim` in `TTIRToTTNN.cpp` to expand dim-2 from 1 to
   Q_seq before lowering to the TTNN SDPA kernel.

3. **tt-mlir / tt-metal (Tier B, not fixed):** TTNN SDPA kernel produces
   wrong results when Q_seq or K_seq is not a multiple of 32.  The
   tiny-random model's video sequence length is `2×2×2 = 8` frames/spatial
   tokens, and the text sequence length is also 8.  Both are non-tile-aligned
   (8 % 32 ≠ 0).  The TTNN SDPA kernel pads internally but does not
   correctly mask the padding, producing PCC = 0.798.  This is the known
   `ttnn-sdpa-nonaligned-kv-pcc-wrong` Tier B bug.

## Fix

**tt-xla** (`remediation/ltx_video-pytorch-tiny_random-single_device-inference`, commit `bc657eba4`):
- `python_package/tt_torch/backend/backend.py`: pass `check_guards=False` to
  both `ep.module()` call sites in `torch_pass_pipeline`, suppressing
  `_guards_fn` insertion.

**tt-mlir** (`remediation/ltx_video-pytorch-tiny_random-single_device-inference`, commits `ae4adfd42`, `50be98b21`):
- `lib/Dialect/TTIR/IR/TTIROps.cpp`: relax `ScaledDotProductAttentionOp::verify()` to allow `mask.dim(2) == 1`.
- `lib/Dialect/TTNN/IR/TTNNOps.cpp`: same relax in TTNN verifier.
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: add `expandMaskQueryDim` helper that RepeatOps a broadcast dim-2 mask from 1 → seqLen before TTNN SDPA; also guard `shouldUseDecode` to require `kv_seq_len % 32 == 0`.

**Proposed fix for Tier B** (not implemented): The TTNN SDPA kernel's padding
path must correctly mask out padding tiles (beyond the actual Q/K sequence
length) so that softmax and output accumulation are not corrupted by
uninitialised values.  This requires changes to the SDPA kernel in tt-metal,
coordinated with the TTNN SDPA op lowering in tt-mlir — a cross-repo change
touching more than 3 files.

## Tier B justification
Indicator: **cross-cutting** — fixing TTNN SDPA precision for non-tile-aligned
seq_len requires coordinated changes to the SDPA kernel in tt-metal and the
lowering pass in tt-mlir, touching more than 3 files across two repos.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole (n150-style WH device on aus-wh-07)
- Duration:    28.23s
- Tier A attempts: 2

## Files changed
- `python_package/tt_torch/backend/backend.py` (tt-xla)
- `lib/Dialect/TTIR/IR/TTIROps.cpp` (tt-mlir)
- `lib/Dialect/TTNN/IR/TTNNOps.cpp` (tt-mlir)
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (tt-mlir)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 50be98b21 |
| tt-xla          | bc657eba4 |
| tt-forge-models | 6f725fe208 |
