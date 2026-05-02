# Remediation Summary: ltx_video-pytorch-LTX_Video_0_9_8_13B_distilled-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ltx_video/pytorch-LTX_Video_0_9_8_13B_distilled-single_device-inference]

## Result
FAIL — SDPA attention mask dim 2 must match query sequence length (tt-mlir); _guards_fn NameError fixed (tt-xla) but second compiler bug blocks pass

## Stack layer
tt-xla, tt-mlir

## Tier
A

## Bug fingerprint
sdpa-mask-q-dim-broadcast

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before fix):
```
torch._dynamo.exc.BackendCompilerFailed: backend='tt' raised:
NameError: name 'L' is not defined

While executing %_guards_fn : [num_users=0] = call_module[target=_guards_fn](args = (%args_0, %args_1, %args_2, %args_3), kwargs = {})
```
(The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is the last printed line of pytest output, not the real error.)

Residual failure after fix (new error exposed):
```
loc("custom-call.100"): error: 'ttir.scaled_dot_product_attention' op Attention mask at dim 2 must match query sequence length
Failed to convert from SHLO to TTIR module
ValueError: Error code: 13
```

## Root cause
Two separate compiler-stack bugs:

1. **tt-xla `_guards_fn` NameError (fixed):** `program.module()` in `torch_pass_pipeline` uses `check_guards=True` by default. This inserts a `_guards_fn` submodule whose generated Python code references `L` (Dynamo's locals dict). When the model has guard expressions whose keys aren't in model inputs, the `L` substitution is incomplete, causing `NameError: name 'L' is not defined` during AOT Autograd re-export inside `run_decompositions`, and again when the decomposed module is interpreted. Inductor strips `_guards_fn` for the same reason.

2. **tt-mlir SDPA mask q-dim broadcast (unfixed):** The LTX Video cross-attention (attn2) produces an attention mask of shape `[batch, heads, 1, kv_seq_len]` where dim 2 = 1 is a broadcast placeholder meaning the same mask applies to every query position. The `ttir.ScaledDotProductAttentionOp` verifier requires dim 2 to exactly equal the query sequence length (no implicit broadcasting). The `TenstorrentScaledDotProductAttentionConversionPattern` in `StableHLOLegalizeCompositePass.cpp` passes the mask through without expanding it.

## Fix
**Applied (tt-xla):** `python_package/tt_torch/backend/backend.py`
- Before calling `_exported.run_decompositions(decompositions)`, patch `_exported.module` to always pass `check_guards=False`.
- Pass `check_guards=False` explicitly when extracting `compiled_graph = program.module(check_guards=False)`.
- This prevents the `_guards_fn` submodule from being inserted into the exported program.

**Proposed (tt-mlir, unfixed per one-fix-per-report rule):** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`
- In `TenstorrentScaledDotProductAttentionConversionPattern::matchAndRewrite`, when the attention mask has shape [..., 1, kvSeqLen] and query sequence length > 1, insert a `ttir::BroadcastOp` to expand dim 2 from 1 to querySeqLen before passing the mask to `ttir::ScaledDotProductAttentionOp`.
- The fix exists in commit `129eaa6f4` on branch `origin/remediation/nomic-embed-text-v1-gguf-single-device-inference` in the tt-mlir repo.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    73.63s (with first fix applied, still failing on second bug)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/backend.py` — `check_guards=False` at both `ep.module()` call sites in `torch_pass_pipeline`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 49aeacbc9c5fa5e5a6f0658fd664236727c80282 |
| tt-forge-models | 1df6fe6061bdaaaca4b929e59897d9abda5dffcf |
