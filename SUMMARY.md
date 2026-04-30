# Remediation Summary: autoformer-pytorch-Tourism_Monthly-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[autoformer/pytorch-Tourism_Monthly-single_device-inference]

## Result
FAIL — pcc=0.9650 < required 0.99 after all loader fixes; gap is not bfloat16 accumulation (CPU bf16 simulation gives pcc~1.000), pointing to a compiler-stack numerical correctness bug

## Stack layer
loader, tt-mlir

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
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9650542548367147. Required: pcc=0.99.
```

Original failure (before loader fixes):
```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors
```

## Root cause
Three distinct loader bugs were fixed; a fourth compiler-stack bug remains unfixed.

**Bug 1 (loader): BFloat16 FFT unsupported.**
`TorchDynamicLoader.load_model()` passes `dtype_override=torch.bfloat16` to any loader
whose `load_model` signature contains a `dtype_override` parameter. The autoformer's
`load_model` accepted `**kwargs` which forwarded `torch_dtype=bfloat16` via
`AutoformerForPrediction.from_pretrained(..., **kwargs)`. PyTorch does not implement
`torch.fft.rfft` for BFloat16, giving `RuntimeError: Unsupported dtype BFloat16`.
Fix: removed `dtype_override` from the signature so `load_model` always loads in float32.

**Bug 2 (loader): Dynamo materializes incompatible 4-D attention mask.**
Under Dynamo tracing, `is_tracing()` returns True, preventing
`_ignore_bidirectional_mask_sdpa` from returning True (which would skip mask creation).
`AutoformerForPrediction.from_pretrained` sets `config._attn_implementation = "eager"`,
which IS in `ALL_MASK_ATTENTION_FUNCTIONS._global_mapping`, so mask creation proceeds
and materialises a `(1, 1, tgt_len, src_len)` 4-D mask. The autocorrelation attention
(line 530 of `modeling_autoformer.py`) then tries to view its `(bsz*heads, tgt_len,
channel)` output as `(bsz, heads, tgt_len, src_len)` where `channel=16 ≠ src_len=24`,
giving a Dynamo fake-tensor view failure. On CPU eager `is_tracing()` is False and the
all-ones mask is skipped (mask=None), so line 530 is never executed.
Fix: reset `model.config._attn_implementation = None` after `from_pretrained`; this
causes `_preprocess_mask_arguments` to early-exit with None unconditionally,
matching CPU-eager behaviour.

**Bug 3 (loader): Scalar loss poisons PCC evaluation.**
`AutoformerForPrediction.forward()` returns `Seq2SeqTSPredictionOutput` which includes
a scalar `loss` field when `future_values` is provided. The evaluator's `_is_single_element`
returns True for `tensor.numel() == 1`, causing `compute_pcc` to return 0.0. Since PCC is
`min()` over all pytree leaves, the scalar loss drives the final PCC to 0.0, masking the
meaningful params comparison.
Fix: added `_AutoformerParamsWrapper(nn.Module)` that intercepts the model output and
returns only `.params` (a tuple of three `(1, prediction_length)` tensors).

**Bug 4 (compiler-stack, Tier B): PCC gap after all loader fixes.**
After the three loader fixes, the test runs on silicon but gives pcc=0.9650542548367147
(required 0.99). Extensive CPU simulation rules out a bfloat16 precision floor:
- bf16 input rounding: pcc~1.0000
- bf16 weight quantization: pcc~1.0000
- bf16 matmul simulation (all nn.Linear in bf16): pcc=0.999992
- full bf16 model (all weights + inputs cast through bf16): pcc=0.999997
None of these explain the 0.9650 gap, ruling out bfloat16 accumulation as the cause.
The FFT op itself is accurate on TT hardware (standalone pcc=1.0 on a (4,24) input).
The root cause is an unidentified numerical correctness bug in the tt-mlir/tt-xla
compiler stack — likely in how the autocorrelation's complex multiply, multi-axis reduce,
or FFT-based delay aggregation is lowered to TT MLIR. Diagnosis requires inspection of
the compiled StableHLO graph and intermediate tensor values.

## Fix
Three loader fixes committed on `remediation/autoformer-pytorch-Tourism_Monthly-single_device-inference`
in `tt_forge_models`:

1. `autoformer/pytorch/loader.py`: removed `dtype_override` from `load_model` signature
2. `autoformer/pytorch/loader.py`: added `model.config._attn_implementation = None` after `from_pretrained`
3. `autoformer/pytorch/loader.py`: added `_AutoformerParamsWrapper` class and wrapped model return

Proposed fix for Bug 4: inspect the StableHLO graph emitted by `backend="tt"` for this
model and compare per-op outputs between TT and CPU execution. Locate the first op where
TT diverges, then triage whether it is a missing/incorrect lowering in tt-mlir or a
precision-mode flag in tt-metal.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The PCC gap cannot be attributed to a single missing lowering pattern or a fix in
one named function. The simulation results rule out bfloat16 arithmetic in matmul,
weights, and inputs — the residual 0.035 PCC gap has no identified single-op root
cause. Diagnosing it requires tracing per-op numerical divergence through the full
compiled graph, which is cross-cutting investigation work before any fix can be
attempted.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    75.54s (0:01:15) for the failing run
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/autoformer/pytorch/loader.py` (three loader fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 0f6251464 |
| tt-forge-models | 5c5213fc1f |
