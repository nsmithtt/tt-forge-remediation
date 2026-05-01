# Remediation Summary: minicpm_sala-causal_lm-pytorch-9B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[minicpm_sala/causal_lm/pytorch-9B-single_device-inference]

## Result
FAIL — ttmlir-f32-precision-not-preserved: TT compiler downcasts float32 Simple GLA state updates to BF16, causing PCC 0.754 across 24 GLA layers (required 0.99)

## Stack layer
tt-mlir

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
Original: `ModuleNotFoundError: No module named 'triton'`

After loader fixes: `AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7541460544010932. Required: pcc=0.99.`

## Root cause

MiniCPM-SALA is a 9B causal LM with a hybrid architecture: 8 standard SDPA
layers and 24 Simple GLA (Gated Linear Attention) layers using the `fla`
(flash-linear-attention) library. Three sequential bugs were found:

**Bug 1 (loader):** `modeling_minicpm_sala.py` imports `fla` ops at module
level. `fla/ops/__init__.py` eagerly imports Triton GPU kernels unavailable
on TT hardware → `ModuleNotFoundError: No module named 'triton'`. Fixed by
injecting pure-PyTorch stub modules into `sys.modules` before
`from_pretrained` is called.

**Bug 2 (loader):** During `torch.compile` tracing,
`_prepare_4d_causal_attention_mask_for_sdpa` returns a 4D `[B,1,T,T]` float
mask (it does not short-circuit to None when `is_tracing_=True`). The GLA
`attn_fn` then does `attention_mask[:, -q_len:]` on dim 1 of size 1 with
start=-18, which XLA's strict bounds check rejects as out of range [-1, 0].
Simple GLA is inherently causal via its lower-triangular decay mask, so the
mask is semantically unnecessary for batch_size=1 single-sequence inference.
Fixed by wrapping `attn_fn` in all GLA modules to drop 4D masks.

**Bug 3 (Tier B, in tt-mlir):** The GLA fused-recurrent kernel maintains
a floating-point state matrix `S` updated as:
`S = S * exp(g_gamma) + k_t ⊗ v_t` (float32 arithmetic in the reference).
With seqlen=18 (< 64 threshold → `mode="fused_recurrent"`), the model runs
24 GLA layers each with 18 timestep accumulations. The TT compiler lowers
these ops but does not preserve float32 through the lowering passes —
explicit `to(torch.float32)` casts in the stub are traced but the TT
compiler downcasts the accumulated state to BF16. After 24×18=432 BF16
accumulations, numerical error accumulates to PCC 0.754 vs the float32 CPU
reference (required: 0.99). This is the same `ttmlir-f32-precision-not-preserved`
bug seen in other models with explicit float32 arithmetic in attention/SSM paths.

## Fix

Two loader fixes committed to
`remediation/minicpm_sala-causal_lm-pytorch-9B-single_device-inference` in
`tt_forge_models`:

- `minicpm_sala/causal_lm/pytorch/loader.py`: Added `_patch_fla_for_tt_device()`
  which injects stub `fla.ops`, `fla.ops.simple_gla`, `fla.ops.simple_gla.fused_recurrent`,
  `fla.ops.utils.index`, and `fla.utils` modules into `sys.modules` before
  `from_pretrained`. Stubs implement `chunk_simple_gla` and
  `fused_recurrent_simple_gla` using pure-PyTorch (no Triton).

- `minicpm_sala/causal_lm/pytorch/loader.py`: Added `_patch_model_gla_attention()`
  which wraps `attn_fn` on all GLA attention modules to replace 4D attention
  masks with None at runtime.

**Proposed Tier B fix:** In `tt-mlir`, the `ComplexDataTypeConversion` pass
(or a precision-preservation pass) needs to honour explicit `torch.float32`
casts that appear in the FX graph after `to(torch.float32)` calls. When an
explicit cast to float32 is traced, the resulting `stablehlo.convert` op
should not be eliminated or folded into bf16. The fix would live in the
StableHLO-to-TTIR lowering in `tt-mlir` and requires auditing every pass
that can silently downcast element types.

## Tier B justification

- **cross-cutting**: preserving float32 through the TT lowering pipeline
  requires changes across multiple passes in tt-mlir (type inference,
  dtype propagation, constant folding) and potentially tt-metal kernel
  dispatch. It is not a single bounded fix in one file.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 399.61s (0:06:39) for the final run (reached PCC evaluation)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/minicpm_sala/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 92f91992e4e8ab90cca3e571e3b2e2a19cfadde6 |
| tt-forge-models | 580a56b95c86b3e67404a2729322ff92029f2ca5 |
