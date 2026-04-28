# Remediation Summary: cog-florence-2-2-large-pytorch-Large-single-device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cog_florence_2_2_large/pytorch-Large-single_device-inference]

## Result
FAIL — SDPA decode crash fixed; second bug PCC=0.099 is Tier B unknown mechanism

## Stack layer
tt-mlir

  - `loader`         — bug was in tt_forge_models or test inputs
  - `tt-xla`         — bug in compiler frontend (PJRT, torch_xla bridge)
  - `tt-mlir`        — bug in compiler core (StableHLO→TTIR lowering)
  - `tt-metal`       — bug in backend runtime / kernels
  - `hardware-class` — model exceeds single-device capacity (XFAIL)
  - `n/a`            — NO_FIX_NEEDED (could not reproduce)

## Tier
B

  - `N/A` — loader fix, no fix needed, or hardware-class XFAIL
  - `A`   — compiler-stack fix attempted (succeeded → SILICON_PASS,
            ran out of attempts → FAIL with explanation)
  - `B`   — compiler-stack bug filed without attempting fix

## Bug fingerprint
sdpa-k-chunk-size-lt-32

  Format: `<area>-<short-description>`. Use the same string verbatim
  whenever a later report hits the same bug — this is how the audit
  groups failures.

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Root TT_FATAL (from device log):
  k_chunk_size=2 (must be >= 32)

The SDPA decode kernel calls get_chunk_size(K_seq) which returns the largest
power-of-2 divisor of K_seq, capped at 512. For Florence-2 cross-attention,
K_seq=578 (576 image tokens + 2 text tokens from "<MORE_DETAILED_CAPTION>").
578 = 2 × 17², so get_chunk_size(578) = 2 < 32 → TT_FATAL.

## Root cause
Two paths independently activated the SDPA decode kernel for Florence-2's
decoder cross-attention (K_seq=578, Q_seq=1):

1. **SDPAFusingPattern** (tt-mlir TTNN fusing pass): the `isDecode` predicate
   checked only `Q_seq == 1`, with no guard on K_seq divisibility. With
   `TTMLIR_ENABLE_OPMODEL=ON` and `enableOpConstraints=true` the pass fires on
   every `MatmulOp+SoftmaxOp` pair, including Florence-2's eager-attention
   `torch.bmm` calls. isDecode=true for cross-attention (Q=1, K=578) →
   `ScaledDotProductAttentionDecodeOp` created → TT_FATAL at runtime.

2. **TTIRToTTNN shouldUseDecode** (composite SDPA path): checked
   `K_seq >= 32` which is true for K=578, but the correct requirement is
   `K_seq % 32 == 0`. This path is used when the model employs
   `F.scaled_dot_product_attention`; Florence-2 uses eager attention so this
   path was secondary, but the check was incorrect in principle.

After fixing both guards the crash is eliminated. The test now runs to
completion but fails with PCC=0.09955934649555423 (required 0.99). This PCC
failure is a pre-existing second bug unmasked by the crash fix. The PCC value
is identical across all test runs regardless of which SDPA path is taken
(confirmed: adding a guard to also block prefill SDPA for Q_seq=1 produced
exactly the same PCC), indicating the root cause is unrelated to SDPA
dispatch.

## Fix
**Committed on branch `remediation/cog-florence-2-2-large-pytorch-Large-single-device-inference`
in tt-mlir (commit 576c82a91):**

`lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`:
- `isDecode` condition: added `kSeqLen > 0 && kSeqLen % 32 == 0` guard.
- Added early return in prefill branch when `qShape[kSeqLenDim] == 1`:
  autoregressive decode with K not divisible by 32 falls back to plain matmul.

`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`:
- `shouldUseDecode()`: changed `keyType.getDimSize(kSeqLenDim) >= 32` to
  `kSeqLen > 0 && kSeqLen % 32 == 0`.

**Proposed fix for the PCC bug (not implemented):**
Diagnosis-first work is needed to identify which op produces wrong results.
The PCC of ~0.099 is too low for BF16 rounding; it indicates a systematic
wrong-result from one or more ops. The candidate areas are:
  - DaViT ChannelAttention uses a transposed attention formulation
    (attention over channel dimension, not spatial). SDPAFusingPattern may
    be producing mathematically equivalent but numerically incorrect SDPA ops
    for this unusual Q_seq=K_seq=C//groups, head_dim=N_patches configuration.
  - Some other tt-mlir op (matmul, elementwise, normalization) may have a
    correctness bug for the specific tensor shapes in this model.

## Tier B justification
`internal-error-unknown-mechanism` — The PCC=0.099 bug is a second
compiler-stack failure revealed after the SDPA crash fix. Its root cause is
unknown without additional instrumentation. The mechanism (which op, which
kernel, which shape) cannot be determined from the test output alone. Diagnosis
requires either enabling TTXLA_LOGGER_LEVEL=DEBUG to dump compiled modules and
identifying which attention/matmul op produces wrong output, or bisecting op by
op. This diagnosis-first work exceeds the scope of a single Tier A fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~5 minutes (compilation ~3 min + inference ~2 min)
- Tier A attempts: 1 (fixed crash; PCC failure is separate bug)

## Files changed
- tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp
- tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 576c82a910ffb34459ca67dd7ebd9870372af2ab |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8d3fa853464e1d39b15b27ca6cb75c7318ea476d |
