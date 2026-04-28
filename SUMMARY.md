# Remediation Summary: aname_tommy_test-causal_lm-pytorch-Test-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aname_tommy_test/causal_lm/pytorch-Test-single_device-inference]

## Result
FAIL — torch_chunk_gated_delta_rule Python for loops (63 iterations of chunk_size=64) cause XLA to generate an enormous unrolled computation graph across 24 GatedDeltaNet layers, making compilation effectively non-terminating

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
xla-python-for-loop-unroll-chunk-gated-delta-rule

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
`Aname-Tommy/Test` is a `qwen3_5_text` model (Qwen3.5 hybrid architecture) with 32 decoder layers: 24 `linear_attention` (GatedDeltaNet) layers and 8 `full_attention` (standard self-attention) layers.

Because the `flash-linear-attention` CUDA library and `causal-conv1d` are not installed (and cannot be used on TT hardware), the GatedDeltaNet layers fall back to `torch_chunk_gated_delta_rule`, a pure-PyTorch implementation. This fallback contains two nested Python for loops:

1. **Inner triangular fix-up loop**: `for i in range(1, chunk_size)` — 63 iterations for `chunk_size=64`. Each iteration performs indexed tensor slicing and in-place update to compute a lower-triangular sequential matrix recurrence.
2. **Chunk processing loop**: `for i in range(0, seq_len // chunk_size)` — 2 iterations for `seq_len=128, chunk_size=64`.

Under XLA/torch.compile, Python for loops are unrolled at trace time. With 24 GatedDeltaNet layers, the total unrolled loop bodies add up to `24 × (63 + 2) = 1,560` tensor operations, each containing multiple matrix multiplications. The resulting graph is enormous, causing XLA compilation to take hours. Locally, the CPU-only forward pass (which directly executes the loops) takes 47 seconds for the 32-layer, 4096-hidden-dim model.

The inner triangular loop computes a sequential lower-triangular matrix recurrence where row `i` depends on the already-updated rows `0..i-1`, making it inherently sequential and impossible to trivially vectorize.

## Fix
A fix would require a vectorized, loop-free implementation of the sequential GatedDeltaNet recurrence in `transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule` that avoids Python-level iteration over `chunk_size`.

The sequential dependency (`attn[i, :i] = attn[i, :i] @ (I + attn[:i, :i])` where `attn[:i, :i]` reflects already-updated rows) means that each row depends on all previous rows, ruling out a straightforward `torch.vmap` vectorization. An algorithm based on parallel prefix scans over the triangular matrix product would be needed — a non-trivial mathematical reformulation.

Alternative: if the `fla` (Flash Linear Attention) library were ported to TT hardware, the CUDA kernel `chunk_gated_delta_rule` would replace the Python fallback. This is the upstream intended path.

The fix lives in tt-xla (the frontend compilation layer) because the root cause is XLA's inability to handle Python for loops over device tensors without full graph unrolling.

## Tier B justification
Applies: **new-infrastructure**

The fix requires a vectorized implementation of the GatedDeltaNet sequential recurrence without Python for loops. The inner loop (`for i in range(1, 64)`) implements a lower-triangular sequential matrix product with data dependencies between iterations — no existing PyTorch primitive can express this in a single operation. Implementing a correct parallel prefix scan over triangular matrices, or porting the `fla` CUDA kernel to TT hardware, both constitute new infrastructure well beyond a single scoped fix.

## Verification
- pytest exit: TIMEOUT (test killed on silicon, never returned within test timeout)
- Hardware: n150
- Duration: >36 minutes observed locally before kill; CPU-only forward 47s
- Tier A attempts: N/A

## Files changed
None — Tier B, no fix attempted

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348e7c9f31a1df6faea36de7eb42a3c01 |
