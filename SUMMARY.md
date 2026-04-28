# Remediation Summary: deepseek/deepseek_r1_distill/pytorch-Distill_Qwen_7B-single_device-inference

## Skill version
9

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_distill/pytorch-Distill_Qwen_7B-single_device-inference]

## Result
FAIL — TT SDPA k_chunk_size constraint with seq_len=22 < 32 corrupts attention outputs

## Failure
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8898229810701653. Required: pcc=0.95.
```

## Root cause
**Layer: compiler core (tt-mlir) / runtime (tt-metal) — SDPA decomposition**

The TT hardware's SDPA kernel requires `k_chunk_size >= 32` (i.e., the key sequence length must be at least 32 tokens). DeepSeek-R1-Distill-Qwen-7B is loaded with transformers 5.2.0, which changed the default `_attn_implementation` from `"eager"` to `"sdpa"` for PyTorch 2.7. The test prompt tokenizes to 22 tokens (seq_len=22 < 32), which triggers the k_chunk_size constraint in the TT SDPA lowering and corrupts the attention scores — producing PCC=0.889 vs the reference CPU output.

Verification of the root cause: swapping `attn_implementation="eager"` in the loader raised PCC from 0.889 → 0.970. This is not a fix (it hides the compiler bug), but it confirms SDPA is the cause of the major PCC drop. It is also consistent with the pattern described in the skill rules: "SDPA chunk-size limits" are classified as compiler-stack bugs.

Note: even with `attn_implementation="eager"`, PCC=0.970 does not reach the default threshold of 0.99. This secondary gap (relative to ~0.98 achieved by similar Qwen2/Llama 7B models) may indicate an additional precision issue, e.g. float32 softmax upcast in `eager_attention_forward` not being respected on TT hardware. This is a separate issue that would need further investigation after the primary SDPA bug is fixed.

## Fix
The fix belongs in **tt-mlir** (and/or tt-metal), specifically in the SDPA decomposition/lowering:

- The SDPA lowering should either:
  1. Pad the key/value tensors to meet the `k_chunk_size >= 32` requirement and mask the padded positions (preserving correctness for short sequences), or
  2. Fall back to a non-chunked SDPA path when `seq_len < k_chunk_size`.

The loader should NOT be changed to use `attn_implementation="eager"` — that would paper over the compiler bug and hide it from future testing. Any transformer model with short sequences (< 32 tokens) would silently regress without a visible test failure.

## Verification
- PCC=0.889 with default transformers 5.2.0 `attn_implementation="sdpa"` (SDPA k_chunk_size bug triggered)
- PCC=0.970 with `attn_implementation="eager"` patched into loader (confirms SDPA is root cause; still fails 0.99 threshold)
- No SILICON_PASS achieved; test left failing per "stop and report" rule for compiler-stack bugs
- Hardware: n150

## Files changed
None — no loader changes were made. The `attn_implementation="eager"` hypothesis test was reverted.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348 |
