# Remediation Summary: gpt_jt_6b_v0-causal_lm-pytorch-gpt-jt-6b-v0-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_jt_6b_v0/causal_lm/pytorch-gpt-jt-6b-v0-single_device-inference]

## Result
FAIL — GPT-J's explicit float32 attention casts (query/key cast to float32 before matmul) are lowered to bfloat16 by TT compiler; accumulated error across 28 transformer layers yields logits PCC=0.9323 < required 0.95

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.5535388644027031. Required: pcc=0.95.

## Root cause
Two loader bugs were masking the true compiler-stack bug and artificially suppressing PCC below the actual hardware precision floor:

**Loader bug 1 (padding=max_length):** `load_inputs` called the tokenizer with `padding="max_length"`, padding the 6-token sample text to 128 tokens using EOS tokens. 122/128 output positions are padding, and TT and CPU produce different values at those padded positions, pulling measured PCC down to ~0.55. Fix: remove `padding="max_length"`.

**Loader bug 2 (DynamicCache in PCC computation):** With `use_cache` not set (defaulting to True), the model returns a `CausalLMOutputWithPast` including a `DynamicCache` object (`past_key_values`). The evaluator's `tree_flatten` traverses this object and includes 28×2=56 key/value tensors (each shape [1, 16, 6, 256]) in the PCC minimum computation. These tensors have float32 precision on CPU but bfloat16 on TT, giving very low PCC that dominates the minimum. Fix: add `inputs["use_cache"] = False`.

**Remaining compiler-stack bug:** After both loader fixes, logits PCC=0.9323, still below the 0.95 threshold. GPT-J's attention implementation (`GPTJAttention._attn`) explicitly casts query and key tensors to float32 before computing the attention matmul and softmax, then casts results back to value dtype (float16). On CPU this gives float32 precision attention. On TT hardware, these float32 casts are not preserved by the compiler — all operations are lowered to bfloat16. Over 28 transformer layers, the accumulated error is sufficient to push logits PCC below 0.95. Confirmed: CPU float16 vs CPU bfloat16 gives PCC=0.9998, ruling out dtype mismatch as the cause; the gap is specifically from float32 attention on CPU vs bfloat16 attention on TT.

## Fix
**Loader fixes** (committed to tt_forge_models remediation branch, then submodule pointer updated in tt-xla):

- `tt-xla/third_party/tt_forge_models/gpt_jt_6b_v0/causal_lm/pytorch/loader.py`:
  - Removed `padding="max_length"` from tokenizer call in `load_inputs`
  - Added `inputs["use_cache"] = False` to exclude DynamicCache tensors from PCC comparison

No compiler-stack fix attempted (Tier B).

## Tier B justification
cross-cutting — preserving float32 precision through every lowering pass in tt-mlir would require modifying the dtype propagation and precision semantics across all StableHLO→TTIR lowering patterns; this cannot be scoped to a single file or function.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    ~148s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/gpt_jt_6b_v0/causal_lm/pytorch/loader.py` (two loader fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ce02dd07fd8169dc43f59afd58387f64f2e5245c |
| tt-forge-models | e68be7307ebfea964cd3714adae5c05f99aaec86 |
