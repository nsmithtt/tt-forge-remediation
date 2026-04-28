# Remediation Summary: pythia-reward_model-pytorch-1B-deduped-tldr-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[pythia/reward_model/pytorch-1B-deduped-tldr-single_device-inference]

## Result
FAIL — GPT-NeoX 1B (head_dim=256, 16 layers) overflows to ~1.07e+38 on TT device in bfloat16 (CPU: 4.8125); reward logit is single-element so PCC=0.0 by design when allclose fails; Tier B compiler bug

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
gpt-neox-bfloat16-overflow-head-dim-256

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.0. Required: pcc=0.95.

## Root cause
The model is `cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr`, a GPT-NeoX architecture with 16 decoder layers, `hidden_size=2048`, `num_attention_heads=8`, `head_dim=256`, and `partial_rotary_factor=0.25`. It is a `GPTNeoXForSequenceClassification` variant that produces a single scalar reward logit (shape `(1, 1)`). Since the output is a single-element tensor, `TorchComparisonEvaluator.compute_pcc()` returns 0.0 whenever `allclose` fails (by design — lines 130–132 of `torch_comparison_evaluator.py`).

The allclose fails because TT device produces `~1.07e+38` (approaching bfloat16 max of `3.39e+38`) while CPU produces `4.8125`. This is catastrophic overflow, not ordinary PCC degradation. Systematic investigation with the 1B model and pretrained weights confirmed overflow appears between 8 and 16 decoder layers. Smaller GPT-NeoX models (70m with head_dim=64, 6 layers; 160m with head_dim=64, 12 layers) show ~5–6× output amplification in bfloat16 but no overflow. The larger head dimension (256 vs 64) and deeper stacking (16 layers) combined with pretrained weight magnitude appear to push intermediate activations past the bfloat16 range during the SDPA or post-attention MLP computations.

Additionally, there was a loader-level issue: `load_inputs()` was calling `tokenizer(..., padding="max_length", max_length=128)`, which pads the natural ~57-token input to 128 tokens. The padding silently changes the pooling position (last non-padding token) used by `GPTNeoXForSequenceClassification.forward`, which was introducing a second source of mismatch. This was fixed in tt_forge_models commit `ff15f9eac5`. However, even after removing the padding, the device still overflows, confirming the root cause is the compiler stack.

This failure is related to `ttmlir-f32-precision-not-preserved` (as seen in the 6.9B causal LM sibling report) but is more severe: the 6.9B model (head_dim=128, 32 layers) shows PCC=0.9439 degradation while this 1B model (head_dim=256, 16 layers) overflows entirely. The head_dim=256 dimension likely stresses bfloat16 accumulation in the SDPA computation more aggressively.

## Fix
**Loader fix applied:** Removed `padding="max_length"` and `max_length=128` from `load_inputs()` in `tt-xla/third_party/tt_forge_models/pythia/reward_model/pytorch/loader.py`. Committed to tt_forge_models as `ff15f9eac5` on branch `remediation/pythia-reward_model-pytorch-1B-deduped-tldr-single_device-inference` (pushed to `nsmith` remote).

**Compiler fix not attempted:** The overflow in bfloat16 across 16 GPT-NeoX layers with head_dim=256 is a cross-cutting precision failure in tt-mlir's lowering pipeline. The underlying fix would require either: (a) using float32 accumulators in bfloat16 matmul lowerings throughout SDPA and MLP computations, or (b) inserting dynamic range normalization. Both are cross-cutting changes affecting every matmul/SDPA lowering — well beyond a single-function Tier A fix.

## Tier B justification
cross-cutting — fixing bfloat16 precision overflow requires changes to matmul accumulation in all SDPA and MLP lowerings across tt-mlir and tt-metal kernels; the root cause mechanism (which computation step first overflows) is not isolated without further profiling, making this an internal-error-unknown-mechanism as well.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~120s (overflow observed; test run aborted after PCC=0.0 confirmed)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/pythia/reward_model/pytorch/loader.py` — removed `padding="max_length"` and `max_length=128` from `load_inputs()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ff15f9eac5bfced929b4cc7d40c78e9057281f23 |
