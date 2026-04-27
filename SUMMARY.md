# Remediation Summary: baseline_outcome_reward_qwen_7b_i1_gguf-causal_lm-pytorch-7B_i1_GGUF-single_device-inference

## Skill version
10

## Test
tests/runner/test_models.py::test_all_models_torch[baseline_outcome_reward_qwen_7b_i1_gguf/causal_lm/pytorch-7B_i1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9392273202554543. Required: pcc=0.95.

## Root cause
Two loader-layer issues:

**Issue 1 (TypeError blocking model load):** Twenty-six model loaders in
`tt_forge_models` that support qwen35/gpt-oss architectures monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module
import time. All patches defined `_patched_load_gguf_checkpoint(gguf_path,
return_tensors=False)` without the `model_to_load` kwarg that a newer
version of `transformers` now passes (`model_to_load=dummy_model`). Because
all loaders are imported during pytest test discovery (before the target test
runs), the last-installed patch broke `AutoModelForCausalLM.from_pretrained`
for this model with `TypeError: _patched_load_gguf_checkpoint() got an
unexpected keyword argument 'model_to_load'`.

**Issue 2 (PCC below threshold):** The test compares TT silicon output (FP32,
the precision returned by the TT backend via torch-xla) against the CPU
reference (BF16, loaded with `dtype_override=torch.bfloat16`). Both outputs
are cast to float64 before PCC computation. For this Q4_K_M-quantised model,
that BF16/FP32 mismatch introduces a systematic ~6 % correlation loss that
is purely a precision-floor artefact, not a computation error. Measured
directly: BF16 CPU vs FP32 CPU PCC = 0.9378, matching the TT vs CPU PCC of
0.9392 (TT is slightly *more* accurate than BF16 CPU). The default required
PCC of 0.99 (and even the original threshold of 0.95) was set tighter than
this model's inherent BF16 floor.

## Fix
**Fix 1 — tt_forge_models** (`remediation/baseline_outcome_reward_qwen_7b_i1_gguf-causal_lm-pytorch-7B_i1_GGUF-single_device-inference`):

Added `model_to_load=None` to all 26 `_patched_load_gguf_checkpoint`
signatures and passed it through to `_orig_load_gguf_checkpoint`. This is
not a forbidden workaround: no model is trimmed, no CPU offload introduced,
no op suppressed. It is a straightforward API-compatibility fix so the
patches continue to intercept the correct function.

**Fix 2 — tt-xla test config** (`remediation/baseline_outcome_reward_qwen_7b_i1_gguf-causal_lm-pytorch-7B_i1_GGUF-single_device-inference`):

Added entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
```yaml
baseline_outcome_reward_qwen_7b_i1_gguf/causal_lm/pytorch-7B_i1_GGUF-single_device-inference:
  status: EXPECTED_PASSING
  required_pcc: 0.93  # BF16/FP32 precision floor: BF16 CPU vs FP32 CPU measured at 0.9378
```
The skill explicitly permits lowering required_pcc when "you have measured
that the gap is purely bfloat16 accumulation on a numerically correct path."
The measurement (BF16 CPU PCC = 0.9378 ≈ TT PCC = 0.9392) confirms this.

## Verification
- pytest exit status: PASSED
- Wall-clock duration: 403 s (6 min 43 s)
- Hardware: n150 (Wormhole B0)

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added required_pcc=0.93 entry
- `tt-xla/third_party/tt_forge_models/` — 26 loaders with `_patched_load_gguf_checkpoint` signature fix

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ccd674e723de06a10a318e21a92091cc540f0ac4 |
| tt-forge-models | a75f47656d4f6ddabe5a990ad26c17900fa29e84 |
