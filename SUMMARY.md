# Remediation Summary: delexa-causal_lm-pytorch-7B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[delexa/causal_lm/pytorch-7B-single_device-inference]

## Result
SILICON_PASS — YaRN RoPE uninit buffers fixed; test passes on n150

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
yarn-rope-uninit-buffers-transformers-5x

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.

(The CI failure "Test exceeded configured timeout and was killed" was the CI runner timing out because the underlying pcc=nan failure caused model re-runs. Locally the test completes in ~111s but fails with pcc=nan before the fix.)

## Root cause
transformers 5.x uses `init_empty_weights` during `from_pretrained`, which initializes the model on a meta device. The `__init__` of `MistralYaRNScaledRotaryEmbedding` calls `self.yarn(device)` and builds `cos_cached`/`sin_cached` buffers, but because the module is initialized on meta during the two-phase loading, these non-persistent buffers (`inv_freq`, `cos_cached`, `sin_cached`) are left with garbage (uninitialized) values — not NaN in float32 but containing values like `-5.5e+11`, `1.2e-18`, etc. When the model is cast to bfloat16, the garbage float32 values produce NaN in bfloat16 (small denormals outside bfloat16's range become NaN). The NaN then propagates through the entire forward pass, yielding pcc=nan in both CPU golden and TT device outputs.

## Fix
In `tt_forge_models/delexa/causal_lm/pytorch/loader.py`, after `AutoModelForCausalLM.from_pretrained()`, iterate over all modules with a `yarn()` method and `inv_freq` buffer. For each such module:
1. Call `module.yarn(device)` to recompute `inv_freq` correctly
2. Recompute `cos_cached` and `sin_cached` from the corrected `inv_freq`

This pattern is specific to models using `MistralYaRNScaledRotaryEmbedding` (custom YaRN RoPE from the Delexa-7b repo).

Branch: `remediation/delexa-causal_lm-pytorch-7B-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    109.30s (0:01:49)
- Tier A attempts: N/A

## Files changed
- tt_forge_models/delexa/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 632e2a007b8102c6f9c23175e3cbf8a175e07395 |
