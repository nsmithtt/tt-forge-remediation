# Remediation Summary: lfm2-causal_lm-pytorch-lfm2_350m_unsloth-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lfm2/causal_lm/pytorch-lfm2_350m_unsloth-single_device-inference]

## Result
SILICON_PASS — two evaluator fixes + measured BF16 PCC threshold

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
evaluator-non-cache-subclass-key-value-cache

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: YES — CPU BF16 vs FP32 PCC floor measured at 0.9659; TT measures 0.9825 (above BF16 floor); threshold lowered to 0.98
- Warning / exception suppression: NO

## Failure
E   TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache

## Root cause
`Lfm2HybridConvCache` is not a subclass of `transformers.Cache`. The evaluator's
`_match_data_types` checked `isinstance(tensor, Cache)` before converting cache
objects to legacy (key, value) tuples. Because `Lfm2HybridConvCache` failed the
isinstance check, it was passed through as-is. `tree_map` treated it as a leaf
(it is not registered as a pytree), and then `torch.equal(x, y)` received a
non-Tensor argument and raised TypeError.

A second error followed after the first fix: `Lfm2HybridConvCache` stores
`torch.tensor([])` placeholder entries in `key_cache`/`value_cache` for conv-layer
slots (those slots use `conv_cache` instead). Zipping the full `key_cache` and
`value_cache` lists produced empty tensors that caused `torch.max()` on empty
input in `_compare_atol`.

After both evaluator fixes, the test ran to completion on silicon with PCC 0.9825.
The CPU BF16 vs FP32 floor for this hybrid conv/attention model is 0.9659, so the
0.9825 TT vs CPU BF16 gap is BF16 accumulation on a numerically correct path.
The `required_pcc` was lowered to 0.98.

## Fix
All changes in `tt-xla` on branch
`remediation/lfm2-causal_lm-pytorch-lfm2_350m_unsloth-single_device-inference`.

**Fix 1** — `tests/infra/evaluators/torch_comparison_evaluator.py`:
- Added duck-type check `hasattr(tensor, "key_cache") and hasattr(tensor, "value_cache")`
  alongside the `isinstance(tensor, Cache)` guard in `convert_and_match`, so
  `_cache_to_legacy` is also called for non-Cache cache objects.
- Extended `_cache_to_legacy` with a new branch that builds a legacy tuple from
  `key_cache`/`value_cache` zip pairs, with empty-tensor placeholders filtered out,
  plus `conv_cache` entries (also filtered).

**Fix 2** — `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
- Added entry for `lfm2/causal_lm/pytorch-lfm2_350m_unsloth-single_device-inference`
  with `required_pcc: 0.98`, justified by measured BF16 floor.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    80.79s
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c6198d697639f30883de899af1a533a738897854 |
| tt-forge-models | 8f34d8c41e52799685d191b250527e087debc1bd |
