# Remediation Summary: mradermacher_dr_tulu_no_rler_8b_i1_gguf-causal_lm-pytorch-DR_Tulu_No_RLER_8B_i1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_dr_tulu_no_rler_8b_i1_gguf/causal_lm/pytorch-DR_Tulu_No_RLER_8B_i1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(The initial CI report cited `ImportError: Please install torch and gguf>=0.10.0`, which was due to a different CI environment; the local reproduction revealed the narrower-signature bug as the actual blocker.)

## Root cause
26 GGUF model loaders in tt_forge_models patched `transformers.load_gguf_checkpoint` at module-import time with a narrow signature `(gguf_path, return_tensors=False)`. When pytest collects `test_all_models_torch` it imports all model loaders, so any one of these 26 patches activates before the dr_tulu test runs. transformers 5.2.0 added a new `model_to_load=dummy_model` keyword argument to its internal `load_gguf_checkpoint` call inside `from_pretrained`. The patched function drops `**kwargs`, causing the TypeError when the dr_tulu loader calls `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`.

Additionally, the dr_tulu model directory had no `requirements.txt`, so CI environments without `gguf>=0.10.0` pre-installed would get the ImportError cited in the original report.

## Fix
Two changes in `tt_forge_models` on branch `remediation/mradermacher-dr-tulu-no-rler-8b-i1-gguf`:

1. **26 GGUF loader patched-function signatures** (cherry-picked from `remediation/mathstral_7b_v0_1_i1_gguf-gguf-model-to-load-kwarg`, commit `4c41e61b4f`): added `**kwargs` to `_patched_load_gguf_checkpoint` and passed them through to `_orig_load_gguf_checkpoint` in all 26 loaders that had the narrow signature.

2. **New `requirements.txt`** for `mradermacher_dr_tulu_no_rler_8b_i1_gguf/causal_lm/pytorch/` (commit `844cfc61ce`): `gguf>=0.10.0`

The corresponding tt-xla submodule pointer was updated on branch `remediation/mradermacher-dr-tulu-no-rler-8b-i1-gguf`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    792.50s (0:13:12)
- Tier A attempts: N/A

## Files changed
- `mradermacher_dr_tulu_no_rler_8b_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- 26 loader files: `_patched_load_gguf_checkpoint` signature widened to accept `**kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 574d9c43fd31466555955f27ffa9cb83158316fe |
| tt-forge-models | 844cfc61ce4e5882f424459a86ad8196eff77b12 |
