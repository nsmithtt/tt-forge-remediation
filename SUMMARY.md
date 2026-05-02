# Remediation Summary: mradermacher_es_qwen_math_base_7b_3k_stage2_6k_t4_ds_o2_aug_kl0_01_step480_i1_gguf-causal_lm-pytorch-es_qwen_math_base_7B_stage2_step480_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_es_qwen_math_base_7b_3k_stage2_6k_t4_ds_o2_aug_kl0_01_step480_i1_gguf/causal_lm/pytorch-es_qwen_math_base_7B_stage2_step480_i1_GGUF-single_device-inference]

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

(The originally reported `AttributeError: 'NoneType' object has no attribute 'config'` is a secondary symptom; the primary error surfaces at modeling_utils.py:4016 when transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` against a narrow-signature patched function.)

## Root cause
Session contamination: 26 GGUF loaders (qwen35 and gpt-oss variants) each installed a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrapper with a narrow signature that did not accept the `model_to_load` keyword argument added in transformers 5.2.0. During pytest collection these loaders are imported before the `mradermacher_es_qwen_math_base_7b` test runs, leaving the narrow-sig contaminated function installed into `transformers.modeling_utils`. The `mradermacher_es_qwen_math_base_7b` loader itself (qwen2 architecture) does not patch anything and relies on the uncontaminated transformers function — which it no longer gets.

## Fix
All 26 narrow-signature `_patched_load_gguf_checkpoint` wrappers were widened to accept `(*args, **kwargs)` in `tt-xla/third_party/tt_forge_models`. This is the same fix as documented in the GGUF model_to_load TypeError memory entry.

Commit in `tt_forge_models`: `fa39e95c38f4059844b7188225251f0b522df91f` (branch `remediation/mradermacher_dao1_30b_a3b_i1_gguf-causal_lm-pytorch-30B_A3B_i1_GGUF-single_device-inference`)

Commit in `tt-xla` bumping the submodule pointer: `e6869eff9193556a0ff35b3f9d5c7ee85528f777`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    385s
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models: widened narrow-sig patches in 26 GGUF loader files (submodule pointer bump in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e6869eff9193556a0ff35b3f9d5c7ee85528f777 |
| tt-forge-models | fa39e95c38f4059844b7188225251f0b522df91f |
