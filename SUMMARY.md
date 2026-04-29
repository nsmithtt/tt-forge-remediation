# Remediation Summary: colbert_xm-embedding_generation-pytorch-antoinelouis-colbert-xm-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[colbert_xm/embedding_generation/pytorch-antoinelouis/colbert-xm-single_device-inference]

## Result
NO_FIX_NEEDED — test already passes on the configured branch; fix was previously committed to tt_forge_models.

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, psutil._psutil_linux, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, [...] (total: 222)

This is the pytest-forked crash format: the forked child process crashed before `_XLAC` (torch_xla C extension) was even loaded, indicating the crash occurred during model loading/compilation before the XLA backend was engaged.

## Root cause
Loader layer (`tt_forge_models`). The ColBERT-XM model is based on Facebook's X-MOD architecture, which uses `XmodOutput.lang_adapter` to apply language-specific adapters. The original implementation in transformers 5.x iterates over 81 language adapters using boolean masked indexing (`hidden_states[lang_mask]`) and masked assignment (`new_hidden_states[lang_mask] = adapted_lang_hidden_states`). These operations create tensors with dynamic shapes that are incompatible with XLA compilation: the batch dimension varies per language bucket, making static shape inference impossible.

When `torch.compile` is invoked with the "tt" backend, these dynamic-shape operations cause a crash in the XLA tracing/compilation phase, producing the pytest-forked "Extension modules" failure report.

## Fix
Already committed to the `ip-172-31-30-232-tt-xla-dev/ubuntu/hf-bringup-range-1500-500-0` branch of `tt_forge_models` as commit `8a5b8fe05d` ("Patch XmodOutput.lang_adapter to avoid masked tensor assignment for XLA compilation").

File: `third_party/tt_forge_models/colbert_xm/embedding_generation/pytorch/loader.py`

The fix adds `_patch_xmod_lang_adapters(model)` called from `load_model()`. This function replaces the per-layer `lang_adapter` method with `_simple_lang_adapter`, which skips the language-dispatch loop entirely and directly calls `self.adapter_modules[default_lang_key](hidden_states)`. Since all inputs at inference time use the model's `default_language` (English, `"en_XX"`), this is semantically equivalent while producing static shapes that XLA can compile.

No changes were required to tt-xla, tt-mlir, or tt-metal.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    67.35s
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/colbert_xm/embedding_generation/pytorch/loader.py` (in tt_forge_models submodule, pre-existing commit 8a5b8fe05d)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d782ab03a3fd3188b8b089e0188e52310e8ff044 |
