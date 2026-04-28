# Remediation Summary: codegemma_7b_it_gguf_smashed-causal_lm-pytorch-CodeGemma_7B_IT_Q4_K_M_GGUF-single_device-inference

## Skill version
5

## Test
`tests/runner/test_models.py::test_all_models_torch[codegemma_7b_it_gguf_smashed/causal_lm/pytorch-CodeGemma_7B_IT_Q4_K_M_GGUF-single_device-inference]`

## Result
FAIL — PCC=0.914975212965678 on TT silicon (required 0.99); root cause is compiler math-fidelity precision loss on Gemma 7B architecture, same as the known issue tracked in gemma/pytorch-1.1_7B_IT (assert_pcc:false, issue #2861)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-gemma-math-fidelity-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.914975212965678. Required: pcc=0.95.
```

## Root cause
Two separate issues were found:

**Issue 1 (loader bug, fixed):** transformers 5.2.0 removed `gemma` (v1) from
`GGUF_CONFIG_MAPPING` and `GGUF_TO_FAST_CONVERTERS`. CodeGemma GGUF files use
`architecture="gemma"`, so the upgrade caused a hard
`ValueError: GGUF model with architecture gemma is not supported yet.`
A separate commit on the same remediation branch also fixed 26 loaders whose
`_patched_load_gguf_checkpoint` still called `load_gguf_checkpoint` with a
fixed `(gguf_path, return_tensors)` signature, dropping the new `model_to_load`
kwarg added in transformers 5.2.0.

**Issue 2 (compiler precision, not fixed):** After the loader fixes, the model
runs to completion but PCC between TT silicon output and CPU reference is
consistently 0.9149752129656837—independent of whether the model is loaded in
FP32 or BF16. The same architecture (Gemma 7B, 28 layers, head_dim=256) is
already tracked in the test config as
`gemma/pytorch-1.1_7B_IT-single_device-inference` with `assert_pcc: false`
referencing tt-xla issue #2861 (ComputeConfig math_fidelity/fp32_dest_acc_en).
The default TT compiler math fidelity produces systematic error across all 28
attention layers, yielding a PCC floor of ~0.915 that cannot be closed by
loader-level changes.

## Fix
**Loader fixes (committed to remediation branch in tt_forge_models):**

1. `3c9fd0c062` — `codegemma_7b_it_gguf_smashed/causal_lm/pytorch/loader.py`:
   Patch `GGUF_CONFIG_MAPPING` and `GGUF_TO_FAST_CONVERTERS` at module import
   time to restore `gemma` (v1) architecture support dropped in transformers 5.2.0.

2. `dbfcf0a2f6` — 26 GGUF loaders: update `_patched_load_gguf_checkpoint`
   signatures from `(gguf_path, return_tensors)` to `(*args, **kwargs)` to
   accept the new `model_to_load` kwarg added in transformers 5.2.0.

**Compiler precision fix (NOT attempted):** Would require enabling
`fp32_dest_acc_en=True` and/or raising math fidelity for all operations in the
Gemma 7B computation graph; this is a cross-cutting change touching every
lowering pass that emits compute kernel configs.

## Tier B justification
`cross-cutting`: Achieving PCC ≥ 0.95 for Gemma 7B requires increased math
fidelity or FP32 destination accumulation across all 28 transformer layers and
their contained operations (matmul, softmax, RMSNorm, GELU). This is not a
single-function or single-file change—it is a compiler-wide precision policy
decision already tracked under tt-xla issue #2861 for the standard (non-GGUF)
Gemma 7B variant.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    452.17s (0:07:32) — model runs, PCC check fails
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/codegemma_7b_it_gguf_smashed/causal_lm/pytorch/loader.py`
  (gemma v1 GGUF arch registration)
- 26 other GGUF loader files (model_to_load kwarg fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | dbfcf0a2f624f77e5d411a08f5c02c2059334826 |
