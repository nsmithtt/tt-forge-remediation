# Remediation Summary: gemma3_16b_glm_heretic_gguf-causal_lm-pytorch-16B_GLM_Heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_16b_glm_heretic_gguf/causal_lm/pytorch-16B_GLM_Heretic_GGUF-single_device-inference]

## Result
XFAIL — Gemma3 16B at BF16 (~32 GB) exceeds n150 12 GB DRAM; TT runtime returns INTERNAL error code 13 at execution time

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-dram-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
(original reported failure; secondary failure after loader fix below)

After loader fix, the test fails with:
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause

**Two issues found and resolved before reaching the hardware capacity ceiling:**

1. **Loader bug (cross-loader gguf patch clobbering)**: 26 GGUF loaders for qwen3.5/gpt-oss models define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` at module level, replacing `transformers.load_gguf_checkpoint` with a narrow-signature function. When pytest collects all loaders in a single process, one of these narrow-sig patches is active when the gemma3_16b test runs. transformers 5.2.0 calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` which the narrow signature rejects with TypeError.

2. **Compiler frontend bug (XLA lazy slice OOB)**: After the loader fix, the test fails with `Value out of range (expected to be in range of [-23, 22], but got -1023)` in `TorchFunctionOverride.__torch_function__`. This is the known `aten-slice-tensor-out-of-bounds-start` bug: `cache_utils.py:214` calls `full_value_states[:, :, -sliding_window+1:, :]` with `sliding_window=1024`, producing `start=-1023` on a sequence-length-23 dimension. The XLA lazy backend rejects out-of-range negative indices instead of clamping them like PyTorch eager.

3. **Hardware capacity ceiling**: After both fixes, the model compiles (taking ~14 minutes for the 46-layer Gemma3 16B graph) and then fails at `_run_cached_graph` with `INTERNAL: Error code: 13`. Gemma3 16B has ~16B parameters; loaded as BF16 via `dtype_override=torch.bfloat16`, that is 16B × 2 bytes ≈ 32 GB, which exceeds n150's 12 GB DRAM. The INTERNAL error 13 is the TT runtime's OOM error surfaced as a StatusOr failure. This is hardware class.

## Fix

**Fix 1 — loader (tt_forge_models):** Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated the `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` call to `_orig_load_gguf_checkpoint(*args, **kwargs)` in all 26 affected loaders. Branch: `remediation/gemma3_16b_glm_heretic_gguf-gguf-model-to-load-kwarg` in tt-forge-models.

**Fix 2 — compiler frontend (tt-xla):** Added `aten.slice.Tensor` handling to `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py`. For slices with statically-known dimension sizes, clamps `start` and `end` to `[-size, size]` before dispatching to XLA, matching PyTorch eager clamping semantics.

**Fix 3 — test config (tt-xla):** Added `KNOWN_FAILURE_XFAIL` entry for `gemma3_16b_glm_heretic_gguf/causal_lm/pytorch-16B_GLM_Heretic_GGUF-single_device-inference` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. Branch: `remediation/gemma3_16b_glm_heretic_gguf-aten-slice-oob` in tt-xla.

## Verification
- pytest exit: xfailed (1 xfailed, 5 warnings)
- Hardware:    n150
- Duration:    867.33s (0:14:27)
- Tier A attempts: N/A

## Files changed
- `python_package/tt_torch/torch_overrides.py` (tt-xla)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)
- 26 `*/causal_lm/pytorch/loader.py` files (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 97e666d746aca15a8110e49e9feed09ab5c22444 |
| tt-forge-models | 48f72a7f3f78db2626ab4046a88d2cb237a899bc |
