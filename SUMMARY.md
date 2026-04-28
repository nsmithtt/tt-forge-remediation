# Remediation Summary: afrique_gemma_12b_gguf-causal_lm-pytorch-AfriqueGemma_12B_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[afrique_gemma_12b_gguf/causal_lm/pytorch-AfriqueGemma_12B_Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Value out of range (expected to be in range of [-22, 21], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_51, 2, -1023, 9223372036854775807), kwargs = {})

Original traceback:
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

(The originally-reported TT_FATAL about ethernet cores is a non-fatal startup-time topology message; each instance is immediately followed by a WARNING that skips the core. The real blocking error was the slice OOB.)

## Root cause
Two bugs combined:

1. **Loader layer**: In a full pytest session, 26 GGUF model loaders import and globally monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time. The patchers used the old signature `(gguf_path, return_tensors=False)`, but transformers 5.x now calls `load_gguf_checkpoint(gguf_path, ..., model_to_load=dummy_model)`. The last patcher installed before afrique_gemma runs rejects `model_to_load` with `TypeError: got an unexpected keyword argument`.

2. **tt-xla compiler frontend**: After the loader bug was fixed, the test failed with an out-of-range slice start. Gemma3's sliding window attention cache computes `full_value_states[:, :, -self.sliding_window + 1 :, :]` where `sliding_window=1024` but the test runs with `seq_len=22`, producing `aten.slice.Tensor(tensor, dim=2, start=-1023, end=INT64_MAX)`. PyTorch eager silently clamps out-of-range starts to 0; the XLA/TT backend raises `RuntimeError: Value out of range`. This is a strict bounds validation in the PJRT/XLA path with no equivalent clamp.

## Fix
1. **tt-forge-models** (`fb114a6a4cde3a434a0adaa4803c27cc3036034d`): Changed all 26 broken GGUF patcher functions from `(gguf_path, return_tensors=False)` signature to `(*args, **kwargs)`, passing through to `_orig_load_gguf_checkpoint(*args, **kwargs)`. This makes patchers forward-compatible with the `model_to_load` kwarg added in transformers 5.x.

2. **tt-xla** (`90ceb400c3d8fda7f1951bdb99b24a4516bfb717`): Added `clamp_out_of_range_slice_starts` FX pass to `torch_pass_pipeline` in `python_package/tt_torch/backend/passes.py` and `python_package/tt_torch/backend/backend.py`. The pass walks the exported FX graph, finds `aten.slice.Tensor` nodes with a negative integer start that is less than `-dim_size`, and clamps the start to `-dim_size`. This matches PyTorch eager semantics and allows XLA compilation to proceed.

Files changed in tt-xla:
- `python_package/tt_torch/backend/passes.py` — new `clamp_out_of_range_slice_starts` function
- `python_package/tt_torch/backend/backend.py` — import and call the new pass

Files changed in tt-forge-models:
- 26 GGUF loader files under various model directories — `_patched_load_gguf_checkpoint` signature fix

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    743.17s (0:12:23)
- Tier A attempts: 1

## Files changed
- tt-xla: `python_package/tt_torch/backend/passes.py`, `python_package/tt_torch/backend/backend.py`
- tt-forge-models: 26 GGUF loader files

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 90ceb400c3d8fda7f1951bdb99b24a4516bfb717 |
| tt-forge-models | fb114a6a4cde3a434a0adaa4803c27cc3036034d |
