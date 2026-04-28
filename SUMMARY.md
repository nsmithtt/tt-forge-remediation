# Remediation Summary: mpt_1b_redpajama_200b_dolly-causal_lm-pytorch-mpt-1b-redpajama-200b-dolly-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[mpt_1b_redpajama_200b_dolly/causal_lm/pytorch-mpt-1b-redpajama-200b-dolly-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
transformers-5x-all-tied-weights-keys-mosaic-gpt

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
/home/ttuser/hf-bringup/tt-xla/python_package/tt_torch/torch_overrides.py:34: UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /pytorch/torch/csrc/autograd/python_variable_indexing.cpp:345.)

## Root cause
Three bugs were found and fixed:

1. **Loader bug (transformers 5.x breaking change)**: `MosaicGPT.__init__` never calls `self.post_init()`, but transformers 5.x moved `all_tied_weights_keys` initialization into `post_init()`. When `_finalize_model_loading` later calls `_adjust_tied_keys_with_tied_pointers`, it accesses `self.all_tied_weights_keys` which does not exist, causing `AttributeError`. Fix: patch `MosaicGPT.__init__` via `get_class_from_dynamic_module` to call `PreTrainedModel.post_init(self)` if `all_tied_weights_keys` is not set.

2. **Loader bug (device-stateful ALiBi attention bias)**: The test framework runs a CPU forward pass for the golden reference before compiling for XLA. `MosaicGPT._attn_bias` caches `self.attn_bias` on CPU with `_attn_bias_initialized=True`. When dynamo later traces for XLA, it reuses the cached CPU tensor while `attention_mask` is an XLA fake tensor, causing `TorchRuntimeError: Unhandled FakeTensor Device Propagation for aten.masked_fill.Scalar, found two different devices cpu, xla:0`. Fix: patch `_attn_bias` to check device type and reset the cache if device changes.

3. **tt-xla compiler bug (aten.slice out-of-bounds negative start)**: MPT ALiBi attention does `attn_bias[:, :, -query_len:, :]` on a bias of shape `(1, 16, 1, 2048)`. When `query_len=7`, the slice start `-7` is out of range for dim2 of size 1 (valid range `[-1, 0]`). PyTorch clamps this silently; XLA/TT raises `RuntimeError: Value out of range (expected to be in range of [-1, 0], but got -7)`. Fix: add `clamp_out_of_range_slice_starts` FX pass in tt-xla that clamps out-of-range negative starts to `-dim_size`.

## Fix

**tt_forge_models** — `remediation/mpt_1b_redpajama_200b_dolly-causal_lm-pytorch-single_device-inference`:
- `mpt_1b_redpajama_200b_dolly/causal_lm/pytorch/loader.py`: added `_patch_mosaic_gpt()` static method with two patches (post_init call and device-aware attn_bias reset), called from `load_model()`.

**tt-xla** — `remediation/mpt_1b_redpajama_200b_dolly-aten-slice-oob`:
- `python_package/tt_torch/backend/passes.py`: added `clamp_out_of_range_slice_starts(gm)` FX pass.
- `python_package/tt_torch/backend/backend.py`: imported and called `clamp_out_of_range_slice_starts` in `torch_pass_pipeline`.
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: added `mpt_1b_redpajama_200b_dolly` with `status: EXPECTED_PASSING`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    123.96s (0:02:03)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models/mpt_1b_redpajama_200b_dolly/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 99a4f6143b30146b7bc42c9b3b5028f52b565c00 |
| tt-forge-models | bf1bb695ac0b07b2b5a9a399bc564889c3992558 |
