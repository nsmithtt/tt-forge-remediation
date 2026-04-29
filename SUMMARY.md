# Remediation Summary: gemma3_12b_qat_gguf-causal_lm-pytorch-12B_IT_QAT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_12b_qat_gguf/causal_lm/pytorch-12B_IT_QAT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

(On local reproduction, the first error surfaced was:
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
This is the co-collected GGUF loader kwargs bug that gates the slice error.)

## Root cause
Two bugs gate this test:

**Bug 1 — loader layer**: Other GGUF loaders collected in the same pytest session
(gpt_oss_swallow, qwen3_5, etc.) monkey-patch `transformers.gguf_utils.load_gguf_checkpoint`
at import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers
5.x added a `model_to_load=` keyword argument that it passes at `modeling_utils.py:4016`.
When `gemma3_12b_qat_gguf`'s `from_pretrained` internally calls `load_gguf_checkpoint`,
it hits the narrow-signature wrapper → `TypeError`. Bug fingerprint:
`gguf-load-checkpoint-model-to-load-kwarg`.

**Bug 2 — tt-xla**: Gemma 3 uses `SlidingWindowCache.update()` which does
`full_value_states[:, :, -sliding_window+1:, :]`. With `sliding_window=1024` and
`seq_len=24` (dim_size=23), the start index is `-1023`, which is below `-dim_size=-23`.
PyTorch CPU silently clamps this to 0, but the XLA/TT backend validates the range strictly
and raises `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)`.
Bug fingerprint: `aten-slice-tensor-out-of-bounds-start`.

## Fix
**Bug 1** — `tt_forge_models` at commit `d845f4ae08`:
Updated 26 GGUF loader wrappers to use `(*args, **kwargs)` signatures so extra keyword
arguments flow through to the real `load_gguf_checkpoint`.
Files: 26 `*/causal_lm/pytorch/loader.py` files in tt_forge_models.

**Bug 2** — `tt-xla` at commit `8cb9ab61f`:
Added `clamp_out_of_range_slice_starts(gm)` FX pass in
`python_package/tt_torch/backend/passes.py`. The pass iterates `aten.slice.Tensor`
nodes, reads dim_size from `node.args[0].meta["val"].shape`, and clamps `start` to
`max(-dim_size, start)` for any out-of-range negative start.
Called from `torch_pass_pipeline` in `python_package/tt_torch/backend/backend.py`
after `bypass_assert_tensor_metadata`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    626.46s (0:10:26)
- Tier A attempts: 1

## Files changed
tt_forge_models (26 files):
- `*/causal_lm/pytorch/loader.py` — `_patched_load_gguf_checkpoint` signature → `(*args, **kwargs)`

tt-xla (2 files):
- `python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts`
- `python_package/tt_torch/backend/backend.py` — import and call `clamp_out_of_range_slice_starts`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 018707d53b96b9740516ab7e0583da777786cafa |
| tt-forge-models | d845f4ae08cfa47c5b9abc4bc563fd71033f46f4 |
