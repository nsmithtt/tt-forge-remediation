# Remediation Summary: gemma3_270m_it_qat_gguf-causal_lm-pytorch-Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_270m_it_qat_gguf/causal_lm/pytorch-Q4_0-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: narrow-sig `load_gguf_checkpoint` clobbering and aten.slice negative start OOB in sliding-window KV cache

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
Original CI error:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced locally as (gguf was installed; same root):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, revealed second bug:
```
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)
While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_21, 2, -511, 9223372036854775807), kwargs = {})
Original traceback: cache_utils.py:214: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

## Root cause

**Bug 1 (loader):** Three loaders alphabetically preceding `gemma3_270m_it_qat_gguf` —
`bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, and
`dmind_3_mini_i1_gguf` — each patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
at module import time with a narrow signature `(gguf_path, return_tensors=False)`. In
transformers 5.2.0 `PreTrainedModel.from_pretrained` calls
`load_gguf_checkpoint(..., model_to_load=dummy_model)`. Because the narrow-sig function
is installed by the time the gemma3 test runs, the call raises `TypeError`. Additionally,
`gemma3_270m_it_qat_gguf` has no `requirements.txt`, so the original CI error was the
`ImportError` raised when `gguf` is absent.

**Bug 2 (tt-xla):** Gemma 3 uses a sliding window of 512 tokens for local-attention layers.
During the KV-cache update it computes `full_value_states[:, :, -sliding_window+1:, :]`
= `full_value_states[:, :, -511:, :]`. When `seq_len < sliding_window` (here seq_len ≈ 23),
the slice start -511 < -23 = -dim_size. PyTorch eager silently clamps this to -dim_size;
the XLA/TT lazy backend instead raises "Value out of range".

## Fix

**Bug 1 — loader fix in tt_forge_models:**
- Cherry-picked commit `073cb3abb8` from `origin/remediation/darkc0de-xortron-gguf-model_to_load-kwarg`,
  which broadens the signature of `_patched_load_gguf_checkpoint` in 26 loaders from
  `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` and updates the inner call
  to `_orig_load_gguf_checkpoint(*args, **kwargs)`. All 26 loaders share the identical two-line pattern.
- Added `gemma3_270m_it_qat_gguf/requirements.txt` declaring `gguf>=0.10.0`.

Branches: `tenstorrent/tt-forge-models@remediation/gemma3_270m_it_qat_gguf-causal_lm-pytorch-Q4_0-single_device-inference`

**Bug 2 — compiler-frontend fix in tt-xla:**
Added `clamp_out_of_range_slice_starts(gm)` FX pass in
`python_package/tt_torch/backend/passes.py`. The pass iterates all `aten.slice.Tensor`
nodes; for any with a static negative `start < -dim_size` it clamps `start` to `-dim_size`,
matching PyTorch eager semantics. Integrated into `torch_pass_pipeline` in
`python_package/tt_torch/backend/backend.py` immediately after `bypass_assert_tensor_metadata`.

Branch: `tenstorrent/tt-xla@remediation/gemma3_270m_it_qat_gguf-causal_lm-pytorch-Q4_0-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    308.80s (0:05:08)
- Tier A attempts: 1

## Files changed
**tt_forge_models (26 loader files + 1 requirements.txt):**
- `gemma3_270m_it_qat_gguf/requirements.txt` (new)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- 23 additional loader files with identical pattern (full list in commit `73df3cddfe`)

**tt-xla:**
- `python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts`
- `python_package/tt_torch/backend/backend.py` — import and call `clamp_out_of_range_slice_starts`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 257cd9c8a06826d5d889166aee38f3cf85a48e1a |
| tt-forge-models | f5a45b59997b7ecc80d0cb0c5e319048cc826f92 |
