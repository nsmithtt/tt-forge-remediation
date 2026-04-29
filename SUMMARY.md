# Remediation Summary: gemma3_heretic_gguf-causal_lm-pytorch-4B_IT_HERETIC_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_heretic_gguf/causal_lm/pytorch-4B_IT_HERETIC_GGUF-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807), kwargs = {})
Original traceback:
  File ".../transformers/models/gemma3/modeling_gemma3.py", line 371, in forward
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two bugs stacked:

1. **Loader layer (tt_forge_models):** 26 GGUF model loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — missing `**kwargs`. Transformers 5.2.0 added `model_to_load=None` to this signature, so when another GGUF loader's patch intercepts the call, `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` is raised. This masked the underlying compiler bug when running a single test in isolation.

2. **tt-xla compiler frontend:** Gemma3 uses `SlidingWindowCache` with `sliding_window=1024`. During inference the cache update does `full_value_states[:, :, -sliding_window+1:, :]` which produces `aten.slice.Tensor` with `start=-1023`. When `seq_len=24`, the valid XLA index range is `[-24, 23]` but the start of `-1023` falls outside this — XLA's kernel validates bounds strictly, while PyTorch CPU silently clamps. This fires in `partition_fx_graph_for_cpu_fallback` before tt-mlir compilation.

## Fix
1. **tt_forge_models** (`remediation/gemma3_heretic_gguf-causal_lm-pytorch-4B_IT_HERETIC_GGUF-single_device-inference`): Added `**kwargs` to the function signature and forwarded it in the call to `_orig_load_gguf_checkpoint` in all 26 affected loader files. Files: 26 `*/causal_lm/pytorch/loader.py` files containing Qwen3.5 and GPT-OSS Swallow GGUF loaders.

2. **tt-xla** (`remediation/gemma3_heretic_gguf-causal_lm-pytorch-4B_IT_HERETIC_GGUF-single_device-inference`):
   - `python_package/tt_torch/backend/passes.py`: Added `clamp_out_of_range_slice_starts(gm)` FX pass that iterates `aten.slice.Tensor` nodes, reads the dimension size from `node.args[0].meta["val"].shape`, and clamps the start index to `max(-dim_size, start)` when it is out of range.
   - `python_package/tt_torch/backend/backend.py`: Imported and called `clamp_out_of_range_slice_starts(compiled_graph)` after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    459.73s (0:07:39)
- Tier A attempts: 1

## Files changed
**tt_forge_models:**
- 26 × `*/causal_lm/pytorch/loader.py` — add `**kwargs` to `_patched_load_gguf_checkpoint` signature and forward in call

**tt-xla:**
- `python_package/tt_torch/backend/passes.py` — add `clamp_out_of_range_slice_starts` FX pass
- `python_package/tt_torch/backend/backend.py` — import and invoke `clamp_out_of_range_slice_starts`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 46c94a60fa2c0af1d24b1626b97a26ba349d0158 |
| tt-forge-models | 400987b194452e8146fec1daace96b1b71aa9eef |
