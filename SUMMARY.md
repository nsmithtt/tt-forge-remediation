# Remediation Summary: flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf/causal_lm/pytorch-v0.3_1.2B_MLX_Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-lfm2-mlx-conv-weight-shape-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache

(actual first failures during reproduction: KeyError: 'lfm2', then TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load', then RuntimeError: size mismatch in conv weights)

## Root cause
Four independent loader bugs, all in tt_forge_models:

1. **Missing lfm2 tokenizer converter** — `transformers.integrations.ggml.GGUF_TO_FAST_CONVERTERS`
   had no entry for the `'lfm2'` architecture, raising `KeyError: 'lfm2'` when
   `AutoTokenizer.from_pretrained` tried to build a fast tokenizer from the GGUF.
   LFM2 uses a GPT2-style BPE tokenizer (`tokenizer.ggml.model = "gpt2"`), so
   `GGUFGPTConverter` is the correct mapping.

2. **Stale `_patched_load_gguf_checkpoint` signature** — 26 loaders that globally
   monkey-patched `load_gguf_checkpoint` used the old two-argument signature
   `(gguf_path, return_tensors=False)`.  `transformers` 5.2.0 added a
   `model_to_load` keyword argument; whichever of those loaders was imported first
   replaced the global function with a version that rejected the new kwarg.

3. **MLX GGUF conv weight shape mismatch** — `Lfm2TensorProcessor.process()`
   assumes `shortconv.conv.weight` arrives as 2-D `[hidden_dim, L_cache]` and
   inserts a dimension at axis 1.  The MLX-exported GGUF stores the weight as 3-D
   `[hidden_dim, L_cache, 1]` (a trailing size-1 dim), so after the expand the
   tensor is `[hidden_dim, 1, L_cache, 1]` (4-D) instead of the required
   `[hidden_dim, 1, L_cache]` (3-D), causing a `RuntimeError` from
   `log_state_dict_report`.

4. **`Lfm2HybridConvCache` in model output** — with `use_cache` defaulting to
   `True`, the model returns an `Lfm2HybridConvCache` object as `past_key_values`.
   The XLA comparison evaluator calls `torch.equal()` on every output leaf and
   raises `TypeError` because `Lfm2HybridConvCache` is not a `Tensor`.

## Fix
All fixes in `flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf/causal_lm/pytorch/loader.py`
in `tt-forge-models`, on branch
`remediation/flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference`:

- **Fix 1**: `GGUF_TO_FAST_CONVERTERS.setdefault("lfm2", GGUFGPTConverter)` at
  module import time.
- **Fix 2**: Patch `Lfm2TensorProcessor.process` to squeeze the trailing size-1
  dim from 3-D conv weights before the `expand_dims(axis=1)`.
- **Fix 3**: After `from_pretrained`, call `model.resize_token_embeddings(len(tokenizer))`
  when the tokenizer vocabulary exceeds the GGUF-declared `vocab_size`.
- **Fix 4**: Set `inputs["use_cache"] = False` in `load_inputs` so the model
  returns only tensor outputs.

Additionally, 26 other loaders that globally patched `load_gguf_checkpoint` with
the stale two-argument signature were updated to use `(*args, **kwargs)`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    159.17s (0:02:39)
- Tier A attempts: N/A

## Files changed
- `flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf/causal_lm/pytorch/loader.py`
- 26 `*/causal_lm/pytorch/loader.py` files with stale `_patched_load_gguf_checkpoint` signature

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e209f7797520e7b445c25616cd6820d43d1b7c47 |
| tt-forge-models | e8fdd3d10f0b8213103dfe8b5c0a3088d272c813 |
