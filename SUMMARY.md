# Remediation Summary: internlm2_chat_1_8b-causal_lm-pytorch-chat_1_8b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[internlm2_chat_1_8b/causal_lm/pytorch-chat_1_8b-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-rope-inv-freq-uninit

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.

## Root cause
Three loader-layer bugs, all caused by transformers 5.x breaking changes:

1. **Special token ID mismatch** (fixed in prior commit): `InternLM2TokenizerFast`
   registers `SLOW_TO_FAST_CONVERTERS` under the slow tokenizer key
   (`"InternLM2Tokenizer"`) rather than the fast tokenizer key
   (`"InternLM2TokenizerFast"`), so the `InternLM2Converter` never runs when
   loading the fast tokenizer.  Transformers 5.x then adds the six special
   tokens (`<|im_start|>` etc.) as new IDs starting at vocab_size (92544+)
   instead of reusing the trained positions (92538-92543).  Input IDs above
   92543 trigger an out-of-range embedding lookup.  Fix: resize the embedding
   table to 92550 and copy trained rows 92538-92543 into the new rows
   92544-92549.

2. **`DynamicCache.from_legacy_cache` removed**: `modeling_internlm2.py` calls
   `DynamicCache.from_legacy_cache(past_key_values)` when `use_cache=True` and
   `past_key_values` is not a `Cache` object.  This method was removed in
   transformers 5.x.  Fix: set `model.config.use_cache = False` after loading.
   Single-forward-pass inference does not need KV caching.  (The test suite
   happened to work around this via a global shim applied at collection time by
   another loader, but the internlm2 loader must be self-contained.)

3. **Uninitialised RoPE `inv_freq` buffers** (root cause of `pcc=nan`):
   Transformers 5.x uses `init_empty_weights()` for efficient model loading.
   The `persistent=False` `inv_freq` buffer in `InternLM2RotaryEmbedding.__init__`
   is computed on the meta device and registered, but meta-tensor values are
   never materialised into real data.  After `from_pretrained`, these buffers
   hold garbage memory values up to ~5×10³⁶.  `cos(5×10³⁶)` produces NaN on
   platforms where the C math library's argument-reduction fails for very large
   inputs.  NaN hidden states propagate through all subsequent layers, producing
   all-NaN logits on both CPU and TT silicon → `pcc=nan`.  Fix: iterate over
   all layers and recompute `inv_freq` from the stored `base` and `dim`
   attributes.

## Fix
All changes are in
`tt-forge-models/internlm2_chat_1_8b/causal_lm/pytorch/loader.py`
on branch
`remediation/internlm2_chat_1_8b-causal_lm-pytorch-chat_1_8b-single_device-inference`.

- **Commit 1** (prior session): resize `tok_embeddings` and `output` from 92544
  to 92550 and copy trained embeddings for the six special tokens.
- **Commit 2** (this session): set `model.config.use_cache = False` to skip the
  removed `from_legacy_cache` code path; iterate over all attention layers to
  recompute `inv_freq` from `rotary.base` and `rotary.dim`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    204.66s (0:03:24)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/internlm2_chat_1_8b/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | da48f3697f3adfffcd7785561509f15b539937b1 |
