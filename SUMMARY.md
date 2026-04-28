# Remediation Summary: florence_2_flux_large-pytorch-Large-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[florence_2_flux_large/pytorch-Large-single_device-inference]

## Result
FAIL — First compiler bug (SDPA decode k_chunk_size assertion) fixed with Tier A guard; second compiler bug (regular SDPA PCC=0.224 for kv_len=1 / kv_len=583 decoder shapes) prevents SILICON_PASS.

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
sdpa-decode-kv-seq-len-not-multiple-of-32

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Underlying assertion (from `sdpa_decode.cpp:66`):
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
```

## Root cause

**Bug 1 (fixed): SDPA decode elected for kv_len not divisible by 32.**

Florence-2-Flux-Large is a BART-based encoder-decoder model with a DaViT visual encoder. During single-step decode (`decoder_input_ids` of length 1), `q_len == 1`, which caused both decision points in tt-mlir to route to `ttnn::transformer::scaled_dot_product_attention_decode`:

- `shouldUseDecode()` in `TTIRToTTNN.cpp` checked only `qSeqLen == 1`.
- `isDecode` in `SDPAFusingPattern.cpp` checked only `qShape[kSeqLenDim] == 1`.

The SDPA decode kernel (`sdpa_decode.cpp`) computes `k_chunk_size` as the largest power-of-2 divisor of `kv_len`, then asserts `k_chunk_size % 32 == 0`. For this model:

- **Self-attention** (`q=k=v shape [1,16,1,64]`): `kv_len=1`, `get_chunk_size(1)=1` → assertion fires.
- **Cross-attention** (`q=[1,16,1,64]`, `k=v=[1,16,583,64]`): `kv_len=583` (DaViT produces 583 visual tokens from a 768×768 image), `get_chunk_size(583)=1` (583 is odd) → assertion fires.

**Bug 2 (unfixed): Regular SDPA produces PCC=0.224 for decoder shapes.**

After adding the `kv_len % 32 == 0` guard (routing the above shapes to regular SDPA instead), the test runs to completion but produces completely wrong outputs: PCC=0.22421660775977295 vs. required 0.99. The regular SDPA path handles these shapes incorrectly — likely because it pads KV tensors to tile width (32) without a corresponding attention mask, causing all 31 zero-padded key positions to contribute `exp(0)=1` to the softmax, diluting the attended value by ≈32×.

Per skill rules, the first fix was committed on the remediation branch and the second bug is filed as FAIL.

## Fix

**Loader fixes (applied via existing tt_forge_models commit `7525fb0bfaf288f0d5ca384901349ae1b5856d8e`):**

1. **`_florence2_compat_load()` in `loader.py`** — Patches `PretrainedConfig.__init__` to add `forced_bos_token_id = None` post-init. transformers 5.x moved this attribute to `GenerationConfig`; Florence-2's custom `Florence2LanguageConfig.__post_init__()` reads `self.forced_bos_token_id` and raises `AttributeError`.

2. **`_tokenizer_compat_load()` in `loader.py`** — Patches `PreTrainedTokenizerFast._add_tokens` to convert `dict` entries in `additional_special_tokens` to `AddedToken` objects. Florence-2-Flux-Large stores 1000+ special tokens as raw dicts in its tokenizer config; transformers 5.x no longer converts these automatically.

**Compiler-stack fix (Tier A, committed to tt-mlir `remediation/florence_2_flux_large-pytorch-Large-single_device-inference`):**

Added `kv_len % 32 == 0` guard to both decision points that elect SDPA decode:

- `shouldUseDecode()` in `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — returns `false` when `kvSeqLen % 32 != 0`.
- `isDecode` in `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp` — sets `isDecode = false` when `kvSeqLen % 32 != 0`.

This prevents the `k_chunk_size % 32 == 0` fatal assertion but does not fix the downstream regular SDPA PCC failure.

**Proposed fix for Bug 2 (for compiler team):**

For decoder self-attention with `q_len=1, kv_len=1` and cross-attention with `q_len=1, kv_len=583`, regular SDPA produces wrong output. The most likely cause: TTNN pads KV tensors to multiples of 32 in the tile layout but does not synthesize an attention mask to zero out the padded positions. Proposed fix: when padding KV in the regular SDPA path, generate or extend the attention mask with `-inf` (or `-1e9`) for padded positions. Alternatively, fix SDPA decode to handle non-power-of-2 `kv_len` by rounding up to the nearest multiple of 32 and padding K/V accordingly. Lives in `tt-metal` SDPA kernel program factories.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    n/a (second bug unfixed)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/florence_2_flux_large/pytorch/loader.py` (via submodule pointer update, existing fix)
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — SDPA decode guard
- `tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp` — SDPA fusing decode guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | def41977a7c2d01e7010ea736d3e0c57520be974 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7525fb0bfaf288f0d5ca384901349ae1b5856d8e |
