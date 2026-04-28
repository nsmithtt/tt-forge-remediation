# Remediation Summary: florence_2-image_captioning-pytorch-Large-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[florence_2/image_captioning/pytorch-Large-single_device-inference]

## Result
FAIL — TTNN SDPA decode kernel does not support k_len < 32 (decoder self-attention with one token gives k_chunk_size=2, violating k_chunk_size % 32 == 0)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
sdpa-k-chunk-size-lt-32

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Device log:
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2 (assert.hpp:104)
ERR| Exception:
{TT_FATAL @ ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:66: k_chunk_size % 32 == 0
Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
 --- ttnn::transformer::scaled_dot_product_attention_decode(...)
```

## Root cause
**Loader layer** (transformers 5.x breaking change): `PretrainedConfig` in transformers 5.x moved generation parameters — including `forced_bos_token_id` — out of `PretrainedConfig` into `GenerationConfig`. The Florence-2 remote code in `configuration_florence2.py` (loaded via `trust_remote_code=True`) accesses `self.forced_bos_token_id` after calling `super().__init__()`, which no longer sets it. This raises `AttributeError`.

Loader fix applied: `_florence2_compat_load()` temporarily patches `PretrainedConfig.__init__` to restore the missing attribute and patches `torch.linspace` to force CPU device (the DaViT vision encoder calls `torch.linspace().item()` during construction, which fails on meta device). `AutoProcessor` replaced with separate `AutoTokenizer` + `AutoImageProcessor` (the combined processor's `TokenizersBackend` is incompatible with the microsoft tokenizer in transformers 5.2.0).

**Compiler layer** (tt-metal): After the loader fix, the model reaches the TTNN SDPA decode kernel. Florence-2 is an encoder-decoder model; the test passes `decoder_input_ids` of shape `(1, 1)` — a single BOS token. This causes k_len=1 in the decoder self-attention. `get_chunk_size(1)` returns 2 (the maximum power-of-2 divisor of 1, min-capped at 2), failing the assertion `k_chunk_size % 32 == 0`.

The assertion is in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:66`:
```cpp
uint32_t s = input_tensor_k.logical_shape()[-2];  // s = 1
uint32_t k_chunk_size = get_chunk_size(s);          // returns 2
TT_FATAL(k_chunk_size % 32 == 0, ...);              // 2 % 32 != 0 → fatal
```

## Fix
**Loader fix** (committed and pushed):
- `third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py`
  - Added `_florence2_compat_load()`: patches `PretrainedConfig.__init__` temporarily to restore `forced_bos_token_id = None`, patches `torch.linspace` to force CPU device
  - Split `AutoProcessor` into `AutoTokenizer` + `AutoImageProcessor` for non-community variants
  - Branch: `remediation/florence_2-image_captioning-pytorch-Large-single_device-inference` in tt-forge-models
  - Commit: `a4760fa395`

**Compiler fix** (proposed, not attempted — Tier B):
The TTNN SDPA decode kernel would need to support k_len < 32. Options:
1. In `scaled_dot_product_attention_decode` in `sdpa_decode.cpp`: detect `s < 32`, pad K and V tensors to 32 positions with zeros, and use `cur_pos` to mask the padded positions during attention computation.
2. Alternatively, fall through to a non-chunked (regular SDPA, non-decode) attention path when sequence length is below the minimum chunk size.

## Tier B justification
**cross-cutting** — Modifying the TTNN SDPA decode kernel to handle k_len < 32 (via K/V tensor padding + masking, or a fallback non-chunked path) would affect all models using `scaled_dot_product_attention_decode`. The change requires tensor manipulation in the kernel dispatch layer and verification that the kernel correctly respects `cur_pos` for masking padded positions. This is a coordinated kernel + validation change, not a scoped one-line fix.

## Verification
- pytest exit: FAIL
- Hardware: n150 (Wormhole B0)
- Duration: 318.84s (0:05:18) — loader loaded model successfully, failure at SDPA decode execution
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py`
  (commit `a4760fa395` on `remediation/florence_2-image_captioning-pytorch-Large-single_device-inference` in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a4760fa395cab897ee5e751250631cb26b282ca8 |
