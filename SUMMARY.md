# Remediation Summary: manga_ocr-pytorch-Base-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[manga_ocr/pytorch-Base-single_device-inference]

## Result
SILICON_PASS — loader bug (unused tokenizer + missing decoder_input_ids) fixed and tt-mlir SDPA decode guard added

## Stack layer
loader + tt-mlir

  - `loader` — removed unused `AutoTokenizer.from_pretrained` (required `fugashi`/`unidic_lite` dependencies that aren't needed for the test), and added missing `decoder_input_ids` to `load_inputs` so the BERT decoder runs
  - `tt-mlir` — `shouldUseDecode()` in `TTIRToTTNN.cpp` now requires `key_seq_len % 32 == 0` before routing to the SDPA decode path, avoiding the `k_chunk_size < 32` kernel assertion

## Tier
A

  - Compiler-stack fix: single-function guard in one file (`TTIRToTTNN.cpp`)

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
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Two bugs combined to produce the failure:

**Loader bug**: `load_model` loaded `AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")`,
which requires `fugashi` and `unidic_lite` (MeCab tokenizer dependencies). These packages are not
installed in the test environment, causing `ModuleNotFoundError` before reaching silicon. Additionally,
`load_inputs` returned only `{"pixel_values": pixel_values}`, but `VisionEncoderDecoderModel.forward()`
requires `decoder_input_ids` for the BERT decoder; without them, the decoder raises
`ValueError: You must specify exactly one of input_ids or inputs_embeds`, which XLA wraps as
`INTERNAL: Error code: 13` during compilation.

**Compiler bug**: With `decoder_input_ids` added (seq_len=1 BOS token), the BERT decoder
self-attention produces SDPA tensors with `query_seq_len=1` and `key_seq_len=1`. The `shouldUseDecode()`
function in `TTIRToTTNN.cpp` routes any `query_seq_len==1` operation to `ttnn::ScaledDotProductAttentionDecodeOp`.
The SDPA decode kernel computes `k_chunk_size = get_chunk_size(key_seq_len=1) = 2`, which fails the
assertion `k_chunk_size % 32 == 0` (line 66 of `sdpa_decode.cpp`):
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
```
The same failure occurs for cross-attention (`key_seq_len=197` from the ViT encoder): `get_chunk_size(197)=2`.

The `get_chunk_size(s)` formula returns values ≥32 only when `s % 32 == 0`.

## Fix
**Loader** (`tt-xla/third_party/tt_forge_models`, branch `remediation/manga_ocr-pytorch-Base-single_device-inference`):
- `manga_ocr/pytorch/loader.py`: Removed `AutoTokenizer.from_pretrained()` call and `self.tokenizer`
  attribute (unused in the test workflow). Added `self.decoder_start_token_id` populated from
  `model.config.decoder_start_token_id` in `load_model`, and added `decoder_input_ids` (a 1-token
  BOS tensor) to the dict returned by `load_inputs`.

**Compiler** (`tt-mlir`, branch `remediation/manga_ocr-pytorch-Base-single_device-inference`):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: Added `key_seq_len % 32 == 0` guard to
  `shouldUseDecode()`. Attention ops with key sequence length not divisible by 32 fall back to
  the regular (prefill) SDPA path, which handles arbitrary sizes via tiling.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    50.37s
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/manga_ocr/pytorch/loader.py`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 967c033a85e3aa9f28426d8fd3ff7d9b7287cea0 |
| tt-xla          | 5846c60eb4fe3c4d6ca917a2c30888229595d5ef |
| tt-forge-models | 49dd201d45272a52595ebaab28a965a3ab931562 |
