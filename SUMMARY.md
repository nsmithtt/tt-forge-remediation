# Remediation Summary: faster_whisper-speech_recognition-pytorch-Distil_small_en-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[faster_whisper/speech_recognition/pytorch-Distil_small_en-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
sdpa-decode-kseqlen-not-divisible-by-32

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Stack trace showed:
```
tt::runtime::ttnn::operations::transformer::run(tt::target::ttnn::ScaledDotProductAttentionDecodeOp const*, tt::runtime::ttnn::ProgramContext&)
```

## Root cause
Two bugs were present. First, a loader bug: `load_inputs` in the
`faster_whisper` pytorch loader returned `[input_features, decoder_input_ids]`
as a positional list. `WhisperForConditionalGeneration.forward` takes
`(input_features, attention_mask, decoder_input_ids, ...)`, so
`decoder_input_ids` landed in the `attention_mask` slot, raising
`ValueError: You have to specify either decoder_input_ids or
decoder_inputs_embeds` before reaching silicon.

After fixing the loader, the original silicon failure reproduced. The
root cause is in tt-mlir: two independent sites select the SDPA decode
kernel path purely on `qSeqLen == 1` without checking whether the key
sequence length satisfies the hardware constraint `kSeqLen % 32 == 0`.
The SDPA decode kernel in tt-metal enforces this: it requires
`k_chunk_size % 32 == 0` AND `kSeqLen % k_chunk_size == 0`, so the
minimum valid chunk size is 32, meaning kSeqLen must be divisible by 32.

For the Whisper decoder at inference time with one decoder token:
- Self-attention: kSeqLen = 1 (not divisible by 32) → TT_FATAL
- Cross-attention: kSeqLen = 1500 (Whisper encoder always pads to 3000
  mel frames → 1500 after 2-stride conv; 1500 % 32 = 28) → TT_FATAL

The two buggy sites:
1. `SDPAFusingPattern.cpp` (TTNN fusing pass): `isDecode = qShape[2] == 1`
2. `TTIRToTTNN.cpp` `shouldUseDecode()`: `return qType.getDimSize(2) == 1`

Both fire and emit `ScaledDotProductAttentionDecodeOp` to the flatbuffer,
which then fails at runtime with error code 13 (TT_FATAL).

## Fix
**Loader fix** (`tt-xla/third_party/tt_forge_models`):
- `faster_whisper/speech_recognition/pytorch/loader.py`: changed
  `load_inputs` to return a dict
  `{"input_features": ..., "decoder_input_ids": ...}` instead of a
  positional list, so the test harness passes both as keyword arguments.

**Compiler fix** (`tt-mlir`):
- `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`: added
  `kShape4D[kSeqLenDim] % 32 == 0` guard to the `isDecode` predicate.
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: added
  `kSeqLen % 32 == 0` guard to `shouldUseDecode()`.

When the guard fails, both sites fall through to the regular
`ScaledDotProductAttentionOp` (non-decode) path, which does not impose
a k_chunk_size divisibility constraint and handles these shapes correctly.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    69.29s
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/faster_whisper/speech_recognition/pytorch/loader.py`
- `tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 299da11b96354458916f8af962493cefca46cb00 |
| tt-xla          | 421b2041c9e47b7fd51359f918d81a7002cdc1c8 |
| tt-forge-models | 24fba5ee2a72b66d91be255d276ff85fc3bb2fc6 |
