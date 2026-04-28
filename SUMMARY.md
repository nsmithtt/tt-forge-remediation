# Remediation Summary: faster_whisper-speech_recognition-pytorch-Large_v3_Turbo-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[faster_whisper/speech_recognition/pytorch-Large_v3_Turbo-single_device-inference]

## Result
FAIL — SDPA decode guard (Tier A, SILICON fixed) reveals second Tier B BF16 precision bug: pcc=0.94 < 0.95 threshold

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

(After SDPA fix, the model runs but fails PCC:)
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9408808372606495. Required: pcc=0.95.

## Root cause
Two bugs were present.

**Bug 1 (Tier A, fixed): SDPA decode kernel — kSeqLen not divisible by 32**

The same root cause as the Distil_small_en variant. Two sites in tt-mlir select
the SDPA decode kernel purely on `qSeqLen == 1` without checking whether the
key sequence length satisfies `kSeqLen % 32 == 0`. The SDPA decode kernel
requires `k_chunk_size % 32 == 0` and for Whisper decoder:
- Self-attention: kSeqLen = 1 → `k_chunk_size` = 1 (not divisible by 32)
- Cross-attention: kSeqLen = 1500 (Whisper encoder pads to 3000 mel → 1500
  after 2-stride conv; 1500 % 32 = 28) → also fails

Both sites (`SDPAFusingPattern.cpp` and `TTIRToTTNN.cpp`) emit
`ScaledDotProductAttentionDecodeOp` which then triggers TT_FATAL at runtime
(surfaced as INTERNAL error code 13).

Fix applied: added `kSeqLen % 32 == 0` guard to both sites so that sequences
with non-aligned key lengths fall through to the regular SDPA path.

**Bug 2 (Tier B, unfixed): ttmlir-f32-precision-not-preserved**

After fixing Bug 1, the model runs to completion but PCC is 0.94 vs 0.95
threshold. The `TorchDynamicLoader` automatically applies `dtype_override=
torch.bfloat16` (detected via signature inspection), so both CPU and TT run
in BF16. The divergence arises because:
- CPU BF16 matmul: internally accumulates in FP32 (PyTorch CPU kernel)
- TT BF16 matmul: accumulates in BF16 (hardware constraint)

Measured: CPU FP16 vs CPU BF16 PCC = 0.9999 (quantization alone is negligible).
Therefore the 0.06 PCC gap is entirely from BF16 accumulation on TT vs FP32
accumulation on CPU, accumulated over 36 layers (32 encoder + 4 decoder, each
with attention and FFN matmuls at 1280 hidden dim).

The same issue was present in the Distil_small_en variant but did not surface
because that model has only 14 layers (12 encoder + 2 decoder) and PCC
remained above 0.95. With 36 layers, the accumulated BF16 error crosses the
threshold.

This is a cross-cutting Tier B issue: every matmul lowering in tt-mlir/tt-metal
would need to preserve FP32 accumulation to fix it.

## Fix
**Loader fix (tt_forge_models):**
- `faster_whisper/speech_recognition/pytorch/loader.py`: changed `load_inputs`
  to return `{"input_features": ..., "decoder_input_ids": ...}` dict instead of
  a positional list `[input_features, decoder_input_ids]`. Positional arg 2 maps
  to `attention_mask`, not `decoder_input_ids`, in `WhisperForConditionalGeneration.forward`.

**Compiler fix (tt-mlir, Tier A — applied but insufficient due to Bug 2):**
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: added `keySeqLen % 32 == 0`
  guard to `shouldUseDecode()`.
- `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`: added
  `kSeqLen % 32 == 0` to `isDecode` predicate; added fallback guard to block
  prefill SDPA when Q_seq == 1 but K not aligned (return failure() instead).

**Proposed fix for Bug 2:** Change the TTNN matmul/attention lowering to use
FP32 accumulation for BF16 inputs where the hardware supports it. This requires
changes to the matmul kernel dispatch in tt-metal and the corresponding MLIR
lowering patterns — a multi-file, cross-repo change.

## Tier B justification
- Indicator: `cross-cutting`
- Explanation: Fixing `ttmlir-f32-precision-not-preserved` requires changing
  accumulation mode for every matmul and attention op in tt-mlir/tt-metal.
  It cannot be scoped to a single function or file without risking silent
  regressions in other models.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    130.45s (run that revealed PCC failure after SDPA fix)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/faster_whisper/speech_recognition/pytorch/loader.py`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
- `tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 87607e3285fe6739a1b05473841c05608214b777 |
| tt-xla          | 276a9c776ec9c5db8af7f9e9c9bbd2d1af3d1a10 |
| tt-forge-models | e1edcf5f28adcf550550d6cf0e2c382f4a50387d |
