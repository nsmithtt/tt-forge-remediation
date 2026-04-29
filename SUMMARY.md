# Remediation Summary: aufklarer_qwen3_asr_1_7b_mlx_8bit-speech_recognition-pytorch-1.7B_MLX_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aufklarer_qwen3_asr_1_7b_mlx_8bit/speech_recognition/pytorch-1.7B_MLX_8bit-single_device-inference]

## Result
FAIL — TT silicon runtime accuracy bug: PCC=0.62 vs required 0.99; CPU bfloat16 gives PCC=0.9995 (same inputs), confirming the gap is not bfloat16 accumulation

## Stack layer
loader, tt-mlir, tt-metal

## Tier
B

## Bug fingerprint
tt-silicon-accuracy-qwen-lm-pcc-failure

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `RuntimeError: Error code: 13` (PJRT kInternal — stablehlo.scatter failed to legalize)
After loader + tt-mlir fixes: `AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.6184789089388705. Required: pcc=0.99.`

## Root cause

### Loader layer (fixed)

Three bugs were fixed in the loader:

1. **MLX quantization_config incompatibility** (`transformers-5x-mlx-quantization-config-no-quant-method`):
   `aufklarer/Qwen3-ASR-1.7B-MLX-8bit`'s `config.json` contains a `quantization_config` in MLX format
   (`{"group_size":64,"bits":8,"mode":"affine"}`) with no `quant_method` field. Transformers ≥5.0 raises
   `ValueError` on such a config. Fixed by stripping `quantization_config` before loading.

2. **Missing `preprocessor_config.json`** (`qwen3-asr-missing-preprocessor-config`):
   The MLX repo does not ship `preprocessor_config.json`, so `AutoProcessor.from_pretrained` fails. Fixed
   by loading the processor from the base `Qwen/Qwen3-ASR-1.7B` model instead.

3. **`DynamicCache` in model output** (`qwen3-asr-dynamic-cache-in-forward-output`):
   `Qwen3ASRWrapper.forward` returned the full thinker output (including `past_key_values: DynamicCache`),
   which caused `torch.equal(x, y)` in the evaluator to raise `TypeError`. Fixed by returning
   `output.logits` only.

### tt-mlir layer (fixed — Tier A)

**`stablehlo.scatter` rank mismatch** (`stablehlo-scatter-index-vector-dim-rank-mismatch`):
XLA compiles `chunk_lengths[chunk_lengths == 0] = val` to a `stablehlo.scatter` where `scatter_indices`
has shape `[K, 1]` (rank 2) while `updates` has shape `[K]` (rank 1). The trailing size-1 dimension is
the `index_vector_dim` — a legitimate stablehlo encoding for scalar index vectors. The pre-existing
`checkBasicLegality` check rejected this with "indices.rank <= updates.rank" failure. Fixed by:
- Relaxing the rank check to allow `indices.rank == updates.rank + 1` when the extra dim is the trailing
  `index_vector_dim` of size 1.
- Adding the symmetric squeeze `[K, 1] → [K]` in `extractElementWiseScatterIndices` so the TTIR scatter
  sees matching ranks.

### tt-metal layer (unfixed — Tier B)

After the loader and tt-mlir fixes, the model compiles and runs on TT silicon in 123.96 s but produces
PCC=0.62 vs the CPU float32 reference (required: 0.99). A CPU bfloat16 vs float32 comparison with the
same fixed-seed inputs gives PCC=0.9995, proving the gap is NOT bfloat16 accumulation — it is a genuine
runtime accuracy bug in the TT silicon execution. The same symptom (PCC below threshold) is seen in
`qwen_2_5/causal_lm/pytorch-1.5B` and `1.5B_Instruct`, both set to `assert_pcc: false` in the test
config as a known workaround. The root cause of the Qwen-class LM accuracy regression on TT silicon is
unknown and requires systematic op-by-op investigation.

## Fix

**Loader fixes** (`tt-xla/third_party/tt_forge_models`, branch `remediation/aufklarer_qwen3_asr_1_7b_mlx_8bit-speech_recognition-pytorch-1.7B_MLX_8bit-single_device-inference`):
- `aufklarer_qwen3_asr_1_7b_mlx_8bit/speech_recognition/pytorch/loader.py`:
  - Strip `config.quantization_config` before `AutoModel.from_pretrained`
  - Load processor from `Qwen/Qwen3-ASR-1.7B` base model via `AutoProcessor.from_pretrained`
  - Construct `Qwen3ASRModel` manually (backend, model, processor, max_new_tokens)
  - Return `output.logits` in `Qwen3ASRWrapper.forward`

**Compiler-stack fix** (`tt-mlir`, branch `remediation/aufklarer_qwen3_asr_1_7b_mlx_8bit-speech_recognition-pytorch-1.7B_MLX_8bit-single_device-inference`):
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:
  - `checkBasicLegality`: permit `indices.rank == updates.rank + 1` when the extra dim is the trailing
    `index_vector_dim` of size 1
  - `extractElementWiseScatterIndices`: add squeeze `[K, 1] → [K]` when indices rank exceeds updates rank

**Proposed fix for the TT silicon accuracy bug**: investigate which op in the Qwen3 LM decoder
(attention, MLP, RoPE, or layer norm) produces results diverging from CPU float32. This likely requires
per-layer output comparison with the `torch_comparison_evaluator` and targeted op-level tests in the
tt-mlir lit suite. No specific file identified yet.

## Tier B justification
Indicator: `internal-error-unknown-mechanism` — which op in the Qwen3 LM produces the incorrect result
is unknown; diagnosis must come before a fix. The issue also manifests in Qwen2.5-1.5B (same LM
architecture), suggesting a cross-cutting precision issue rather than a single-file change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    123.96s (model ran to completion; failed at PCC comparison)
- Tier A attempts: 1 (scatter rank-mismatch fix; resolved compile error but revealed accuracy bug)

## Files changed
- `tt-xla/third_party/tt_forge_models/aufklarer_qwen3_asr_1_7b_mlx_8bit/speech_recognition/pytorch/loader.py`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 40dee8b76063dab72f82053e9f8e4fd96dc2caa0 |
| tt-xla          | 83cf9dc81f1b7349c2dd480084ffd92c31fd2f50 |
| tt-forge-models | da937b0193bffcd3d7ec3a44ac56c05e0b12a5e0 |
