# Remediation Summary: aufklarer_qwen3_asr_0_6b_mlx_8bit/speech_recognition/pytorch-0.6B_MLX_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aufklarer_qwen3_asr_0_6b_mlx_8bit/speech_recognition/pytorch-0.6B_MLX_8bit-single_device-inference]

## Result
FAIL — PCC=0.723 on TT silicon after scatter legalization fix; WH BF16 matmul accumulation floor in 46-layer multimodal model (18 audio + 28 text decoder)

## Stack layer
loader, tt-mlir

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
Original failure:
```
E   RuntimeError: Error code: 13
loc("scatter.53"): error: failed to legalize operation 'stablehlo.scatter'
```

After loader fixes and scatter legalization fix:
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7233858152171995. Required: pcc=0.99.
```

## Root cause

Three bugs were found and fixed in order:

**Bug 1 — loader (fixed)**: `aufklarer/Qwen3-ASR-0.6B-MLX-8bit` has a `quantization_config` in MLX format (missing `quant_method`). Transformers ≥4.57 `get_hf_quantizer()` raises `ValueError` on this. Also the `dtype` kwarg was used instead of `torch_dtype`.

**Bug 2 — loader (fixed)**: `load_inputs()` received `dtype_override=torch.bfloat16` from the dynamic loader but did not cast `input_features` to bfloat16, causing `Input type (float) and bias type (c10::BFloat16) should be the same` in the Conv layers.

**Bug 3 — tt-mlir Tier A scatter (fixed)**: The line
`chunk_lengths[chunk_lengths == 0] = self.n_window * 2`
in `Qwen3ASRAudioEncoder.forward` generates a StableHLO scatter with `scatter_indices` shape `[K, 1]` (index_vector_dim=1) but `updates` shape `[K]`. `checkBasicLegality` in `StableHLOToTTIRScatterOpConversionPattern` rejected this with `"TTIR scatter requires indices.rank <= updates.rank"`. The fix allows rank(indices) == rank(updates) + 1 when the extra trailing dimension is the index_vector_dim, and `extractElementWiseScatterIndices` squeezes it away.

**Bug 4 — tt-mlir Tier B (unfixed)**: After all three fixes, the model compiles and runs on silicon but gives PCC=0.723 (required: 0.99). The model has 18 audio encoder layers (d_model=896, encoder_ffn_dim=3584) plus 28 text decoder layers (hidden=1024, intermediate_size=3072), totaling 46 layers of BF16 matmul accumulation. The audio encoder output feeds directly into 13 of ~37 sequence positions in the text decoder, compounding errors through 28 additional attention layers. PCC=0.723 is consistent with the known WH BF16 matmul accumulation floor (`ttmlir-f32-precision-not-preserved`) observed in Gemma 7B (PCC=0.915), Qwen3-4B (PCC=0.864), and GPT-J 6B (PCC=0.75). The multimodal amplification (audio encoder BF16 errors propagating through text decoder) explains the unusually low PCC for a 0.6B model.

## Fix

**Loader fixes** in `tt-xla/third_party/tt_forge_models/aufklarer_qwen3_asr_0_6b_mlx_8bit/speech_recognition/pytorch/loader.py`:
- In `_load_model_wrapper`: import `qwen_asr` before `AutoConfig.from_pretrained()` (to register `qwen3_asr` model type), delete `quantization_config` attribute from the config, pass modified config to `Qwen3ASRModel.from_pretrained`, use `torch_dtype` kwarg (not `dtype`).
- In `load_inputs`: cast all floating-point tensors to `dtype_override` after processor call.

**tt-mlir scatter legalization fix** in `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:
- In `checkBasicLegality`: allow rank(indices) == rank(updates) + 1 when the extra trailing dimension equals index_vector_dim.
- In `extractElementWiseScatterIndices`: squeeze the trailing dimension from indices when rank(indices) == rank(updates) + 1.

**Remaining issue (FAIL)**: WH BF16 matmul accumulation gives PCC=0.723, well below the required 0.99. Proposed fix: preserve F32 accumulation through the TTIR-to-TTNN lowering pipeline (`ttmlir-f32-precision-not-preserved`), tracked in tt-xla #2861.

## Tier B justification
cross-cutting — preserving FP32 matmul accumulation requires changes across every matmul lowering pass in tt-mlir, touching many files and affecting all models.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    158.83s (0:02:38)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/aufklarer_qwen3_asr_0_6b_mlx_8bit/speech_recognition/pytorch/loader.py` (loader bugs 1 and 2)
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (scatter legalization fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | e07ffbb311734ba404674239fae8ef935bd8b775 |
| tt-xla          | 31dac87b0e446d42b5e2b401d6ec4c498d9e3ea7 |
| tt-forge-models | 36f3671e3857b51d72d36ffff4aafe422a3c46b6 |
