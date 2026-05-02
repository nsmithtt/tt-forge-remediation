# Remediation Summary: ministral_14b_instruct_bnb_4bit-causal_lm-pytorch-ministral_3_14b_instruct_2512_unsloth_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/ministral_14b_instruct_bnb_4bit/pytorch-unsloth/Ministral-3-14B-Instruct-2512-unsloth-bnb-4bit-single_device-inference]

## Result
FAIL — loader bugs fixed; Tier B compilation timeout blocks silicon run: TT MLIR compiler hangs >90 min compiling the 14B VLM LM graph at 9240-token sequence length

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
tt-mlir-vlm-long-sequence-compilation-timeout

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 2310 (input tensor's size at dimension 0), but got split_sizes=[2320]
```

## Root cause
**Loader bugs (fixed):**

Three bugs in the `ministral_14b_instruct_bnb_4bit/pytorch` loader:

1. **Missing `bitsandbytes>=0.46.1` dependency**: No `requirements.txt` existed, so BnB was unavailable and the model could not be loaded.

2. **BnB `Linear4bit` modules not dequantized**: After `from_pretrained`, the model contains `bnb.nn.Linear4bit` modules. TT hardware has no BnB kernel support, so these must be dequantized to `nn.Linear(bfloat16)` before device transfer. Unsloth's BnB models store pre-dequantized BF16 weights inside `Linear4bit` containers (no `quant_state`), so the replacement must handle both the properly-quantized and pre-dequantized cases.

3. **`split_sizes` computed via int64 arithmetic on TT device → bfloat16 rounding**: The original `get_image_features` computed `split_sizes = (image_sizes // ratio).prod(dim=-1).tolist()` with `image_sizes` as a TT tensor. TT promotes int64 to bfloat16 internally; `bfloat16(2310) = 2320` (7 mantissa bits), causing `torch.split` to fail with the reported error. Fix: compute on CPU via `torch.as_tensor(image_sizes, device="cpu").to(torch.int64)` before the arithmetic.

Additionally, `generate_block_attention_mask` in `transformers.models.pixtral.modeling_pixtral` uses in-place tensor assignment on a TT tensor, and `masked_scatter` for vision token injection causes OOM on TT. Both are replaced with CPU-built or scatter-free equivalents.

**Remaining Tier B bug:**

After the loader fixes, the test reaches silicon compilation but the TT MLIR compiler hangs for >90 minutes without completing. The Pixtral vision encoder produces 9240 image tokens for the candy.JPG test image. Together with text tokens, the main Ministral 14B LM graph is traced at sequence length ~9240. At this sequence length, the XLA graph contains ~40 × (full attention matmul 9240×9240 + FFN) ops, and TT MLIR compilation exceeds 90 minutes of wall clock time (251 threads, 52% CPU — actively compiling, not deadlocked).

The 3B GGUF text-only variant compiles and runs in ~8 minutes at short sequence lengths; the 14B VLM at 9240 tokens is ~185× more compute per attention layer and the compile time does not scale linearly.

## Fix
**Loader fixes (tt-forge-models `670538ac45`):**

1. `mistral/ministral_14b_instruct_bnb_4bit/pytorch/requirements.txt` (new): adds `bitsandbytes>=0.46.1`.

2. `mistral/ministral_14b_instruct_bnb_4bit/pytorch/loader.py` (new): implements `_dequantize_bnb4_to_bf16()` to replace all `bnb.nn.Linear4bit` → `nn.Linear(bfloat16)`, handling both `quant_state` (true 4-bit) and plain-BF16 (unsloth pre-dequantized) cases; and `_patch_mistral3_for_tt_device()` which:
   - Overrides `Mistral3Model.get_image_features` to compute `split_sizes` on CPU with explicit `torch.int64`
   - Replaces `generate_block_attention_mask` with a CPU-built version
   - Overrides `Mistral3Model.forward` to do vision-token scatter via `nonzero()` indexing rather than `masked_scatter`

**Proposed fix for Tier B (not attempted):**

The TT MLIR compiler should either: (a) cap the dynamic sequence length used for tracing/compilation to a smaller representative value (e.g. 512 or 2048), or (b) improve compilation throughput for large attention graphs. This is a cross-cutting change touching the StableHLO→TTIR lowering path and/or graph partitioning.

## Tier B justification
- **cross-cutting**: Fixing the >90-minute compile time for long sequences requires changes to the TT MLIR compiler's graph-size handling, tile/grid assignment, or compile-time caching — not a scoped single-function fix. The root cause is the quadratic scaling of attention graph size with sequence length, and no single file or pattern covers the fix.

## Verification
- pytest exit: TIMEOUT (exit 124, 90-minute timeout — 2 runs)
- Hardware:    blackhole-p150b
- Duration:    >90 min (compilation does not complete)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mistral/ministral_14b_instruct_bnb_4bit/pytorch/loader.py` (new)
- `tt-xla/third_party/tt_forge_models/mistral/ministral_14b_instruct_bnb_4bit/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 82038d9ced765a2e9bb092c31394e2ff3d8f1343 |
| tt-forge-models | 670538ac457e02a3eb7cdfcd6aa7d0185b8578ab |
