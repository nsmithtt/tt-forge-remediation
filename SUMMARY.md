# Remediation Summary: deepseek/deepseek_r1_distill/pytorch-Distill_Qwen_7B_unsloth_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_distill/pytorch-Distill_Qwen_7B_unsloth_bnb_4bit-single_device-inference]

## Result
FAIL — sdpa-k-chunk-size-lt-32 compiler bug causes PCC=0.9348 (required 0.99) after loader fixes applied

## Stack layer
tt-mlir

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
Original failure (before loader fixes):
```
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

After adding bitsandbytes requirement:
```
RuntimeError: Creating a Parameter from an instance of type Params4bit requires that detach() returns an instance of the same type, but return type Tensor was found instead. To use the type as a Parameter, please correct the detach() semantics defined by its __torch_dispatch__() implementation.
```

After dequantization fix (final failure):
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9348857551550884. Required: pcc=0.99.
```

## Root cause
Two loader bugs were fixed before reaching the compiler-stack failure:

**Loader bug 1 — missing bitsandbytes dependency**: The `DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT` variant loads `unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit`, a BNB 4-bit quantized checkpoint that requires `bitsandbytes>=0.46.1`. No `requirements.txt` existed for this loader directory.

**Loader bug 2 — BNB Params4bit incompatible with TT device**: After installing bitsandbytes, the model loaded as BNB 4-bit quantized (`Params4bit` parameters). The test framework calls `model.to(tt_device)` which fails because `Params4bit.detach()` returns a plain `Tensor`, not a `Params4bit`. BNB quantized tensors are CUDA-only and cannot be transferred to XLA/TT devices. Fix: dequantize all `Linear4bit` layers to regular `nn.Linear` with bfloat16 weights before returning the model.

**Compiler-stack bug (Tier B) — sdpa-k-chunk-size-lt-32**: After the two loader fixes the model loaded and compiled successfully, but PCC=0.9348 vs the required 0.99. This is the same SDPA k_chunk_size constraint bug documented in report `deepseek-deepseek_r1_distill-pytorch-Distill_Qwen_7B-single_device-inference`: the test prompt tokenizes to 22 tokens (seq_len=22 < 32), and the TT SDPA kernel requires `k_chunk_size >= 32`, causing corrupted attention outputs. The Qwen2 model architecture is shared between the base and BNB variants, so the PCC degradation is identical in nature.

## Fix
**Loader fixes (committed to tt-forge-models remediation branch):**

1. `deepseek/deepseek_r1_distill/pytorch/requirements.txt` (new file): Added `bitsandbytes>=0.46.1`.

2. `deepseek/deepseek_r1_distill/pytorch/loader.py`: Added `_dequantize_bnb_model()` static method that iterates through all `Linear4bit` modules and replaces each with a dequantized `nn.Linear`. Uses `bitsandbytes.functional.dequantize_4bit(weight.data, quant_state)` to recover float16 weights, then converts to bfloat16. Called in `load_model()` for the `DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT` variant.

**Compiler-stack fix (proposed, not attempted):**

The SDPA lowering in tt-mlir / tt-metal should either pad key/value tensors to meet the `k_chunk_size >= 32` requirement (masking padded positions) or fall back to a non-chunked path when `seq_len < k_chunk_size`. This is the same fix proposed in the existing report for `Distill_Qwen_7B`.

## Tier B justification
Indicator: **cross-cutting** — fixing the SDPA k_chunk_size constraint requires changes in tt-mlir and/or tt-metal affecting all models with short sequences (< 32 tokens). This is the same bug already classified as Tier B in the prior report for the base `Distill_Qwen_7B` variant. Attempting a second compiler-stack fix for the same underlying bug in this report is out of scope.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 126.98s (0:02:06)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1_distill/pytorch/requirements.txt` (new)
- `deepseek/deepseek_r1_distill/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aa8de6050fc76b60ff1e92bf71cc5ed8e16dccc7 |
| tt-forge-models | d344e0b8884eba889418c8d185a1968aa0b35db5 |
