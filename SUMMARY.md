# Remediation Summary: blacksheep_rp_12b_gguf-causal_lm-pytorch-12B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[blacksheep_rp_12b_gguf/causal_lm/pytorch-12B_GGUF-single_device-inference]

## Result
FAIL — loader fixed (gguf requirement); remaining PCC=0.949 < 0.99 is a Tier B compiler precision issue

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Two issues:

1. **Loader bug (fixed):** The `blacksheep_rp_12b_gguf` loader was missing a `requirements.txt` specifying `gguf>=0.10.0`. When the RequirementsManager cleaned up after a previous test, `gguf` was removed from the venv, causing `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` to raise an `ImportError`. The configured branch (`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-0`) already had the `apply_chat_template` fallback fix but lacked the `requirements.txt` entry.

2. **Compiler precision issue (unfixed, Tier B):** After the loader fix, the test ran to completion on Blackhole p150b silicon in 9m53s and produced PCC=0.949. The required threshold is 0.99. This gap (4%) is consistent with the BF16 matmul accumulation floor seen on TT silicon for large language models (same class as Gemma 7B PCC≈0.915, Qwen3 4B PCC=0.864, GPT-J 6B PCC=0.75 on Wormhole n150). The BlackSheep-RP-12B model (Mistral/Llama-based 12B architecture, 40 transformer layers) accumulates significant BF16 rounding error through its depth.

## Fix
Loader fix applied: added `blacksheep_rp_12b_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0` to the remediation branch `remediation/blacksheep_rp_12b_gguf-causal_lm-pytorch-12B_GGUF-single_device-inference` in `tt-forge-models`. The tt-xla submodule was advanced on branch `remediation/blacksheep-rp-12b-gguf` to point to this fix.

The PCC precision failure is not fixed. To fix it, BF16 accumulation in all matmul lowerings in tt-mlir would need to be preserved at FP32 precision (tracked as tt-xla #2861 for the Wormhole variant).

## Tier B justification
The BF16 matmul precision floor is a cross-cutting issue requiring FP32 accumulation preservation through every matmul lowering in the compiler. This touches more than 3 files across multiple passes and is the same underlying issue as existing Tier B reports for Gemma, Qwen3, and GPT-J.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    593.25s (0:09:53)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `blacksheep_rp_12b_gguf/causal_lm/pytorch/requirements.txt` (added, `gguf>=0.10.0`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 59280e780a75318bc82a95cb5766027ab81f7b2c |
| tt-forge-models | 751a52c5a492a8098628ab5ef1ee79c629b9f0a6 |
