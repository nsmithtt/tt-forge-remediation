# Remediation Summary: l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf/causal_lm/pytorch-L3_8B_Stheno_v3_3_32K_Ultra_NEO_V1_IMATRIX_GGUF-single_device-inference]

## Result
FAIL — TT BF16 computation diverges from CPU BF16 at PCC 0.9654 (required 0.99); measured CPU BF16 vs CPU FP32 PCC = 0.999 so this is not a BF16 floor

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-llama-gqa

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```
After the loader fix, the test runs to completion but fails PCC:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9654606045033249. Required: pcc=0.99.
```

## Root cause
Two bugs, in two layers:

**Loader bug (fixed):** The `l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf/causal_lm/pytorch/` directory had no `requirements.txt`, so the test runner never installed `gguf>=0.10.0` before importing the loader. The loader also called `apply_chat_template` unconditionally, which would fail on GGUF files without an embedded chat template.

**Compiler-stack bug (Tier B, unfixed):** After fixing the loader, the model runs end-to-end on silicon but produces logits with PCC 0.9654 vs the CPU BF16 reference. The CPU BF16 floor for this model is measured at 0.999 (CPU BF16 vs CPU FP32), so the 3.5% PCC gap is a real TT computation divergence, not BF16 accumulation noise. The model is LLaMA-3 8B with GQA (32 Q heads, 8 KV heads) and an extended RoPE context (rope_theta=2000000 in rope_scaling, max_seq_len=32768). The PCC deficit is consistent with the known TT BF16 precision issue affecting LLaMA models (see tt-xla issue #2944, which causes the vanilla LLaMA 3.1 8B to require required_pcc=0.98), but this GGUF variant shows an even larger gap despite running both CPU and TT in BF16. The gap likely comes from accumulated BF16 matmul rounding errors amplified by the large rope_theta value.

## Fix
**Loader fixes applied** in `tt_forge_models` on branch `remediation/l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf`:

1. Added `l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0` so the test runner installs gguf before importing the loader.

2. Guarded `apply_chat_template` in `load_inputs()` with `if self.tokenizer.chat_template is not None:` to handle GGUF files that do not embed chat template metadata.

**Proposed compiler-stack fix:** Investigate TT SDPA and matmul accumulation precision for LLaMA GQA models. The fix would live in `tt-mlir`, likely in the SDPA lowering or the BF16 matmul accumulator logic. This is the same root cause as issue #2944 but manifests more severely for models with large rope_theta. A targeted investigation should start with SDPA softmax precision on TT for sequences with high RoPE frequency.

## Tier B justification
`cross-cutting`: Improving BF16 accumulation precision for LLaMA-class matmuls and SDPA would require changes across multiple lowering passes in tt-mlir. The existing LLaMA 3.1 8B PCC issue (#2944) has been open and unfixed for some time, indicating this requires deeper infrastructure work beyond a scoped single-file fix.

## Verification
- pytest exit: FAIL (PCC 0.9654 < 0.99 required; ImportError resolved by loader fix)
- Hardware:    wormhole
- Duration:    441.32s (0:07:21)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: `l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt_forge_models`: `l3_8b_stheno_v3_3_32k_ultra_neo_v1_imatrix_gguf/causal_lm/pytorch/loader.py` (apply_chat_template guard)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5de2adeed74993389918d3ef54b25618999c8e39 |
