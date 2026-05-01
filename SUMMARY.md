# Remediation Summary: granitelib_rag/causal_lm/pytorch-hallucination-detection-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granitelib_rag/causal_lm/pytorch-hallucination-detection-single_device-inference]

## Result
FAIL — PCC=0.9342 < 0.99 required; BF16 matmul precision floor on BH p150b for 40-layer dense attention transformer

## Stack layer
tt-mlir

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
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9342334750535178. Required: pcc=0.99.
```

## Root cause
The GraniteLib RAG loader loads `ibm-granite/granite-4.0-micro`, which has `model_type=granitemoehybrid` but is in fact a 40-layer dense attention-only transformer (no MoE experts: `num_local_experts=0`, no Mamba2 SSM: all `layer_types=['attention', ...]`). Hidden size 2560, intermediate size 8192. On BH p150b hardware, the WH/BH BF16 matmul accumulation error compounds over 40 layers, producing PCC=0.934 vs the required 0.99. This is the same well-known precision floor observed in Gemma 7B (PCC~0.915), Qwen3 4B (PCC=0.864), GPT-J 6B (PCC=0.75), and BlackSheep 12B (PCC=0.949) on n150/p150b. No F32 workaround is feasible at model scale (~2.4B params).

Secondary observation: the loader populates `adapter_subfolder` for the LoRA adapter but never actually loads it via PEFT — the model runs as the bare granite-4.0-micro base model. This is a loader completeness issue but does not affect the PCC failure (both CPU and TT run the same un-adapted model, so PCC comparison is self-consistent).

## Fix
None available. The BF16 precision accumulation error in tt-mlir's matmul lowering is a known compiler-level Tier B bug affecting all large multi-layer dense transformers on WH/BH hardware. Fix requires compiler-level F32 accumulation or mixed-precision path in tt-mlir. Track as `ttmlir-bf16-matmul-precision-floor`.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    151s
- Tier A attempts: N/A

## Files changed
None (no fix available)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
