# Remediation Summary: kairos_14b_v1_m4_robust_i1_gguf-causal_lm-pytorch-14b_v1_m4_robust_i1_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kairos_14b_v1_m4_robust_i1_gguf/causal_lm/pytorch-14B_v1_m4_robust_i1_GGUF-single_device-inference]

## Result
FAIL — PCC=0.9897 < required 0.99 on BH p150b after loader fix; Tier B ttmlir-bf16-matmul-precision-floor

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
Original failure:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix (residual):
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9897144194191673. Required: pcc=0.99.
```

## Root cause
Two issues in sequence:

**Loader (fixed):** During test collection, `dmind_3_mini_i1_gguf` and ~25 other qwen3.5/gpt-oss GGUF loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a function of fixed signature `(gguf_path, return_tensors=False)`. transformers 5.2.0 added a `model_to_load` kwarg to `load_gguf_checkpoint`. Because `modeling_utils.py` does a lazy `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint` at each GGUF `from_pretrained` call, the patched narrow-signature function is used and rejects the new kwarg. The kairos loader has no own patch, so it is a victim of cross-loader contamination.

**Compiler (unfixed):** After the loader fix, the model (Qwen2, 48 layers, hidden_size=5120, Q4_K_M GGUF) compiles and runs on BH p150b but produces PCC=0.9897 vs required 0.99. This is the known WH/BH BF16 matmul precision floor: large hidden dimensions accumulate rounding error across 48 transformer blocks, yielding a systematic deviation below the 0.99 threshold. Same root cause as `ttmlir-bf16-matmul-precision-floor` seen in BlackSheep-RP 12B (PCC=0.949) and other large dense models.

## Fix
**Loader fix — cross-loader contamination** (tt_forge_models `d83214f0b8`): Changed all 26 narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrappers to `_patched_load_gguf_checkpoint(*args, **kwargs)` and forwarded args/kwargs to the original function. This allows `model_to_load` and any future kwargs from transformers to flow through without TypeError.

**Loader fix — requirements.txt** (tt_forge_models `a381f9e753`): Added `gguf>=0.10.0` to `kairos_14b_v1_m4_robust_i1_gguf/causal_lm/pytorch/requirements.txt` to ensure the GGUF loader is available.

**Compiler fix (proposed):** Increase matmul math fidelity from BF16 to F32 accumulation for large models on WH/BH silicon. This is a cross-cutting change across tt-mlir lowering passes and is infeasible as a targeted fix.

## Tier B justification
**cross-cutting**: Fixing the BF16 matmul precision floor requires changing math fidelity across all matmul lowering passes in tt-mlir. It affects every large model using BF16 and cannot be scoped to a single function or file.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    576.95s (0:09:36)
- Tier A attempts: N/A

## Files changed
- `kairos_14b_v1_m4_robust_i1_gguf/causal_lm/pytorch/requirements.txt` — new file, `gguf>=0.10.0`
- `tt_forge_models/<26 loaders>/causal_lm/pytorch/loader.py` — `*args, **kwargs` fix in `_patched_load_gguf_checkpoint`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 710ac8155a0f1a28a4e5e45979e5e8a7e2fa7a29 |
| tt-forge-models | a381f9e753d1585a402402a592ba242ef2769178 |
