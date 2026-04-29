# Remediation Summary: babylm_2025_submission_strict-causal_lm-pytorch-babylm_2025_submission_strict-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[babylm_2025_submission_strict/causal_lm/pytorch-babylm_2025_submission_strict-single_device-inference]

## Result
FAIL — PCC=0.9851 below required 0.99; loader triton stub fixed but TT silicon BF16 matmul precision gap is Tier B

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision

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
E   ModuleNotFoundError: No module named 'triton'
```

After loader fix, secondary failure:
```
AssertionError: PCC comparison failed: pcc=0.9850798096917568 < Required: pcc=0.99
```

## Root cause
Two distinct bugs:

**1. Loader bug (fixed):** The BabyLM 2025 Submission Strict model uses the xQwen architecture with `flash-linear-attention` (fla) and `mlstm_kernels` (xlstm) dependencies. Both packages unconditionally import `triton` at module level — `fla/ops/__init__.py` imports from `fla.ops.abc.chunk` which does `import triton` before any kernel guards. Since Tenstorrent hardware has no CUDA/triton, this causes `ModuleNotFoundError: No module named 'triton'` during model loading.

The fix installs a no-op `sys.modules` stub for `triton` and its submodules before any fla/mlstm_kernels imports are triggered. A key implementation detail: `torch._inductor.runtime.triton_compat` must be pre-loaded before the stub is injected, so PyTorch caches `HAS_TRITON=False` and does not try to use the stub for code generation.

**2. Compiler-stack precision bug (Tier B, unfixed):** After the loader fix, the model runs on TT silicon but produces PCC=0.9851 vs CPU BF16 baseline. The measured CPU BF16 vs FP32 floor for this model is 0.9907 — just above the 0.99 threshold. The gap between TT silicon (0.9851) and CPU BF16 (0.9907) is ~0.006, indicating accumulated WH BF16 matmul precision error beyond the inherent quantization noise. This is the same cross-cutting Wormhole BF16 matmul precision deficit seen in Gemma 7B (PCC ~0.915), Qwen3 4B (PCC=0.864), and GPT-J 6B (PCC=0.75). The model is 270M parameters with hybrid mLSTM+attention layers (mLSTM uses native_autograd CPU fallback; attention layers run on TT silicon), so the 0.006 gap is a mild but real manifestation of the same WH BF16 accumulation error in the attention/linear matmuls.

## Fix
**Loader fix** (`loader.py` in `tt_forge_models`):
- Added `_install_triton_stub()` function that installs a minimal no-op `triton` stub in `sys.modules` with all submodules (`triton.language`, `triton.language.extra`, `triton.language.extra.libdevice`, `triton.language.core`, `triton.runtime`) and all attributes used by fla/mlstm_kernels at import time
- Pre-loads `torch._inductor.runtime.triton_compat` before stub injection to keep `HAS_TRITON=False` in PyTorch's inductor cache
- Called at module level before any `trust_remote_code` loading can trigger fla/mlstm_kernels imports

Files changed:
- `babylm_2025_submission_strict/causal_lm/pytorch/loader.py` in `tt_forge_models`

**Precision bug** (proposed fix for Tier B review):
The fix would require preserving float32 intermediate precision in WH matmul lowerings in tt-mlir/tt-metal, or using higher-precision accumulation modes. This is a cross-cutting change touching every matmul lowering path, making it Tier B. The human reviewer should treat this as a known WH BF16 precision deficit and file accordingly.

## Tier B justification
Which indicator: **cross-cutting**

The WH BF16 matmul precision deficit is not a single lowering pattern — it affects every BF16 matrix multiplication on Wormhole hardware. Fixing it requires either preserving float32 precision through all matmul lowering passes in tt-mlir/tt-metal, or changing how the hardware accumulates BF16 products, which touches every matmul-related file and kernel across multiple repos.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 188.98s (0:03:08)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/babylm_2025_submission_strict/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fee37571faa0c042895a812494601861d87a8a41 |
| tt-forge-models | b590ccacc49e928147f6aec062d34face664734d |
