# Remediation Summary: gemma_3n_gguf-causal_lm-pytorch-E4B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3n_gguf/causal_lm/pytorch-E4B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
ttnn-pow-scalar-negative-float-rejected

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
loc("power.302"): error: 'ttnn.pow_scalar' op exponent must be non-negative; but got -5.000000e-01
ValueError: Error code: 13
```

## Root cause
Three bugs, fixed in order:

**Bug 1 (loader)**: `get_gguf_hf_weights_map` in transformers knew `gemma3_text → gemma3` but not
`gemma3n_text → gemma3n`. When loading Gemma3n weights, the GGUF utility raised
`NotImplementedError: Unknown gguf model_type: gemma3n_text in gguf-py` (line 529 of
`transformers/modeling_gguf_pytorch_utils.py`). The existing monkey-patch in the loader set
`model_type = "gemma3n_text"` but never registered the reverse mapping needed for
`get_gguf_hf_weights_map`.

**Bug 2 (loader)**: `load_shard_spec` iterated every attention layer and accessed
`layer.self_attn.k_proj.weight` unconditionally. Gemma3n uses KV-sharing — the last N layers
reuse KV states from an earlier layer and have no `k_proj` / `v_proj` of their own, raising
`AttributeError: 'Gemma3nTextAttention' object has no attribute 'k_proj'`.

**Bug 3 (tt-mlir)**: `PowScalarOp::verify()` rejected all negative float exponents with
`exponent must be non-negative; but got -5.000000e-01`. Gemma3n's RMSNorm computes
`x * (x.pow(-0.5))` (rsqrt). The SFPU hardware kernel (`calculate_unary_power`) handles
negative float exponents via an `IS_POSITIVE_EXPONENT` template parameter — the verifier
restriction was overly conservative and incorrect.

## Fix

**Bug 1** — `tt-xla/third_party/tt_forge_models/gemma_3n_gguf/causal_lm/pytorch/loader.py`:
Added step 6 to `_patch_transformers_gemma3n_gguf()`: monkey-patches
`gguf_utils.get_gguf_hf_weights_map` to remap `model_type="gemma3n_text"` → `"gemma3n"` before
delegating to the original. Commit: `0f881827bc` on tt-forge-models branch
`worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-range-300-200-1`.

**Bug 2** — same file, `load_shard_spec`:
Wrapped `k_proj`/`v_proj` accesses in `if hasattr(layer.self_attn, "k_proj"):`.
Commit: `8cac31bc9c` on same branch.

**Bug 3** — `tt-mlir/lib/Dialect/TTNN/IR/TTNNOps.cpp`, `PowScalarOp::verify()`:
Removed the `< 0.0` check on float exponents; the function now returns `success()` for any
`FloatAttr`, since the SFPU kernel supports negative floats. The `< 0` check for integer
exponents is retained (integer negative powers are genuinely unsupported).
Commit: `dfd3ef528` on branch
`remediation/gemma_3n_gguf-causal_lm-pytorch-E4B_IT_GGUF-single_device-inference` in tt-mlir.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    838.70s (0:13:58)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma_3n_gguf/causal_lm/pytorch/loader.py`
- `tt-mlir/lib/Dialect/TTNN/IR/TTNNOps.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | dfd3ef5282325eb15522c9d1cb8c52fdff0992ea |
| tt-xla          | b0bc3475d3a33c06e82f0954845c67f29cc36791 |
| tt-forge-models | 8cac31bc9c (on tt-xla branch worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-range-300-200-1) |
