# Remediation Summary: asym_w8w8_int8_static_per_tensor_tiny_llama/causal_lm/pytorch-asym_w8w8_int8_static_per_tensor_tiny_llama-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[asym_w8w8_int8_static_per_tensor_tiny_llama/causal_lm/pytorch-asym_w8w8_int8_static_per_tensor_tiny_llama-single_device-inference]

## Result
FAIL — `stablehlo.round_nearest_even` is not legalized in tt-mlir; compilation aborts with `ValueError: Error code: 13`

## Failure
Original reported failure:
```
E   RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/device/chip_helpers/silicon_sysmem_manager.cpp:326: tt::exception
```

Reproduced locally as:
```
loc("round-nearest-even.2"): error: failed to legalize operation 'stablehlo.round_nearest_even'
ValueError: Error code: 13
```

in `torch_xla._XLAC._xla_warm_up_cache` during compilation of the quantization forward pass.

## Root cause
**Layer: MLIR (tt-mlir)** — the StableHLO-to-TTNN lowering pass has no lowering rule for `stablehlo.round_nearest_even`.

The model is `nm-testing/asym-w8w8-int8-static-per-tensor-tiny-llama` (TinyLlama 1.1B with asymmetric W8A8 INT8 static per-tensor quantization via `compressed_tensors`). Its quantization forward pass calls `torch.round()` (via `round_to_quantized_type_args` in `compressed_tensors`) to snap scaled activations to the nearest integer before clamping. `torch.round` in PyTorch maps to `stablehlo.round_nearest_even` in the XLA/StableHLO IR. This op is not handled in any lowering pass in tt-mlir, so the compiler rejects it.

Minimal repro:
```python
import torch, torch_xla, torch_xla.core.xla_model as xm
x = torch.tensor([1.5, 2.5], dtype=torch.bfloat16).to(xm.xla_device())
result = torch.round(x)
torch_xla._XLAC._xla_warm_up_cache([result], [])
# → "failed to legalize operation 'stablehlo.round_nearest_even'"
# → ValueError: Error code: 13
```

## Fix
The fix must live in **tt-mlir** (compiler layer). Two approaches:

1. **Preferred** — Add a lowering pattern for `stablehlo.round_nearest_even` → `ttnn::round` in the StableHLO-to-TTNN lowering pass (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIR.cpp` or equivalent).  `ttnn::round` uses the same round-half-to-even semantics, so the numerical result is identical.

2. **Alternative** — Add a canonicalization/conversion pass that rewrites `stablehlo.round_nearest_even` to `stablehlo.round_nearest_afz` before TTNN lowering if `ttnn::round_nearest_afz` is already supported.  Note that this changes rounding semantics for midpoint values (e.g. 0.5→0 vs 0.5→1), which affects quantization accuracy.

Approach 1 is strictly correct; approach 2 is a lossy workaround.

## Verification
Test did not pass. No silicon run was performed.

## Files changed
None — this is a compiler-stack bug; no loader-layer fix is possible or appropriate.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e8db7752cf (branch: worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-start65-2) |
