# Remediation Summary: deepseek-deepseek_tng_r1t2_chimera-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_tng_r1t2_chimera/pytorch-single_device-inference]

## Result
FAIL — torch._grouped_mm (CUDA-only grouped GEMM) crashes TT XLA module compiler after histc fix is applied

## Stack layer
tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
grouped-mm-cuda-only-no-tt-xla-lowering

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
NotImplementedError: "histogram_cpu" not implemented for 'Int'

While executing %histc : [num_users=1] = call_function[target=torch.ops.aten.histc.default](args = (%_to_copy_62, 256, 0, 255), kwargs = {})
Original traceback:
  File ".../transformers/integrations/moe.py", line 271, in grouped_mm_experts_forward
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
  File ".../tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

## Root cause

**Primary bug (Tier A, fixed)**: PyTorch 2.9 introduced `torch.nn.functional.grouped_mm` / `torch._grouped_mm`, which enabled the transformers `grouped_mm` MoE experts implementation by default (via `PreTrainedModel.get_correct_experts_implementation` returning `"grouped_mm"` when `_experts_implementation=None`). The `grouped_mm_experts_forward` function in `transformers/integrations/moe.py` contains device-type branching:

```python
# With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
num_tokens_per_expert = torch.histc(histc_input, ...)
```

On TT hardware, `device.type` is neither `"cpu"` nor `"cuda"`, so the code takes the `int` branch. `torch.histc` has no TT or CUDA kernel available on this machine, so it falls back to `histogram_cpu` which does not support integer input.

A Tier A fix was applied in `tt_torch/torch_overrides.py`: the `TorchFunctionOverride.__torch_function__` now casts integer inputs to float before dispatching `torch.histc`, matching the CPU path semantics. This fix is applied at graph-capture time so the compiled graph contains `histc(float)` rather than `histc(int)`.

**Secondary bug (Tier B, blocks test)**: After the histc fix, `grouped_mm_experts_forward` reaches `torch._grouped_mm` — a PyTorch 2.9 CUDA-only grouped GEMM primitive. On TT hardware, the XLA module compiler (`module_builder.cc`) crashes with a fatal segfault when it encounters `_grouped_mm` in the compiled graph. `torch._grouped_mm` has no XLA/StableHLO lowering and no safe CPU fallback path for TT device tensors, causing the crash.

The three available MoE experts implementations all fail on TT:
- `grouped_mm`: histc int (fixed by Tier A) then `_grouped_mm` segfault (Tier B)
- `eager`: data-dependent `index_add_` → lowered to `ttnn::embedding_backward` → INTERNAL error
- `batched_mm`: `gate_up_proj[expert_ids_clamped]` 3D tensor indexing → lowered to `ttnn::embedding` → circular buffer CB overflow (row size 4096×1024 exceeds L1)

## Fix
**Tier A fix applied** in `tt-xla`, file `python_package/tt_torch/torch_overrides.py`:
- Added a guard in `TorchFunctionOverride.__torch_function__` that converts integer inputs to float before calling `torch.histc`, so the CPU float histc kernel handles the fallback correctly.
- Commit: `002044358e0893b90e866dac24fa8f9b49933c54` on branch `remediation/deepseek-deepseek_tng_r1t2_chimera-pytorch-single_device-inference`

**Proposed fix for Tier B**: Add a XLA/StableHLO lowering for `torch._grouped_mm` in the tt-mlir compiler, OR add a safe device→CPU transfer + CPU fallback path in tt-xla's PJRT bridge for grouped GEMM ops. Without this, all three MoE implementations fail on TT hardware.

## Tier B justification (FAIL with Tier=B only)
new-infrastructure

`torch._grouped_mm` (PyTorch 2.9 grouped GEMM) has no XLA/StableHLO lowering in tt-mlir. Adding it requires implementing a new op in the MLIR compiler stack. Additionally, the fallback `eager` and `batched_mm` experts implementations each trigger separate TT kernel failures (embedding_backward INTERNAL error and embedding CB overflow respectively), indicating that the full MoE forward pass with `n_routed_experts=256` and `moe_intermediate_size=2048` is not yet supported on TT.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~104s (to segfault after histc fix)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — cast integer histc inputs to float in `TorchFunctionOverride.__torch_function__`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 002044358e0893b90e866dac24fa8f9b49933c54 |
| tt-forge-models | 6359be9bdc6a1a0830ba5cd0ed6396d8f611a216 |
