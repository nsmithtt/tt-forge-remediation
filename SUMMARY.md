# Remediation Summary: academic_ds-causal_lm-pytorch-9B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[academic_ds/causal_lm/pytorch-9B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
deepseek-v3-grouped-mm-histc-int-not-cpu

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

While executing %histc : [num_users=1] = call_function[target=torch.ops.aten.histc.default](args = (%_to_copy_37, 64, 0, 63), kwargs = {})
Original traceback:
  File "transformers/integrations/moe.py", line 271, in grouped_mm_experts_forward
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
  File "tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

(The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` reported in the ticket is only the final printed line of pytest output; the actual failure is the `NotImplementedError` above.)

## Root cause
`ByteDance-Seed/academic-ds-9B` is a DeepSeek V3 architecture with 64 routed MoE experts.
transformers selects the `grouped_mm_experts_forward` path by default whenever
`torch.__version__ >= 2.9.0` (via `is_grouped_mm_available()`).  That function
at line 270 of `transformers/integrations/moe.py` branches on `device.type`:

```python
histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
num_tokens_per_expert = torch.histc(histc_input, ...)
```

When the model runs on the TT XLA device, `device.type` is not `"cpu"`, so
`expert_ids_g.int()` is selected.  The CPU `histogram_cpu` kernel (which handles
the fallback for TT) does not support integer input, causing the `NotImplementedError`.
Even if the `histc` issue were patched in isolation, `torch._grouped_mm` (called
next to compute the grouped matrix multiply) is also CUDA-only and would fail.

The bug is in the loader layer: the model needs a TT-compatible MoE forward that
avoids both `histc(int)` and `torch._grouped_mm`.

## Fix
Added `_deepseek_v3_naive_moe_forward` to
`python_package/tt_torch/torch_overrides.py` in `tt-xla` and monkey-patched
`DeepseekV3NaiveMoe.forward = _deepseek_v3_naive_moe_forward`, following the
same pattern already used for `GptOssExperts`.

The replacement forward has two paths:
- **CPU** (`device.type == "cpu"`): reproduces the original per-expert eager loop
  verbatim — used as the golden reference.
- **Device** (TT XLA): a static dense-BMM path with no data-dependent control
  flow and no CUDA-only ops, compilable by `torch.compile`.  Replicates hidden
  states across all E experts (`repeat`+`view`), computes gate-up and down
  projections via `torch.bmm`, and applies routing weights via a `scatter_`
  into a dense `[T, E]` matrix.

Commit: `5f8e101e092fc87d3047cc9613f6ddf18bc664ce` on branch
`remediation/academic_ds-causal_lm-pytorch-9B-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    166.13s (0:02:46)
- Tier A attempts: N/A

## Files changed
- `tt-xla: python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5f8e101e092fc87d3047cc9613f6ddf18bc664ce |
| tt-forge-models | 0f7b734348c38b2040cbf7f00e4e93a9e3a46aaa |
