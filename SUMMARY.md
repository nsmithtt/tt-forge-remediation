# Remediation Summary: dart_v2_moe_sft/causal_lm/pytorch-dart-v2-moe-sft-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[dart_v2_moe_sft/causal_lm/pytorch-dart-v2-moe-sft-single_device-inference]

## Result
SILICON_PASS

## Failure
Fatal Python error: Segmentation fault

Stack trace indicated crash at `torch_overrides.py:34` (`return func(*args, **(kwargs or {}))`) inside `__torch_function__`, called from `dynamo_bridge.py:partition_fx_graph_for_cpu_fallback`.

## Root cause
**Compiler frontend (tt-xla)** — `MixtralExperts.forward` in transformers 5.2.x uses `nonzero()` and a data-dependent for loop:

```python
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
for expert_idx in expert_hit:
    ...
```

`nonzero()` returns a tensor whose shape depends on the runtime values, making the loop count data-dependent. XLA/torch.compile cannot trace through data-dependent control flow — the op triggered a segfault in `partition_fx_graph_for_cpu_fallback` during StableHLO compilation.

The model is `p1atdev/dart-v2-moe-sft`, a Mixtral-based MoE with 4 experts and top-k=2. The same bug exists for `GptOssExperts`, and `torch_overrides.py` already had a monkey patch for it. Mixtral was not covered.

## Fix
Added `_mixtral_experts_forward` to `tt-xla/python_package/tt_torch/torch_overrides.py` with the same CPU/device split as the existing GptOss patch:

- **CPU path**: exact copy of the original per-expert loop (preserves numerics for PCC golden reference)
- **Device path**: dense bmm across all experts (`[E, T, H] @ [E, H, 2*inter]`), then one-hot routing weights combine — no `nonzero()`, no data-dependent control flow

Weight layout notes:
- `gate_up_proj`: stored as `[E, 2*inter, H]` (F.linear convention), transposed to `[E, H, 2*inter]` for bmm
- `down_proj`: stored as `[E, H, inter]`, transposed to `[E, inter, H]` for bmm

Also added `MixtralExperts.forward = _mixtral_experts_forward` monkey-patch registration (with `try/except ImportError`) to bypass the `@use_experts_implementation` decorator, matching the existing GptOss pattern.

This is not a forbidden workaround: no model trimming, no CPU offload, no shape changes. The full Mixtral model runs on device with a mathematically equivalent implementation.

## Verification
- pytest exit status: PASSED
- Wall-clock duration: ~70s (first run, includes compilation)
- e2e_perf avg_time: ~0.013s per inference call (cached)
- Hardware: Blackhole p150b

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `_mixtral_experts_forward` and monkey-patch registration

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f4fd3e1b6885606609f203af415a0b43250deef6 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
