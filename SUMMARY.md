# Remediation Summary: chinese_mixtral/causal_lm/pytorch-Chinese_Mixtral_Instruct-single_device-inference

## Skill version
5

## Test
`tests/runner/test_models.py::test_all_models_torch[chinese_mixtral/causal_lm/pytorch-Chinese_Mixtral_Instruct-single_device-inference]`

## Result
XFAIL — Mixtral-8x7B total weights (~46.7B params, ~93 GB bfloat16) exceed n150 single-device DRAM capacity

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
oom-mixtral-8x7b-exceeds-n150-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Fatal Python error: Segmentation fault
```

Followed (after loader fix) by:
```
TT_FATAL: Out of Memory: Not enough space to allocate 1879048192 B DRAM buffer across 8 banks,
where each bank needs to store 234881024 B, but bank size is 4273390016 B
(allocated: 4141406464 B, free: 131983552 B, largest free block: 123228608 B)
```

## Root cause

Two issues:

### 1. Segfault during graph partition (loader layer — fixed)

`MixtralExperts.forward` under transformers 5.2 (via the `@use_experts_implementation`
decorator) dispatches experts using a Python for-loop whose iteration count is determined at
runtime by `nonzero(expert_hit)`. XLA/torch.compile cannot statically trace a loop with a
dynamic trip count, so `partition_fx_graph_for_cpu_fallback` crashes with a segfault while
executing a node inside that loop.

Setting `model.config._experts_implementation = "batched_mm"` after `from_pretrained()`
switches `MixtralExperts.forward` to `batched_mm_experts_forward`, which uses only static
tensor operations (`einsum`, `scatter`) and is fully XLA-traceable.

### 2. OOM during device execution (hardware capacity — XFAIL)

After the segfault fix, compilation succeeds and the graph is dispatched to the n150 device.
The device OOMs immediately during weight upload. `hfl/chinese-mixtral-instruct` is a
Mixtral-8x7B sparse MoE model: 32 layers × 8 experts × 3 weight matrices
([14336, 4096] gate/up, [4096, 14336] down) gives approximately 46.7B total parameters.
In bfloat16 that is ~93 GB, far exceeding the n150's ~12 GB per-chip DRAM. This is a
genuine hardware capacity ceiling, not a compiler bug.

## Fix

### Applied (loader layer)

`chinese_mixtral/causal_lm/pytorch/loader.py` (in tt_forge_models,
`remediation/chinese-mixtral-causal-lm-pytorch-segfault`): added

```python
model.config._experts_implementation = "batched_mm"
```

after `AutoModelForCausalLM.from_pretrained(...)` to replace the dynamic
expert-dispatch for-loop with static batched-MM operations that XLA can trace.

### Applied (test config, hardware-class XFAIL)

`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
(in tt-xla, `remediation/chinese-mixtral-causal-lm-pytorch-Chinese_Mixtral_Instruct-single-device-inference`):
added

```yaml
chinese_mixtral/causal_lm/pytorch-Chinese_Mixtral_Instruct-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "OOM: Mixtral-8x7B total parameters (~46.7B, ~93 GB bfloat16) exceed n150 single-device DRAM capacity"
```

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    n150
- Duration:    290.78s (0:04:50)
- Tier A attempts: N/A

## Files changed
- `chinese_mixtral/causal_lm/pytorch/loader.py` (tt_forge_models — remediation/chinese-mixtral-causal-lm-pytorch-segfault)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 60dfb63a606c15e7c238acd2839372bcddf0472f |
| tt-forge-models | 642dfca2efd5a8d1b3584bbf8e3be6c8894586c3 |
