# Remediation Summary: mirothinker-causal_lm-pytorch-huihui_v1_5_30B_abliterated-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[mirothinker/causal_lm/pytorch-huihui_v1_5_30B_abliterated-single_device-inference]

## Result
XFAIL — model weights ~60 GB in bf16 exceed n150 single-device DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen3-moe-30b-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

The crash occurs in `tt_torch/torch_overrides.py:34 __torch_function__` while
`torch_xla/_dynamo/dynamo_bridge.py:762 partition_fx_graph_for_cpu_fallback`
runs the model on CPU to partition the FX graph.

## Root cause
**Primary: hardware capacity.** The model `huihui-ai/Huihui-MiroThinker-v1.5-30B-abliterated`
is a `Qwen3MoeForCausalLM` with 48 MoE layers, 128 experts per layer, hidden size 2048,
and MoE intermediate size 768. In bf16 the total weight footprint is approximately:

- Per MoE layer: 128 × (2 × 768 × 2048 + 2048 × 768) × 2 bytes ≈ 1.21 GB
- 48 layers × 1.21 GB = 58.0 GB
- Attention (48 layers) + embeddings ≈ 2.2 GB
- **Total ≈ 60 GB** — far exceeding the 12 GB DRAM of the n150.

**Secondary: incidental compilation crash.** Because PyTorch ≥ 2.9.0 is installed,
`PreTrainedModel.__init__` auto-selects `_experts_implementation = "grouped_mm"` for
`Qwen3MoeExperts`. The `grouped_mm_experts_forward` function in
`transformers/integrations/moe.py` calls:

```python
histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
num_tokens_per_expert = torch.histc(histc_input, ...)
```

During `partition_fx_graph_for_cpu_fallback` the tensors reside on the XLA device
(device.type != "cpu"), so the `.int()` path is taken. `torch.histc` for integer input
on CPU is not implemented and crashes (SIGSEGV or `NotImplementedError` depending on the
execution context).

This compilation crash is a secondary compiler-frontend bug — but even if it were fixed,
the model would still fail at device execution due to the hardware capacity limitation.
The compilation crash does not change the XFAIL disposition.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` with reason
citing hardware capacity.

No loader or compiler-stack changes were made. The incidental compilation bug
(`grouped_mm` + XLA device type + `torch.histc`) should be tracked as a separate
issue in tt-xla (`torch_overrides.py` needs a monkey patch for `Qwen3MoeExperts.forward`
analogous to the existing GptOss patches).

## Verification
- pytest exit: FAIL (SIGSEGV on first run; NotImplementedError with debug instrumentation)
- Hardware:    n150
- Duration:    ~415s (debug run) / ~12 min (first run before segfault)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 47bb14ee4f8a7218085fa9011af9848be0aec7e2 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
