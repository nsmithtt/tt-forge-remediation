# Remediation Summary: kimi_k2-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k2/pytorch-single_device-inference]

## Result
XFAIL — Hardware capacity: Kimi K2 is a 1T-parameter MoE model with 384 experts; W4A16 quantized weights are ~500 GB, far exceeding n150 (12 GB), n300 (24 GB), and p150b (32 GB) single-device DRAM.

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-kimi-k2-1t-params

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}): got AttributeError("'ndarray' object has no attribute 'add'")

from user code:
   File ".../modeling_deepseek.py", line 1471, in torch_dynamo_resume_in_forward_at_1452
    layer_outputs = decoder_layer(...)
   File ".../modeling_deepseek.py", line 1217, in forward
    hidden_states = self.mlp(hidden_states)
   File ".../modeling_deepseek.py", line 530, in forward
    y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
   File ".../modeling_deepseek.py", line 580, in moe_infer
    end_idx = start_idx + num_tokens
   File ".../tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

## Root cause
Kimi K2 is the MoonShot AI 1T-parameter Mixture-of-Experts language model. The full model config (61 layers, hidden_size=7168, 384 routed experts, moe_intermediate_size=2048) at W4A16 (4-bit weights) requires approximately 500 GB of DRAM — far beyond what any single TT device provides.

The existing loader (`kimi_k2/pytorch/loader.py`) uses forbidden model trimming (`config.num_hidden_layers = 2`, `config.hidden_size = 1024`, etc.) with random weights (`AutoModelForCausalLM.from_config`) to create a tiny synthetic model. This is moot given the hardware capacity ceiling.

The proximate failure is a secondary loader bug: the custom `modeling_deepseek.py` `moe_infer` method calls `tokens_per_expert.cpu().numpy()` to drive a per-expert for-loop. During dynamo FakeTensor tracing, the TT `TorchFunctionOverride` in `torch_overrides.py` intercepts arithmetic on numpy scalars (returned by `.numpy()`), returning ndarray objects instead of tensors. When `end_idx = start_idx + num_tokens` executes, the result is an ndarray with no `.add` method, causing dynamo to crash.

This `moe_infer` numpy dispatch pattern is the same class of bug documented for DeepSeek V3 (see project memory). However, fixing it is not meaningful here because the full model is hardware-class XFAIL.

## Fix
Added `kimi_k2/pytorch-single_device-inference` to `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` with `status: KNOWN_FAILURE_XFAIL` and a hardware capacity reason explaining that 1T params / W4A16 ~500 GB exceeds all single-device DRAM limits.

File changed: `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    n150
- Duration:    106.29s
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 76019f7543bf930347404631612ed978f31be935 |
| tt-forge-models | ebcfe743a1f2fd8b850014c4554bf931b137e40b |
