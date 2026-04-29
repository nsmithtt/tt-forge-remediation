# Remediation Summary: deepseek_r1_zero-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_zero/pytorch-single_device-inference]

## Result
XFAIL — DeepSeek-R1-Zero expert weights (~16 GB BF16 for 256 experts × moe_intermediate_size=2048) exceed single-device DRAM on n150 (12 GB); the full 671B real model far exceeds all single-device DRAM.

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-deepseek-r1-zero-expert-weights-exceed-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Initial failure: `NotImplementedError: "histogram_cpu" not implemented for 'Int'`

```
transformers/integrations/moe.py line 271, in grouped_mm_experts_forward
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
```

After loader fix: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` (OOM)

## Root cause

Two bugs in sequence:

**1. Loader bug — grouped_mm histc on Int under TT XLA device**

`transformers` 5.x defaults to `grouped_mm` experts implementation on PyTorch 2.9+ when `torch.nn.functional.grouped_mm` is available. The `grouped_mm_experts_forward` function (moe.py:270) picks Int dtype for `histc_input` when `device.type != "cpu"`. Under TT's XLA device (`device.type = "xla"`), it passes an Int tensor to `torch.histc`. The TT backend's `__torch_function__` falls through to the CPU implementation, which raises `NotImplementedError: "histogram_cpu" not implemented for 'Int'`.

Fix: set `model.config._experts_implementation = "batched_mm"` after model creation. The `batched_mm` path uses vectorized matmul without any `torch.histc` call.

**2. Hardware capacity — expert weight OOM on n150**

The loader preserves the real model's MoE configuration: 256 routed experts with `moe_intermediate_size = 2048`. Expert weight sizes per MoE layer:
- `gate_up_proj`: [256, 4096, 1024] = 1.07B elements × 2 bytes ≈ 2 GB
- `down_proj`: [256, 1024, 2048] = 537M elements × 2 bytes ≈ 1 GB

For 5 MoE layers: ~16.1 GB of expert weights alone, exceeding n150 DRAM (12 GB). The full DeepSeek-R1-Zero model has 671B parameters (≈1.34 TB at BF16), far exceeding all single TT device DRAM.

## Fix

**Loader fix (loader.py):**
Added `model.config._experts_implementation = "batched_mm"` and `model.config.use_cache = False` after `AutoModelForCausalLM.from_config(...)` to avoid the CUDA-specific `grouped_mm` path that uses `torch.histc` on Int tensors.

**Test config (test_config_inference_single_device.yaml):**
Added `KNOWN_FAILURE_XFAIL` entry for `deepseek/deepseek_r1_zero/pytorch-single_device-inference` due to hardware capacity ceiling.

## Verification
- pytest exit: FAIL (INTERNAL: Error code: 13 — OOM after histc fix)
- Hardware:    n150
- Duration:    183.18s (second run, after histc fix)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1_zero/pytorch/loader.py` — add `_experts_implementation = "batched_mm"` and `use_cache = False`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add `KNOWN_FAILURE_XFAIL` entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7d9a87768fb98480f82b9a93982bdb5b06623780 |
| tt-forge-models | bf1ba5e1234c30f77a5c04efb63a1643af8f3fb7 |
