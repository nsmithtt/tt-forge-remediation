# Remediation Summary: gigachat3_10b_a1_8b_bf16-causal_lm-pytorch-GigaChat3-10B-A1_8B-bf16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gigachat3_10b_a1_8b_bf16/causal_lm/pytorch-GigaChat3-10B-A1.8B-bf16-single_device-inference]

## Result
XFAIL — model is ~22 GB BF16 (DeepSeekV3 MoE, 10B params, 26 layers × 64 experts) but n150 has only 12 GB DRAM; INTERNAL Error code 13 on device execution

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
deepseekv3-moe-histc-int-xla

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
```
(from `grouped_mm_experts_forward` in `transformers/integrations/moe.py`),
followed after the batched_mm loader fix by:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
(from `_run_cached_graph` during device execution — DRAM capacity exceeded)

## Root cause
Two separate issues:

1. **Loader bug**: `grouped_mm_experts_forward` in transformers selects `.int()` for non-CPU devices when computing expert token counts via `torch.histc`. XLA does not implement `histogram_cpu` for integer tensors, raising `NotImplementedError`. Fix: switch to `batched_mm_experts_forward` by setting `model.config._experts_implementation = "batched_mm"` after `from_pretrained`.

2. **Hardware capacity ceiling**: GigaChat3-10B-A1.8B-bf16 uses a DeepSeekV3 MoE architecture (`DeepseekV3ForCausalLM`) with 64 routed experts per layer, 26 layers, hidden_size=1536, moe_intermediate_size=1280. The full model is approximately 22 GB in BF16 (expert weight tensor alone is `[64, 2560, 1536] × 26 × 4 bytes ≈ 16 GB`). The n150 device has approximately 12 GB DRAM. The model cannot be allocated on device, causing INTERNAL Error code 13 during `_run_cached_graph` execution.

## Fix
1. **Loader fix** in `tt_forge_models`: Set `model.config._experts_implementation = "batched_mm"` in `gigachat3_10b_a1_8b_bf16/causal_lm/pytorch/loader.py` after `AutoModelForCausalLM.from_pretrained(...)` to avoid `torch.histc` on integer dtype.
   - Branch: `remediation/gigachat3_10b_a1_8b_bf16-causal_lm-pytorch-GigaChat3-10B-A1_8B-bf16-single_device-inference` in `tenstorrent/tt-forge-models`

2. **Test config update** in `tt-xla`: Added `KNOWN_FAILURE_XFAIL` entry for this test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.
   - Branch: `remediation/gigachat3_10b_a1_8b_bf16-causal_lm-pytorch-GigaChat3-10B-A1_8B-bf16-single_device-inference` in `tenstorrent/tt-xla`

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    n150
- Duration:    341.18s (0:05:41)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gigachat3_10b_a1_8b_bf16/causal_lm/pytorch/loader.py` — set `_experts_implementation = "batched_mm"`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2c625a332eb41a0dbdbf827416cb9bba66790921 |
| tt-forge-models | 27ab64a78ff4679b8ac54fbf779a07d2cf9a2193 |
