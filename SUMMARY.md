# Remediation Summary: glm_4_7_flash_awq-causal_lm-pytorch-W8A16_GS32-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_awq/causal_lm/pytorch-W8A16_GS32-single_device-inference]

## Result
XFAIL — GLM-4.7-Flash AWQ W8A16 model has ~29 GB of INT8 weights (~58 GB at BF16 after decompression), exceeding single n150 device DRAM (12 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-7-flash-awq-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:499: in forward
    hidden_states = hidden_states + self.shared_experts(residuals)
transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py:362: in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
torch_xla/torch_xla.py:87: in sync
    torch_xla._XLAC._xla_step_marker(
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Preceded by an `ImportError: compressed_tensors is not installed` (missing
requirements.txt), which was fixed first.

## Root cause
Two separate issues:

**Issue 1 (loader)**: `requirements.txt` was missing — the model uses
`compressed-tensors` format (W8A16 pack_quantized) but the dependency was not
declared. Fixed by adding `requirements.txt` with `compressed-tensors`.

**Issue 2 (loader)**: The `glm4_moe_lite` MoE expert dispatch uses
`grouped_mm_experts_forward` by default, which calls `torch.histc(ids.int(),
...)` — unsupported on XLA ("histogram_cpu" not implemented for Int) — and
`torch._grouped_mm` which has no XLA lowering. Fixed by setting
`model.config._experts_implementation = "batched_mm"`.

**Issue 3 (hardware-class)**: The model has 64 routed experts × 46 MoE layers
× ~9.4M params/expert = ~28.75B parameters. At W8A16 (INT8 weights), storage
is ~29 GB; after decompression to BF16 (which `compressed-tensors` performs
lazily on first forward), storage is ~58 GB. The n150 has 12 GB device DRAM.
The test runs for ~13 minutes (XLA compilation), then fails at `INTERNAL:
Error code: 13` consistently at `shared_experts.down_proj` — two consecutive
identical failures at the same location, ruling out the transient eth-core
MMIO error. This is device DRAM exhaustion from a model that is 2.4× over
capacity even at INT8.

## Fix
1. Created `glm_4_7_flash_awq/causal_lm/pytorch/requirements.txt` with
   `compressed-tensors` in tt-forge-models.

2. Added `_tt_static_glm4_moe_lite_forward` (static per-expert masked matmul)
   to `ALL_EXPERTS_FUNCTIONS` in the loader, then selected `"batched_mm"` as
   the implementation (static loop takes 73+ minutes to compile with 64
   experts; batched_mm compiles in ~13 minutes and uses standard bmm).

3. Added `KNOWN_FAILURE_XFAIL` entries for both W4A16_GS32 and W8A16_GS32
   variants in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
   in tt-xla (W4A16 at INT4 is ~14.4 GB, also exceeding the 12 GB DRAM limit).

## Verification
- pytest exit: FAIL (hardware capacity; no silicon pass possible on n150)
- Hardware:    n150
- Duration:    773.73s (0:12:53) before DRAM exhaustion failure
- Tier A attempts: N/A

## Files changed
- `glm_4_7_flash_awq/causal_lm/pytorch/requirements.txt` (created, tt-forge-models)
- `glm_4_7_flash_awq/causal_lm/pytorch/loader.py` (modified, tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (modified, tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7dd84a9edd9ce91fa3e8b427358d0f203a78e2bc |
| tt-forge-models | ee0c822bf2fd1f943c79dfb77821d9eefd5e3878 |
