# Remediation Summary: gpt_oss_20b_counsel_mindbuddi_i1_gguf/causal_lm/pytorch-20B_Counsel_MindBuddi_i1_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_counsel_mindbuddi_i1_gguf/causal_lm/pytorch-20B_Counsel_MindBuddi_i1_GGUF-single_device-inference]

## Result
XFAIL — 20B model bfloat16 weights (~40 GB) exceed single-device DRAM (~32 GB); OOM during input tensor layout conversion at inference

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
device-dram-oom-20b-bfloat16

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two distinct failures observed across the investigation:

**Original failure (runs 2–3):** `Fatal Python error: Segmentation fault` inside
`partition_fx_graph_for_cpu_fallback` → `torch._ops.py:841` →
`tt_torch/torch_overrides.py:34` → `torch._ops.py:841`.

**After fix (run 5):** `TT_FATAL: Out of Memory: Not enough space to allocate
1061683200 B DRAM buffer across 8 banks, where each bank needs to store
132710400 B, but bank size is 4273390016 B (allocated: 4111273024 B,
free: 162116992 B, largest free block: 66355200 B)`

Stack at OOM:
```
tt::pjrt::FlatbufferLoadedExecutableInstance::prepareInputTensor
tt::pjrt::FlatbufferLoadedExecutableInstance::execute
xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice
```

## Root cause

**Segfault (fixed):** The `gpt-oss` GGUF model loads as `Qwen3MoeForCausalLM`
because `gpt_oss_swallow_120b_rl_v0_1_gguf/loader.py` monkey-patches the
GGUF architecture map and renames `model_type: gpt-oss → qwen3_moe` at import
time. The existing GPT-OSS monkey-patches in `torch_overrides.py` target
`GptOssExperts`, not `Qwen3MoeExperts`. In PyTorch ≥ 2.9 the
`@use_experts_implementation` decorator dispatches `Qwen3MoeExperts.forward`
to `grouped_mm_experts_forward` (uses `torch._grouped_mm`), which is not
XLA-aware. When `partition_fx_graph_for_cpu_fallback` dry-runs the FX graph on
real device tensors, `torch._grouped_mm` is called → segfault.

**OOM (hardware-class):** A 20-billion parameter model at bfloat16 requires
approximately 40 GB of device DRAM for weight constants. The single Wormhole
device provides approximately 32 GB (8 banks × ~4 GB). After loading ~30.6 GB
of model constants, the PJRT executor cannot allocate an additional ~1 GB input
buffer due to memory fragmentation (free: ~1.24 GB total, but largest
contiguous block only ~503 MB). This is a genuine hardware capacity ceiling,
not a compiler bug.

## Fix

**Loader / tt-xla fix (addresses segfault):**

`tt-xla` commit `a9ac127f0` — `python_package/tt_torch/torch_overrides.py`:
Added `_qwen3_moe_experts_forward` function that replaces
`grouped_mm_experts_forward` for `Qwen3MoeExperts`. The replacement uses a
static-shape dense `torch.bmm` path on device (XLA-compatible) and a
per-expert loop on CPU (golden reference). Monkey-patches
`Qwen3MoeExperts.forward` at import time, bypassing the
`@use_experts_implementation` decorator dispatch to `torch._grouped_mm`.

`tt-xla` commit `ecf716a1a` — `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
Marked `gpt_oss_20b_counsel_mindbuddi_i1_gguf/causal_lm/pytorch-20B_Counsel_MindBuddi_i1_GGUF-single_device-inference`
as `KNOWN_FAILURE_XFAIL` (hardware capacity ceiling).

## Verification
- pytest exit: FAIL (OOM — hardware capacity, see XFAIL disposition)
- Hardware:    wormhole (n150)
- Duration:    1157.29s (0:19:17)
- Tier A attempts: 1 (segfault fix succeeded; subsequent OOM is hardware-class)

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `_qwen3_moe_experts_forward` and `Qwen3MoeExperts.forward` monkey-patch
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ecf716a1a1864271469d8007f40e95f55a5dc72b |
| tt-forge-models | db6756f5c07d28dc821b7fc7bd9e66e2d2941bb1 |
